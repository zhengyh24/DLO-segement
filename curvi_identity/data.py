"""Offline supervised preparation. Never imported by inference."""
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from .config import Config
from .features import extract_pair,normalize_patch,reverse_features,augment
from .graph import build_graph
from .io import read_json,write_json,new_directory,read_mask,rgb_hash,resolve_relative

ROOT=Path(__file__).resolve().parents[1]

def parquet_module():
    sys.path.insert(0,str(ROOT/'third_party/wireseg_deps'))
    import pyarrow.parquet as pq
    return pq

def decode_masks(value,shape):
    sys.path.insert(0,str(ROOT/'third_party/padlos_deps'))
    from pycocotools import mask as M
    rles=json.loads(value) if isinstance(value,str) else value
    masks=[]
    for item in rles:
        item=dict(item)
        if isinstance(item['counts'],list):item=M.frPyObjects(item,*item['size'])
        mask=M.decode(item).astype(bool)
        if mask.shape!=shape:raise ValueError('RLE coordinates differ from native RGB')
        masks.append(mask)
    return masks

def arc_labels(graph,masks,purity):
    # Overlapping GT has no unique visible identity and cannot label an arc.
    owner=np.zeros(graph.visible.shape,np.int32);count=np.zeros(graph.visible.shape,np.uint16)
    for i,mask in enumerate(masks,1):count+=mask;owner[mask]=i
    owner[count!=1]=0
    assigned={};stats={'supervised_arcs':0,'ignored_arcs':0,'overlap_gt_pixels':int((count>1).sum())}
    for ident,node in graph.nodes.items():
        coords=node['coords'];labels=owner[coords[:,0],coords[:,1]]
        hist=np.bincount(labels,minlength=len(masks)+1);hist[0]=0
        winner=int(hist.argmax()) if len(hist) else 0
        if winner and hist[winner]/len(labels)>=purity:assigned[ident]=winner;stats['supervised_arcs']+=1
        else:stats['ignored_arcs']+=1
    return assigned,stats

def group_map(path):
    obj=read_json(path);records=obj['records'];mapping={};seen={}
    if not obj.get('grouping_basis'):raise ValueError('Document actual grouping_basis; scene_family alone is insufficient')
    kind=obj.get('grouping_kind','verified_scene')
    if kind not in ('verified_scene','condition_holdout_pilot'):raise ValueError('Unknown grouping_kind')
    if kind=='condition_holdout_pilot' and obj.get('independent_scene_split') is not False:
        raise ValueError('Condition pilot must explicitly disclaim independent_scene_split')
    for r in records:
        name,group,split=r['file_name'],r['group_id'],r['split']
        if not isinstance(group,str) or not group.strip() or split not in ('train','validation','test'):
            raise ValueError('Each image needs a real group_id and train/validation/test split')
        if name in mapping:raise ValueError(f'Duplicate file_name in groups: {name}')
        if kind=='verified_scene' and group.startswith('condition:'):
            raise ValueError('Condition groups require an explicit condition_holdout_pilot protocol')
        if kind=='condition_holdout_pilot':
            condition='/'.join(name.replace('\\','/').split('/')[:-1])
            if group!='condition:'+condition or len(condition.split('/'))!=2:
                raise ValueError('Condition group must match the observed family/count parent path')
        if group in seen and seen[group]!=split:raise ValueError(f'Group crosses splits: {group}')
        if group in ('data_center','flying_wire','wire_on_plane','real_easy','real_hard'):
            raise ValueError('scene_family is not a simulation run / acquisition group')
        seen[group]=split;mapping[name]=r
    if not mapping:raise ValueError('Empty group mapping')
    return obj,mapping

def iter_rows(root,columns):
    root=Path(root);files=sorted((root/'data').glob('*.parquet'))
    if not files:raise ValueError('Expected completed local data/*.parquet shards')
    pq=parquet_module()
    for p in files:
        if p.resolve().is_relative_to((root/'real').resolve()):raise ValueError('Real subset is not a training source')
        parquet=pq.ParquetFile(p)
        missing=set(columns)-set(parquet.schema_arrow.names)
        if missing:raise ValueError(f'{p.name}: missing columns {sorted(missing)}')
        row_index=0
        for batch in parquet.iter_batches(batch_size=8,columns=columns):
            for row in batch.to_pylist():
                yield p,row_index,row;row_index+=1

def decoded_rgb(value):
    raw=value.get('bytes') if isinstance(value,dict) else value
    if not isinstance(raw,bytes):raise ValueError('Parquet must contain embedded RGB bytes')
    with Image.open(io.BytesIO(raw)) as im:
        if im.getexif().get(274,1)!=1:raise ValueError('Synthetic importer requires EXIF=1; do not guess GT orientation')
        return np.asarray(im.convert('RGB')).copy()

def inventory(root,output):
    out=new_directory(output);records=[]
    for p,i,row in iter_rows(root,['file_name','scene_family']):
        records.append(dict(row,shard=p.name,row=i))
    write_json(out/'inventory.json',{'records':records,'ground_truth_read':False,
               'note':'Supply externally verified simulation/acquisition groups; no automatic random split.'})
    return {'images':len(records),'output':str(out)}

def export_images(root,groups,output):
    obj,mapping=group_map(groups);out=new_directory(output);records=[];seen=set()
    for p,i,row in iter_rows(root,['file_name','image']):
        name=row['file_name']
        if name not in mapping or mapping[name]['split']=='test':continue
        if name in seen:raise ValueError(f'Duplicate source image: {name}')
        seen.add(name);rgb=decoded_rgb(row['image']);dest=out/f'{len(records):06d}.png';Image.fromarray(rgb).save(dest)
        records.append({'file_name':name,'rgb_path':dest.name,'rgb_sha256':rgb_hash(rgb),
                        'foreground':None,'producer':None,'gt_used':None})
    expected={k for k,v in mapping.items() if v['split']!='test'}
    if seen!=expected:raise ValueError(f'{len(expected-seen)} mapped images missing from completed shards')
    write_json(out/'predictions.template.json',{'records':records})
    write_json(out/'groups.json',obj)
    return {'images':len(records),'output':str(out),'training_performed':False}

def prepare(root,groups,predictions,output,config=Config()):
    group_obj,mapping=group_map(groups);predictions=Path(predictions)
    pred_obj=read_json(predictions);pred={r['file_name']:r for r in pred_obj['records']}
    if len(pred)!=len(pred_obj['records']):raise ValueError('Duplicate prediction file_name')
    out=new_directory(output);(out/'pairs').mkdir();records=[];audits=[];seen=set();hash_splits={}
    for shard,row_index,row in iter_rows(root,['file_name','image','masks_rle','scene_family']):
        name=row['file_name']
        if name not in mapping or mapping[name]['split']=='test':continue
        if row['scene_family'].startswith('real'):raise ValueError('Real regression is excluded from training preparation')
        if name in seen:raise ValueError(f'Duplicate source file_name: {name}')
        seen.add(name);split=mapping[name]['split'];group=mapping[name]['group_id']
        rgb=decoded_rgb(row['image']);digest=rgb_hash(rgb)
        if digest in hash_splits and hash_splits[digest]!=split:raise ValueError('Identical RGB crosses train/validation')
        hash_splits[digest]=split
        cache=pred.get(name)
        if not cache or not cache.get('foreground') or not cache.get('producer') or cache.get('gt_used') is not False:
            raise ValueError(f'{name}: need explicit prediction foreground/producer/gt_used=false')
        if cache.get('rgb_sha256')!=digest:raise ValueError(f'{name}: prediction RGB hash mismatch')
        fg=read_mask(resolve_relative(predictions.parent,cache['foreground']))>0
        if fg.shape!=rgb.shape[:2]:raise ValueError(f'{name}: prediction is not native resolution')
        graph=build_graph(fg,config)  # GT never participates in candidates/crops.
        assigned,stats=arc_labels(graph,decode_masks(row['masks_rle'],fg.shape),config.purity)
        support=read_json(resolve_relative(predictions.parent,cache['support'])) if cache.get('support') else {}
        n=positive=0
        for c in graph.candidates:
            if c['a'] not in assigned or c['b'] not in assigned:continue
            label=int(assigned[c['a']]==assigned[c['b']]);s,l,v=extract_pair(rgb,graph,c,config,support.get(c['key']))
            relative=f'pairs/{len(records):08d}.npz'
            np.savez_compressed(out/relative,small=s,large=l,geometry=v,label=np.float32(label))
            records.append({'path':relative,'file_name':name,'group_id':group,'split':split,
                            'rgb_sha256':digest,'label':label,'candidate':c['key']})
            n+=1;positive+=label
        audits.append({'file_name':name,'shard':str(shard.resolve()),'row':row_index,'split':split,'group_id':group,
                       'rgb_sha256':digest,'scene_family':row['scene_family'],'prediction_producer':cache['producer'],
                       'graph':graph.audit,'supervision':stats,'pairs':n,'positive':positive,
                       'ignored_pairs':len(graph.candidates)-n})
    expected={k for k,v in mapping.items() if v['split']!='test'}
    if seen!=expected:raise ValueError(f'{len(expected-seen)} mapped images missing; refuse partial dataset preparation')
    if not records:raise ValueError('No supervised candidate pairs. Check foreground and grouping; no empty manifest emitted')
    manifest={'schema':1,'kind':'predicted_native_pair_supervision','config':config.to_dict(),
              'groups':group_obj,'prediction_manifest':pred_obj,'images':audits,'records':records,
              'training_performed':False,'real_subset_used':False}
    write_json(out/'manifest.json',manifest)
    return {'pairs':len(records),'images':len(audits),'positive':sum(r['label'] for r in records),'output':str(out)}

class PairDataset:
    def __init__(self,manifest,split,seed=0):
        if split not in ('train','validation'):raise ValueError('Training loader refuses test split')
        self.path=Path(manifest);obj=read_json(self.path)
        self.records=[r for r in obj['records'] if r['split']==split]
        self.training=split=='train';self.rng=np.random.default_rng(seed)
        if not self.records:raise ValueError(f'No pairs for {split}')
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        import torch
        r=self.records[index]
        with np.load(self.path.parent/r['path'],allow_pickle=False) as z:
            s,l,g=z['small'],z['large'],z['geometry'];label=float(z['label'])
        if label!=r['label'] or label not in (0,1):raise ValueError('Pair label differs from manifest')
        if self.training and self.rng.random()<.5:s,l,g=reverse_features(s,l,g)
        s,l=normalize_patch(s),normalize_patch(l)
        if self.training:s,l,g=augment(s,l,g,self.rng)
        return tuple(torch.from_numpy(a.copy()) for a in (s,l,g))+ (torch.tensor(label,dtype=torch.float32),)
