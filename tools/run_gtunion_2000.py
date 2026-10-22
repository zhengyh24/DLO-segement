"""Explicit GT-union TRAINING experiment; immutable old run, real-frontend validation."""
import sys,os,json,shutil,time,subprocess,traceback
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from curvi_identity.io import read_json,read_rgb,read_mask
from curvi_identity.config import Config
from run_curvi_2000 import save,sha
OLD=ROOT/'run';OUT=ROOT/'run_gtunion_2000_20261014'

def union(masks,shape):
    result=np.zeros(shape,bool)
    for mask in masks:result|=mask
    return result

def validate(obj,cfg):
    assert obj['kind']=='gt_union_train_predicted_validation_v1'
    assert obj['real_subset_used'] is False
    for field in ('small_size','large_size','min_arc','junction_radius','max_gap','max_candidates','purity'):
        assert obj['config'][field]==getattr(cfg,field)
    groups={};hashes={}
    for im in obj['images']:
        assert im['split'] in ('train','validation')
        assert im['input_origin']==('gt_binary_union' if im['split']=='train' else 'frozen_predicted_foreground')
        for table,key in ((groups,im['group_id']),(hashes,im['rgb_sha256'])):
            assert key not in table or table[key]==im['split']
            table[key]=im['split']
    for split in ('train','validation'):
        assert {r['label'] for r in obj['records'] if r['split']==split}=={0,1}

def setup():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return
    for name in ('groups.json','config.json','inference_inputs.json'):shutil.copyfile(OLD/name,OUT/name)
    cfg=read_json(OUT/'config.json');cfg['dataset_root']='offline_labels_not_network_input';save(OUT/'config.json',cfg)
    save(OUT/'protocol.json',dict(training_images=2000,validation_images=300,epochs=10,
        training_foreground='OR union of all GT instance masks, binary only',
        input_channels='RGB + binary union + union-derived skeleton + union-derived candidate A/B partitions',
        gt_instance_identity='Offline supervision only; never used to build graph or candidate inputs',
        checkpoint_selection='minimum bidirectional BCE on original predicted-foreground validation pairs',
        ideal_validation='oracle foreground diagnostic only; not deployable inference or independent test',
        old_run=str(OLD),from_scratch=True,split_sha256=sha(OLD/'groups.json'),
        limitation='Union skeleton cannot separate all parallel adhesions; ideal-to-predicted foreground domain gap'))

def export():
    from curvi_identity.data import decode_masks
    rows={r['file_name']:r for r in read_json(OLD/'selected_labels.json')['records']}
    inputs=read_json(OLD/'inference_inputs.json')['records'];binary=[]
    dest=OUT/'binary_union';dest.mkdir(exist_ok=True)
    for r in inputs:
        masks=decode_masks(rows[r['file_name']]['masks_rle'],(1024,1024))
        fg=union(masks,(1024,1024));path=dest/(r['name']+'.png')
        Image.fromarray(fg.astype(np.uint8)*255).save(path)
        binary.append(dict(name=r['name'],image=r['image'],file_name=r['file_name'],
            rgb_sha256=r['rgb_sha256'],binary_foreground=str(path),binary_sha256=sha(path)))
    save(OUT/'binary_inputs.json',dict(records=binary,instance_identity_present=False,
         oracle_foreground=True,warning='GT-derived binary geometry; no instance identity'))
    print('BINARY_UNIONS_EXPORTED',len(binary),flush=True)

def labels():
    from curvi_identity.data import decode_masks,arc_labels
    from curvi_identity.graph import Graph
    cfg=Config.read(OUT/'config.json');groups=read_json(OUT/'groups.json')
    inputs={r['file_name']:r for r in read_json(OUT/'binary_inputs.json')['records']}
    rows={r['file_name']:r for r in read_json(OLD/'selected_labels.json')['records']}
    old=read_json(OLD/'prepared/manifest.json');images=[];records=[];ideal_images=[];ideal_records=[]
    for i,r in enumerate(sorted(groups['records'],key=lambda r:(r['shard'],r['row'])),1):
        inp=inputs[r['file_name']];dest=OUT/'prepared/graphs'/inp['name'];obj=read_json(dest/'graph.json')
        with np.load(dest/'arrays.npz') as z:part,sk,arcs=z['partition'],z['skeleton'],z['arcs']
        nodes={int(k):dict(coords=np.argwhere(arcs==k)) for k in np.unique(arcs) if k}
        graph=Graph(part>0,sk,arcs,part,nodes,obj['candidates'],obj['audit'])
        assigned,stats=arc_labels(graph,decode_masks(rows[r['file_name']]['masks_rle'],part.shape),cfg.purity)
        rr=[]
        for index,c in enumerate(graph.candidates):
            if c['a'] not in assigned or c['b'] not in assigned:continue
            rr.append(dict(file_name=r['file_name'],group_id=r['group_id'],split=r['split'],
                rgb_sha256=inp['rgb_sha256'],label=int(assigned[c['a']]==assigned[c['b']]),
                candidate=c['key'],candidate_index=index,graph_item=str(dest/'graph.json')))
        im=dict(file_name=r['file_name'],group_id=r['group_id'],split=r['split'],rgb_sha256=inp['rgb_sha256'],
                graph=graph.audit,supervision=stats,pairs=len(rr),positive=sum(x['label'] for x in rr),input_origin='gt_binary_union')
        ideal_images.append(im);ideal_records.extend(rr)
        if r['split']=='train':images.append(im);records.extend(rr)
        if i%50==0:print({'labelled':i,'total':len(groups['records'])},flush=True)
    for im in old['images']:
        if im['split']=='validation':images.append(dict(im,input_origin='frozen_predicted_foreground'))
    for r in old['records']:
        if r['split']=='validation':records.append(dict(r,graph_item=str(OLD/'prepared'/r['graph_item'])))
    manifest=dict(schema=1,kind='gt_union_train_predicted_validation_v1',storage_kind='native_graph_lazy_v1',
        config=cfg.to_dict(),groups=groups,images=images,records=records,real_subset_used=False,
        training_performed=False,gt_instance_identity_in_input=False)
    validate(manifest,cfg);save(OUT/'prepared/manifest.json',manifest)
    save(OUT/'prepared/ideal_manifest.json',dict(manifest,kind='gt_union_oracle_diagnostic_v1',images=ideal_images,records=ideal_records))
    summary={}
    for split in ('train','validation'):
        rr=[r for r in records if r['split']==split];ii=[im for im in images if im['split']==split]
        summary[split]=dict(images=len(ii),pairs=len(rr),positive=sum(x['label'] for x in rr),images_without_pairs=sum(x['pairs']==0 for x in ii))
    save(OUT/'preparation_summary.json',summary);print(summary,flush=True)

def check():
    from curvi_identity.graph import build_graph
    from curvi_identity.features import extract_pair
    from curvi_cloud_data import CloudPairDataset
    cfg=Config.read(OUT/'config.json')
    # Change instance count/membership while keeping binary union identical.
    a=np.zeros((128,128),bool);b=a.copy();a[61:67,8:120]=1;b[8:120,61:67]=1
    variants=[[a,b],[b,a],[a|b],[(a|b)&(np.indices(a.shape)[1]<64),(a|b)&(np.indices(a.shape)[1]>=64)]]
    graphs=[build_graph(union(v,a.shape),cfg) for v in variants];base=graphs[0]
    assert base.candidates
    tested=0
    for graph in graphs[1:]:
        assert graph.candidates==base.candidates
        np.testing.assert_array_equal(graph.partition,base.partition)
        np.testing.assert_array_equal(graph.skeleton,base.skeleton)
        for c in base.candidates:
            x=extract_pair(np.zeros((*a.shape,3),np.uint8),base,c,cfg)
            y=extract_pair(np.zeros((*a.shape,3),np.uint8),graph,c,cfg)
            for xx,yy in zip(x,y):np.testing.assert_array_equal(xx,yy)
            tested+=1
    ds=CloudPairDataset(OUT/'prepared/manifest.json','train',cfg.seed)
    for i in range(min(16,len(ds))):
        small,large,g,y=ds.raw(i)
        assert small.shape==(7,128,128) and large.shape==(7,256,256)
        assert set(np.unique(small[3:])).issubset({0,1}) and set(np.unique(large[3:])).issubset({0,1})
        assert np.array_equal(g[-3:],np.zeros(3))
    save(OUT/'input_isolation_check.json',dict(passed=True,invariant_candidate_feature_comparisons=tested,
        variants='reordered IDs, merged instances, different ownership with exactly the same binary union',
        real_pairs_checked=min(16,len(ds)),gt_identity_in_input=False,
        qualification='GT binary shape is intentionally used; this is not a GT-free foreground experiment'))
    print('GT_IDENTITY_INPUT_ISOLATION_PASS',tested,flush=True)

def train():
    import train_curvi_2000 as trainer
    from curvi_cloud_data import CloudPairDataset
    assert read_json(OUT/'input_isolation_check.json')['passed']
    trainer.PairDataset=CloudPairDataset;trainer.validate_manifest=validate
    trainer.main(OUT)

def evaluate_all():
    import evaluate_gtunion_2000 as evaluator
    import run_cloud_curvi as old_adapter
    evaluator.selected_rows=old_adapter.selected_labels
    binary=read_json(OUT/'binary_inputs.json')['records']
    for regime in ('predicted','ideal'):
        out=OUT/f'evaluation_{regime}';out.mkdir(exist_ok=True);(out/'fit').mkdir(exist_ok=True);(out/'prepared').mkdir(exist_ok=True)
        for name in ('config.json','groups.json','inference_inputs.json'):shutil.copyfile(OUT/name,out/name)
        shutil.copyfile(OUT/'fit/best.pt',out/'fit/best.pt')
        shutil.copyfile(OUT/('prepared/manifest.json' if regime=='predicted' else 'prepared/ideal_manifest.json'),out/'prepared/manifest.json')
        if regime=='predicted':shutil.copyfile(OLD/'predictions.json',out/'predictions.json')
        else:save(out/'predictions.json',dict(records=[dict(file_name=r['file_name'],foreground=r['binary_foreground'],rgb_sha256=r['rgb_sha256'],gt_used=True) for r in binary]))
        save(out/'input_regime.json',dict(oracle_foreground=regime=='ideal',training_foreground='GT binary union'))
        subprocess.run([sys.executable,'-u','tools/predict_gtunion_2000.py','--output',str(out)],cwd=ROOT,check=True)
        # Evaluator checks identity isolation separately from oracle foreground provenance.
        evaluator.main(out)
        result=read_json(out/'evaluation.json');result.update(training_foreground='GT binary union',
            evaluation_foreground=regime,oracle_foreground=regime=='ideal',independent_test=False)
        result['scope']='Single-seed condition validation; '+('GT-union oracle diagnostic, NOT actual segmentation performance' if regime=='ideal' else 'original predicted foreground; used for checkpoint selection')
        save(out/'evaluation.json',result)
        report=(out/'results.md').read_text()
        report=report.replace('SAM/HQ冻结，原生1024×1024输入，训练仅更新PairNet。','训练输入为GT二值并集；验证前景为'+regime+'；原生1024×1024，训练仅更新PairNet。')
        (out/'results.md').write_text('# Input provenance\n\n'+result['scope']+'\n\n'+report)

def step(name,command):
    for path,digest in read_json(OUT/'source_hashes.json').items():assert sha(ROOT/path)==digest,path
    log=OUT/'logs'/f'{name}_{int(time.time())}.log'
    with log.open('w') as f:
        child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        save(OUT/'status.json',dict(stage=name,pid=os.getpid(),child_pid=child.pid,log=str(log),time=time.time()))
        rc=child.wait()
    if rc:raise RuntimeError(f'{name} failed ({rc}); see {log}')

def run():
    setup();(OUT/'logs').mkdir(exist_ok=True)
    freeze=OUT/'source_hashes.json'
    if not freeze.exists():
        paths=list((ROOT/'curvi_identity').glob('*.py'))+list((ROOT/'tools').glob('*.py'))
        save(freeze,{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
        for p in paths:
            dst=OUT/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    for stage,marker in [('export','binary_inputs.json'),('graphs','graphs_complete.json'),('labels','prepared/manifest.json'),('check','input_isolation_check.json'),('train','fit/completion.json'),('evaluate','evaluation_complete.json')]:
        if (OUT/marker).exists():continue
        args=['tools/gtunion_graphs.py',str(OUT)] if stage=='graphs' else ['tools/run_gtunion_2000.py',stage]
        step(stage,[sys.executable,'-u',*args])
        if stage in ('graphs','evaluate'):save(OUT/marker,dict(complete=True))
    save(OUT/'status.json',dict(stage='complete',time=time.time()))

if __name__=='__main__':
    os.chdir(ROOT)
    try:
        stage=sys.argv[1]
        if stage=='run':
            import fcntl
            OUT.mkdir(exist_ok=True)
            with (OUT/'run.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
        else:{'export':export,'labels':labels,'check':check,'train':train,'evaluate':evaluate_all}[stage]()
    except Exception:
        OUT.mkdir(exist_ok=True);save(OUT/'failure.json',dict(error=traceback.format_exc(),time=time.time()))
        save(OUT/'status.json',dict(stage='failed',time=time.time()));raise
