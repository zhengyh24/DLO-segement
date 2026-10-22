"""Frozen small development/check sets for a second accuracy iteration."""
from collections import Counter
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from cable_evidence_fusion import load_cached,complete_visible,proposal_graph,Proposal,fuse_residual,proposal_features
from cable_combination_backend import stitch_ids,global_path_cover
from run_evidence_fusion_experiment import SOURCE,sha,instance_scores,summary

OUT=Path('outputs/combination_optimization_20261012/v1')
PREVIOUS=Path('outputs/accuracy_optimization_20261012')
METHODS=['stitch_geometry','stitch_consensus','stitch_matching','cover_shape','cover_evidence','cover_then_stitch']


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        try:temp.replace(path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.15*(attempt+1))


def prepare(out):
    records=load(SOURCE/'dataset_manifest.json')['records']
    old=load(PREVIOUS/'v1/protocol.json')
    for p,digest in old['frozen_original_source_sha256'].items():
        if sha(p)!=digest:raise ValueError(f'Original source changed: {p}')
    selector=load(PREVIOUS/'selection_v2/selector_protocol.json')
    for p,digest in selector['source_sha256'].items():
        if sha(p)!=digest:raise ValueError(f'Previous champion changed: {p}')
    original_dev={tuple(k) for k in old['development_cases']}
    dev={tuple(k) for k in load(PREVIOUS/'roi_probe_v2/roi_protocol.json')['cases']}
    def ordered(pool):return sorted(pool,key=lambda r:hashlib.sha256(('combination-v1/'+r['dataset']+'/'+r['name']).encode()).hexdigest())
    for group,n in [('real_easy',6),('real_hard',2),('one_synthetic_scene',2)]:
        pool=[r for r in records if r['group']==group and (r['dataset'],r['name']) in original_dev and (r['dataset'],r['name']) not in dev]
        dev.update((r['dataset'],r['name']) for r in ordered(pool)[:n])
    check=set()
    for group,n in [('c1',2),('c2',2),('c3',2),('real_easy',10),('real_hard',6),('one_synthetic_scene',2)]:
        pool=[r for r in records if r['group']==group and (r['dataset'],r['name']) not in original_dev]
        check.update((r['dataset'],r['name']) for r in ordered(pool)[:n])
    assert len(dev)==24 and len(check)==24 and not dev&check
    paths=[Path(__file__),Path(__file__).with_name('cable_combination_backend.py'),
           Path(__file__).with_name('cable_evidence_fusion.py'),Path(__file__).with_name('run_evidence_fusion_experiment.py')]
    protocol={'created_at':datetime.now().isoformat(),'scope':'Previously inspected development/challenge regression; not blind testing',
              'source_sha256':{str(p):sha(p) for p in paths},'methods':METHODS,'new_training':False,
              'development_cases':sorted([list(k) for k in dev]),'checking_cases':sorted([list(k) for k in check]),
              'sample_rule':'Reuse all 14 existing ROI cases; fill development by name hash within previous development; hash checking from previous remaining',
              'stage1_baseline_sha256':sha(PREVIOUS/'selection_v2/evaluation_all.json'),
              'ground_truth_used_for_inference':False,'native_resolution':True,
              'stitch_max_gap_width_factor':5.,'stitch_stop_benefit':.45,'matching_time_limit_seconds':3.,
              'cover_count_cost':.025,'cover_incoherence_cost':.70,'cover_duplicate_cost':.5,'cover_solver_time_limit_seconds':3.,
              'references':['https://arxiv.org/abs/2503.12094','https://proceedings.mlr.press/v229/viswanath23a.html','https://arxiv.org/abs/1601.00847'],
              'not_paper_reproductions':True}
    target=out/'protocol.json'
    if target.exists():
        stored=load(target)
        if stored['source_sha256']!=protocol['source_sha256']:raise ValueError('Code changed; use a new version directory')
        return records,stored
    dump(target,protocol);return records,protocol


def select(records,protocol,split):
    keys={tuple(k) for k in protocol['development_cases' if split=='development' else 'checking_cases']}
    return [r for r in records if (r['dataset'],r['name']) in keys]


def cached(r):
    visible,base,proposals=load_cached(SOURCE,r['dataset'],r['name'])
    for p in proposals:proposal_features(p)
    rgb=np.asarray(Image.open(r['image']).convert('RGB'))
    case=PREVIOUS/'selection_v2'/r['dataset']/'coverage_color'/r['name']
    current=np.asarray(Image.open(case/'instance_ids.png'))
    selection=load(case/'inference.json')['selected_backend']
    geom=np.asarray(Image.open(PREVIOUS/'v1'/r['dataset']/'fill_geodesic'/r['name']/'instance_ids.png'))
    prop=np.asarray(Image.open(PREVIOUS/'v1'/r['dataset']/'proposal_graph'/r['name']/'instance_ids.png'))
    return rgb,visible,current,geom,prop,proposals,selection


def save_prediction(out,r,method,ids,visible,audit):
    if ids.shape!=visible.shape or np.any((ids>0)&~visible):raise ValueError('Native foreground constraint violated')
    dest=out/r['dataset']/method/r['name'];dest.mkdir(parents=True,exist_ok=True)
    Image.fromarray(ids.astype(np.uint16)).save(dest/'instance_ids.png')
    Image.fromarray(visible.astype(np.uint8)*255).save(dest/'visible_foreground.png')
    Image.fromarray((visible & (ids==0)).astype(np.uint8)*255).save(dest/'unknown_visible.png')
    audit.update(status='ok' if ids.any() else 'empty',image_sha256=r['image_sha256'],method=method,
                 ground_truth_used_for_inference=False,hidden_pixels_added=0)
    dump(dest/'inference.json',audit)


def predict(out,records,methods):
    start=time.perf_counter()
    for i,r in enumerate(records,1):
        todo=[m for m in methods if not (out/r['dataset']/m/r['name']/'inference.json').exists()]
        if not todo:continue
        began=time.perf_counter();rgb,visible,current,geom,prop,proposals,selection=cached(r)
        covers={}
        for m in todo:
            t=time.perf_counter()
            if m.startswith('stitch_'):
                ids,audit=stitch_ids(rgb,visible,current,proposals,use_evidence=m!='stitch_geometry',
                                    global_match=m=='stitch_matching',only_proposal=m=='stitch_matching',selection=selection)
            else:
                use=m!='cover_shape'
                if use not in covers:covers[use]=global_path_cover(rgb,visible,geom,prop,proposals,current,use)
                ids,audit=covers[use];audit=dict(audit)
                if m=='cover_then_stitch':
                    ids,post=stitch_ids(rgb,visible,ids,proposals,True,True)
                    audit={'cover':audit,'stitch':post}
            audit['seconds']=time.perf_counter()-t
            save_prediction(out,r,m,ids,visible,audit)
        dump(out/'progress.json',{'phase':'prediction','index':i,'selected_cases':len(records),'case':r['name'],
                                  'updated_at':datetime.now().isoformat(),'elapsed_seconds':time.perf_counter()-start})
        print(f'combination [{i}/{len(records)}] {r["dataset"]}/{r["name"]}: {time.perf_counter()-began:.2f}s',flush=True)


def roi_predict(out,records):
    available={tuple(k) for k in load(PREVIOUS/'roi_probe_v2/roi_protocol.json')['cases']}
    records=[r for r in records if (r['dataset'],r['name']) in available]
    for i,r in enumerate(records,1):
        methods=['roi_uniform_fill','roi_uniform_evidence']
        if all((out/r['dataset']/m/r['name']/'inference.json').exists() for m in methods):continue
        rgb,visible,current,geom,prop,proposals,selection=cached(r)
        cache=PREVIOUS/'roi_probe_v2'/r['dataset']/r['name']
        new_visible=np.asarray(Image.open(cache/'uniform_mask.png'))>0
        details=load(cache/'inference.json')['policies']['uniform']
        for row in details['proposals']:
            if not row.get('accepted'):continue
            box=tuple(row['crop_xyxy']);mask=np.asarray(Image.open(row['mask_path']))>0
            proposals.append(Proposal(box,mask,'text_roi',.5,sha(row['mask_path'])))
        ids=complete_visible(current,new_visible)
        save_prediction(out,r,'roi_uniform_fill',ids,new_visible,{'reused_queries':details['text_queries'],'new_queries':0,'semantic_frontend_changed':True})
        proposal,groups,graph=proposal_graph(proposals,new_visible.shape)
        ids,audit=fuse_residual(current,new_visible,proposal,groups,False)
        save_prediction(out,r,'roi_uniform_evidence',ids,new_visible,dict(audit,graph=graph,reused_queries=details['text_queries'],new_queries=0,semantic_frontend_changed=True))
        print(f'ROI association [{i}/{len(records)}] {r["dataset"]}/{r["name"]}',flush=True)


def evaluate(out,records,methods,split):
    from evaluate_full_comparison import truth_masks,binary,bootstrap
    original=load(PREVIOUS/'selection_v2/evaluation_all.json')
    baseline={(r['dataset'],r['name'],r['method']):r for r in original['rows']}
    tables={};rows=[]
    for i,r in enumerate(records,1):
        for m in methods:
            dest=out/r['dataset']/m/r['name']
            if not (dest/'inference.json').exists():raise ValueError(f'Missing saved prediction: {dest}')
            result=dest/'evaluation.json'
            if result.exists():rows.append(load(result));continue
            ids=np.asarray(Image.open(dest/'instance_ids.png'))
            gt=truth_masks(r,tables);union=np.logical_or.reduce(gt) if gt else np.zeros(ids.shape,bool)
            visible=np.asarray(Image.open(dest/'visible_foreground.png'))>0
            row={'dataset':r['dataset'],'name':r['name'],'group':r['group'],'method':m,
                 'prediction_sha256':sha(dest/'instance_ids.png'),'status':load(dest/'inference.json')['status'],
                 'coverage':float(np.count_nonzero(ids)/max(visible.sum(),1)),**binary(ids>0,union),**instance_scores(ids,gt)}
            dump(result,row);rows.append(row)
        for m in ['coverage_color','ours_final','padlos_final']:rows.append(dict(baseline[(r['dataset'],r['name'],m)]))
        if i%6==0 or i==len(records):print(f'evaluate combination [{i}/{len(records)}]',flush=True)
    summaries={};paired={};groups={}
    for d in ['c135','wire300','synthetic20']:
        summaries[d]={m:summary([r for r in rows if r['dataset']==d and r['method']==m]) for m in methods+['coverage_color','ours_final','padlos_final']}
        paired[d]={}
        for m in methods:
            chosen=[r for r in rows if r['dataset']==d and r['method']==m]
            if chosen:paired[d][m]={k:bootstrap(np.array([r[k]-baseline[(d,r['name'],'coverage_color')][k] for r in chosen])) for k in ['foreground_iou','instance_iou']}
    dump(out/f'evaluation_{split}.json',{'summaries':summaries,'paired_vs_previous_champion':paired,'rows':rows,
             'scope':'Fixed small-scale development/checking regression; not a blind test','GT_used_for_inference':False})
    for d,s in summaries.items():
        print(d,json.dumps({m:{k:v.get(k) for k in ['n','foreground_iou','instance_iou','pq_style']} for m,v in s.items()},ensure_ascii=False),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','predict','evaluate','roi_predict','roi_evaluate'])
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--split',choices=['development','checking'],default='development')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);a=p.parse_args()
    records,protocol=prepare(a.output);records=select(records,protocol,a.split)
    if a.phase=='prepare':print(Counter(r['dataset'] for r in records),flush=True)
    elif a.phase=='predict':predict(a.output,records,a.methods)
    elif a.phase=='evaluate':evaluate(a.output,records,a.methods,a.split)
    else:
        available={tuple(k) for k in load(PREVIOUS/'roi_probe_v2/roi_protocol.json')['cases']}
        records=[r for r in records if (r['dataset'],r['name']) in available]
        if a.phase=='roi_predict':roi_predict(a.output,records)
        else:evaluate(a.output,records,['roi_uniform_fill','roi_uniform_evidence'],'roi')


if __name__=='__main__':main()
