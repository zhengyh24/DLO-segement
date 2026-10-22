"""Equal-budget SAM3.1 foveated text rescan; semantic probe, not a final backend.

Uniform and identity-disagreement policies get equal counts, up to four native
512-pixel crops. Small images may have fewer unique boxes, equally for both.
Saved predictions precede all GT access. Guarded output reuses the same queries.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from model_paths import SAM31_REPO, SAM31_CHECKPOINT
import sys
import time
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from run_evidence_fusion_experiment import SOURCE, prepare, dump, sha

OUT=Path('outputs/accuracy_optimization_20261012/roi_probe_v2')


def jobs(visible, base, proposal, side=512, budget=4):
    h,w=visible.shape
    lut=np.zeros(int(proposal.max())+1,np.uint16)
    for ident in np.unique(proposal[proposal>0]):
        count=np.bincount(base[proposal==ident],minlength=int(base.max())+1);count[0]=0
        if count.sum():lut[ident]=int(count.argmax())
    disagreement=visible & ((base==0) | (lut[proposal]!=base))
    sk=skeletonize(visible)
    degree=cv2.filter2D(sk.astype(np.uint8),cv2.CV_16U,np.ones((3,3),np.uint8))
    junction=sk & (degree>=4)
    uncertainty=disagreement.astype(np.float32)+.5*cv2.dilate(junction.astype(np.uint8),np.ones((21,21),np.uint8))
    def box(cx,cy):
        ww,hh=min(side,w),min(side,h)
        x=int(np.clip(round(cx-ww/2),0,w-ww));y=int(np.clip(round(cy-hh/2),0,h-hh))
        return [x,y,x+ww,y+hh]
    uniform=[box(w*q,h*t) for q,t in [(0,0),(1,0),(0,1),(1,1)]]
    uniform=list(dict.fromkeys(tuple(b) for b in uniform))[:budget]
    candidates=[]
    for y in sorted(set(list(range(0,max(h-side,0)+1,side//2))+[max(0,h-side)])):
        for x in sorted(set(list(range(0,max(w-side,0)+1,side//2))+[max(0,w-side)])):
            b=box(x+side/2,y+side/2)
            a,c,d,e=b
            score=float(uncertainty[c:e,a:d].sum())
            candidates.append((score,tuple(b)))
    selected=[]
    for score,b in sorted(candidates,key=lambda v:(-v[0],v[1])):
        if b in selected:continue
        cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        if any(np.hypot(cx-(a[0]+a[2])/2,cy-(a[1]+a[3])/2)<side*.35 for a in selected):continue
        selected.append(b)
        if len(selected)>=len(uniform):break
    # Small images cannot meet the separation heuristic. Keep query count equal
    # by filling with distinct remaining boxes, then explicit repeats if needed.
    for score,b in sorted(candidates,key=lambda v:(-v[0],v[1])):
        if len(selected)>=len(uniform):break
        if b not in selected:selected.append(b)
    while len(selected)<len(uniform):selected.append(uniform[len(selected)])
    assert len(selected)==len(uniform)
    return {'uniform':uniform,'disagreement':selected},int(disagreement.sum())


def record_set(protocol,records):
    dev={tuple(k) for k in protocol['development_cases']}
    available=[r for r in records if (r['dataset'],r['name']) in dev]
    chosen=[]
    for group,n in [('c1',2),('c2',2),('c3',2),('real_easy',4),('real_hard',4)]:
        chosen+=sorted([r for r in available if r['group']==group],key=lambda r:r['name'])[:n]
    return chosen


def predict(records):
    from stage_a_geomprompt_sam31 import load_sam31,run_crop_text,Crop
    import torch
    from cable_evidence_fusion import Proposal,proposal_features
    torch.set_num_threads(4)
    predictor=load_sam31(SAM31_CHECKPOINT,SAM31_REPO)
    for idx,r in enumerate(records,1):
        dest=OUT/r['dataset']/r['name']
        if (dest/'inference.json').exists():continue
        image=Image.open(r['image']).convert('RGB')
        b=SOURCE/r['dataset'];c=Path('outputs/accuracy_optimization_20261012/v1')/r['dataset']
        visible=np.asarray(Image.open(b/'ours_hq'/r['name']/'mask.png'))>0
        base=np.asarray(Image.open(b/'ours_mbest'/r['name']/'instance_ids.png'))
        proposal=np.asarray(Image.open(c/'proposal_graph'/r['name']/'instance_ids.png'))
        policies,uncertainty=jobs(visible,base,proposal)
        outputs={};audits={};began=time.perf_counter()
        for policy,boxes in policies.items():
            union=visible.copy();guarded=visible.copy();rows=[]
            started=time.perf_counter()
            for k,box in enumerate(boxes):
                x0,y0,x1,y1=box
                crop=Crop(f'{policy}_{k:02d}',x0,y0,x1,y1,'budgeted_accuracy_probe')
                records_,local=run_crop_text(predictor,image,crop,dest/'crop_cache','cable',preserve_aspect=True,
                                            proposal_dir=dest/'proposals')
                union[y0:y1,x0:x1]|=local
                for row in records_:
                    if not row.get('accepted'):continue
                    mask=np.asarray(Image.open(row['mask_path']))>0
                    p=Proposal(tuple(box),mask,'text',.5,sha(row['mask_path']));proposal_features(p)
                    ratio=float(np.count_nonzero(mask & visible[y0:y1,x0:x1])/max(mask.sum(),1))
                    accepted=p.quality>=.45 and ratio>=.20
                    if accepted:guarded[y0:y1,x0:x1]|=mask
                    row['guarded_acceptance']={'accepted':accepted,'shape_quality':p.quality,'overlap_with_original':ratio}
                rows+=records_
            if (guarded & ~visible).sum()>max(256,visible.sum()*.4):
                guarded=visible.copy();rollback=True
            else:rollback=False
            outputs[policy]=union
            if policy=='disagreement':outputs['disagreement_guarded']=guarded
            audits[policy]={'seconds':time.perf_counter()-started,'crop_boxes':[list(v) for v in boxes],
                            'text_queries':len(boxes),'proposals':rows,'guarded_rollback':rollback}
        dest.mkdir(parents=True,exist_ok=True)
        for name,mask in outputs.items():Image.fromarray(mask.astype(np.uint8)*255).save(dest/f'{name}_mask.png')
        dump(dest/'inference.json',{'policies':audits,'original_foreground_pixels':int(visible.sum()),
                  'identity_disagreement_pixels':uncertainty,'ground_truth_used_for_inference':False,
                  'seconds':time.perf_counter()-began,'native_size':list(image.size),
                  'semantic_probe_only':True,'image_sha256':r['image_sha256']})
        dump(OUT/'progress.json',{'index':idx,'selected_cases':len(records),'case':r['name'],'updated_at':datetime.now().isoformat()})
        print(f'ROI [{idx}/{len(records)}] {r["dataset"]}/{r["name"]}: {time.perf_counter()-began:.1f}s',flush=True)


def evaluate(records):
    from evaluate_full_comparison import truth_masks,binary,bootstrap
    tables={};rows=[]
    for r in records:
        dest=OUT/r['dataset']/r['name']
        if not (dest/'inference.json').exists():raise ValueError('Missing ROI prediction')
        gt=truth_masks(r,tables);truth=np.logical_or.reduce(gt)
        for method in ['original','uniform','disagreement','disagreement_guarded']:
            path=SOURCE/r['dataset']/'ours_hq'/r['name']/'mask.png' if method=='original' else dest/f'{method}_mask.png'
            mask=np.asarray(Image.open(path))>0
            rows.append({'dataset':r['dataset'],'name':r['name'],'method':method,**binary(mask,truth)})
    summary={}
    for d in ['c135','wire300']:
        summary[d]={m:{'n':sum(r['dataset']==d and r['method']==m for r in rows),
                         **{k:float(np.mean([r[k] for r in rows if r['dataset']==d and r['method']==m])) for k in ['foreground_iou','precision','recall']}}
                    for m in ['original','uniform','disagreement','disagreement_guarded']}
    dump(OUT/'evaluation.json',{'summaries':summary,'rows':rows,'scope':'14 development images, semantic-only equal-crop-count probe',
                               'not_final_instance_results':True,'maximum_new_queries_per_policy':4,
                               'actual_query_count':'Audit in each saved inference.json; both policies have exactly equal counts per case'})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['predict','evaluate']);a=p.parse_args()
    records,protocol=prepare(OUT);records=record_set(protocol,records)
    payload={'source_sha256':sha(__file__),'cases':[[r['dataset'],r['name']] for r in records],
             'side':512,'maximum_crops_per_policy':4,'text_prompt':'cable','same_loaded_sam31_weights':True,
             'native_outputs':True,'guard_shape_quality':.45,'guard_original_overlap':.20,'guard_growth_cap':.4,
             'gt_used_for_inference':False,'not_paper_reproduction':True,'scope':'Development only, semantic masks only'}
    target=OUT/'roi_protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=payload:raise ValueError('ROI source changed; use a new version')
    dump(target,payload)
    if a.phase=='predict':predict(records)
    else:evaluate(records)


if __name__=='__main__':main()
