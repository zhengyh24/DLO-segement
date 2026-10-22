"""Batch identity inference: binary foreground provenance explicit; no instance-label importer."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from curvi_identity.io import read_json,read_rgb,read_mask,rgb_hash
from curvi_identity.graph import build_graph,decode
from curvi_identity.pipeline import load_model,score_graph,unknown_mask


def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)


def threshold_ids(graph,scores,threshold):
    # Deliberately remove exclusivity, margin and conflict veto; same candidates/scores/threshold.
    parent={int(k):int(k) for k in np.unique(graph.partition) if k}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for c,s in zip(graph.candidates,scores):
        if min(s)>=threshold:parent[find(c['a'])]=find(c['b'])
    lookup=np.zeros(int(graph.partition.max())+1,np.uint16);ids={}
    for x in parent:
        root=find(x)
        if root not in ids:ids[root]=len(ids)+1
        lookup[x]=ids[root]
    return lookup[graph.partition]


def main(out):
    oracle=read_json(out/'input_regime.json')['oracle_foreground']
    torch.set_num_threads(4);target=out/'validation_predictions';target.mkdir(exist_ok=True)
    model,cfg,ckpt=load_model(out/'fit/best.pt','cuda')
    # Split metadata is descriptive only. All segmentation inputs come from RGB-only cache manifests.
    names={r['file_name'] for r in read_json(out/'groups.json')['records'] if r['split']=='validation'}
    records=[r for r in read_json(out/'inference_inputs.json')['records'] if r['file_name'] in names]
    pred={r['file_name']:r for r in read_json(out/'predictions.json')['records']}
    for index,r in enumerate(records,1):
        dest=target/r['name'];dest.mkdir(exist_ok=True)
        if (dest/'audit.json').exists():continue
        start=time.perf_counter();rgb=read_rgb(r['image']);fg=read_mask(pred[r['file_name']]['foreground'])>0
        if rgb_hash(rgb)!=r['rgb_sha256'] or fg.shape!=rgb.shape[:2]:raise ValueError('Input mismatch')
        graph=build_graph(fg,cfg);scores=score_graph(rgb,graph,model,cfg,'cuda')
        full,audit=decode(graph,scores,cfg);threshold=threshold_ids(graph,scores,cfg.threshold)
        for method,ids in [('no_link',graph.partition),('full_default',full),('bidirectional_threshold_only',threshold)]:
            if np.any((ids>0)!=fg):raise ValueError('Identity variant changed foreground')
            Image.fromarray(ids.astype(np.uint16)).save(dest/(method+'.png'))
        unknown=unknown_mask(graph,audit);Image.fromarray(unknown.astype(np.uint8)*255).save(dest/'unknown.png')
        save(dest/'audit.json',dict(file_name=r['file_name'],rgb_sha256=r['rgb_sha256'],native_shape=list(fg.shape),
            ground_truth_used=oracle,gt_instance_identity_used=False,oracle_foreground=oracle,foreground_changed_pixels=0,graph=graph.audit,decoding=audit,
            scores=[dict(candidate=c['key'],forward=float(s[0]),reverse=float(s[1])) for c,s in zip(graph.candidates,scores)],
            unknown_pixels=int(unknown.sum()),unknown_semantics='No accepted association; not calibrated uncertainty',
            seconds_identity_only=time.perf_counter()-start,checkpoint_epoch=ckpt['epoch']))
        print({'validation_inferred':index,'total':len(records)},flush=True)
    save(target/'completion.json',dict(images=len(records),ground_truth_used=oracle,gt_instance_identity_used=False,oracle_foreground=oracle,checkpoint_epoch=ckpt['epoch']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.output.resolve())
