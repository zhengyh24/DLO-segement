"""Linux migration adapter; frozen data/algorithm, compact lazy crop storage."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
from curvi_identity.io import read_json,read_rgb,read_mask,rgb_hash
from curvi_identity.config import Config
from run_curvi_2000 import save,sha,finalize
OUT=ROOT/'run'


def remap(obj):
    if isinstance(obj,str):
        return obj.replace('D:/cable_identity_runs/curvi_2000_20261014',OUT.as_posix()).replace('D:\\cable_identity_runs\\curvi_2000_20261014',OUT.as_posix())
    if isinstance(obj,list):return [remap(x) for x in obj]
    if isinstance(obj,dict):return {k:remap(v) for k,v in obj.items()}
    return obj


def prepare_paths():
    if (OUT/'cloud_migration.json').exists():return
    for name in ('groups.json','frontend_protocol.json','protocol.json','rgb_dedup.json','geometry_duplicate_check.json'):
        save(OUT/name,read_json(OUT/'local_original'/name))
    cfg=read_json(OUT/'local_original/config.json');cfg['dataset_root']=(OUT/'selected_labels.json').as_posix();save(OUT/'config.json',cfg)
    inputs=remap(read_json(OUT/'local_original/inference_inputs.json'));save(OUT/'inference_inputs.json',inputs)
    expected={r['name']:r for r in inputs['records']};reused=0
    for directory in (OUT/'frontend/scale2300/ours_coarse').iterdir():
        p=directory/'inference.json'
        if not p.exists():continue
        info=read_json(p)
        if info['image_sha256']!=expected[directory.name]['image_sha256'] or info['ground_truth_used'] is not False:raise ValueError('Copied frontend mismatch')
        for p in directory.rglob('*.json'):
            obj=read_json(p);save(p,remap(obj))
        reused+=1
    for r in inputs['records']:
        if sha(r['image'])!=r['image_sha256']:raise ValueError('Uploaded RGB mismatch')
    save(OUT/'cloud_migration.json',dict(reused_coarse=reused,images=len(inputs['records']),preserved_split=True,
        preserved_algorithm=True,changes=['Linux path adapter','lazy native crop storage','RTX4090/Python3.12.15 environment'],
        local_original_records='local_original/',offline_labels_only=True))


def selected_labels(groups,columns):
    labels={r['file_name']:r for r in read_json(OUT/'selected_labels.json')['records']}
    for r in sorted(groups['records'],key=lambda r:(r['shard'],r['row'])):
        row=labels[r['file_name']]
        yield r,{k:row[k] for k in columns}


def prepare_pairs():
    from curvi_identity.data import group_map,decode_masks,arc_labels
    from curvi_identity.graph import build_graph
    from curvi_identity.train import validate_manifest
    groups,_=group_map(OUT/'groups.json');cfg=Config.read(OUT/'config.json')
    inputs={r['file_name']:r for r in read_json(OUT/'inference_inputs.json')['records']}
    predictions=read_json(OUT/'predictions.json');pred={r['file_name']:r for r in predictions['records']}
    prepared=OUT/'prepared';prepared.mkdir(exist_ok=True);records=[];images=[]
    for r,row in selected_labels(groups,['file_name','masks_rle']):
        inp=inputs[r['file_name']];dest=prepared/'graphs'/inp['name'];done=dest/'done.json'
        if done.exists():obj=read_json(done)
        else:
            if shutil.disk_usage(OUT).free<2*1024**3:raise RuntimeError('Less than 2 GiB free; stop without deletion')
            dest.mkdir(parents=True,exist_ok=True);rgb=read_rgb(inp['image']);cache=pred[r['file_name']]
            if rgb_hash(rgb)!=cache['rgb_sha256']:raise ValueError('Prediction/RGB mismatch')
            fg=read_mask(cache['foreground'])>0;graph=build_graph(fg,cfg)
            assigned,stats=arc_labels(graph,decode_masks(row['masks_rle'],fg.shape),cfg.purity)
            current=[]
            for index,c in enumerate(graph.candidates):
                if c['a'] not in assigned or c['b'] not in assigned:continue
                current.append(dict(file_name=r['file_name'],group_id=r['group_id'],split=r['split'],rgb_sha256=inp['rgb_sha256'],
                    label=int(assigned[c['a']]==assigned[c['b']]),candidate=c['key'],candidate_index=index,
                    graph_item=(dest/'graph.json').relative_to(prepared).as_posix()))
            np.savez_compressed(dest/'arrays.npz',partition=graph.partition,skeleton=graph.skeleton)
            save(dest/'graph.json',dict(image=inp['image'],candidates=graph.candidates,audit=graph.audit))
            audit=dict(file_name=r['file_name'],split=r['split'],group_id=r['group_id'],rgb_sha256=inp['rgb_sha256'],
                scene_family=r['scene_family'],prediction_producer=cache['producer'],graph=graph.audit,supervision=stats,
                pairs=len(current),positive=sum(p['label'] for p in current),ignored_pairs=len(graph.candidates)-len(current))
            obj=dict(image=audit,records=current);save(done,obj)
        records.extend(obj['records']);images.append(obj['image'])
        if len(images)%10==0:
            save(OUT/'pair_progress.json',dict(images=len(images),total=len(groups['records']),pairs=len(records)));print({'prepared':len(images),'pairs':len(records)},flush=True)
    manifest=dict(schema=1,kind='predicted_native_pair_supervision',storage_kind='native_graph_lazy_v1',config=cfg.to_dict(),
        groups=groups,prediction_manifest=predictions,images=images,records=records,training_performed=False,real_subset_used=False)
    validate_manifest(manifest,cfg);save(prepared/'manifest.json',manifest)
    summary={}
    for split in ('train','validation'):
        rr=[r for r in records if r['split']==split];ii=[r for r in images if r['split']==split]
        summary[split]=dict(images=len(ii),pairs=len(rr),positive=sum(r['label'] for r in rr),negative=sum(1-r['label'] for r in rr),
            images_without_pairs=[r['file_name'] for r in ii if not r['pairs']])
    save(OUT/'preparation_summary.json',summary)


def step(stage,args):
    for relative,digest in read_json(OUT/'cloud_source_hashes.json').items():
        if sha(ROOT/relative)!=digest:raise ValueError('Cloud source changed after freeze: '+relative)
    logs=OUT/'logs';logs.mkdir(exist_ok=True);path=logs/f'{stage}_{int(time.time())}.log'
    with path.open('w') as log:
        proc=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),child_pid=proc.pid,log=str(path),time=time.time()))
        rc=proc.wait()
    if rc:raise RuntimeError(f'{stage} exit {rc}: {path}')


def run():
    prepare_paths()
    frozen=OUT/'cloud_source_hashes.json'
    if not frozen.exists():
        sources=[p for folder in ('tools','curvi_identity','vendor','third_party/hqsam_deps') for p in (ROOT/folder).rglob('*.py')]
        hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
        save(frozen,hashes)
        for source in sources:
            target=OUT/'cloud_source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for stage in ('coarse','hq'):
        if not (OUT/f'{stage}_status.json').exists():step(stage,['tools/curvi_cloud_frontend.py',stage,'--output',str(OUT)])
    finalize(OUT)
    if not (OUT/'prepared/manifest.json').exists():step('pairs',['tools/run_cloud_curvi.py','pairs'])
    if not (OUT/'fit/completion.json').exists():step('training',['tools/run_cloud_curvi.py','train'])
    if not (OUT/'validation_predictions/completion.json').exists():step('identity_inference',['tools/predict_curvi_2000.py','--output',str(OUT)])
    if not (OUT/'evaluation.json').exists():step('evaluation',['tools/run_cloud_curvi.py','evaluate'])
    save(OUT/'status.json',dict(stage='complete',time=time.time(),report=str(OUT/'results.md')))


if __name__=='__main__':
    os.chdir(ROOT);p=argparse.ArgumentParser();p.add_argument('stage',choices=['paths','run','pairs','train','evaluate']);a=p.parse_args()
    try:
        if a.stage=='paths':prepare_paths()
        elif a.stage=='pairs':prepare_pairs()
        elif a.stage=='train':
            import train_curvi_2000 as trainer
            from curvi_cloud_data import CloudPairDataset
            trainer.PairDataset=CloudPairDataset;trainer.main(OUT)
        elif a.stage=='evaluate':
            import evaluate_curvi_2000 as evaluator
            evaluator.selected_rows=selected_labels;evaluator.main(OUT)
        else:
            import fcntl
            with (OUT/'cloud_run.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
    except Exception:
        failure=dict(stage=a.stage,time=time.time(),error=traceback.format_exc())
        save(OUT/'failure.json',failure);save(OUT/'status.json',dict(stage='failed',detail=failure));raise
