"""Usable accuracy candidate: frozen frontend plus observable identity selection.

Run fresh RGB through separate GPU stages, or reproduce the backend from a
trusted prediction cache. Labels and evaluation modules are never imported.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from PIL import Image
from cable_evidence_fusion import load_cached, predict_methods
from cable_identity_selector import select_identity

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        try:temp.replace(path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.15*(attempt+1))


def fuse(image,cache,dataset,name,dest):
    began=time.perf_counter()
    rgb=np.asarray(Image.open(image).convert('RGB'))
    visible,base,proposals=load_cached(cache,dataset,name)
    evidence={}
    for m,ids,audit in predict_methods(rgb,visible,base,proposals,['fill_geodesic','proposal_graph']):
        evidence[m]=(ids,audit)
    coarse=json.loads((cache/dataset/'ours_coarse'/name/'inference.json').read_text(encoding='utf-8'))
    if sha(image)!=coarse['image_sha256']:raise ValueError('RGB does not match cached native geometry')
    ids,selection=select_identity(rgb,visible,base,evidence['fill_geodesic'][0],evidence['proposal_graph'][0],
                                  coarse['crop_count'],'coverage_color')
    seconds=time.perf_counter()-began
    dest.mkdir(parents=True,exist_ok=True)
    Image.fromarray(ids.astype(np.uint16)).save(dest/'instance_ids.png')
    Image.fromarray(visible.astype(np.uint8)*255).save(dest/'visible_foreground.png')
    Image.fromarray((visible & (ids==0)).astype(np.uint8)*255).save(dest/'unknown_visible.png')
    from cable_identity_graph import render
    render(rgb,ids,dest/'instances_overlay.png')
    dump(dest/'inference.json',{'status':'ok' if ids.any() else 'empty','method':'coverage_color',
        'seconds_including_cached_input_read_excluding_export':seconds,'native_size':[rgb.shape[1],rgb.shape[0]],
        'image_sha256':sha(image),'selection':selection,'proposal_graph':evidence['proposal_graph'][1]['graph'],
        'ground_truth_used_for_inference':False,'hidden_pixels_added':0,
        'assigned_visible_pixels':int(np.count_nonzero(ids)),
        'unknown_visible_pixels':int(np.count_nonzero(visible & (ids==0)))})
    return seconds


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--dataset',default='c135')
    p.add_argument('--case')
    p.add_argument('--stage',choices=['all','coarse','hq','geometry','fusion'],default='all')
    a=p.parse_args()
    a.image=a.image.resolve();a.output=a.output.resolve()
    name=a.case or a.image.stem
    paths=[Path(__file__),ROOT/'tools/cable_evidence_fusion.py',ROOT/'tools/cable_identity_selector.py',
           ROOT/'tools/full_comparison_inference.py',ROOT/'tools/stage_a_geomprompt_sam31.py',
           ROOT/'tools/mbest_backend.py',ROOT/'tools/refine_observed_ids.py']
    manifest={'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        'image':str(a.image),'image_sha256':sha(a.image),'native_RGB':True,
        'ground_truth_used_for_inference':False,'method':'SAM3.1/GeoMPrompt + HQ + proposal/geometric identity selector',
        'cache_root':str(a.cache_root) if a.cache_root else None,'cache_dataset':a.dataset if a.cache_root else None,
        'cache_case':name if a.cache_root else None,'experimental_status':'Challenge-regression candidate; not established universal SOTA'}
    target=a.output/'run_manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Different input/code. Use a new output directory.')
    dump(target,manifest)
    if a.cache_root:
        seconds=fuse(a.image,a.cache_root,a.dataset,name,a.output/'candidate')
        print(f'Cached backend completed: {seconds:.3f}s; {a.output / "candidate/instance_ids.png"}',flush=True)
        return
    record={'dataset':'fresh','name':name,'image':str(a.image),'image_sha256':manifest['image_sha256']}
    if a.stage=='all':
        started=time.perf_counter();times={}
        for stage in ['coarse','hq','geometry','fusion']:
            began=time.perf_counter()
            command=[sys.executable,str(ROOT/'tools/padlos_runtime_runner.py'),str(Path(__file__)),
                     '--image',str(a.image),'--output',str(a.output),'--case',name,'--stage',stage]
            process=subprocess.run(command,capture_output=True,encoding='utf-8',errors='replace')
            (a.output/f'{stage}.log').write_text(process.stdout+'\n'+process.stderr,encoding='utf-8')
            times[stage]=time.perf_counter()-began
            print(f'{stage}: exit={process.returncode}, {times[stage]:.2f}s',flush=True)
            if process.returncode:
                raise RuntimeError(f'{stage} failed; see {a.output / (stage+".log")}')
        dump(a.output/'pipeline_status.json',{'status':'complete','stage_process_seconds':times,
            'end_to_end_cold_process_seconds':time.perf_counter()-started,
            'timing_scope':'includes separate process startup/model loading/export; not resident deployment FPS',
            'completed_at':datetime.now().isoformat()})
    elif a.stage in ('coarse','hq'):
        import full_comparison_inference as frozen
        # Redirect output at runtime only; never edit or write into the old run.
        frozen.OUT=a.output
        getattr(frozen,a.stage)([record])
    elif a.stage=='geometry':
        rgb=np.asarray(Image.open(a.image).convert('RGB'))
        visible=np.asarray(Image.open(a.output/'fresh/ours_hq'/name/'mask.png'))>0
        from mbest_backend import predict
        from refine_observed_ids import fill_observed_ids
        began=time.perf_counter()
        try:ids,audit=predict(rgb,visible)
        except Exception:
            ids=np.zeros(visible.shape,np.uint16);audit={'status':'failed','error':traceback.format_exc()}
        ids,audit['visible_band_fill']=fill_observed_ids(ids,visible)
        dest=a.output/'fresh/ours_mbest'/name;dest.mkdir(parents=True,exist_ok=True)
        Image.fromarray(ids).save(dest/'instance_ids.png')
        audit.update(ground_truth_used_for_inference=False,seconds=time.perf_counter()-began)
        dump(dest/'inference.json',audit)
    else:
        print(f'Backend seconds: {fuse(a.image,a.output,"fresh",name,a.output / "candidate"):.3f}',flush=True)


if __name__=='__main__':main()
