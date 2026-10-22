"""Second-round research candidate from a compatible frozen frontend cache."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from cable_evidence_fusion import load_cached,proposal_features,proposal_graph,complete_visible
from cable_identity_selector import select_identity
from cable_combination_backend import global_path_cover
from cable_dual_continuation import dual_stitch
from run_combination_experiment import load,dump,sha


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--dataset',required=True);p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest={'image_sha256':sha(a.image),'cache_root':str(a.cache_root.resolve()),'dataset':a.dataset,'case':a.case,
        'method':'adaptive_agreement_stitch','source_sha256':{str(path):sha(path) for path in [Path(__file__),
          Path(__file__).with_name('cable_combination_backend.py'),Path(__file__).with_name('cable_dual_continuation.py'),
          Path(__file__).with_name('cable_evidence_fusion.py'),Path(__file__).with_name('cable_identity_selector.py')]},
        'ground_truth_used_for_inference':False,'status':'research_candidate_after_48_case_regression'}
    target=a.output/'run_manifest.json'
    if target.exists() and load(target)!=manifest:raise ValueError('Input/code changed; use a new directory')
    dump(target,manifest)
    began=time.perf_counter();rgb=np.asarray(Image.open(a.image).convert('RGB'))
    visible,original,proposals=load_cached(a.cache_root,a.dataset,a.case)
    coarse=load(a.cache_root/a.dataset/'ours_coarse'/a.case/'inference.json')
    if coarse['image_sha256']!=manifest['image_sha256']:raise ValueError('Native RGB differs from frontend cache')
    geom=complete_visible(original,visible)
    pid,groups,graph=proposal_graph(proposals,visible.shape)
    prop=complete_visible(pid,visible)
    previous,previous_audit=select_identity(rgb,visible,original,geom,prop,coarse['crop_count'],'coverage_color')
    ids,cover=global_path_cover(rgb,visible,geom,prop,proposals,previous,coarse['crop_count']>1)
    ids,stitch=dual_stitch(rgb,visible,ids,proposals,original,True)
    seconds=time.perf_counter()-began
    a.output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(ids.astype(np.uint16)).save(a.output/'instance_ids.png')
    Image.fromarray(visible.astype(np.uint8)*255).save(a.output/'visible_foreground.png')
    Image.fromarray((visible & (ids==0)).astype(np.uint8)*255).save(a.output/'unknown_visible.png')
    from cable_identity_graph import render
    render(rgb,ids,a.output/'instances_overlay.png')
    dump(a.output/'inference.json',{'method':'adaptive_agreement_stitch','status':'ok' if ids.any() else 'empty',
         'seconds_cached_backend_excluding_export':seconds,'cover':cover,'stitch':stitch,
         'previous_selector':previous_audit,'native_shape':list(ids.shape),'ground_truth_used_for_inference':False,'hidden_pixels_added':0})
    print(f'Combination candidate saved: {seconds:.2f}s cached backend; {a.output / "instance_ids.png"}',flush=True)


if __name__=='__main__':main()
