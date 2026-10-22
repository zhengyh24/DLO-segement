"""Adapter around pinned official mBEST, with observed-mask clipping only.

Author code is retained under third_party/mBEST (GPL-3.0). This wrapper does not
claim to recreate unavailable hidden pixels and reports unsupported failures.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

COMMIT='b8c19b217185449838b0b55f406c4577a5394ba0'


def predict(rgb, mask):
    if rgb.shape[:2]!=mask.shape:
        raise ValueError('Image/mask geometry mismatch')
    if not mask.any():
        return np.zeros(mask.shape,np.uint16),{'status':'empty','path_count':0}
    repo=Path(__file__).resolve().parents[1]/'third_party/mBEST'
    sys.path.insert(0,str(repo/'_deps'))
    sys.path.insert(0,str(repo))
    os.environ.setdefault('MPLBACKEND','Agg')
    from mBEST import mBEST
    # Unique nonzero RGB encodings avoid the author's 18-color display limit.
    palette=[[i&255,(i>>8)&255,(i>>16)&255] for i in range(1,65535)]
    model=mBEST(colors=palette)
    model.set_image(np.ascontiguousarray(rgb[:,:,::-1]))
    paths,colored=model.run(mask.astype(np.uint8),verbose=False)
    colored=colored.astype(np.uint32)
    ids=(colored[:,:,0]+(colored[:,:,1]<<8)+(colored[:,:,2]<<16)).astype(np.uint16)
    outside=int(np.count_nonzero((ids>0)&~mask))
    ids[~mask]=0
    return ids,{'status':'ok','path_count':len(paths),'clipped_outside_visible_pixels':outside,
                'unassigned_visible_pixels':int(np.count_nonzero(mask&(ids==0)))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fill-visible-band',action='store_true')
    a=p.parse_args()
    rgb=np.asarray(Image.open(a.image).convert('RGB'))
    mask=np.asarray(Image.open(a.mask).convert('L'))>0
    if rgb.shape[:2]!=mask.shape:
        if abs(rgb.shape[1]/rgb.shape[0]-mask.shape[1]/mask.shape[0])>.01:
            raise ValueError('RGB/mask aspect ratios differ')
        rgb=np.asarray(Image.fromarray(rgb).resize((mask.shape[1],mask.shape[0]),Image.Resampling.LANCZOS))
    a.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    try:
        ids,audit=predict(rgb,mask)
    except Exception as e:
        audit={'status':'failed','error':f'{type(e).__name__}: {e}'}
        # A failure is a missing prediction, not silently an easy-image exclusion.
        ids=np.zeros(mask.shape,np.uint16)
    if a.fill_visible_band:
        from refine_observed_ids import fill_observed_ids
        ids,audit['visible_band_fill']=fill_observed_ids(ids,mask)
    audit.update(commit=COMMIT,source='https://github.com/StructuresComp/mBEST',
                 seconds=time.perf_counter()-start,ground_truth_used_for_inference=False)
    Image.fromarray(ids).save(a.output/'instance_ids.png')
    from cable_identity_graph import render
    render(rgb,ids,a.output/'instances_overlay.png')
    (a.output/'inference.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(audit,flush=True)


if __name__=='__main__':
    main()
