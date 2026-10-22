"""RGB-only native inference: no data loader, labels, or GT arguments."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from sacm_frontend.model import CableForeground

def starts(n,tile,stride):
    if not 0<stride<=tile:raise ValueError('Require 0 < stride <= tile')
    return sorted(set([*range(0,max(1,n-tile+1),stride),max(0,n-tile)]))

@torch.inference_mode()
def predict_native(model,rgb,tile,stride,device):
    model.eval();h,w=rgb.shape[:2];total=np.zeros((h,w),np.float32);count=np.zeros((h,w),np.float32)
    for y in starts(h,tile,stride):
        for x in starts(w,tile,stride):
            crop=rgb[y:y+tile,x:x+tile];hh,ww=crop.shape[:2]
            tensor=torch.from_numpy(crop.copy()).permute(2,0,1)[None].to(device).float()/255
            tensor=F.pad(tensor,(0,tile-ww,0,tile-hh))
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                prob=model(tensor).float().sigmoid()[0,0,:hh,:ww].cpu().numpy()
            total[y:y+hh,x:x+ww]+=prob;count[y:y+hh,x:x+ww]+=1
    if not np.all(count>0):raise RuntimeError('Uncovered native pixels')
    return total/count

def load_model(checkpoint,device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=CableForeground(**ck['model_config']);model.load_state_dict(ck['model'],strict=True)
    return model.to(device),ck

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    device=torch.device(a.device);model,ck=load_model(a.checkpoint,device)
    # Use encoded pixel coordinates; do not auto-rotate EXIF.
    with Image.open(a.image) as im:rgb=np.array(im.convert('RGB'))
    a.out.mkdir(parents=True,exist_ok=False);cfg=ck['config']
    prob=predict_native(model,rgb,cfg['tile'],cfg['stride'],device)
    np.save(a.out/'probability.npy',prob)
    Image.fromarray((prob>=cfg['threshold']).astype(np.uint8)*255).save(a.out/'mask.png')
    (a.out/'inference.json').write_text(json.dumps(dict(image=str(a.image),shape=list(prob.shape),
        ground_truth_read=False,native_pixels=True,threshold=cfg['threshold'],checkpoint=str(a.checkpoint)),indent=2))
    print('NATIVE_RGB_INFERENCE_COMPLETE')
if __name__=='__main__':main()
