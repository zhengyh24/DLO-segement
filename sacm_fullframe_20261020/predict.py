"""RGB-only inference. No GT imports, labels, threshold fitting or per-image selection."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sacm_frontend.predict import predict_native
from .common import load,write,sha


@torch.inference_mode()
def probability(model,rgb,mode,device):
    if mode=='slide':return predict_native(model,rgb,512,384,device)
    assert mode=='full'
    model.eval()
    x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device).float()/255
    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
        p=model(x).float().sigmoid()[0,0].cpu().numpy()
    assert p.shape==rgb.shape[:2] and np.isfinite(p).all()
    return p


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--mode',choices=['slide','full'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--threshold',type=float,default=.5);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda');model,_=load(a.checkpoint,device)
    with Image.open(a.image) as im:rgb=np.array(im.convert('RGB'))
    prob=probability(model,rgb,a.mode,device);np.save(out/'probability.npy',prob)
    Image.fromarray((prob>=a.threshold).astype('uint8')*255).save(out/'mask.png')
    write(out/'inference.json',dict(image=a.image,checkpoint_sha256=sha(a.checkpoint),mode=a.mode,
          shape=list(prob.shape),threshold=a.threshold,ground_truth_read=False,resized=False))

if __name__=='__main__':main()
