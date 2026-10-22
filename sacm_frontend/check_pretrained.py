"""Real pretrained ViT-B 512-native forward/backward/reload software witness."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sacm_frontend.model import CableForeground
from sacm_frontend.data import load_record,read_manifest
from sacm_frontend.losses import foreground_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--encoder',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.manual_seed(20261014);torch.set_num_threads(4)
    device=torch.device('cuda');torch.cuda.reset_peak_memory_stats();begin=time.time()
    obj=read_manifest(a.manifest,True);row=next(r for r in obj['records'] if r['split']=='train')
    rgb,masks=load_record(a.manifest.parent,row);h,w=rgb.shape[:2];tile=512
    # Explicit software-only GT-centered crop to witness positive supervision.
    ys,xs=np.where(masks.any(0));assert len(xs)>0
    y=max(0,min(h-tile,int(ys[len(ys)//2])-tile//2));x=max(0,min(w-tile,int(xs[len(xs)//2])-tile//2))
    rgb=rgb[y:y+tile,x:x+tile];masks=masks[:,y:y+tile,x:x+tile]
    assert rgb.shape[:2]==(tile,tile) and masks.any()
    image=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device).float()/255
    labels=torch.from_numpy(masks.copy()).to(device);valid=torch.ones(tile,tile,dtype=torch.bool,device=device)
    model=CableForeground('vit_b',True).to(device);source=model.load_encoder(a.encoder)
    before=model.encoder.patch_embed.proj.weight.detach().clone()
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    logits=model(image);loss,terms=foreground_loss(logits,labels,valid,True,.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.blocks[0].mlp.adapter.up.weight.grad.abs().sum()>0
    assert model.external[0].up.weight.grad.abs().sum()>0
    assert model.detail[0].weight.grad.abs().sum()>0
    assert model.encoder.patch_embed.proj.weight.grad is None
    optim.step();assert torch.equal(before,model.encoder.patch_embed.proj.weight)
    model.eval()
    with torch.no_grad():expected=model(image).cpu()
    torch.save(model.state_dict(),a.out/'reload_fixture.pt')
    # Release optimizer state before loading a second real encoder on an 8GB GPU.
    del optim,model,logits,loss;torch.cuda.empty_cache()
    restored=CableForeground('vit_b',True).to(device)
    restored.load_state_dict(torch.load(a.out/'reload_fixture.pt',weights_only=True,map_location=device));restored.eval()
    with torch.no_grad():actual=restored(image).cpu()
    assert torch.equal(actual,expected)
    result=dict(passed=True,device=torch.cuda.get_device_name(),native_crop=[tile,tile],
        positive_pixels=int(masks.any(0).sum()),source=source,terms=terms,
        frozen_encoder_unchanged=True,adapter_and_detail_gradients=True,exact_reload=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-begin,
        ground_truth_centered_crop='software witness ONLY; not production sampling or inference',accuracy_claim=False)
    (a.out/'check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
