"""Bounded same-image, different-instance memory learning diagnostic.

Two same-color perpendicular bars are an artificial fixture, not real cables.
Step two intentionally receives NO seed/history to isolate persistent state.
This is a fitting test with different prompt locations, not generalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .data import exit_targets
from .losses import tracking_loss
from .model import InstanceTracker


def make_prompts(size, locations, device):
    y, x = torch.meshgrid(torch.arange(size,device=device), torch.arange(size,device=device), indexing="ij")
    return torch.stack([torch.exp(-((y-sy)**2+(x-sx)**2)/8) for sy,sx in locations])[:,None]


def iou(prediction, target, valid):
    p, t = (prediction > 0) & valid.bool(), target.bool() & valid.bool()
    intersection = (p & t).sum((1,2,3)).float()
    union = (p | t).sum((1,2,3)).float()
    return (intersection/union.clamp_min(1)).cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--updates",type=int,default=300)
    args = p.parse_args()
    if not 1 <= args.updates <= 1000:
        p.error("Bounded diagnostic requires 1..1000 updates")
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4);torch.manual_seed(21)
    model=InstanceTracker().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    target=torch.zeros(2,1,64,64,device=device)
    target[0,0,30:35,:]=1;target[1,0,:,30:35]=1
    union=target.amax(0,keepdim=True)
    rgb=(.1+.7*union).expand(2,3,-1,-1).contiguous()
    valid=torch.ones_like(target);valid[:,:,30:35,30:35]=0
    spatial=torch.ones_like(valid)
    other=union.expand_as(target)*(1-target)
    seed=make_prompts(64,((32,10),(10,32)),device)
    zeros=torch.zeros_like(seed)
    batch={"target":target,"valid":valid,"other":other,"seed_target":union.expand_as(target),
           "exits":exit_targets(target,zeros,valid)}
    started=time.perf_counter();first_loss=None
    with (outdir/"train.jsonl").open("w",encoding="utf-8") as handle:
        for step in range(1,args.updates+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            first=model(rgb,seed,zeros,spatial_valid=spatial)
            second=model(rgb,zeros,zeros,query=first["query"],anchor=first["anchor"],spatial_valid=spatial)
            loss=(tracking_loss(first,batch)[0]+tracking_loss(second,batch)[0])/2
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError("Nonfinite diagnostic update")
            optimizer.step()
            if first_loss is None:first_loss=float(loss.detach())
            if step == 1 or step % 50 == 0 or step == args.updates:
                record={"step":step,"loss":float(loss.detach()),"seconds":time.perf_counter()-started}
                handle.write(json.dumps(record)+"\n");handle.flush();print(json.dumps(record),flush=True)
    model.eval()
    with torch.no_grad():
        # Different seed locations, same fitted image; NOT a new scene.
        seed=make_prompts(64,((32,20),(20,32)),device)
        first=model(rgb,seed,zeros,spatial_valid=spatial)
        second=model(rgb,zeros,zeros,query=first["query"],anchor=first["anchor"],spatial_valid=spatial)
        swapped=model(rgb,zeros,zeros,query=first["query"].flip(0),anchor=first["anchor"].flip(0),spatial_valid=spatial)
    scores=iou(second["mask_logits"],target,valid)
    swap_scores=iou(swapped["mask_logits"],target.flip(0),valid)
    result={"status":"completed","kind":"artificial_memory_learnability_smoke",
            "optimizer_steps":args.updates,"skipped_updates":0,"first_loss":first_loss,
            "last_loss":float(loss.detach()),"second_step_target_iou":scores,
            "swapped_state_opposite_target_iou":swap_scores,
            "threshold_met":min(scores+swap_scores)>=.9,
            "predeclared_software_threshold":.9,
            "elapsed_seconds":time.perf_counter()-started,
            "gt_initial_seed":True,"second_step_seed_and_history_zero":True,
            "same_rgb_for_both_instances":True,"training_images":1,
            "novel_scene_generalization":False,"deployment_accuracy_result":False,
            "limitations":"Same artificial crossing fitted repeatedly; changing prompt/state verifies conditional learning only.",
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    torch.save({"schema":1,"kind":"smoke","optimizer_steps":args.updates,"model_config":{},"model":model.state_dict(),
                "diagnostic":result},outdir/"fixture.pt")
    Image.fromarray((rgb[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(outdir/"input.png")
    for k in range(2):
        Image.fromarray((second["mask_logits"][k,0].sigmoid().cpu().numpy()*255).astype(np.uint8)).save(outdir/f"instance_{k+1}_probability.png")
    (outdir/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":main()
