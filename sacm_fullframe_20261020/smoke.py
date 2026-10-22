"""Real four-sample accumulation and exact compact reload, not a scored experiment."""
import argparse
from pathlib import Path
import torch
from .preflight import SYN,REAL,PARENT,PARENT_SHA
from .data import manifests,schedule,sample
from .common import load,delta,write,now
from sacm_frontend.losses import foreground_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--arm',choices=['crop','full'],required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261020)
    device=torch.device('cuda');model,parent=load(PARENT,device);ds=manifests(SYN,REAL)
    train={k:(root,[r for r in rows if r['split']=='train']) for k,(root,rows) in ds.items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.);model.train();samples=[]
    for e in schedule(20261020,0,2000,64)[:4]:
        root,rows=train[e['domain']];x,m,v,meta=sample(root,rows[e['index']],e,a.arm);x=x[None].to(device)
        with torch.autocast('cuda'):z=model(x);loss,_=foreground_loss(z,m.to(device),v.to(device))
        scaler.scale(loss/4).backward();samples.append(dict(domain=e['domain'],**meta))
    scaler.unscale_(optimizer);assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    scaler.step(optimizer);scaler.update();assert {int(v['step']) for v in optimizer.state.values()}=={1}
    ck=delta(model,parent,PARENT,PARENT_SHA);torch.save(ck,out/'witness_delta.pt')
    restored,_=load(out/'witness_delta.pt',device)
    for k,v in model.state_dict().items():assert torch.equal(v,restored.state_dict()[k]),k
    model.eval();restored.eval()
    with torch.inference_mode(),torch.autocast('cuda'):u=model(x);v=restored(x)
    assert torch.equal(u,v)
    write(out/'receipt.json',dict(status='passed',time=now(),arm=a.arm,samples=samples,actual_optimizer_updates=1,
          delta_all_state_exact=True,reload_prediction_exact=True,accuracy_result=False))

if __name__=='__main__':main()
