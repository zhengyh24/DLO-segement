"""Read-only data audit and isolated numerical/VRAM witnesses; never a training result."""
import argparse,collections,gc,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .common import load,sha,write,now
from .data import manifests,sample,schedule
from .model import NativeForeground
from sacm_frontend.model import CableForeground
from sacm_frontend.data import load_record
from sacm_frontend.losses import foreground_loss

PARENT='runs/followup_20261014/continuation_B_control/best.pt'
PARENT_SHA='aba103d22cda8cdf1521f608900ce54b8f8dd98070bb25787e11ed3b8cbd9baf'
SYN='data/pilot2300/manifest.json'
REAL='data/real_adaptation_20261014/manifest.json'


def audit(out):
    assert sha(PARENT)==PARENT_SHA
    assert sha(SYN)=='f1a7826c6e45055eda3581b94862ab30b6f2c1614570d2c853f530e1c89ae473'
    assert sha(REAL)=='5613326d6e3d308ac979243c1cb8b5e216d0b93959814c05f05add019a8b84a3'
    ds=manifests(SYN,REAL);summary={}
    for domain,(root,rows) in ds.items():
        dimensions=collections.Counter();splits=collections.Counter()
        for row in rows:
            rgb,masks=load_record(root,row);splits[row['split']]+=1
            dimensions[(row['split'],*rgb.shape[:2])]+=1
        summary[domain]=dict(images=len(rows),splits=dict(splits),dimensions=[dict(split=k[0],h=k[1],w=k[2],n=v) for k,v in sorted(dimensions.items())])
    assert summary['synthetic']['splits']['train']==2000 and summary['real']['splits']['train']==64
    free=shutil.disk_usage('.').free
    assert free>=3*1024**3+250*1024**2,'Insufficient reserve for initial five-cycle pair'
    write(out,dict(time=now(),status='passed',datasets=summary,parent_sha256=PARENT_SHA,
          free_bytes=free,torch=torch.__version__,cuda=torch.version.cuda,
          gpus=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())],
          process_snapshot=subprocess.check_output(['ps','-eo','pid,args'],text=True),
          gpu_snapshot=subprocess.check_output(['nvidia-smi'],text=True)))


def mechanics(out):
    torch.manual_seed(20261020);torch.set_num_threads(4)
    old=CableForeground('tiny_test',True,False)
    new=NativeForeground('tiny_test',True,False,7);new.load_state_dict(old.state_dict())
    # Nonzero relative embeddings exercise decomposed biases and their interpolation.
    with torch.no_grad():
        for block in old.encoder.blocks:
            block.attn.rel_pos_h.normal_(0,.1);block.attn.rel_pos_w.normal_(0,.1)
    new.load_state_dict(old.state_dict())
    x=torch.rand(1,3,64,80);a=old(x);b=new(x)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    a.square().mean().backward();b.square().mean().backward();maxgrad=0.
    for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6)
            maxgrad=max(maxgrad,float((p.grad-q.grad).abs().max()))
    z=torch.randn(1,1,70,90,requires_grad=True);valid=torch.zeros(70,90,dtype=torch.bool);valid[:65,:81]=True
    masks=torch.zeros(1,70,90,dtype=torch.bool);masks[0,10:30,15:40]=True
    loss,_=foreground_loss(z,masks,valid);loss.backward()
    assert torch.count_nonzero(z.grad[0,0][~valid])==0
    odd=new(torch.rand(1,3,65,81));assert odd.shape==(1,1,65,81)
    # Check semantic interpolation target is padded 80x96, not a squeezed 65x81 grid.
    captured={}
    def hook(module,args,output):captured['semantic']=output.detach()
    handle=new.semantic.register_forward_hook(hook)
    rgb=torch.rand(1,3,65,81);pred=new(rgb);handle.remove()
    semantic=torch.nn.functional.interpolate(captured['semantic'],size=(80,96),mode='bilinear',align_corners=False)[...,:65,:81]
    expected=new.head(torch.cat([semantic,new.detail(rgb)],1))
    torch.testing.assert_close(pred,expected,rtol=0,atol=0)
    counts=collections.Counter()
    for cycle in range(20):
        rows=schedule(20261020,cycle,2000,64)
        assert len(rows)==4000
        for i in range(0,4000,4):assert collections.Counter(r['domain'] for r in rows[i:i+4])=={'synthetic':2,'real':2}
        counts.update(r['index'] for r in rows if r['domain']=='real')
        assert max(counts.values())-min(counts.values())<=1
    write(out,dict(status='passed',time=now(),fp32_forward_maxdiff=float((a-b).abs().max()),
          fp32_gradient_maxdiff=maxgrad,odd_shape=[65,81],padding_loss_gradient_zero=True,
          global_attention='all keys for each query; query-chunked exact algebra',schedule_20_cycles_verified=True))


def witness(out,h,w,row_id=None,parity=False):
    torch.manual_seed(20261020);torch.set_num_threads(4)
    device=torch.device('cuda');model,parent=load(PARENT,device)
    if row_id:
        ds=manifests(SYN,REAL);root,rows=ds['real'];row=next(r for r in rows if r['id']==row_id)
        x,masks,valid,meta=sample(root,row,dict(flip_h=False,flip_v=False,crop_seed=0),'full')
        h,w=x.shape[-2:]
    else:
        x=torch.rand(3,h,w);masks=torch.zeros(2,h,w,dtype=torch.bool)
        masks[0,h//3:h//3+2,:]=True;masks[1,:,w//2:w//2+2]=True;valid=torch.ones(h,w,dtype=torch.bool)
    x=x[None].to(device);masks=masks.to(device);valid=valid.to(device)
    torch.cuda.reset_peak_memory_stats();start=time.time();params=[p for p in model.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    model.train()
    shapes={}
    handles=[]
    for i,projection in enumerate(model.projections):
        def hook(module,args,output,index=i):shapes[str(index)]=list(output.shape)
        handles.append(projection.register_forward_hook(hook))
    with torch.autocast('cuda'):
        logits=model(x);loss,_=foreground_loss(logits,masks,valid)
    assert logits.shape==(1,1,h,w) and torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(optim)
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    nonzero=sum(bool(p.grad.abs().sum()>0) for p in params if p.grad is not None)
    assert nonzero>0
    torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(optim);scaler.update()
    for handle in handles:handle.remove()
    optim.zero_grad(set_to_none=True);model.eval()
    with torch.inference_mode(),torch.autocast('cuda'):prediction=model(x)
    assert prediction.shape==logits.shape and torch.isfinite(prediction).all()
    for name,p in model.named_parameters():
        if not p.requires_grad:assert torch.equal(p.cpu(),parent['model'][name])
    torch.cuda.synchronize();elapsed=time.time()-start
    receipt=dict(status='passed',time=now(),hw=[h,w],row_id=row_id,seconds=elapsed,
                 peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                 loss=float(loss),nonzero_gradient_parameters=nonzero,projection_shapes=shapes,
                 source='real RGB and aligned GT' if row_id else 'random synthetic mechanics witness',
                 accuracy_result=False,forward_backward_prediction=True)
    if parity:
        del model,optim,params,logits,prediction,loss;gc.collect();torch.cuda.empty_cache()
        old=CableForeground(**parent['model_config']).to(device);old.load_state_dict(parent['model']);old.eval()
        with torch.inference_mode(),torch.autocast('cuda'):a=old(x).float().cpu()
        del old;gc.collect();torch.cuda.empty_cache()
        new,_=load(PARENT,device);new.eval()
        with torch.inference_mode(),torch.autocast('cuda'):b=new(x).float().cpu()
        torch.testing.assert_close(a,b,rtol=.01,atol=.015)
        receipt['amp_original_logit_maxdiff']=float((a-b).abs().max())
        receipt['amp_original_probability_maxdiff']=float((a.sigmoid()-b.sigmoid()).abs().max())
    write(out,receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','mechanics','witness']);p.add_argument('--out',required=True)
    p.add_argument('--height',type=int,default=1024);p.add_argument('--width',type=int,default=1024);p.add_argument('--row-id');p.add_argument('--parity',action='store_true');a=p.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    try:
        if a.action=='audit':audit(a.out)
        elif a.action=='mechanics':mechanics(a.out)
        else:witness(a.out,a.height,a.width,a.row_id,a.parity)
    except Exception as e:
        write(a.out,dict(status='failed',time=now(),error=repr(e),traceback=traceback.format_exc(),requested_hw=[a.height,a.width],row_id=a.row_id));raise

if __name__=='__main__':main()
