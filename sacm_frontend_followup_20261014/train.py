"""Portable native foreground trainer; separate output for every ablation/seed."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sacm_frontend.model import CableForeground
from sacm_frontend.data import NativeCrops,read_manifest
from sacm_frontend_followup_20261014.losses import foreground_loss
from sacm_frontend.evaluate import evaluate

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--encoder',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--fork-from',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--allow-condition-pilot',action='store_true')
    p.add_argument('--software-smoke',action='store_true',help='Permits tiny random encoder only; no accuracy claim')
    a=p.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8'))
    if cfg['tile']%16 or not 0<cfg['stride']<=cfg['tile']:raise ValueError('Invalid tile/stride')
    if cfg['epochs']<1 or cfg['accumulation']<1:raise ValueError('Invalid training budget')
    if cfg['backbone']=='tiny_test' and not a.software_smoke:raise ValueError('Tiny model is test-only')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    device=torch.device(a.device);obj=read_manifest(a.manifest,a.allow_condition_pilot)
    ds=NativeCrops(a.manifest,cfg['tile'],cfg['seed'],a.allow_condition_pilot)
    val=[r for r in obj['records'] if r['split']=='validation']
    if not val:raise ValueError('Validation required')
    if a.resume and a.fork_from:raise ValueError('Resume and fork are mutually exclusive')
    if a.resume:
        if a.resume.resolve()!=(a.out/'last.pt').resolve() or not a.out.is_dir():
            raise ValueError('Resume only the current last.pt in its original output directory; use a new run for forks')
        if not (a.out/'best.pt').is_file():
            raise ValueError('Incomplete checkpoint pair: best.pt missing; preserve this run for recovery')
    else:
        a.out.mkdir(parents=True,exist_ok=False)
    mc=dict(backbone=cfg['backbone'],detail=cfg['detail'],recompute=True)
    model=CableForeground(**mc);start=0;best=-1.;best_epoch=-1;provenance={}
    ck=None
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        if ck['config']!=cfg or ck['manifest_sha256']!=sha(a.manifest):raise ValueError('Resume config/data changed')
        saved_best=torch.load(a.out/'best.pt',map_location='cpu',weights_only=True)
        if 'best_epoch' not in ck or saved_best['epoch']!=ck['best_epoch']:
            raise ValueError('Checkpoint pair mismatch: last.pt references a different best epoch; preserve for recovery')
        del saved_best
        if any(int(p.stem.split('_')[1])>ck['epoch'] for p in a.out.glob('epoch_*.json')):
            raise ValueError('Newer epoch records exist; refuse history overwrite. Preserve and inspect interrupted run')
        if ck['epoch']+1>=cfg['epochs']:
            print('RUN_ALREADY_COMPLETE; no files modified',flush=True)
            return
        model.load_state_dict(ck['model']);start=ck['epoch']+1;best=ck['best'];best_epoch=ck['best_epoch'];provenance=ck['provenance']
    elif a.fork_from:
        ck=torch.load(a.fork_from,map_location='cpu',weights_only=True)
        if ck['manifest_sha256']!=sha(a.manifest):raise ValueError('Fork data mismatch')
        for key,value in ck['config'].items():
            if key not in ('epochs','balanced','balanced_weight') and cfg.get(key)!=value:
                raise ValueError('Fork may only change budget and balanced loss weight')
        if cfg['epochs']<=ck['epoch']+1:raise ValueError('Fork needs additional epochs')
        model.load_state_dict(ck['model'],strict=True);start=ck['epoch']+1
        provenance=dict(ck['provenance'],fork_parent=str(a.fork_from),fork_parent_sha256=sha(a.fork_from),fork_parent_epoch=ck['epoch'],selection='best among new continuation epochs only')
    elif a.encoder:
        provenance=model.load_encoder(a.encoder);provenance['sha256']=sha(a.encoder)
    elif not (a.software_smoke and cfg['backbone']=='tiny_test'):
        raise ValueError('Pretrained encoder required; no silent random production run')
    model.to(device)
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    if ck:
        optim.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda' and ck['cuda_rng']:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    meta=dict(config=cfg,model_config=mc,provenance=provenance,start_epoch=start,new_epochs=cfg['epochs']-start,manifest_sha256=sha(a.manifest),
        grouping_kind=obj['grouping_kind'],independent_scene_split=obj['independent_scene_split'],
        software_smoke=a.software_smoke,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        train_images=len(ds),validation_images=len(val),torch=str(torch.__version__),frozen_dependency_sha256={p.name:sha(p) for p in Path('sacm_frontend').glob('*.py')},
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (a.out/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for epoch in range(start,cfg['epochs']):
        begin=time.time();model.train();ds.epoch=epoch;optim.zero_grad(set_to_none=True);losses=[]
        order=np.random.default_rng(cfg['seed']+epoch).permutation(len(ds));acc=cfg['accumulation']
        for step,index in enumerate(order):
            rgb,masks,valid=ds[int(index)];rgb=rgb[None].to(device);masks=masks.to(device);valid=valid.to(device)
            window=min(acc,len(ds)-(step//acc)*acc)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(rgb);loss,terms=foreground_loss(logits,masks,valid,cfg['balanced'],cfg['topology'],cfg.get('balanced_weight',.25))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss/window).backward();losses.append(float(loss.detach()))
            if (step+1)%acc==0 or step+1==len(ds):
                scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                scaler.step(optim);scaler.update();optim.zero_grad(set_to_none=True)
            if step%10==0:print(json.dumps(dict(epoch=epoch,step=step,loss=losses[-1],terms=terms)),flush=True)
        result=evaluate(model,a.manifest.parent,val,cfg,device);improved=result['macro_iou']>best
        if improved:best_epoch=epoch
        best=max(best,result['macro_iou']);result.update(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.time()-begin,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
        (a.out/f'epoch_{epoch:03d}.json').write_text(json.dumps(result,indent=2))
        state=dict(model=model.state_dict(),optimizer=optim.state_dict(),scaler=scaler.state_dict(),
            model_config=mc,config=cfg,epoch=epoch,best=best,best_epoch=best_epoch,provenance=provenance,
            manifest_sha256=sha(a.manifest),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        torch.save(state,a.out/'last.tmp');(a.out/'last.tmp').replace(a.out/'last.pt')
        if improved:
            torch.save(dict(model=model.state_dict(),model_config=mc,config=cfg,provenance=provenance,epoch=epoch),a.out/'best.tmp')
            (a.out/'best.tmp').replace(a.out/'best.pt')
        print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    (a.out/'completion.json').write_text(json.dumps(dict(epochs=cfg['epochs'],start_epoch=start,new_epochs=cfg['epochs']-start,best_validation_iou=best,
        software_smoke=a.software_smoke,independent_scene_split=obj['independent_scene_split'])))
if __name__=='__main__':main()
