"""Same PairNet/default objective, with epoch resume and persistent progress."""
import argparse
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from curvi_identity.config import Config
from curvi_identity.data import PairDataset
from curvi_identity.io import read_json
from curvi_identity.model import PairNet
from curvi_identity.train import evaluate,validate_manifest
from run_curvi_2000 import save,sha,DEFAULT_OUT


def checkpoint(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def main(out):
    cfg=Config.read(out/'config.json')
    if not cfg.training_enabled: raise RuntimeError('Training disabled')
    manifest=out/'prepared/manifest.json';obj=read_json(manifest);validate_manifest(obj,cfg)
    digest=sha(manifest);fit=out/'fit';fit.mkdir(exist_ok=True)
    torch.set_num_threads(4);random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    trainset=PairDataset(manifest,'train',cfg.seed);valset=PairDataset(manifest,'validation',cfg.seed)
    generator=torch.Generator().manual_seed(cfg.seed)
    loader=DataLoader(trainset,batch_size=cfg.batch_size,shuffle=True,num_workers=0,generator=generator)
    validation=DataLoader(valset,batch_size=cfg.batch_size,shuffle=False,num_workers=0)
    model=PairNet().to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    history=[];steps=0;best=float('inf');first=0
    if (fit/'last.pt').exists():
        last=torch.load(fit/'last.pt',map_location='cpu',weights_only=True)
        if last['manifest_sha256']!=digest or last['config']!=cfg.to_dict(): raise ValueError('Resume input/config changed')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);history=last['history']
        steps=last['optimizer_steps'];best=last['best_bce'];first=last['epoch']
        generator.set_state(last['generator_state']);trainset.rng.bit_generator.state=last['augment_rng']
        random.setstate(last['python_rng']);ns=last['numpy_rng'];np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),*ns[2:]))
        torch.set_rng_state(last['torch_rng']);torch.cuda.set_rng_state_all(last['cuda_rng'])
        print({'resume_after_epoch':first,'steps':steps},flush=True)
    elif (fit/'history.json').exists(): raise RuntimeError('History exists without epoch resume checkpoint; inspect interrupted run')
    if not (fit/'initial.json').exists():
        labels=[r['label'] for r in obj['records'] if r['split']=='train'];prior=float(np.mean(labels))
        val_y=np.asarray([r['label'] for r in obj['records'] if r['split']=='validation'])
        prior_bce=float(-(val_y*np.log(prior)+(1-val_y)*np.log(1-prior)).mean())
        save(fit/'initial.json',dict(untrained=evaluate(model,validation,'cuda'),train_positive_prior=prior,
            constant_prior_validation_bce=prior_bce,validation_positive_fraction=float(val_y.mean()),
            parameters=sum(p.numel() for p in model.parameters()),manifest=manifest.as_posix(),manifest_sha256=digest))
    for epoch in range(first,cfg.epochs):
        epoch_start=time.perf_counter();model.train();total=0.;count=0
        for batch,(s,l,g,y) in enumerate(loader,1):
            s,l,g,y=(a.to('cuda') for a in (s,l,g,y));optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(s,l,g),y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step();steps+=1;total+=float(loss.detach())*len(y);count+=len(y)
            if batch==1 or batch%100==0:
                progress=dict(epoch=epoch+1,epochs=cfg.epochs,batch=batch,batches=len(loader),optimizer_steps=steps,
                    train_bce_so_far=total/count,epoch_seconds=time.perf_counter()-epoch_start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                save(fit/'progress.json',progress);print(progress,flush=True)
        metrics=evaluate(model,validation,'cuda')
        row=dict(epoch=epoch+1,train_bce=total/count,validation=metrics,optimizer_steps=steps,seconds=time.perf_counter()-epoch_start)
        history.append(row)
        common=dict(schema=1,architecture='PairNet-v1',trained=True,optimizer_steps=steps,model=model.state_dict(),
            config=cfg.to_dict(),epoch=epoch+1,validation=metrics,selection='minimum validation bidirectional BCE',
            prepared_manifest=manifest.as_posix(),manifest_sha256=digest)
        if metrics['bce']<best:
            best=metrics['bce'];checkpoint(fit/'best.pt',common)
        ns=np.random.get_state()
        checkpoint(fit/'last.pt',dict(common,optimizer=optimizer.state_dict(),history=history,best_bce=best,
            generator_state=generator.get_state(),augment_rng=trainset.rng.bit_generator.state,python_rng=random.getstate(),
            numpy_rng=(ns[0],ns[1].tolist(),*ns[2:]),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        save(fit/'history.json',history);print(row,flush=True)
    save(fit/'completion.json',dict(optimizer_steps=steps,epochs=cfg.epochs,best_validation_bce=best,
        current_invocation_seconds=time.perf_counter()-started,total_epoch_seconds=sum(r['seconds'] for r in history),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        manifest_sha256=digest,scope='Single-seed condition-holdout validation, not independent test'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUT);a=p.parse_args();main(a.output.resolve())
