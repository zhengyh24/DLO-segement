"""One paired stage (5/10/20 complete cycles), resumable at cycle boundaries.

Overflow retries repeat the SAME accumulation group, preserving effective batches.
Retries are separately counted as extra compute/exposures, never hidden.
"""
import argparse,collections,json,os,random,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from sacm_frontend.losses import foreground_loss
from .common import load,delta,write,sha,now
from .data import manifests,schedule,sample
from .evaluate import evaluate,fixed


def save_ck(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def run(config,out,target):
    cfg=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert target in (5,10,20)
    for key in ('parent','synthetic_manifest','real_manifest'):
        assert sha(cfg[key])==cfg[key+'_sha256'],key+' changed'
    assert cfg['lr']==1e-4 and cfg['weight_decay']==.01 and cfg['accumulation']==4
    ds=manifests(cfg['synthetic_manifest'],cfg['real_manifest'])
    train={k:(root,[r for r in rows if r['split']=='train']) for k,(root,rows) in ds.items()}
    assert len(train['synthetic'][1])==2000 and len(train['real'][1])==64
    valroot,realrows=ds['real'];val=[r for r in realrows if r['split']=='validation']
    assert len(val)==25
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');model,parent=load(cfg['parent'],device,cfg['query_chunk'])
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    source={str(p):sha(p) for p in sorted(Path('sacm_fullframe_20261020').glob('*.py'))}
    metadata=dict(config=cfg,source_sha256=source,config_sha256=sha(config),created_at=now(),
                  initialization='same frozen synthetic B_control parent; optimizer reset',
                  selection='mean validation macro IoU at .5 across both inference modes; shared selected checkpoint for the 2 inference cells',
                  independent_scene_split=False,loss='BCE + 0.4 Dice',native_geometry=True)
    start_cycle=0;best=-1.;best_cycle=-1;actual_total=0;skips_total=0
    if (out/'last.pt').exists():
        old=json.loads((out/'run.json').read_text());assert old['config_sha256']==sha(config)
        assert old['source_sha256']==source,'Frozen source changed; do not resume'
        model,ck=load(out/'last.pt',device,cfg['query_chunk']);params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay']);optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start_cycle=ck['cycle']+1;best=ck['best'];best_cycle=ck['best_cycle'];actual_total=ck['actual_updates'];skips_total=ck['amp_skips']
        assert not (out/f'cycle_{start_cycle:03d}.json').exists(),'Partial commit; preserve and diagnose'
    else:
        assert not (out/'run.json').exists(),'Incomplete pre-cycle run; diagnose rather than overwrite'
        write(out/'run.json',metadata)
        baseline={mode:evaluate(model,valroot,val,mode,device) for mode in ('slide','full')}
        write(out/'before_training.json',dict(time=now(),evaluations=baseline))
    if start_cycle>=target:
        assert (out/f'stage_{target:02d}_completion.json').exists();return
    for cycle in range(start_cycle,target):
        if shutil.disk_usage('.').free < 3*1024**3+85*1024**2:raise RuntimeError('3GiB reserve plus atomic checkpoint headroom would be violated')
        started=time.time();torch.cuda.reset_peak_memory_stats();gpu_ms=0.;cycle_steps=0;cycle_skips=0
        totals=collections.Counter();usage=collections.Counter();presentations=collections.Counter();losses=[]
        entries=schedule(seed,cycle,len(train['synthetic'][1]),len(train['real'][1]))
        schedule_path=out/f'schedule_{cycle:03d}.json';write(schedule_path,entries)
        model.train()
        with (out/f'samples_{cycle:03d}.jsonl').open('x',encoding='utf-8') as log:
            for offset in range(0,len(entries),4):
                group=entries[offset:offset+4]
                for attempt in range(5):
                    optimizer.zero_grad(set_to_none=True);batch_losses=[]
                    for entry in group:
                        root,rows=train[entry['domain']];row=rows[entry['index']]
                        rgb,masks,valid,meta=sample(root,row,entry,cfg['arm'])
                        rgb=rgb[None].to(device);masks=masks.to(device);valid=valid.to(device)
                        event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True);event0.record()
                        with torch.autocast('cuda'):
                            logits=model(rgb);loss,_=foreground_loss(logits,masks,valid,False,0.)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                        scaler.scale(loss/4).backward();event1.record();event1.synchronize();gpu_ms+=event0.elapsed_time(event1)
                        batch_losses.append(float(loss.detach()));presentations[entry['domain']+':'+row['id']]+=1
                        for key in ('input_pixels','valid_pixels','encoder_padded_pixels'):totals['presented_'+key]+=meta[key]
                        log.write(json.dumps(dict(cycle=cycle,group=offset//4,attempt=attempt,**entry,**meta))+'\n')
                        del rgb,masks,valid,logits,loss
                    scaler.unscale_(optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(params,1.)
                    scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()<scale:
                        cycle_skips+=1
                        if attempt==4:raise RuntimeError('Repeated AMP overflow; preserve failure, no budget change')
                        continue
                    assert torch.isfinite(norm)
                    cycle_steps+=1;actual_total+=1;losses.extend(batch_losses)
                    for entry in group:usage[entry['domain']+':'+train[entry['domain']][1][entry['index']]['id']]+=1
                    break
                if offset%128==0:
                    log.flush();write(out/'progress.json',dict(time=now(),cycle=cycle,completed_samples=offset+4,
                          total_samples=4000,actual_updates=actual_total,cycle_amp_skips=cycle_skips,
                          train_loss=float(np.mean(batch_losses)),elapsed_seconds=time.time()-started,
                          gpu_active_seconds=gpu_ms/1000))
                    print(json.dumps(dict(cycle=cycle,samples=offset+4,updates=actual_total,loss=float(np.mean(batch_losses)))),flush=True)
        assert cycle_steps==1000 and sum(usage.values())==4000
        assert all(usage['synthetic:'+r['id']]==1 for r in train['synthetic'][1])
        skips_total+=cycle_skips;optimizer.zero_grad(set_to_none=True)
        train_seconds=time.time()-started
        evaluations={mode:evaluate(model,valroot,val,mode,device) for mode in ('slide','full')}
        score=float(np.mean([fixed(evaluations[m])['macro_iou'] for m in evaluations]))
        if score>best:best=score;best_cycle=cycle
        actual_saved={int(v['step']) for v in optimizer.state.values() if 'step' in v}
        assert actual_saved=={actual_total},(actual_saved,actual_total)
        result=dict(time=now(),cycle=cycle+1,selection_score=score,evaluations=evaluations,
                    samples=4000,actual_optimizer_updates=cycle_steps,total_actual_updates=actual_total,
                    amp_skips=cycle_skips,extra_presentations=4*cycle_skips,loss=float(np.mean(losses)),
                    unique_syn=2000,real_usage=dict(usage),presentation_usage=dict(presentations),
                    pixels=dict(totals),schedule_sha256=sha(schedule_path),train_wall_seconds=train_seconds,
                    gpu_train_seconds=gpu_ms/1000,total_wall_seconds=time.time()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        ck=dict(**delta(model,parent,cfg['parent'],cfg['parent_sha256']),cycle=cycle,best=best,best_cycle=best_cycle,
                actual_updates=actual_total,amp_skips=skips_total,config=cfg,source_sha256=source)
        save_ck(out/f'weights_cycle_{cycle+1:03d}.pt',ck)
        ck.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        save_ck(out/'last.pt',ck);write(out/f'cycle_{cycle:03d}.json',result)
        print(json.dumps(dict(cycle=cycle+1,score=score,best=best,best_cycle=best_cycle+1)),flush=True)
    write(out/f'stage_{target:02d}_completion.json',dict(time=now(),cycles=target,best_cycle=best_cycle+1,
          best_validation_score=best,last_validation_score=score,actual_updates=actual_total,amp_skips=skips_total,
          selected_checkpoint=str(out/f'weights_cycle_{best_cycle+1:03d}.pt'),last_checkpoint=str(out/f'weights_cycle_{target:03d}.pt')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cycles',type=int,required=True);a=p.parse_args()
    lock=Path(a.out).parent/(Path(a.out).name+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:run(a.config,a.out,a.cycles)
    except Exception as e:
        write(Path(a.out)/('failure_'+str(time.time_ns())+'.json'),dict(time=now(),error=repr(e),traceback=traceback.format_exc()));raise
    finally:lock.unlink()

if __name__=='__main__':main()
