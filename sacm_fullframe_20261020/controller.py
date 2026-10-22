"""Bounded paired controller. Does not launch seeds/data expansion before evidence review."""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from .common import write,now


def decision(root,stage):
    scores={arm:[json.loads((root/arm/f'cycle_{i:03d}.json').read_text())['selection_score'] for i in range(stage)] for arm in ('crop','full')}
    improving={k:max(v[-2:])-max(v[:-2])>=.002 for k,v in scores.items()}
    degraded={k:all(x<max(v[:-2])-.03 for x in v[-2:]) for k,v in scores.items()}
    return dict(stage=stage,extend=stage<20 and any(improving.values()) and not any(degraded.values()),
                improving=improving,persistent_degradation=degraded,scores=scores,
                rule='extend both if either last-two maximum exceeds earlier best by >=0.002; stop both if either last two each >=0.03 below earlier best; hard cap20')


def pair(root,command):
    jobs=[]
    for gpu,arm in enumerate(('crop','full')):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        log=(root/(arm+'_controller.log')).open('a',encoding='utf-8')
        process=subprocess.Popen(command(arm),env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((arm,process,log))
    while any(p.poll() is None for _,p,_ in jobs):
        write(root/'controller_state.json',dict(time=now(),stage='running',processes=[dict(arm=arm,pid=p.pid,returncode=p.poll()) for arm,p,_ in jobs]))
        time.sleep(30)
    codes={arm:p.returncode for arm,p,_ in jobs}
    for _,_,log in jobs:log.close()
    assert all(v==0 for v in codes.values()),codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    lock=root/'controller.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for stage in (5,10,20):
            # Forecast remaining cycle weights + optimizer temp files, logs and final masks.
            current=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            projected=stage*2*13*1024**2+2*40*1024**2+100*1024**2
            if shutil.disk_usage('.').free-max(0,projected-current)<3*1024**3:
                write(root/'capacity_stop.json',dict(time=now(),stage_requested=stage,free=shutil.disk_usage('.').free,projected_total=projected,current=current))
                if stage==5:raise RuntimeError('No room for initial pair plus 3GiB reserve')
                stage=5 if stage==10 else 10
                break
            pair(root,lambda arm:[sys.executable,'-m','sacm_fullframe_20261020.train','--config',str(root/f'{arm}_config.json'),'--out',str(root/arm),'--cycles',str(stage)])
            d=decision(root,stage);write(root/f'decision_{stage:02d}.json',dict(time=now(),**d))
            if not d['extend']:break
        pair(root,lambda arm:[sys.executable,'-m','sacm_fullframe_20261020.final_evaluation','--root',str(root),'--arm',arm,'--cycles',str(stage)])
        write(root/'controller_state.json',dict(time=now(),stage='paired_training_and_foreground_complete',cycles=stage,
              next='Archive native masks; paired region diagnosis and frozen backend review, then decide seeds/expansion from validation evidence.',replace_frontend=False))
    except Exception as e:
        write(root/'controller_failure.json',dict(time=now(),error=repr(e),traceback=traceback.format_exc()));raise
    finally:lock.unlink()

if __name__=='__main__':main()
