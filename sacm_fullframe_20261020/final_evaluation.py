"""After training has stopped: all four train/inference cells, best and last."""
import argparse,json,time
from pathlib import Path
import torch
from .common import load,write,sha,now
from .data import manifests
from .evaluate import evaluate,fixed,select_threshold


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',choices=['crop','full'],required=True)
    p.add_argument('--cycles',type=int,required=True);a=p.parse_args();root=Path(a.root);run=root/a.arm
    cfg=json.loads((root/f'{a.arm}_config.json').read_text());ds=manifests(cfg['synthetic_manifest'],cfg['real_manifest'])
    stage=json.loads((run/f'stage_{a.cycles:02d}_completion.json').read_text());out=root/f'{a.arm}_evaluation';out.mkdir(exist_ok=True)
    device=torch.device('cuda');torch.set_num_threads(4);summary=[]
    for label,checkpoint in [('best',stage['selected_checkpoint']),('last',stage['last_checkpoint'])]:
        model,ck=load(checkpoint,device,cfg['query_chunk'])
        for mode in ('slide','full'):
            dest=out/label/mode;dest.mkdir(parents=True,exist_ok=True)
            root_real,rows=ds['real'];valrows=[r for r in rows if r['split']=='validation'];testrows=[r for r in rows if r['split']=='test']
            start=time.time();cal=evaluate(model,root_real,valrows,mode,device);threshold=select_threshold(cal)
            write(dest/'calibration.json',dict(**cal,selected_threshold=threshold,checkpoint_sha256=sha(checkpoint)))
            for split,rr in [('validation',valrows),('historical_check',testrows)]:
                # Re-run RGB-only inference for compact native masks at only the two locked thresholds.
                result=evaluate(model,root_real,rr,mode,device,[.5,threshold],dest/f'{split}_masks')
                write(dest/f'{split}.json',result)
            synroot,synrows=ds['synthetic'];syn=evaluate(model,synroot,[r for r in synrows if r['split']=='validation'],mode,device,[.5,threshold])
            write(dest/'synthetic_validation.json',syn)
            summary.append(dict(arm=a.arm,checkpoint_kind=label,checkpoint=checkpoint,cycle=ck['cycle']+1,mode=mode,
                                validation_fixed05=fixed(cal),validation_threshold=threshold,seconds=time.time()-start))
        del model;torch.cuda.empty_cache()
    write(out/'completion.json',dict(time=now(),records=summary,independent_scene_split=False,
          status='foreground_evaluation_complete',backend_evaluation_complete=False,replace_frontend=False))

if __name__=='__main__':main()
