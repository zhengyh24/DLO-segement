import argparse
from pathlib import Path
from .config import Config
from .io import write_json

def main():
    p=argparse.ArgumentParser(description='Native cable identity research prototype. Training is OFF by default.')
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('smoke','self-test','inventory','export-images','prepare','train','infer'):
        q=sub.add_parser(command);q.add_argument('--output',type=Path,required=True)
        if command=='self-test':q.add_argument('--group',choices=['graph','model','data','all'],default='all')
        if command in ('inventory','export-images','prepare'):q.add_argument('--dataset-root',type=Path)
        if command in ('export-images','prepare'):q.add_argument('--groups',type=Path,required=True)
        if command=='prepare':q.add_argument('--predictions',type=Path,required=True)
        if command in ('prepare','train'):q.add_argument('--config',type=Path)
        if command=='train':q.add_argument('--manifest',type=Path,required=True)
        if command in ('train','infer'):q.add_argument('--device',default='cpu')
        if command=='infer':
            for name in ('image','foreground','checkpoint'):q.add_argument('--'+name,type=Path,required=True)
            q.add_argument('--baseline',type=Path);q.add_argument('--support',type=Path)
    a=p.parse_args();config=Config.read(a.config) if getattr(a,'config',None) else Config()
    if hasattr(a,'dataset_root') and a.dataset_root is None:a.dataset_root=Path(config.dataset_root)
    if a.command=='smoke':
        a.output.mkdir(parents=True,exist_ok=True);result={'kind':'cli_smoke_only','training_performed':False}
        write_json(a.output/'cli_smoke.json',result)
    elif a.command=='self-test':
        from .tests import run_tests
        a.output.mkdir(parents=True,exist_ok=True);result=run_tests(a.group,report=True)
        write_json(a.output/f'{a.group}_self_test.json',result)
        if not result['passed']:raise SystemExit(1)
    elif a.command=='inventory':
        from .data import inventory
        result=inventory(a.dataset_root,a.output)
    elif a.command=='export-images':
        from .data import export_images
        result=export_images(a.dataset_root,a.groups,a.output)
    elif a.command=='prepare':
        from .data import prepare
        result=prepare(a.dataset_root,a.groups,a.predictions,a.output,config)
    elif a.command=='train':
        from .train import train
        result=train(a.manifest,a.output,config,a.device)
    else:
        from .pipeline import infer
        result=infer(a.image,a.foreground,a.checkpoint,a.output,a.baseline,a.support,a.device)
    print(result)
