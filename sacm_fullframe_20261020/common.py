import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
from .model import NativeForeground


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def now(): return datetime.now(timezone.utc).isoformat()


def load(path,device,query_chunk=128):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('format')=='native_fullframe_delta_v1':
        assert sha(ck['base_checkpoint'])==ck['base_sha256']
        base=torch.load(ck['base_checkpoint'],map_location='cpu',weights_only=True)
        state=dict(base['model']);state.update(ck['model_delta'])
    else:
        assert 'model' in ck,'Require frozen full parent or native delta'
        state=ck['model']
    model=NativeForeground(**ck['model_config'],query_chunk=query_chunk)
    model.load_state_dict(state,strict=True)
    return model.to(device),ck


def delta(model,parent,path,parent_sha):
    keys={n for n,p in model.named_parameters() if p.requires_grad}|{n for n,_ in model.named_buffers()}
    state=model.state_dict()
    for k,v in state.items():
        if k not in keys:
            assert torch.equal(v.detach().cpu(),parent['model'][k]),'Frozen parameter changed: '+k
    return dict(format='native_fullframe_delta_v1',base_checkpoint=path,base_sha256=parent_sha,
                model_config=parent['model_config'],model_delta={k:state[k].detach().cpu().clone() for k in keys})
