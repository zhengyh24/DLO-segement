"""Local R29 shallow/deep fusion adaptation, not the official ISDNet module.

Zero-initialized residual spatial fusion preserves the original parent logits.
Original sacm_frontend/model.py and all old checkpoint formats remain untouched.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from sacm_frontend.model import CableForeground

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

class ResidualFusionHead(nn.Module):
 def __init__(self,base):
  super().__init__();assert base.in_channels==32 and base.out_channels==1
  self.base=base
  self.residual=nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1))
  nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
 def forward(self,features):return self.base(features)+self.residual(features)

def add_fusion(model):
 assert model.has_detail and isinstance(model.head,nn.Conv2d)
 old=model.head;model.head=ResidualFusionHead(old).to(device=old.weight.device,dtype=old.weight.dtype)
 return model

def build(config):
 cfg=dict(config);fusion=cfg.pop('residual_fusion',False);model=CableForeground(**cfg)
 return add_fusion(model) if fusion else model

def pack(model,parent,base_path,expected_sha):
 assert sha(base_path)==expected_sha
 keys={n for n,p in model.named_parameters() if p.requires_grad}|{n for n,_ in model.named_buffers()}
 state=model.state_dict();base=parent['model']
 for k,v in state.items():
  if k not in keys:assert k in base and torch.equal(v.detach().cpu(),base[k].cpu()),'Frozen tensor changed: '+k
 return dict(format='sacm_residual_fusion_delta_v1',base_checkpoint=str(base_path),base_sha256=expected_sha,model_delta={k:v.detach().cpu().clone() for k,v in state.items() if k in keys})

def load_model(path,device):
 ck=torch.load(path,map_location='cpu',weights_only=True);model=build(ck['model_config'])
 if ck.get('format')=='sacm_residual_fusion_delta_v1':
  assert Path(path).resolve()!=Path(ck['base_checkpoint']).resolve()
  assert sha(ck['base_checkpoint'])==ck['base_sha256'],'Base checkpoint SHA mismatch'
  base=torch.load(ck['base_checkpoint'],map_location='cpu',weights_only=True);assert 'model' in base
  state=dict(base['model'])
  if ck['model_config'].get('residual_fusion',False):
   for k in ['weight','bias']:state['head.base.'+k]=state.pop('head.'+k)
  expected=set(model.state_dict());assert set(ck['model_delta'])<=expected
  trainable={n for n,p in model.named_parameters() if p.requires_grad};assert trainable<=set(ck['model_delta']),'Missing trainable tensor'
  state.update(ck['model_delta']);assert set(state)==expected
 elif 'model' in ck:state=ck['model']
 else:raise ValueError('Only full-parent or residual-fusion v1 checkpoint accepted')
 model.load_state_dict(state,strict=True)
 return model.to(device),ck
