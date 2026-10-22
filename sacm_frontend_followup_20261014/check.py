import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sacm_frontend.losses import foreground_loss as old
from sacm_frontend_followup_20261014.losses import foreground_loss as new
torch.manual_seed(20261014)
z=torch.randn(1,1,21,25,requires_grad=True);m=torch.rand(3,21,25)>.9;v=torch.ones(21,25,dtype=torch.bool);v[-3:]=False
for balanced in [False,True]:
 x,_=old(z,m,v,balanced,0);y,_=new(z,m,v,balanced,0,.25)
 assert torch.equal(x,y)
x,_=old(z,m,v,False,0);y,_=new(z,m,v,True,0,0)
assert torch.equal(x,y)
loss,_=new(z,m,v,True,0,.05);loss.backward()
assert torch.isfinite(z.grad).all() and not z.grad[0,0,-3:].any()
print('LOSS_CHECK_PASS: default exact parity, zero-weight control, finite gradient, padding ignored')
