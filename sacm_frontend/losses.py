"""Offline supervision only. Inference never imports this module."""
import torch
from torch.nn import functional as F

def erode(x):return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
def skeleton(x,iterations=10):
    skel=F.relu(x-F.max_pool2d(erode(x),3,1,1))
    for _ in range(iterations):
        x=erode(x);delta=F.relu(x-F.max_pool2d(erode(x),3,1,1));skel=skel+F.relu(delta-skel*delta)
    return skel
def foreground_loss(logits,instances,valid,balanced=False,topology=0.):
    if logits.shape[0]!=1:raise ValueError('Use accumulation for effective batch >1')
    z=logits[0,0].float();valid=valid.bool()
    target=instances.bool().any(0).float() if len(instances) else torch.zeros_like(z)
    p=z.sigmoid();bce=F.binary_cross_entropy_with_logits(z[valid],target[valid])
    dice=1-(2*(p*target*valid).sum()+1)/((p*valid).sum()+(target*valid).sum()+1)
    instance_term=z.sum()*0
    if balanced:
        terms=[];areas=[]
        for mask in instances.bool():
            keep=mask&valid
            if keep.any():terms.append(F.softplus(-z[keep]).mean());areas.append(keep.sum().float())
        if terms:
            area=torch.stack(areas);weights=(area/(area.mean()/10).clamp_min(1)).clamp(max=1)
            instance_term=(torch.stack(terms)*weights).sum()/weights.sum()
        background=valid&(target==0)
        if background.any():instance_term=instance_term+F.softplus(z[background]).mean()
    cl=z.sum()*0
    if topology:
        ys,xs=torch.where(valid);hh,ww=int(ys.max())+1,int(xs.max())+1
        pp,tt=p[:hh,:ww][None,None],target[:hh,:ww][None,None]
        sp,st=skeleton(pp),skeleton(tt)
        precision=((sp*tt).sum()+1)/(sp.sum()+1);recall=((st*pp).sum()+1)/(st.sum()+1)
        cl=1-2*precision*recall/(precision+recall)
    total=bce+.4*dice+.25*instance_term+topology*cl
    return total,{k:float(v.detach()) for k,v in dict(bce=bce,dice=dice,instance=instance_term,cldice=cl).items()}
