"""Explicitly gated training entry. Importing this module never trains."""
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data import PairDataset
from .io import read_json,write_json,new_directory
from .model import PairNet

def reverse_batch(s,l,g):
    order=[0,1,2,3,4,6,5];r=g.clone();r[:,0:2]*=-1
    r[:,3:5]=g[:,5:7];r[:,5:7]=g[:,3:5];r[:,7:9]=g[:,[8,7]];r[:,9]*=-1;r[:,10:12]=g[:,[11,10]]
    return s[:,order],l[:,order],r

def validate_manifest(obj,config):
    if obj.get('kind')!='predicted_native_pair_supervision' or obj.get('real_subset_used') is not False:
        raise ValueError('Require native prediction-supervised synthetic manifest')
    for field in ('small_size','large_size','min_arc','junction_radius','max_gap','max_candidates','purity'):
        if obj['config'][field]!=getattr(config,field):raise ValueError(f'Prepared data configuration mismatch: {field}')
    groups={};hashes={}
    for r in obj['images']:
        split=r['split']
        if split not in ('train','validation'):raise ValueError('Training preparation must exclude test')
        for table,key in ((groups,r['group_id']),(hashes,r['rgb_sha256'])):
            if key in table and table[key]!=split:raise ValueError('Cross-split group or identical RGB leakage')
            table[key]=split
    for split in ('train','validation'):
        labels={r['label'] for r in obj['records'] if r['split']==split}
        if labels!={0,1}:raise ValueError(f'{split} must contain both same-instance and different-instance pairs')

def evaluate(model,loader,device):
    model.eval();loss_sum=0.;total=0;tp=fp=fn=tn=0
    with torch.inference_mode():
        for s,l,g,y in loader:
            s,l,g,y=(a.to(device) for a in (s,l,g,y))
            a=model(s,l,g);b=model(*reverse_batch(s,l,g))
            loss_sum+=float((torch.nn.functional.binary_cross_entropy_with_logits(a,y,reduction='sum')+
                            torch.nn.functional.binary_cross_entropy_with_logits(b,y,reduction='sum'))/2)
            p=torch.minimum(a.sigmoid(),b.sigmoid())>=.5;positive=y>.5
            tp+=int((p&positive).sum());fp+=int((p&~positive).sum());fn+=int((~p&positive).sum());tn+=int((~p&~positive).sum());total+=len(y)
    return {'bce':loss_sum/total,'pairs':total,'tp_at_0_5':tp,'fp_at_0_5':fp,'fn_at_0_5':fn,'tn_at_0_5':tn,
            'scope':'pair classification only, not instance segmentation accuracy'}

def train(manifest,output,config=Config(),device='cpu'):
    if not config.training_enabled:
        raise RuntimeError('Training disabled. Wait for user confirmation of dataset completion; then explicitly enable in a NEW config.')
    obj=read_json(manifest);validate_manifest(obj,config)
    trainset=PairDataset(manifest,'train',config.seed);valset=PairDataset(manifest,'validation',config.seed)
    out=new_directory(output);write_json(out/'config.json',config.to_dict());shutil.copyfile(manifest,out/'prepared_manifest.json')
    source=out/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,source/p.name)
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config.seed)
    generator=torch.Generator().manual_seed(config.seed)
    loader=DataLoader(trainset,batch_size=config.batch_size,shuffle=True,num_workers=0,generator=generator)
    validation=DataLoader(valset,batch_size=config.batch_size,shuffle=False,num_workers=0)
    model=PairNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    best=float('inf');steps=0;history=[]
    for epoch in range(config.epochs):
        model.train();total=0.;count=0
        for s,l,g,y in loader:
            s,l,g,y=(a.to(device) for a in (s,l,g,y));optimizer.zero_grad(set_to_none=True)
            logits=model(s,l,g);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradients')
            optimizer.step();steps+=1;total+=float(loss.detach())*len(y);count+=len(y)
        metrics=evaluate(model,validation,device)
        row={'epoch':epoch+1,'train_bce':total/count,'validation':metrics,'optimizer_steps':steps}
        history.append(row);write_json(out/'history.json',history);print(row,flush=True)
        if metrics['bce']<best:
            best=metrics['bce']
            torch.save({'schema':1,'architecture':'PairNet-v1','trained':True,'optimizer_steps':steps,
                        'model':model.state_dict(),'config':config.to_dict(),'epoch':epoch+1,
                        'validation':metrics,'selection':'minimum validation bidirectional BCE',
                        'prepared_manifest':'prepared_manifest.json'},out/'best.pt')
    return {'output':str(out),'optimizer_steps':steps,'best_validation_bce':best}
