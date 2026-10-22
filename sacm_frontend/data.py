"""Supervised native crop loading; not imported by predict.py."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def read_manifest(path,allow_condition=False):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if obj['grouping_kind']=='condition_holdout_pilot' and not allow_condition:
        raise ValueError('Condition holdout is NOT independent: pass --allow-condition-pilot explicitly')
    if obj['grouping_kind'] not in ('verified_scene','condition_holdout_pilot'):
        raise ValueError('Unsupported grouping protocol')
    if obj['grouping_kind']=='condition_holdout_pilot' and obj.get('independent_scene_split') is not False:
        raise ValueError('Condition holdout must disclaim independence')
    seen={};names=set();hashes={}
    for r in obj['records']:
        if r['split'] not in ('train','validation','test'):raise ValueError('Bad split')
        if r['id'] in names:raise ValueError('Repeated sample')
        names.add(r['id'])
        for store,key in ((seen,r['group_id']),(hashes,r['rgb_sha256'])):
            if key in store and store[key]!=r['split']:raise ValueError('Group/RGB crosses splits')
            store[key]=r['split']
        for key in ('image','masks'):
            p=Path(r[key])
            if p.is_absolute() or '..' in p.parts:raise ValueError('Manifest paths must be portable relative paths')
    return obj

def load_record(root,row):
    with Image.open(root/row['image']) as im:
        if im.getexif().get(274,1)!=1:raise ValueError('Non-upright synthetic image')
        rgb=np.array(im.convert('RGB'))
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=row['rgb_sha256']:
        raise ValueError('RGB changed after split was frozen')
    with np.load(root/row['masks'],allow_pickle=False) as z:masks=z['masks'].astype(bool)
    if masks.ndim!=3 or masks.shape[1:]!=rgb.shape[:2]:raise ValueError('Native label shape mismatch')
    return rgb,masks

class NativeCrops(torch.utils.data.Dataset):
    def __init__(self,manifest,tile,seed,allow_condition=False):
        self.root=Path(manifest).parent;obj=read_manifest(manifest,allow_condition)
        self.rows=[r for r in obj['records'] if r['split']=='train']
        if not self.rows:raise ValueError('Empty train split')
        self.tile,self.seed,self.epoch=tile,seed,0
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        rgb,masks=load_record(self.root,self.rows[index]);h,w=rgb.shape[:2];t=self.tile
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        y=int(rng.integers(max(1,h-t+1)));x=int(rng.integers(max(1,w-t+1)))
        rgb=rgb[y:y+t,x:x+t];masks=masks[:,y:y+t,x:x+t]
        if rng.random()<.5:rgb=rgb[:,::-1];masks=masks[:,:,::-1]
        if rng.random()<.5:rgb=rgb[::-1];masks=masks[:,::-1]
        hh,ww=rgb.shape[:2];valid=np.zeros((t,t),bool);valid[:hh,:ww]=True
        rgb=np.pad(rgb,((0,t-hh),(0,t-ww),(0,0)))
        masks=np.pad(masks,((0,0),(0,t-hh),(0,t-ww)))
        return torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255,torch.from_numpy(masks.copy()),torch.from_numpy(valid)
