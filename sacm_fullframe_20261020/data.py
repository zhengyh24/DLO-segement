"""Supervision only. Paired complete-cycle schedules independent of input strategy."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sacm_frontend.data import load_record, read_manifest


def manifests(synthetic,real):
    syn=read_manifest(synthetic,True)
    real_obj=json.loads(Path(real).read_text(encoding='utf-8'))
    assert real_obj['grouping_kind']=='historical_rgb_cluster_holdout'
    assert real_obj['independent_scene_split'] is False
    groups={}; hashes={}; ids=set()
    for r in real_obj['records']:
        assert r['split'] in ('train','validation','test') and r['id'] not in ids
        ids.add(r['id'])
        for db,key in ((groups,r['group_id']),(hashes,r['rgb_sha256'])):
            assert db.get(key,r['split'])==r['split'];db[key]=r['split']
        for key in ('image','masks'):
            p=Path(r[key]);assert not p.is_absolute() and '..' not in p.parts
    assert not ({r['rgb_sha256'] for r in syn['records']}&{r['rgb_sha256'] for r in real_obj['records']})
    return {'synthetic':(Path(synthetic).parent,syn['records']), 'real':(Path(real).parent,real_obj['records'])}


def schedule(seed,cycle,ns,nr):
    assert ns%2==0 and nr>0
    rng=np.random.default_rng(np.random.SeedSequence([seed,cycle,20261020]))
    syn=rng.permutation(ns)
    # Circular selection gives cumulative per-real-image counts differing by <=1.
    perm=np.random.default_rng(np.random.SeedSequence([seed,971])).permutation(nr)
    real=perm[(np.arange(ns)+cycle*ns)%nr].copy();rng.shuffle(real)
    rows=[]
    for s,r in zip(syn,real):
        pair=[('synthetic',int(s)),('real',int(r))]
        if rng.random()<.5:pair.reverse()
        for domain,index in pair:
            rows.append(dict(domain=domain,index=index,flip_h=bool(rng.integers(2)),flip_v=bool(rng.integers(2)),crop_seed=int(rng.integers(2**32))))
    assert len(rows)==2*ns and Counter(x['index'] for x in rows if x['domain']=='synthetic')==Counter(range(ns))
    return rows


def sample(root,row,entry,arm):
    rgb,masks=load_record(root,row)
    original=list(rgb.shape[:2]);y=x=0
    if arm=='crop':
        rng=np.random.default_rng(entry['crop_seed']);h,w=rgb.shape[:2]
        y=int(rng.integers(max(1,h-512+1)));x=int(rng.integers(max(1,w-512+1)))
        rgb=rgb[y:y+512,x:x+512];masks=masks[:,y:y+512,x:x+512]
    else:assert arm=='full'
    if entry['flip_h']:rgb=rgb[:,::-1];masks=masks[:,:,::-1]
    if entry['flip_v']:rgb=rgb[::-1];masks=masks[:,::-1]
    h,w=rgb.shape[:2]
    th,tw=(512,512) if arm=='crop' else (h,w)
    valid=np.zeros((th,tw),bool);valid[:h,:w]=True
    rgb=np.pad(rgb,((0,th-h),(0,tw-w),(0,0)))
    masks=np.pad(masks,((0,0),(0,th-h),(0,tw-w)))
    meta=dict(id=row['id'],original_hw=original,input_hw=[th,tw],valid_pixels=h*w,
              input_pixels=th*tw,encoder_padded_pixels=(th+(-th)%16)*(tw+(-tw)%16),crop_yx=[y,x])
    return torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255,torch.from_numpy(masks.copy()),torch.from_numpy(valid),meta
