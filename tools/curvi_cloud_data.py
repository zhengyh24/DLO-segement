"""Storage adapter: identical native pair features, extracted lazily in RAM."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from curvi_identity.io import read_json,read_rgb,rgb_hash
from curvi_identity.graph import Graph
from curvi_identity.config import Config
from curvi_identity.features import extract_pair,reverse_features,normalize_patch,augment


class CloudPairDataset:
    def __init__(self,manifest,split,seed=0):
        if split not in ('train','validation'):raise ValueError('Refuse test split')
        self.path=Path(manifest);obj=read_json(self.path)
        if obj.get('storage_kind')!='native_graph_lazy_v1':raise ValueError('Wrong lazy storage schema')
        self.config=Config(**obj['config']);self.records=[r for r in obj['records'] if r['split']==split]
        self.training=split=='train';self.rng=np.random.default_rng(seed);self.cache={}
        if not self.records:raise ValueError('Empty split')
    def __len__(self):return len(self.records)
    def raw(self,index):
        r=self.records[index];key=r['file_name']
        if key not in self.cache:
            p=self.path.parent/r['graph_item'];obj=read_json(p);rgb=read_rgb(obj['image'])
            if rgb_hash(rgb)!=r['rgb_sha256']:raise ValueError('Lazy RGB hash mismatch')
            with np.load(p.parent/'arrays.npz',allow_pickle=False) as z:
                partition=z['partition'];skeleton=z['skeleton'].astype(bool)
            visible=partition>0
            graph=Graph(visible,skeleton,None,partition,{},obj['candidates'],obj['audit'])
            # Approximately 14 GiB for 2000 native RGB/graphs, within 54 GiB host RAM.
            self.cache[key]=(rgb,graph)
        rgb,graph=self.cache[key];c=graph.candidates[r['candidate_index']]
        if c['key']!=r['candidate']:raise ValueError('Candidate identity mismatch')
        return (*extract_pair(rgb,graph,c,self.config),r['label'])
    def __getitem__(self,index):
        import torch
        s,l,g,label=self.raw(index)
        if label not in (0,1):raise ValueError('Invalid label')
        if self.training and self.rng.random()<.5:s,l,g=reverse_features(s,l,g)
        s,l=normalize_patch(s),normalize_patch(l)
        if self.training:s,l,g=augment(s,l,g,self.rng)
        return tuple(torch.from_numpy(a.copy()) for a in (s,l,g))+(torch.tensor(label,dtype=torch.float32),)
