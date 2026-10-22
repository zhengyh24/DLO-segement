"""Native crops: no annotations and no image resizing."""
import numpy as np
from .config import Config

GEOMETRY_NAMES = ('dx','dy','distance','ax','ay','bx','by','a_facing','b_facing',
                  'log_width_ratio','a_width','b_width','corridor_foreground',
                  'proposal_positive','proposal_negative','support_available')

def reverse_candidate(c):
    r=dict(c)
    for a,b in [('a','b'),('a_end','b_end'),('a_tip','b_tip')]:r[a],r[b]=c[b],c[a]
    return r

def crop_native(array, center, size):
    x,y=np.floor(np.asarray(center)+.5).astype(int)
    x0,y0=x-size//2,y-size//2
    out=np.zeros((size,size)+array.shape[2:],array.dtype)
    h,w=array.shape[:2];xa,ya=max(0,x0),max(0,y0);xb,yb=min(w,x0+size),min(h,y0+size)
    if xb>xa and yb>ya:out[ya-y0:yb-y0,xa-x0:xb-x0]=array[ya:yb,xa:xb]
    return out

def extract_pair(rgb, graph, candidate, config=Config(), support=None):
    if rgb.dtype!=np.uint8 or rgb.shape!=(*graph.visible.shape,3):
        raise ValueError('Expected uint8 native RGB matching foreground')
    c=candidate;a,b=c['a_tip'],c['b_tip'];delta=np.asarray(b['xy'])-a['xy']
    distance=float(np.linalg.norm(delta));unit=delta/max(distance,1e-6)
    da,db=np.asarray(a['direction']),np.asarray(b['direction'])
    xy=np.round(np.linspace(a['xy'],b['xy'],max(2,int(distance)+1))).astype(int)
    corridor=float(graph.visible[xy[:,1],xy[:,0]].mean())
    pos=neg=available=0.
    if support is not None:
        available=float(bool(support.get('support_available',False)))
        if available:pos=float(support['positive']);neg=float(support['negative'])
        if not (0<=pos<=1 and 0<=neg<=1):raise ValueError('Support must be in [0,1]')
    n=config.small_size
    geom=np.asarray([*list(delta/n),distance/n,*da,*db,float(da@unit),float(db@-unit),
                     np.log(max(a['width'],1)/max(b['width'],1)),a['width']/n,b['width']/n,
                     corridor,pos,neg,available],np.float32)
    # Crop first: avoid building a full HxWx7 array for every pair.
    def window(size):
        rgb_patch=crop_native(rgb,c['center'],size)
        part=crop_native(graph.partition,c['center'],size)
        channels=[rgb_patch,crop_native(graph.visible,c['center'],size)[...,None],
                  crop_native(graph.skeleton,c['center'],size)[...,None],
                  (part==c['a'])[...,None],(part==c['b'])[...,None]]
        return np.concatenate(channels,axis=2).transpose(2,0,1).astype(np.uint8)
    return window(config.small_size),window(config.large_size),geom

def reverse_features(small,large,geometry):
    order=[0,1,2,3,4,6,5]
    g=geometry.copy();g[0:2]*=-1;g[3:5]=geometry[5:7];g[5:7]=geometry[3:5]
    g[7:9]=geometry[[8,7]];g[9]*=-1;g[10:12]=geometry[[11,10]]
    return small[order].copy(),large[order].copy(),g

def normalize_patch(patch):
    p=patch.astype(np.float32);p[:3]/=255.
    return p

def augment(small,large,g,rng):
    small,large,g=small.copy(),large.copy(),g.copy()
    for axis,indices in [(-1,[0,3,5]),(-2,[1,4,6])]:
        if rng.random()<.5:
            small=np.flip(small,axis).copy();large=np.flip(large,axis).copy();g[indices]*=-1
    gain=rng.uniform(.85,1.15);bias=rng.uniform(-.05,.05)
    for p in (small,large):p[:3]=np.clip(p[:3]*gain+bias,0,1)
    return small,large,g
