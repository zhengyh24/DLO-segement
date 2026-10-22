"""Native inference path. No dataset importer or annotation argument."""
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from .config import Config
from .features import extract_pair,reverse_features,normalize_patch
from .graph import build_graph,decode
from .io import new_directory,read_json,write_json,read_rgb,read_mask,rgb_hash
from .model import PairNet

def load_model(checkpoint,device='cpu'):
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if obj.get('architecture')!='PairNet-v1' or obj.get('trained') is not True or obj.get('optimizer_steps',0)<1:
        raise ValueError('Inference requires a trained PairNet-v1 checkpoint; random weights are smoke-only')
    config=Config(**obj['config']);model=PairNet();model.load_state_dict(obj['model'],strict=True)
    return model.to(device).eval(),config,obj

def score_graph(rgb,graph,model,config=Config(),device='cpu',support=None):
    scores=[];support=support or {};model.eval()
    with torch.inference_mode():
        for start in range(0,len(graph.candidates),config.batch_size):
            samples=[]
            for c in graph.candidates[start:start+config.batch_size]:
                s,l,g=extract_pair(rgb,graph,c,config,support.get(c['key']))
                samples.extend([(s,l,g),reverse_features(s,l,g)])
            tensors=[]
            for i in range(3):
                a=np.stack([normalize_patch(r[i]) if i<2 else r[i] for r in samples])
                tensors.append(torch.from_numpy(a).to(device))
            scores.extend(model(*tensors).sigmoid().cpu().numpy().reshape(-1,2).tolist())
    return np.asarray(scores,np.float32).reshape(-1,2)

def unknown_mask(graph,audit):
    verified={edge[k] for edge in audit['selected'] for k in ('a','b')}
    return graph.visible & ~np.isin(graph.partition,list(verified))

def infer(image,foreground,checkpoint,output,baseline=None,support=None,device='cpu'):
    model,config,ckpt=load_model(checkpoint,device)
    rgb=read_rgb(image);fg=read_mask(foreground)>0
    if fg.shape!=rgb.shape[:2]:raise ValueError('RGB/foreground must share native shape')
    base=read_mask(baseline) if baseline else None;proposals=read_json(support) if support else None
    started=time.perf_counter();graph=build_graph(fg,config);graph_seconds=time.perf_counter()-started
    started=time.perf_counter();scores=score_graph(rgb,graph,model,config,device,proposals)
    scoring_seconds=time.perf_counter()-started
    started=time.perf_counter();ids,audit=decode(graph,scores,config,base);decoding_seconds=time.perf_counter()-started
    unknown=unknown_mask(graph,audit);out=new_directory(output)
    np.save(out/'instances.npy',ids);Image.fromarray(ids).save(out/'instances.png')
    Image.fromarray(fg.astype(np.uint8)*255).save(out/'visible.png')
    Image.fromarray(unknown.astype(np.uint8)*255).save(out/'unknown.png')
    result={'kind':'trained_native_identity_inference','checkpoint':str(Path(checkpoint).resolve()),
            'optimizer_steps':ckpt['optimizer_steps'],'config':config.to_dict(),'rgb_sha256':rgb_hash(rgb),
            'native_shape':list(fg.shape),'graph':graph.audit,'decoding':audit,
            'scores':[{'candidate':c['key'],'forward':float(s[0]),'reverse':float(s[1])} for c,s in zip(graph.candidates,scores)],
            'unknown_pixels':int(unknown.sum()),'unknown_semantics':'visible partitions with no accepted association; not calibrated identity uncertainty',
            'timing_seconds':{'graph':graph_seconds,'scoring':scoring_seconds,'decoding':decoding_seconds},
            'timing_scope':'identity module only; excludes foreground model, checkpoint loading and output IO',
            'ground_truth_used':False,'foreground_changed_pixels':int(np.count_nonzero((ids>0)!=fg))}
    write_json(out/'audit.json',result)
    return {'output':str(out),'native_shape':list(fg.shape),'selected_links':len(audit['selected']),
            'foreground_changed_pixels':result['foreground_changed_pixels']}
