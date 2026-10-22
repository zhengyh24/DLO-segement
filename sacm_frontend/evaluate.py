"""Offline foreground evaluation; no inference decisions depend on labels."""
import numpy as np
from .data import load_record
from .predict import predict_native

def evaluate(model,root,rows,cfg,device):
    records=[]
    for row in rows:
        rgb,masks=load_record(root,row)
        prob=predict_native(model,rgb,cfg['tile'],cfg['stride'],device);pred=prob>=cfg['threshold']
        gt=masks.any(0) if len(masks) else np.zeros(pred.shape,bool)
        tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
        coverage=[float(pred[m].mean()) for m in masks if m.any()]
        records.append(dict(id=row['id'],iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
            tp=tp,fp=fp,fn=fn,instance_foreground_coverage=coverage))
    if not records:raise ValueError('Empty validation split')
    tp=sum(r['tp'] for r in records);fp=sum(r['fp'] for r in records);fn=sum(r['fn'] for r in records)
    return dict(macro_iou=float(np.mean([r['iou'] for r in records])),
        precision=tp/(tp+fp) if tp+fp else 1.,recall=tp/(tp+fn) if tp+fn else 1.,
        records=records,meaning='Foreground only; not matched instance IoU or PQ')
