"""Offline metrics after RGB-only predictions; thresholds chosen on validation only."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .predict import probability

THRESHOLDS=[i/20 for i in range(1,20)]+[.975,.99]


def evaluate(model,root,rows,mode,device,thresholds=THRESHOLDS,save_dir=None):
    ts=np.array(sorted(set(thresholds)),np.float32);records=[]
    if save_dir:Path(save_dir).mkdir(parents=True,exist_ok=True)
    for row in rows:
        with Image.open(root/row['image']) as im:rgb=np.array(im.convert('RGB'))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['rgb_sha256']
        prob=probability(model,rgb,mode,device)
        with np.load(root/row['masks'],allow_pickle=False) as z:masks=z['masks'].astype(bool)
        gt=masks.any(0);assert prob.shape==gt.shape
        index=np.searchsorted(ts,prob,side='right')
        pos=np.bincount(index[gt],minlength=len(ts)+1);neg=np.bincount(index[~gt],minlength=len(ts)+1)
        tp=np.cumsum(pos[::-1])[::-1][1:];fp=np.cumsum(neg[::-1])[::-1][1:];fn=int(gt.sum())-tp
        iou=np.divide(tp,tp+fp+fn,out=np.ones_like(tp,dtype=float),where=(tp+fp+fn)>0)
        records.append(dict(id=row['id'],dataset=row.get('dataset','synthetic'),iou=iou.tolist(),
                            tp=tp.tolist(),fp=fp.tolist(),fn=fn.tolist(),native_hw=list(gt.shape)))
        if save_dir:
            for t in ts:
                Image.fromarray((prob>=t).astype('uint8')*255).save(Path(save_dir)/f'{row["id"]}__t{float(t):.5f}.png')
    assert records
    curve=[]
    for j,t in enumerate(ts):
        tp=sum(r['tp'][j] for r in records);fp=sum(r['fp'][j] for r in records);fn=sum(r['fn'][j] for r in records)
        domains={d:float(np.mean([r['iou'][j] for r in records if r['dataset']==d])) for d in sorted({r['dataset'] for r in records})}
        curve.append(dict(threshold=float(t),macro_iou=float(np.mean([r['iou'][j] for r in records])),
                          precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),domains=domains))
    return dict(mode=mode,curve=curve,records=records,n=len(records),inference_uses_gt=False)


def fixed(result):return next(r for r in result['curve'] if abs(r['threshold']-.5)<1e-6)


def select_threshold(result):
    # Deterministic ties prefer nearest .5, then lower threshold.
    return max(result['curve'],key=lambda r:(r['macro_iou'],-abs(r['threshold']-.5),-r['threshold']))['threshold']
