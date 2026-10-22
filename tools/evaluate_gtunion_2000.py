"""Offline GT evaluation after all RGB-only identity predictions are saved."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from curvi_identity.io import read_json,read_mask
from curvi_identity.data import decode_masks
from run_curvi_2000 import selected_rows,save

METHODS=('no_link','bidirectional_threshold_only','full_default')


def instance_metrics(ids,gt):
    # Same primary metric as evaluate_full_comparison.instances, using label
    # contingency counts rather than materializing one dense mask per fragment.
    labels,compact=np.unique(ids,return_inverse=True);compact=compact.reshape(ids.shape)
    keep=labels!=0;areas=np.bincount(compact.ravel(),minlength=len(labels))[keep]
    overlap=np.zeros((len(gt),len(areas)))
    for i,g in enumerate(gt):
        inter=np.bincount(compact[g],minlength=len(labels))[keep]
        overlap[i]=inter/np.maximum(1,int(g.sum())+areas-inter)
    rr,cc=linear_sum_assignment(-overlap);iou=float(overlap[rr,cc].sum()/max(len(gt),1))
    rr,cc=linear_sum_assignment(-((overlap>.5)*1000+overlap));valid=overlap[rr,cc]>.5
    tp=int(valid.sum())
    union=np.logical_or.reduce(gt) if gt else np.zeros(ids.shape,bool);fg=ids>0
    pt=int((fg&union).sum());pf=int((fg&~union).sum());pn=int((~fg&union).sum())
    return dict(instance_iou=iou,matched_iou_sum=float(overlap[rr[valid],cc[valid]].sum()),
                tp=tp,fp=len(areas)-tp,fn=len(gt)-tp,gt_instances=len(gt),pred_instances=len(areas),
                foreground_iou=pt/max(1,pt+pf+pn),foreground_precision=pt/max(1,pt+pf),foreground_recall=pt/max(1,pt+pn))


def aggregate(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    return dict(images=len(rows),instance_iou=float(np.mean([r['instance_iou'] for r in rows])),
                pq_style=sum(r['matched_iou_sum'] for r in rows)/max(1,tp+.5*fp+.5*fn),tp=tp,fp=fp,fn=fn,
                instance_precision=tp/max(1,tp+fp),instance_recall=tp/max(1,tp+fn),
                foreground_iou=float(np.mean([r['foreground_iou'] for r in rows])))


def pair_metrics(labels,scores,threshold):
    y=np.asarray(labels)>0;p=np.asarray(scores)>=threshold
    tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum());tn=int((~p&~y).sum())
    return dict(pairs=len(y),threshold=threshold,tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(1,tp+fp),
                recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),accuracy=(tp+tn)/max(1,len(y)))


def main(out):
    completion=read_json(out/'validation_predictions/completion.json')
    groups=read_json(out/'groups.json');groups=dict(groups,records=[r for r in groups['records'] if r['split']=='validation'])
    assert completion['images']==len(groups['records'])
    inputs={r['file_name']:r for r in read_json(out/'inference_inputs.json')['records']};rows=[];score_map={}
    for index,(r,row) in enumerate(selected_rows(groups,['file_name','masks_rle']),1):
        dest=out/'validation_predictions'/inputs[r['file_name']]['name'];audit=read_json(dest/'audit.json')
        if audit['gt_instance_identity_used'] is not False:raise ValueError('Instance identity input leakage')
        if audit['oracle_foreground'] != read_json(out/'input_regime.json')['oracle_foreground']:raise ValueError('Foreground provenance mismatch')
        gt=decode_masks(row['masks_rle'],tuple(audit['native_shape']))
        score_map[r['file_name']]={s['candidate']:min(s['forward'],s['reverse']) for s in audit['scores']}
        for method in METHODS:
            metrics=instance_metrics(read_mask(dest/(method+'.png')),gt)
            rows.append(dict(file_name=r['file_name'],condition=r['group_id'],method=method,**metrics))
        if index%10==0:print({'evaluated':index,'total':len(groups['records'])},flush=True)
    overall={m:aggregate([r for r in rows if r['method']==m]) for m in METHODS}
    conditions={c:{m:aggregate([r for r in rows if r['method']==m and r['condition']==c]) for m in METHODS}
                for c in sorted({r['condition'] for r in rows})}
    manifest=read_json(out/'prepared/manifest.json');labels=[];scores=[]
    for r in manifest['records']:
        if r['split']=='validation':labels.append(r['label']);scores.append(score_map[r['file_name']][r['candidate']])
    pairs={str(t):pair_metrics(labels,scores,t) for t in (.5,.8)}
    baseline={r['file_name']:r for r in rows if r['method']=='no_link'}
    delta=[dict(file_name=r['file_name'],condition=r['condition'],delta_instance_iou=r['instance_iou']-baseline[r['file_name']]['instance_iou'],
                before=baseline[r['file_name']],after=r) for r in rows if r['method']=='full_default']
    delta.sort(key=lambda r:r['delta_instance_iou'])
    summary=dict(scope='Single-seed synthetic condition-holdout validation used for checkpoint selection; not independent scene/real test',
        checkpoint=completion,metrics=overall,conditions=conditions,pair_metrics=pairs,
        wins=sum(r['delta_instance_iou']>1e-9 for r in delta),ties=sum(abs(r['delta_instance_iou'])<=1e-9 for r in delta),
        losses=sum(r['delta_instance_iou'] < -1e-9 for r in delta),worst=delta[:10],best=delta[-10:][::-1],rows=rows)
    ntrain=sum(r['split']=='train' for r in manifest['images']);nval=len(groups['records'])
    lines=['# 2000张训练图探索性结果','',
        f'本清单训练{ntrain}张、验证{nval}张。采用条件整组隔离；缺少实际仿真scene/run标识，因此仅称条件留出。验证集用于选择检查点，非独立盲测。',
        '',f"最佳检查点：第{completion['checkpoint_epoch']}轮；单一随机种子20261013。SAM/HQ冻结，原生1024×1024输入，训练仅更新PairNet。",'',
        '| 方法 | 验证图数 | 实例IoU % | PQ % | TP | FP | FN |','|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v=overall[m];lines.append(f"| {m} | {v['images']} | {100*v['instance_iou']:.2f} | {100*v['pq_style']:.2f} | {v['tp']} | {v['fp']} | {v['fn']} |")
    di=100*(overall['full_default']['instance_iou']-overall['no_link']['instance_iou'])
    dp=100*(overall['full_default']['pq_style']-overall['no_link']['pq_style'])
    lines+=['',f'完整解码相对不连接片段：实例IoU变化{di:+.2f}个百分点，PQ变化{dp:+.2f}个百分点。逐图{summary["wins"]}提升、{summary["ties"]}持平、{summary["losses"]}退化。',
        '', '三者使用相同前景、相同原子段与候选；没有用GT逐图选方法或调解码阈值。no_link是图分段基线，不是coverage_color或mBEST。该结果不能证明超过正式全量基线。',
        '', '## 配对判断（不是实例分割精度）','', '| 阈值 | 配对数 | 精确率 | 召回率 | F1 |','|---|---:|---:|---:|---:|']
    for t,v in pairs.items():lines.append(f"| {t} | {v['pairs']} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |")
    lines+=['','## 分条件结果','']
    for c,methods in conditions.items():
        v=methods['full_default'];b=methods['no_link'];lines.append(f"- {c}：实例IoU {b['instance_iou']:.4f} → {v['instance_iou']:.4f}；PQ {b['pq_style']:.4f} → {v['pq_style']:.4f}。")
    lines+=['','## 退化与改进案例','']
    for title,items in [('退化排序',delta[:5]),('改进排序',delta[-5:][::-1])]:
        lines.append(title+'：')
        for r in items:lines.append(f"- {r['file_name']}：实例IoU变化{100*r['delta_instance_iou']:+.2f}个百分点。")
    lines+=['',f'所有{nval}张验证图均计入，包括无候选、漏前景或没有监督配对的图。身份关联无法恢复前端未检出的线缆，也不绘制遮挡区。',
            '', '原始记录：evaluation.json、fit/history.json、fit/initial.json、fit/completion.json、validation_predictions/。仍需同输入coverage_color/mBEST对照、更多随机种子和真实历史回归，才能判断实际收益。']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(out/'evaluation.json',summary)
    print(overall,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output.resolve())
