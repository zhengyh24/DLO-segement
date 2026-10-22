"""Separate prediction/evaluation commands for a versioned cached experiment."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from PIL import Image

SOURCE = Path('outputs/full_comparison_20261009')
DEFAULT_OUT = Path('outputs/accuracy_optimization_20261012/v1')
METHODS = ['fill_geodesic', 'proposal_graph', 'hybrid_residual', 'hybrid_merge', 'trace_geometry', 'trace_evidence']


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    records = json.loads((SOURCE/'dataset_manifest.json').read_text(encoding='utf-8'))['records']
    frozen = json.loads((SOURCE/'frozen_protocol.json').read_text(encoding='utf-8'))
    for path, digest in frozen['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Original frozen source changed: {path}')
    quota = {'c1': 6, 'c2': 6, 'c3': 6, 'real_easy': 24, 'real_hard': 12, 'one_synthetic_scene': 4, 'unlabelled_cabinet': 4}
    chosen = []
    for group, n in quota.items():
        candidates = sorted((r for r in records if r['group'] == group), key=lambda r: hashlib.sha256(('20261012/'+r['dataset']+'/'+r['name']).encode()).hexdigest())
        chosen += [(r['dataset'], r['name']) for r in candidates[:n]]
    protocol = {'created_at': datetime.now().isoformat(), 'source_root': SOURCE.as_posix(),
                'methods': METHODS, 'seed': 20261012, 'new_training': False,
                'selection': 'sha256 stratified sampling using names/groups only, no scores',
                'development_cases': [list(k) for k in chosen], 'development_counts': dict(Counter(d for d,n in chosen)),
                'scope': 'Already-inspected challenge/development regression; NOT an independent blind test.',
                'inference_inputs': ['raw native RGB', 'cached HQ visible mask', 'accepted text/HQ point proposals', 'cached mBEST IDs'],
                'native_resolution_preserved': True, 'label_access': 'Separate evaluate command only, after saved predictions',
                'references': ['https://arxiv.org/abs/2503.12094', 'https://proceedings.mlr.press/v229/viswanath23a.html', 'https://arxiv.org/abs/1601.00847'],
                'not_reproductions': True,
                'frozen_original_source_sha256': frozen['source_sha256'],
                'source_sha256': {p.as_posix(): sha(p) for p in [Path(__file__), Path(__file__).with_name('cable_evidence_fusion.py')]},
                'original_evaluation_sha256': {d: sha(SOURCE/d/('evaluation_exif_compatible.json' if d == 'wire300' else 'evaluation.json')) for d in ['c135','wire300','synthetic20','cabinet4']}}
    target = out/'protocol.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if old['source_sha256'] != protocol['source_sha256']:
            raise ValueError('Source changed. Use a new version directory.')
    else:
        dump(target, protocol)
    return records, json.loads(target.read_text(encoding='utf-8'))


def selected(records, protocol, split):
    dev = {tuple(k) for k in protocol['development_cases']}
    return [r for r in records if split == 'all' or ((r['dataset'], r['name']) in dev) == (split == 'development')]


def predict(out, records, methods):
    from cable_evidence_fusion import load_cached, predict_methods
    start = time.perf_counter()
    for idx, r in enumerate(records, 1):
        todo = [m for m in methods if not (out/r['dataset']/m/r['name']/'inference.json').exists()]
        if not todo:
            continue
        began = time.perf_counter()
        visible, base, proposals = load_cached(SOURCE, r['dataset'], r['name'])
        rgb = np.asarray(Image.open(r['image']).convert('RGB'))
        prior = time.perf_counter()
        try:
            for method, ids, audit in predict_methods(rgb, visible, base, proposals, todo):
                dest = out/r['dataset']/method/r['name']
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(ids).save(dest/'instance_ids.png')
                Image.fromarray((visible & (ids == 0)).astype(np.uint8)*255).save(dest/'unknown_visible.png')
                audit.update(status='ok' if ids.any() else 'empty', image_sha256=r['image_sha256'],
                             native_size=[rgb.shape[1], rgb.shape[0]], method=method,
                             cumulative_case_seconds=time.perf_counter()-began,
                             elapsed_since_previous_output=time.perf_counter()-prior)
                dump(dest/'inference.json', audit)
                prior = time.perf_counter()
        except Exception:
            dump(out/'errors'/f'{r["dataset"]}_{r["name"]}.json', {'traceback': traceback.format_exc(), 'methods': todo})
            raise
        dump(out/'progress.json', {'phase':'prediction','last_case':[r['dataset'],r['name']], 'index':idx,
                                  'selected_cases':len(records),'methods':methods,'elapsed_seconds':time.perf_counter()-start,
                                  'updated_at':datetime.now().isoformat()})
        print(f'predict [{idx}/{len(records)}] {r["dataset"]}/{r["name"]}: {time.perf_counter()-began:.2f}s', flush=True)


def instance_scores(ids, truth):
    """Exact same Hungarian IoU / PQ threshold, using contingency counts."""
    from scipy.optimize import linear_sum_assignment
    values = np.unique(ids); values = values[values > 0]
    areas = np.bincount(ids.ravel(), minlength=int(ids.max()) + 1)
    overlap = np.zeros((len(truth), len(values)))
    for i, g in enumerate(truth):
        count = np.bincount(ids[g], minlength=len(areas))[values]
        overlap[i] = count / np.maximum(int(g.sum()) + areas[values] - count, 1)
    a, b = linear_sum_assignment(-overlap)
    instance_iou = float(overlap[a, b].sum()/max(len(truth), 1))
    a, b = linear_sum_assignment(-((overlap > .5)*1000 + overlap))
    valid = overlap[a, b] > .5
    tp = int(valid.sum())
    return {'instance_iou':instance_iou, 'tp_instances':tp, 'fp_instances':len(values)-tp,
            'fn_instances':len(truth)-tp, 'matched_iou_sum':float(overlap[a[valid],b[valid]].sum()),
            'gt_count':len(truth), 'pred_count':len(values)}


def summary(rows):
    if not rows: return {}
    keys = ['tp_instances','fp_instances','fn_instances','matched_iou_sum','gt_count','pred_count']
    sums = {k:sum(r[k] for r in rows) for k in keys}
    tp, fp, fn = sums['tp_instances'], sums['fp_instances'], sums['fn_instances']
    return {'n':len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in ['foreground_iou','precision','recall','instance_iou','coverage']},
            'pq_style':sums['matched_iou_sum']/max(tp+.5*fp+.5*fn,1),
            'instance_precision_iou_gt_05':tp/max(tp+fp,1), 'instance_recall_iou_gt_05':tp/max(tp+fn,1), **sums}


def evaluate(out, records, methods, split):
    from evaluate_full_comparison import truth_masks, binary, bootstrap
    tables, rows = {}, []
    original_rows = {}
    for d in ['c135','wire300','synthetic20','cabinet4']:
        original = json.loads((SOURCE/d/('evaluation_exif_compatible.json' if d == 'wire300' else 'evaluation.json')).read_text(encoding='utf-8'))
        for row in original['rows']:
            original_rows[(d,row['name'],row['method'])] = row
    for idx, r in enumerate(records, 1):
        for method in methods:
            dest = out/r['dataset']/method/r['name']
            result = dest/'evaluation.json'
            if result.exists():
                rows.append(json.loads(result.read_text(encoding='utf-8')))
                continue
            if not (dest/'inference.json').exists():
                raise ValueError(f'Missing saved prediction: {dest}')
            if r['dataset'] == 'cabinet4':
                continue
            # Prediction is loaded before label access. No metric enters inference.
            ids = np.asarray(Image.open(dest/'instance_ids.png'))
            gt = truth_masks(r,tables)
            union = np.logical_or.reduce(gt) if gt else np.zeros(ids.shape,bool)
            visible = np.asarray(Image.open(SOURCE/r['dataset']/'ours_hq'/r['name']/'mask.png')) > 0
            audit = json.loads((dest/'inference.json').read_text(encoding='utf-8'))
            row = {'dataset':r['dataset'],'name':r['name'],'group':r['group'],'method':method,
                   'prediction_sha256':sha(dest/'instance_ids.png'),'status':audit['status'],
                   'coverage':float(np.count_nonzero(ids)/max(visible.sum(),1)),
                   **binary(ids>0,union),**instance_scores(ids,gt)}
            dump(result,row); rows.append(row)
        if idx % 5 == 0 or idx == len(records):
            print(f'evaluate [{idx}/{len(records)}]',flush=True)
    for r in records:
        if r['dataset'] == 'cabinet4': continue
        visible = np.asarray(Image.open(SOURCE/r['dataset']/'ours_hq'/r['name']/'mask.png')) > 0
        base = np.asarray(Image.open(SOURCE/r['dataset']/'ours_mbest'/r['name']/'instance_ids.png'))
        for method in ['ours_final','padlos_final']:
            row = dict(original_rows[(r['dataset'],r['name'],method)])
            row['coverage'] = float(np.count_nonzero(base)/max(visible.sum(),1)) if method == 'ours_final' else 0.
            rows.append(row)
    summaries, paired = {}, {}
    for d in ['c135','wire300','synthetic20']:
        summaries[d] = {m:summary([r for r in rows if r['dataset']==d and r['method']==m]) for m in methods+['ours_final','padlos_final']}
        paired[d] = {}
        for m in methods:
            chosen = [r for r in rows if r['dataset']==d and r['method']==m]
            if not chosen: continue
            paired[d][m] = {}
            for control in ['ours_final','padlos_final']:
                paired[d][m][control] = {k:bootstrap(np.array([r[k]-original_rows[(d,r['name'],control)][k] for r in chosen])) for k in ['foreground_iou','instance_iou']}
    report = {'split':split,'summaries':summaries,'paired':paired,'rows':rows,
              'caveats':['All data previously inspected; development/challenge regression, not blind evaluation.',
                         'Image bootstrap is exploratory and not a scene-level confidence interval.',
                         'No GT used for inference; same cached HQ foreground for all variants.',
                         'Final ID union may be smaller than HQ foreground; unknown identity saved separately.',
                         'CABINET has no GT and is excluded from numerical averages.']}
    dump(out/f'evaluation_{split}.json',report)
    for d, s in summaries.items():
        print(d, json.dumps({m:{k:v.get(k) for k in ['n','foreground_iou','instance_iou','pq_style','coverage']} for m,v in s.items()},ensure_ascii=False),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','predict','evaluate'])
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--split',choices=['development','remaining','all'],default='development')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    a=p.parse_args()
    records,protocol=prepare(a.output)
    records=selected(records,protocol,a.split)
    if a.phase == 'predict': predict(a.output,records,a.methods)
    elif a.phase == 'evaluate': evaluate(a.output,records,a.methods,a.split)
    else: print(protocol['development_counts'],flush=True)


if __name__=='__main__':
    main()
