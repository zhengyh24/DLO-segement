"""Common evaluator, separate from inference, with paired uncertainty estimates.

The paper's exact strict component Accuracy evaluator is not publicly released;
we do not relabel IoU-based instance recall as that metric.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
from full_comparison_inference import OUT, ROOT, dump

YY, XX = np.mgrid[-2:3, -2:3]
DISK = ((XX*XX + YY*YY) <= 4).astype(np.uint8)


def masks_from_ids(ids):
    return [ids == x for x in np.unique(ids) if x != 0]


def wire_masks_in_raw_rgb_frame(masks, image_path):
    """RLE labels use EXIF-upright coordinates; saved inference uses raw RGB pixels.

    Apply the inverse metadata transform, without resampling or score-based choices.
    """
    with Image.open(image_path) as image:
        orientation = image.getexif().get(274, 1)
        upright = ImageOps.exif_transpose(image)
        expected = (upright.height, upright.width)
    if any(mask.shape != expected for mask in masks):
        raise ValueError(f'Wire RLE not in expected EXIF frame: {image_path}, {expected}')
    inverse = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
               4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
               6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE,
               8: Image.Transpose.ROTATE_270}.get(orientation)
    if inverse is None:
        return masks
    return [np.asarray(Image.fromarray(mask.astype(np.uint8)).transpose(inverse)) > 0
            for mask in masks]


def truth_masks(r, wire_tables):
    if r['dataset'] == 'c135':
        truth = np.asarray(Image.open(r['truth']).convert('RGB'))
        return [np.all(truth == c, axis=2) for c in np.unique(truth.reshape(-1, 3), axis=0) if np.any(c)]
    if r['dataset'] == 'synthetic20':
        return [np.asarray(Image.open(p).convert('L')) > 0 for p in r['truth_instances']]
    if r['dataset'] == 'wire300':
        sys.path.insert(0, str(ROOT/'third_party/wireseg_deps'))
        sys.path.insert(0, str(ROOT/'third_party/padlos_deps'))
        import pyarrow.parquet as pq
        from pycocotools import mask as mask_utils
        if r['shard'] not in wire_tables:
            wire_tables[r['shard']] = pq.read_table(r['shard'], columns=['masks_rle']).to_pylist()
        value = wire_tables[r['shard']][r['row']]['masks_rle']
        encodings = json.loads(value) if isinstance(value, str) else value
        if isinstance(encodings, dict):
            encodings = [encodings]
        result = []
        for enc in encodings:
            if isinstance(enc['counts'], list):
                enc = mask_utils.frPyObjects(enc, *enc['size'])
            elif isinstance(enc['counts'], str):
                enc = {**enc, 'counts': enc['counts'].encode('ascii')}
            arr = mask_utils.decode(enc)
            if arr.ndim == 3:
                result.extend([arr[:,:,i].astype(bool) for i in range(arr.shape[2])])
            else:
                result.append(arr.astype(bool))
        return wire_masks_in_raw_rgb_frame(result, r['image'])
    return []


def binary(pred, truth):
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    fg = tp/max(tp+fp+fn, 1)
    bg = tn/max(tn+fp+fn, 1)
    return {'tp_pixels': tp, 'fp_pixels': fp, 'fn_pixels': fn,
            'foreground_iou': fg, 'two_class_miou': (fg+bg)/2,
            'precision': tp/max(tp+fp, 1), 'recall': tp/max(tp+fn, 1),
            'dice': 2*tp/max(2*tp+fp+fn, 1)}


def skel(mask):
    s = skeletonize(mask)
    return s, cv2.dilate(s.astype(np.uint8), DISK) > 0, int(s.sum())


def skel_f1(p, g):
    sp, dp, npix = p
    sg, dg, ngix = g
    precision = np.count_nonzero(sp & dg)/max(npix, 1)
    recall = np.count_nonzero(sg & dp)/max(ngix, 1)
    return 2*precision*recall/max(precision+recall, 1e-12)


def instances(pred, gt):
    overlap = np.zeros((len(gt), len(pred)))
    pa = [int(x.sum()) for x in pred]
    ga = [int(x.sum()) for x in gt]
    ps, gs = [skel(x) for x in pred], [skel(x) for x in gt]
    skeleton_scores = np.zeros_like(overlap)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            intersection = np.count_nonzero(g & p)
            overlap[i, j] = intersection/max(ga[i]+pa[j]-intersection, 1)
            skeleton_scores[i, j] = skel_f1(ps[j], gs[i])
    rows, cols = linear_sum_assignment(-overlap)
    instance_iou = float(overlap[rows, cols].sum()/max(len(gt), 1))
    rows, cols = linear_sum_assignment(-((overlap>.5)*1000+overlap))
    valid = overlap[rows, cols] > .5
    matched = overlap[rows[valid], cols[valid]]
    tp, fp, fn = int(valid.sum()), len(pred)-int(valid.sum()), len(gt)-int(valid.sum())
    sr, sc = linear_sum_assignment(-skeleton_scores)
    skeletal_tp = int((skeleton_scores[sr, sc] >= .8).sum())
    return {'gt_count': len(gt), 'pred_count': len(pred), 'instance_iou': instance_iou,
            'tp_instances': tp, 'fp_instances': fp, 'fn_instances': fn,
            'matched_iou_sum': float(matched.sum()),
            'instance_skeleton_f1_gt_normalized': float(skeleton_scores[sr, sc].sum()/max(len(gt), 1)),
            'tp_skeleton_instances': skeletal_tp, 'fp_skeleton_instances': len(pred)-skeletal_tp,
            'fn_skeleton_instances': len(gt)-skeletal_tp}


def aggregate(rows):
    if not rows:
        return {}
    metrics = ['foreground_iou', 'two_class_miou', 'precision', 'recall', 'dice', 'semantic_skeleton_f1',
               'instance_iou', 'instance_skeleton_f1_gt_normalized']
    result = {'n': len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in metrics}}
    sums = {k: sum(r[k] for r in rows) for k in ['tp_pixels','fp_pixels','fn_pixels',
        'tp_instances','fp_instances','fn_instances','matched_iou_sum','gt_count','pred_count',
        'tp_skeleton_instances','fp_skeleton_instances','fn_skeleton_instances']}
    result.update(sums)
    tp, fp, fn = (sums[k] for k in ['tp_instances','fp_instances','fn_instances'])
    result['instance_precision_iou_gt_05'] = tp/max(tp+fp, 1)
    result['instance_recall_iou_gt_05'] = tp/max(tp+fn, 1)
    result['pq_style'] = sums['matched_iou_sum']/max(tp+.5*fp+.5*fn, 1)
    tp, fp, fn = (sums[k] for k in ['tp_skeleton_instances','fp_skeleton_instances','fn_skeleton_instances'])
    result['instance_precision_skeleton_ge_08'] = tp/max(tp+fp, 1)
    result['instance_recall_skeleton_ge_08'] = tp/max(tp+fn, 1)
    result['micro_foreground_iou'] = sums['tp_pixels']/max(sums['tp_pixels']+sums['fp_pixels']+sums['fn_pixels'], 1)
    result['process_failures'] = sum(r.get('status') not in ('ok','empty') for r in rows)
    return result


def bootstrap(delta):
    rng = np.random.default_rng(20261009)
    distribution = np.mean(delta[rng.integers(0, len(delta), size=(10000, len(delta)))], axis=1)
    return {'mean_delta': float(delta.mean()), 'paired_image_bootstrap_95_ci': np.quantile(distribution,[.025,.975]).tolist(),
            'wins': int((delta>1e-9).sum()), 'ties': int((np.abs(delta)<=1e-9).sum()), 'losses': int((delta< -1e-9).sum())}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['c135','wire300','synthetic20','cabinet4'])
    p.add_argument('--padlos-method', choices=['padlos_acm','padlos_acm_exif_compatible'], default='padlos_acm')
    p.add_argument('--report-name', choices=['evaluation','evaluation_exif_compatible'], default='evaluation')
    a = p.parse_args()
    manifest = json.loads((OUT/'dataset_manifest.json').read_text(encoding='utf-8'))
    wire_tables = {}
    for dataset in a.datasets:
        records = [r for r in manifest['records'] if r['dataset'] == dataset]
        missing = [(r['name'], m) for r in records for m in ['ours_mbest',a.padlos_method]
                   if not (OUT/dataset/m/r['name']/'inference.json').exists()]
        if missing:
            raise RuntimeError(f'{dataset}: {len(missing)} missing predictions, first={missing[:4]}')
        rows = []
        for idx, r in enumerate(records, 1):
            if dataset == 'cabinet4':
                for method in ['ours_mbest',a.padlos_method]:
                    dest = OUT/dataset/method/r['name']
                    ids = np.asarray(Image.open(dest/'instance_ids.png'))
                    audit = json.loads((dest/'inference.json').read_text(encoding='utf-8'))
                    rows.append({'name': r['name'], 'method': method, 'pred_count': len(np.unique(ids[ids>0])),
                        'assigned_pixels': int((ids>0).sum()), 'status': audit.get('status'), 'ground_truth_available': False})
                continue
            gt = truth_masks(r, wire_tables)
            rgb_size = Image.open(r['image']).size
            shape = (rgb_size[1], rgb_size[0])
            assert all(x.shape == shape for x in gt), (r['name'], shape, [x.shape for x in gt])
            gt_union = np.logical_or.reduce(gt) if gt else np.zeros(shape, bool)
            for method in ['ours_final','ours_frontend','padlos_final','padlos_frontend']:
                source = 'ours_mbest' if method.startswith('ours') else a.padlos_method
                dest = OUT/dataset/source/r['name']
                ids = np.asarray(Image.open(dest/'instance_ids.png'))
                prediction = ids>0
                if method.endswith('frontend'):
                    front = 'ours_hq' if method.startswith('ours') else 'padlos_pgm'
                    prediction = np.asarray(Image.open(OUT/dataset/front/r['name']/'mask.png'))>0
                audit_source = OUT/dataset/front/r['name'] if method.endswith('frontend') else dest
                audit = json.loads((audit_source/'inference.json').read_text(encoding='utf-8'))
                row = {'dataset': dataset, 'name': r['name'], 'group': r['group'], 'method': method,
                       'status': audit.get('status', 'ok'), **binary(prediction, gt_union),
                       'semantic_skeleton_f1': skel_f1(skel(prediction), skel(gt_union))}
                # The frontend rows share their method's final IDs; don't attribute instance metrics to just PGM/HQ.
                if method.endswith('final'):
                    row.update(instances(masks_from_ids(ids), gt))
                else:
                    row.update({k: 0 for k in ['instance_iou','instance_skeleton_f1_gt_normalized','tp_instances',
                        'fp_instances','fn_instances','matched_iou_sum','gt_count','pred_count','tp_skeleton_instances',
                        'fp_skeleton_instances','fn_skeleton_instances']})
                rows.append(row)
            if idx%10 == 0 or idx == len(records):
                print(f'evaluate {dataset} [{idx}/{len(records)}]', flush=True)
        if dataset == 'cabinet4':
            dump(OUT/dataset/f'{a.report_name}.json', {'rows': rows, 'qualitative_only': True})
            continue
        methods = ['ours_final','ours_frontend','padlos_final','padlos_frontend']
        summaries = {m: aggregate([r for r in rows if r['method']==m]) for m in methods}
        groups = {g: {m: aggregate([r for r in rows if r['method']==m and r['group']==g]) for m in methods}
                  for g in sorted({r['group'] for r in rows})}
        paired = {}
        for key in ['foreground_iou','two_class_miou','semantic_skeleton_f1','instance_iou','instance_skeleton_f1_gt_normalized']:
            ours = np.array([r[key] for r in rows if r['method']=='ours_final'])
            baseline = np.array([r[key] for r in rows if r['method']=='padlos_final'])
            paired[key] = bootstrap(ours-baseline)
        report = {'summaries': summaries, 'by_group': groups, 'paired_ours_minus_padlos': paired,
                  'baseline_output_method': a.padlos_method,
                  'rows': rows, 'caveats': ['Strict paper Accuracy/Ncorr is not claimed; official evaluation code unavailable.',
                      'Image bootstrap does not account for correlated scene views.',
                      'C135 used in past development; Wire300 selected by model disagreement.',
                      'Wire RLE labels aligned from EXIF-upright to raw RGB using inverse EXIF metadata only.',
                      ('PADLOS RGB is decoded to pixel-identical raw PNG before ACM; algorithms, weights and parameters unchanged.'
                       if a.padlos_method.endswith('exif_compatible') else
                       'Released PADLOS ACM reads RGB via OpenCV EXIF rotation, whereas saved PGM masks use raw coordinates; affected Wire cases have a runtime compatibility confound, not clean algorithm accuracy evidence.')]}
        dump(OUT/dataset/f'{a.report_name}.json', report)
        csv_name = 'per_image.csv' if a.report_name == 'evaluation' else 'per_image_exif_compatible.csv'
        with (OUT/dataset/csv_name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
