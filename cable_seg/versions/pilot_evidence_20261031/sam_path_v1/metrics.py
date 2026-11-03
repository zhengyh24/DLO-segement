"""Frozen native-coordinate instance metrics; every image remains in the score."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_iou(predictions, targets, valid=None):
    pred, gt = np.asarray(predictions, bool), np.asarray(targets, bool)
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[1:] != gt.shape[1:]:
        raise ValueError("Masks must be [instances,height,width] in identical coordinates")
    if valid is not None:
        valid = np.asarray(valid, bool).reshape(pred.shape[1:])
        pred, gt = pred & valid, gt & valid
    pixels = int(np.prod(pred.shape[1:]))
    p, g = pred.reshape(len(pred), pixels), gt.reshape(len(gt), pixels)
    intersection = p.astype(np.float64) @ g.astype(np.float64).T
    union = p.sum(1)[:, None] + g.sum(1)[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_scores(iou):
    """Hungarian sum-IoU and strictly >.5 maximum-cardinality PQ matching."""
    iou = np.asarray(iou, dtype=np.float64)
    npred, ngt = iou.shape
    if not npred or not ngt:
        return 0., 0, 0.
    rows, cols = linear_sum_assignment(-iou)
    sum_iou = float(iou[rows, cols].sum())
    eligible = iou > .5
    # An additional TP outweighs every possible change in summed IoU.
    objective = eligible * (min(npred, ngt) + 1 + iou)
    rows, cols = linear_sum_assignment(-objective)
    keep = eligible[rows, cols]
    return sum_iou, int(keep.sum()), float(iou[rows[keep], cols[keep]].sum())


def score_image(predictions, targets, valid=None, *, failed=False):
    pred, gt = np.asarray(predictions, bool), np.asarray(targets, bool)
    raw_predictions = len(pred)
    if valid is not None:
        mask = np.asarray(valid, bool).reshape(pred.shape[1:])
        pred, gt = pred & mask, gt & mask
    # Predictions wholly in ignored annotation regions have no evaluated pixels.
    pred = pred[pred.reshape(len(pred), int(np.prod(pred.shape[1:]))).any(1)]
    gt = gt[gt.reshape(len(gt), int(np.prod(gt.shape[1:]))).any(1)]
    iou = pairwise_iou(pred, gt, valid)
    sum_iou, tp, pq_iou = match_scores(iou)
    npred, ngt = len(pred), len(gt)
    p_union = pred.any(0)
    g_union = gt.any(0)
    if valid is not None:
        mask = np.asarray(valid, bool).reshape(p_union.shape)
        p_union, g_union = p_union & mask, g_union & mask
    union = int((p_union | g_union).sum())
    foreground_iou = float((p_union & g_union).sum() / union) if union else 1.
    instance_iou = sum_iou / ngt if ngt else float(npred == 0)
    if failed:
        # Failure is zero even for an empty GT image; it is not a successful abstention.
        instance_iou = foreground_iou = pq_iou = 0.
        tp = 0
    return dict(instance_iou=instance_iou, union_iou=foreground_iou, tp=tp,
                fp=npred-tp, fn=ngt-tp, matched_iou=pq_iou,
                n_pred=npred, n_gt=ngt, empty_prediction=npred == 0,
                n_pred_before_ignore=raw_predictions, failed=bool(failed), gt_available=True)


def summarize(rows):
    if not rows:
        raise ValueError("An empty evaluation split cannot be scored")
    tp = sum(r.get("tp", 0) for r in rows)
    fp = sum(r.get("fp", 0) for r in rows)
    fn = sum(r.get("fn", 0) for r in rows)
    denominator = tp + .5 * (fp + fn)
    unknown_gt = sum(not r.get("gt_available", True) for r in rows)
    failures = sum(r["failed"] for r in rows)
    pq = sum(r.get("matched_iou", 0.) for r in rows) / denominator if denominator else float(failures == 0)
    return dict(images=len(rows), instance_iou_macro=float(np.mean([r["instance_iou"] for r in rows])),
                union_iou_macro=float(np.mean([r["union_iou"] for r in rows])),
                pq_global=None if unknown_gt else pq, tp=tp, fp=fp, fn=fn,
                failures=failures, empty_predictions=sum(r["empty_prediction"] for r in rows),
                unreadable_ground_truth=unknown_gt,
                metric_protocol="GT-normalized Hungarian macro IoU; strict IoU>0.5 max-TP global PQ",
                empty_gt_policy="empty prediction and empty GT score 1; any inference failure scores 0",
                complete=unknown_gt == 0)
