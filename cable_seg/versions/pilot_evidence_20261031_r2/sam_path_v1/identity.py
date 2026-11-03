"""Offline visible-identity diagnostics; never import this module into forward.

These are explicit diagnostic definitions, not substitutes for frozen IoU/PQ.
An event ROI is a polygon or a list of native pixel coordinates. Events and GT
are consumed only after RGB-only predictions have been computed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DIFFICULTY_PROTOCOL = {
    "version": "fixed_rgb_gt_proxies_v1",
    "instance_count_bins": [4, 8, 16],
    "major_component_min_pixels": 16,
    "extra_major_component_bins": [0, 2],
    "proximity": "overlap of different-instance 3px Chebyshev dilations / their union; thresholds 0 and .01",
    "appearance": "minimum pairwise mean-RGB Euclidean distance / sqrt(3); thresholds .1 and .25; input RGB in [0,1]",
    "limitations": "fixed descriptive proxies, not annotated crossings, occlusions, hidden lengths, or texture identity; no GT reaches model forward",
}


IDENTITY_PROTOCOL = {
    "version": "visible_identity_v1",
    "merge_split_edge": "intersection >= max(16 pixels, fraction * GT area)",
    "fractions": [0.05, 0.10, 0.20],
    "primary_fraction": 0.10,
    "whole_visible": "one prediction precision>=0.9 and recall>=0.9; every GT 8-connected component of >=16 pixels has recall>=0.8",
    "events": "both regions require >=0.5 coverage by one exclusive predicted ID; same/different identity tested after coverage; uncovered is incorrect",
    "hidden_pixels": "never evaluated as visible foreground",
}


def _arrays(predictions, targets, valid=None):
    pred, gt = np.asarray(predictions, bool), np.asarray(targets, bool)
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[1:] != gt.shape[1:]:
        raise ValueError("Expected prediction and GT masks [N,H,W] in identical coordinates")
    if valid is not None:
        mask = np.asarray(valid, bool).reshape(pred.shape[1:])
        pred, gt = pred & mask, gt & mask
    if np.any(pred.sum(0) > 1):
        raise ValueError("Identity diagnostics require exclusive predicted pixel ownership")
    pred = pred[pred.reshape(len(pred), int(np.prod(pred.shape[1:]))).any(1)]
    gt = gt[gt.reshape(len(gt), int(np.prod(gt.shape[1:]))).any(1)]
    return pred, gt


def visible_identity(predictions, targets, valid=None, *, failed=False):
    pred, gt = _arrays(predictions, targets, valid)
    pa = pred.sum((1, 2), dtype=np.int64)
    ga = gt.sum((1, 2), dtype=np.int64)
    ids = np.zeros(pred.shape[1:], dtype=np.int32)
    for j, mask in enumerate(pred):
        ids[mask] = j + 1
    overlap = np.stack([np.bincount(ids[g], minlength=len(pred)+1)[1:] for g in gt]) if len(gt) else np.zeros((0, len(pred)), np.int64)
    sensitivity = {}
    for fraction in IDENTITY_PROTOCOL["fractions"]:
        edges = overlap >= np.maximum(16, fraction * ga[:, None])
        merges = int((edges.sum(0) > 1).sum())
        splits = int((edges.sum(1) > 1).sum())
        sensitivity[str(fraction)] = dict(merged_predictions=merges, fragmented_gt=splits,
            merge_rate=merges / len(pred) if len(pred) else None,
            fragmentation_rate=splits / len(gt) if len(gt) else None)
    recovered, detail = 0, []
    for i, g in enumerate(gt):
        precision = overlap[i] / np.maximum(pa, 1)
        recall = overlap[i] / max(ga[i], 1)
        candidates = np.flatnonzero((precision >= .9) & (recall >= .9))
        components, count = ndimage.label(g, structure=np.ones((3, 3), bool))
        areas = np.bincount(components.ravel())
        substantial = np.flatnonzero(areas >= 16)
        substantial = substantial[substantial != 0]
        winning_query = None
        for j in candidates:
            hits = np.bincount(components[pred[j]], minlength=count+1)
            if np.all(hits[substantial] / areas[substantial] >= .8):
                winning_query = int(j)
                break
        ok = winning_query is not None and not failed
        recovered += int(ok)
        detail.append(dict(gt_index=i, recovered=ok, predicted_index=winning_query,
                           substantial_components=len(substantial)))
    return dict(n_pred=len(pred), n_gt=len(gt), failed=bool(failed),
                whole_visible_recovered=recovered,
                whole_visible_recovery_rate=recovered/len(gt) if len(gt) else None,
                sensitivity=sensitivity, instances=detail)


def region_mask(region, shape):
    """Native coordinates only. No inference-generated event selection."""
    if isinstance(region, np.ndarray):
        if region.shape != tuple(shape):
            raise ValueError("Event region has a different image shape")
        return np.asarray(region, bool)
    height, width = shape
    if "polygon_xy" in region:
        points = region["polygon_xy"]
        if len(points) < 3:
            raise ValueError("Event polygon needs at least three vertices")
        for x, y in points:
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError("Event polygon coordinate outside native image")
        image = Image.new("1", (width, height))
        ImageDraw.Draw(image).polygon([tuple(p) for p in points], fill=1)
        return np.asarray(image, bool)
    points = np.asarray(region.get("pixels_xy", []), dtype=np.int64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
        raise ValueError("Event region needs nonempty pixels_xy or polygon_xy")
    if (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
        raise ValueError("Event pixel outside native image")
    result = np.zeros(shape, bool)
    result[points[:, 1], points[:, 0]] = True
    return result


def identity_events(predictions, events, valid=None, *, failed=False):
    pred = np.asarray(predictions, bool)
    if pred.ndim != 3 or np.any(pred.sum(0) > 1):
        raise ValueError("Expected exclusive predicted masks")
    valid = np.ones(pred.shape[1:], bool) if valid is None else np.asarray(valid, bool).reshape(pred.shape[1:])
    pred = pred & valid
    results = []
    for event in events:
        row = dict(id=event["id"], same_instance=bool(event["same_instance"]),
                   ambiguous=bool(event.get("ambiguous", False)))
        if row["ambiguous"]:
            row.update(eligible=False, reason="annotated_ambiguous")
            results.append(row)
            continue
        if len(event["regions"]) != 2:
            raise ValueError("An identity event requires exactly two visible regions")
        ownership, coverage = [], []
        for region in event["regions"]:
            mask = region_mask(region, pred.shape[1:]) & valid
            area = int(mask.sum())
            if area == 0:
                raise ValueError("Event region is empty after annotation validity")
            fractions = pred[:, mask].sum(1) / area
            j = int(fractions.argmax()) if len(pred) else None
            fraction = float(fractions[j]) if j is not None else 0.
            ownership.append(j if fraction >= .5 else None)
            coverage.append(fraction)
        covered = all(j is not None for j in ownership) and not failed
        correct = covered and ((ownership[0] == ownership[1]) == row["same_instance"])
        row.update(eligible=True, covered=covered, correct=bool(correct),
                   predicted_indices=ownership, coverage=coverage)
        results.append(row)
    return summarize_events(results, include_records=True)


def summarize_events(events, *, include_records=False):
    eligible = [r for r in events if r["eligible"]]
    counts = {}
    for label, same in (("positive", True), ("negative", False)):
        subset = [r for r in eligible if r["same_instance"] == same]
        correct = sum(r["correct"] for r in subset)
        counts[label] = dict(events=len(subset), correct=correct,
                             accuracy=correct/len(subset) if subset else None)
    accuracies = [counts[k]["accuracy"] for k in ("positive", "negative")]
    result = dict(**counts, eligible_events=len(eligible), ambiguous_events=len(events)-len(eligible),
                  covered_events=sum(r["covered"] for r in eligible),
                  coverage=sum(r["covered"] for r in eligible)/len(eligible) if eligible else None,
                  balanced_accuracy=float(np.mean(accuracies)) if all(a is not None for a in accuracies) else None)
    if include_records:
        result["events"] = events
    return result


def summarize_identity(rows):
    observed = [r["identity"] for r in rows if r.get("identity") is not None]
    n_gt = sum(r["n_gt"] for r in observed)
    recovered = sum(r["whole_visible_recovered"] for r in observed)
    complete = len(rows) == len(observed)
    observed_rate = recovered/n_gt if n_gt else None
    result = dict(protocol=IDENTITY_PROTOCOL, images=len(rows), observed_images=len(observed),
                  complete=complete, whole_visible_recovered=recovered, n_gt=n_gt,
                  whole_visible_recovery_rate=observed_rate if complete else None,
                  observed_whole_visible_recovery_rate=observed_rate, sensitivity={})
    for fraction in IDENTITY_PROTOCOL["fractions"]:
        key = str(fraction)
        merges = sum(r["sensitivity"][key]["merged_predictions"] for r in observed)
        fragments = sum(r["sensitivity"][key]["fragmented_gt"] for r in observed)
        n_pred = sum(r["n_pred"] for r in observed)
        result["sensitivity"][key] = dict(merged_predictions=merges, fragmented_gt=fragments,
            merge_rate=merges/n_pred if n_pred and complete else None,
            fragmentation_rate=fragments/n_gt if n_gt and complete else None,
            observed_merge_rate=merges/n_pred if n_pred else None,
            observed_fragmentation_rate=fragments/n_gt if n_gt else None)
    if any(r.get("identity_events") is not None for r in rows):
        result["events"] = summarize_events([event for row in rows for event in (row.get("identity_events") or {}).get("events", [])])
    return result


def attention_trace(outputs):
    """Latent expected positions/entropy, not a certified physical centerline."""
    aux = outputs.get("aux", {})
    if "path_attention" not in aux:
        return dict(available=False, reason="model_has_no_path_attention")
    attention = aux["path_attention"].detach().float().cpu().numpy()
    positions = aux["path_positions"].detach().float().cpu().numpy()
    if attention.ndim != 5 or len(attention) != 1 or positions.shape[:3] != attention.shape[:3]:
        raise ValueError("Expected single-image [1,Q,T,H,W] attention and [1,Q,T,2] positions")
    if not np.isfinite(attention).all() or not np.isfinite(positions).all():
        raise ValueError("Nonfinite path diagnostics")
    p = attention[0].reshape(*attention.shape[1:3], -1)
    entropy = -(p*np.log(p.clip(1e-30))).sum(-1)
    return dict(available=True, interpretation="soft latent expected coordinates, not physical trajectory GT",
                coordinate_frame="normalized padded feature grid, x/y order", attention_shape=list(attention.shape),
                positions_xy=positions[0].tolist(), entropy_nats=entropy.tolist(),
                normalized_entropy=(entropy/np.log(max(p.shape[-1], 2))).tolist(),
                step_distance=np.linalg.norm(np.diff(positions[0], axis=1), axis=-1).tolist())


def difficulty_metadata(rgb, targets, valid=None):
    """Predeclared bins independent of any prediction, after the RGB forward."""
    gt = np.asarray(targets, bool)
    rgb = np.asarray(rgb, np.float32)
    if gt.ndim != 3 or rgb.shape != (*gt.shape[1:], 3) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Difficulty expects native masks and finite HWC RGB in [0,1]")
    if valid is not None: gt = gt & np.asarray(valid, bool).reshape(gt.shape[1:])
    gt = gt[gt.reshape(len(gt), int(np.prod(gt.shape[1:]))).any(1)]
    dilated_count = np.zeros(gt.shape[1:], np.uint16)
    total_components, major_components, means = 0, 0, []
    for mask in gt:
        labels, count = ndimage.label(mask, np.ones((3, 3), bool))
        areas = np.bincount(labels.ravel())[1:]
        total_components += count
        major_components += int((areas >= 16).sum())
        dilated_count += ndimage.maximum_filter(mask, size=7, mode="constant").astype(np.uint16)
        means.append(rgb[mask].mean(0))
    shared, union = int((dilated_count > 1).sum()), int((dilated_count > 0).sum())
    proximity = shared/union if union else 0.
    appearance = None
    if len(means) >= 2:
        means = np.asarray(means)
        distance = np.linalg.norm(means[:, None]-means[None], axis=-1)/np.sqrt(3.)
        np.fill_diagonal(distance, np.inf)
        appearance = float(distance.min())
    n = len(gt)
    extra = max(0, major_components-n)
    return dict(gt_instances=n, gt_visible_components=int(total_components), gt_major_components=major_components,
        extra_major_components=extra, proximity_shared_pixels=shared, proximity_overlap_fraction=proximity,
        minimum_mean_rgb_distance=appearance, mean_rgb_similarity_proxy=None if appearance is None else 1-appearance,
        bins=dict(instance_count="0" if n == 0 else "1-4" if n <= 4 else "5-8" if n <= 8 else "9-16" if n <= 16 else "17+",
                  visible_fragmentation="none" if extra == 0 else "1-2" if extra <= 2 else "3+",
                  proximity_proxy="none" if shared == 0 else "low" if proximity <= .01 else "high",
                  appearance_proxy="single/empty" if appearance is None else "similar" if appearance <= .1 else "medium" if appearance <= .25 else "different"),
        interpretation=DIFFICULTY_PROTOCOL["limitations"])


def difficulty_strata(rows):
    """All images retained, including failed/unreadable inputs as unavailable."""
    from .metrics import summarize
    result = {}
    for axis in ("instance_count", "visible_fragmentation", "proximity_proxy", "appearance_proxy"):
        buckets = {}
        for row in rows:
            label = (row.get("difficulty") or {}).get("bins", {}).get(axis, "unavailable")
            buckets.setdefault(label, []).append(row)
        result[axis] = [dict(bin=label, **summarize(group)) for label, group in sorted(buckets.items())]
    return result


def attention_components(outputs, targets, valid=None, prediction=None):
    """Fractional grid coverage assigns soft mass to visible GT components.

    Assignment is a diagnostic argmax, never a ground-truth trajectory or a
    discrete tracking claim. Fractional area pooling preserves thin masks.
    """
    import torch
    from torch.nn import functional as F
    from scipy.optimize import linear_sum_assignment
    from .metrics import pairwise_iou
    trace = attention_trace(outputs)
    if not trace["available"]: return trace, None
    attention = outputs["aux"]["path_attention"].detach().float().cpu().numpy()[0]
    gt = np.asarray(targets, bool)
    if valid is not None: gt = gt & np.asarray(valid, bool).reshape(gt.shape[1:])
    components, coverage = [], []
    for instance, mask in enumerate(gt):
        labels, count = ndimage.label(mask, np.ones((3, 3), bool))
        areas = np.bincount(labels.ravel())
        for component in range(1, count+1):
            if areas[component] < 16: continue
            field = torch.from_numpy((labels == component).astype(np.float32))[None, None]
            coverage.append(F.adaptive_avg_pool2d(field, attention.shape[-2:])[0, 0].numpy())
            components.append(dict(component_id=len(components)+1, gt_instance=instance, pixels=int(areas[component])))
    flat = attention.reshape(*attention.shape[:2], -1)
    mass = np.einsum("qtn,kn->qtk", flat, np.asarray(coverage).reshape(len(coverage), -1), optimize=True) if coverage else np.zeros((*attention.shape[:2], 0), np.float32)
    visible_mass = mass.sum(-1)
    dominant = mass.argmax(-1)+1 if coverage else np.zeros(attention.shape[:2], np.int64)
    dominant[visible_mass <= 0] = 0
    maximum = mass.max(-1) if coverage else np.zeros(attention.shape[:2], np.float32)
    lookup = np.array([-1]+[c["gt_instance"] for c in components])
    assignments = lookup[dominant]
    matched = []
    if prediction is not None and len(prediction["masks"]) and len(gt):
        iou = pairwise_iou(prediction["masks"], gt, valid)
        rr, cc = linear_sum_assignment(-iou)
        for pred_index, gt_index in zip(rr, cc):
            query = prediction["queries"][int(pred_index)]
            relevant = [i for i, item in enumerate(components) if item["gt_instance"] == gt_index]
            matched.append(dict(query=query, matched_gt_instance=int(gt_index), mask_iou=float(iou[pred_index, gt_index]),
                attention_mass_by_step=mass[query][:, relevant].sum(-1).tolist() if relevant else [0.]*attention.shape[1]))
    trace.update(component_protocol="GT visible 8-connected components >=16px; fractional adaptive area pooling onto attention grid; offline only",
        component_lookup=components, gt_visible_mass_by_step=visible_mass.tolist(),
        dominant_component_by_step=dominant.tolist(), dominant_gt_instance_by_step=assignments.tolist(),
        dominant_component_mass_by_step=maximum.tolist(),
        dominant_instance_changes=((assignments[:, 1:] != assignments[:, :-1]) & (assignments[:, 1:] >= 0) & (assignments[:, :-1] >= 0)).sum(-1).tolist(),
        matched_prediction_gt=matched,
        caveat="argmax mass is not a calibrated identity decision; low mass and missing small fragments remain visible")
    arrays = dict(attention=attention.astype(np.float16), positions_xy=np.asarray(trace["positions_xy"], np.float32),
                  component_mass=mass.astype(np.float16))
    return trace, arrays


def save_diagnostic_sample(folder, sample, prediction, attention_arrays=None):
    """Fixed-sample visualization only; native prediction IDs stay lossless."""
    import hashlib
    from pathlib import Path
    from scipy.optimize import linear_sum_assignment
    from .metrics import pairwise_iou
    name = hashlib.sha256(str(sample["id"]).encode()).hexdigest()[:16]
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    h, w = sample["meta"]["original_size"]
    rgb = np.clip(sample["image"].numpy()[:, :h, :w].transpose(1, 2, 0)*255, 0, 255).astype(np.uint8)
    gt = sample["masks"].numpy()[:, :h, :w].astype(bool)
    pred = np.asarray(prediction["masks"][:, :h, :w], bool)
    palette = np.random.default_rng(20261031).integers(40, 235, size=(len(gt)+len(pred)+1, 3), dtype=np.uint8)
    mapping = {j: len(gt)+j+1 for j in range(len(pred))}
    if len(gt) and len(pred):
        rr, cc = linear_sum_assignment(-pairwise_iou(pred, gt))
        mapping.update({int(j): int(i)+1 for j, i in zip(rr, cc)})
    ids = np.zeros((h, w), np.uint16)
    gt_overlay, pred_overlay = rgb.copy(), rgb.copy()
    for j, mask in enumerate(gt): gt_overlay[mask] = (.45*rgb[mask]+.55*palette[j+1]).astype(np.uint8)
    for j, mask in enumerate(pred):
        ids[mask] = j+1
        pred_overlay[mask] = (.45*rgb[mask]+.55*palette[mapping[j]]).astype(np.uint8)
    panels = []
    for title, array in zip(("RGB", "Visible GT", "Prediction (GT-aligned colors)"), (rgb, gt_overlay, pred_overlay)):
        image = Image.fromarray(array); image.thumbnail((420, 420), Image.Resampling.BILINEAR)
        panel = Image.new("RGB", (420, image.height+25), "white")
        panel.paste(image, (0, 25)); ImageDraw.Draw(panel).text((6, 5), title, fill="black")
        panels.append(panel)
    combined = Image.new("RGB", (1260, max(p.height for p in panels)), "white")
    for i, panel in enumerate(panels): combined.paste(panel, (i*420, 0))
    combined.save(folder/(name+"_panel.png"))
    np.savez_compressed(folder/(name+"_prediction.npz"), instances=ids)
    files = dict(panel=name+"_panel.png", prediction=name+"_prediction.npz",
                 visualization="thumbnail only; inference/native prediction not resized")
    if attention_arrays is not None:
        np.savez_compressed(folder/(name+"_attention.npz"), **attention_arrays)
        files.update(attention=name+"_attention.npz", attention_storage="FP16 quantization of original soft attention and component mass; positions FP32")
    return files
