"""OFFLINE GT-prompt diagnostic, deliberately isolated from RGB deployment.

Oracle prompts measure sensitivity to correct instance/point supervision. They
are not an automatic method, an accuracy upper bound, or an equal-budget control
when their group/point counts differ from the automatic planner.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .types import CropSpec, PromptPlan, PromptSpec


FLAGS = {"ground_truth_used": True, "ground_truth_used_for_inference": True,
         "diagnostic_only": True, "deployment_allowed": False}


def _starts(length):
    if length <= 1008:
        return [0]
    return sorted(set([*range(0, length-1008+1, 752), length-1008]))


def _disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))


def _distance(mask):
    # Explicit zero border makes a full-frame foreground's distance finite.
    padded = np.pad(mask.astype(np.uint8), 1)
    return cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def _inner_canvas(shape, margin=4):
    valid = np.zeros(shape, bool)
    h, w = shape
    if h > 2*margin and w > 2*margin:
        valid[margin:h-margin, margin:w-margin] = True
    return valid


def _ordered_visible_paths(skeleton):
    """Walk separate unbranched visible fragments; never connect hidden gaps."""
    degree = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U,
                         np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - skeleton
    junctions = (degree > 2) & skeleton
    clean = skeleton & ~(cv2.dilate(junctions.astype(np.uint8), _disk(2)) > 0)
    count, labels = cv2.connectedComponents(clean.astype(np.uint8), connectivity=8)
    paths = []
    for index in range(1, count):
        pixels = {tuple(map(int, p)) for p in np.argwhere(labels == index)}
        neighbors = {p: sorted((p[0]+dy, p[1]+dx) for dy in (-1, 0, 1)
                               for dx in (-1, 0, 1) if (dy or dx)
                               and (p[0]+dy, p[1]+dx) in pixels) for p in pixels}
        if any(len(n) > 2 for n in neighbors.values()):
            continue
        ends = sorted(p for p, adjacent in neighbors.items() if len(adjacent) <= 1)
        cursor = ends[0] if ends else min(pixels)
        visited, ordered = set(), []
        while cursor not in visited:
            ordered.append(cursor)
            visited.add(cursor)
            following = [p for p in neighbors[cursor] if p not in visited]
            if not following:
                break
            cursor = following[0]
        if len(visited) == len(pixels):
            paths.append(np.asarray(ordered, np.int32))
    return sorted(paths, key=lambda p: tuple(p[0]))


def _positive_points(target, other, count):
    interior = _distance(target)
    # Thin instances have no two-pixel interior; use their exact centerline and
    # explicitly disclose the reduced clearance instead of inventing pixels.
    clearance = min(2.0, float(interior.max()))
    safe = (target & (interior >= clearance) & _inner_canvas(target.shape)
            & ~(cv2.dilate(other.astype(np.uint8), _disk(3)) > 0))
    paths = _ordered_visible_paths(skeletonize(target) & safe)
    if not paths:
        return (), {"visible_safe_paths": 0, "positive_clearance_pixels": clearance}
    coords, weights = [], []
    for path in paths:
        if len(path) == 1:
            local_weights = np.ones(1)
        else:
            edges = np.linalg.norm(np.diff(path.astype(float), axis=0), axis=1)
            local_weights = np.r_[edges, edges[-1]]
        coords.extend(map(tuple, path))
        weights.extend(local_weights)
    # Quantiles use only visible arc length. A gap contributes no guessed arc.
    cumulative = np.cumsum(weights)
    fractions = {1: (.5,), 2: (.3, .7), 3: (.2, .5, .8)}[count]
    chosen = []
    for fraction in fractions:
        desired = fraction*cumulative[-1]
        order = np.argsort(np.abs(cumulative-desired), kind="stable")
        for index in order:
            y, x = coords[int(index)]
            point = (float(x), float(y))
            if point not in chosen:
                chosen.append(point)
                break
    return tuple(chosen), {"visible_safe_paths": len(paths),
                           "visible_safe_centerline_pixels": len(coords),
                           "positive_clearance_pixels": clearance,
                           "thin_centerline_exception": clearance < 2.0}


def _spread_points(safe, score, count):
    coords = np.argwhere(safe)
    if not len(coords) or count == 0:
        return ()
    scores = score[safe].astype(float)
    selected = []
    for _ in range(min(count, len(coords))):
        if selected:
            distances = np.min(np.stack([np.sum((coords-coords[i])**2, axis=1)
                                        for i in selected]), axis=0)
            utility = scores + np.sqrt(distances)
        else:
            utility = scores.copy()
        if selected:
            utility[selected] = -np.inf
        # np.argwhere/argmax make exact ties deterministic in row-major order.
        selected.append(int(np.argmax(utility)))
    return tuple((float(coords[i, 1]), float(coords[i, 0])) for i in selected)


def _negative_points(target, other, occupancy, count, mode):
    if count == 0:
        return ()
    inner = _inner_canvas(target.shape)
    background = occupancy == 0
    background_dist = _distance(background)
    bg_safe = background & inner & (background_dist >= 3)
    other_dist = _distance(other & ~target & (occupancy == 1))
    other_safe = (other & ~target & inner & (occupancy == 1) & (other_dist >= 2)
                  & ~(cv2.dilate(target.astype(np.uint8), _disk(3)) > 0))
    if mode == "background":
        return _spread_points(bg_safe, background_dist, count)
    if mode == "other_gt":
        return _spread_points(other_safe, other_dist, count)
    others = _spread_points(other_safe, other_dist, (count+1)//2)
    backs = _spread_points(bg_safe, background_dist, count-len(others))
    if len(others)+len(backs) < count:
        others = _spread_points(other_safe, other_dist, count-len(backs))
    return others + backs


def _box(target, positive, mode):
    if mode == "none":
        return None
    h, w = target.shape
    if mode == "expanded":
        yy, xx = np.where(target)
        margin = 8
        a, b, c, d = xx.min(), yy.min(), xx.max()+1, yy.max()+1
    else:
        xx, yy = np.asarray(positive).T
        margin = 32
        a, b, c, d = xx.min(), yy.min(), xx.max()+1, yy.max()+1
    return (float(max(0, a-margin)), float(max(0, b-margin)),
            float(min(w, c+margin)), float(min(h, d+margin)))


def oracle_plan(rgb, gt_masks, positive_count=3, negative_count=0,
                negative_mode="background", box_mode="none"):
    """Return a visibly GT-contaminated plan for explicit offline diagnostics.

    Native 1008 tiles overlap by at least 256 pixels. Each tile reserves up to
    six independent GT groups in input-index order, before checking point
    availability. Unsafe/over-budget objects are logged, never silently replaced
    by easier later instances. GT arrays must contain visible masks only.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty native uint8 RGB")
    if positive_count not in (1, 2, 3) or negative_count not in (0, 1, 2, 3):
        raise ValueError("Expected 1-3 positive and 0-3 negative points")
    if negative_mode not in ("background", "other_gt", "mixed") or box_mode not in ("none", "expanded", "local"):
        raise ValueError("Unknown oracle negative/box mode")
    if positive_count + negative_count + (0 if box_mode == "none" else 2) > 8:
        raise ValueError("Requested point/box tokens exceed fixed SAM slots")
    masks = []
    h, w = rgb.shape[:2]
    occupancy = np.zeros((h, w), np.uint32)
    for gt in gt_masks:
        a = np.asarray(gt)
        if a.shape != (h, w) or a.ndim != 2 or not np.isfinite(a).all():
            raise ValueError("Every GT instance must be a finite native HxW visible mask")
        mask = a > 0
        masks.append(mask)
        occupancy += mask.astype(np.uint32)
    crops, prompts, rows, covered = [], [], [], set()
    for r, y0 in enumerate(_starts(h)):
        for c, x0 in enumerate(_starts(w)):
            x1, y1 = min(w, x0+1008), min(h, y0+1008)
            crop = CropSpec(f"oracle_r{r:03d}_c{c:03d}", (x0, y0, x1, y1))
            crops.append(crop)
            local = [m[y0:y1, x0:x1] for m in masks]
            present = [i for i, m in enumerate(local) if m.any()]
            eligible = present[:6]
            omitted = [{"gt_index": i, "reason": "six_group_capacity"} for i in present[6:]]
            produced = []
            local_occupancy = occupancy[y0:y1, x0:x1]
            for i in eligible:
                target = local[i]
                other = (local_occupancy - target.astype(np.uint32)) > 0
                positive, details = _positive_points(target, other, positive_count)
                if not positive:
                    omitted.append({"gt_index": i, "reason": "no_safe_visible_centerline"})
                    continue
                negative = _negative_points(target, other, local_occupancy, negative_count, negative_mode)
                metadata = {**FLAGS, "gt_index": i, "requested_positive_points": positive_count,
                            "actual_positive_points": len(positive), "requested_negative_points": negative_count,
                            "actual_negative_points": len(negative), "negative_mode": negative_mode,
                            "box_mode": box_mode, **details}
                prompts.append(PromptSpec(f"{crop.crop_id}_gt{i:04d}", crop.crop_id, positive, negative,
                                          _box(target, positive, box_mode), metadata))
                produced.append(i)
                covered.add(i)
            rows.append({"crop_id": crop.crop_id, "visible_gt_indices": present,
                         "reserved_gt_indices": eligible, "prompted_gt_indices": produced,
                         "omitted_gt": omitted})
    visible = {i for i, m in enumerate(masks) if m.any()}
    audit = {**FLAGS, "schema": "sam_once_oracle_diagnostic_v1", "native_resolution": True,
             "canvas_size": 1008, "overlap": 256, "max_groups_per_crop": 6,
             "gt_instance_count": len(masks), "visible_gt_instance_count": len(visible),
             "covered_gt_indices": sorted(covered), "uncovered_gt_indices": sorted(visible-covered),
             "empty_gt_indices": [i for i, m in enumerate(masks) if not m.any()],
             "requested_positive_count": positive_count, "requested_negative_count": negative_count,
             "negative_mode": negative_mode, "box_mode": box_mode, "crops": rows,
             "actual_groups": len(prompts), "actual_positive_points": sum(len(p.positive) for p in prompts),
             "actual_negative_points": sum(len(p.negative) for p in prompts),
             "actual_boxes": sum(p.box is not None for p in prompts),
             "selection_policy": "first six visible GT input indices per tile; never replace unsafe groups",
             "comparison_limits": "GT-assisted diagnostic; not deployment or guaranteed upper bound; actual input budgets may differ"}
    # No GT-derived ridge channel is exported; GT is used explicitly for prompts.
    return PromptPlan(crops, prompts, np.zeros((h, w), np.float32), audit).validate(rgb)
