"""Ground-truth-free, conservative transactions for experimental A/B feedback.

The checks below measure internal consistency, not true segmentation accuracy.
Visible pixels and labels outside the requested ROI are frozen exactly. Hidden
connections are never painted, and a rejected transaction keeps the old result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def feedback_roi(row: dict, image_size, source_size, default_side: int = 512) -> list[int]:
    width, height = image_size
    sx, sy = width / source_size[0], height / source_size[1]
    if "roi_xyxy" in row:
        x0, y0, x1, y1 = row["roi_xyxy"]
        bounds = (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
        # Conflict pixels can be tiny; retain context and exclusive anchors.
        cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        half = max(default_side/2, (bounds[2]-bounds[0])/2, (bounds[3]-bounds[1])/2)
        bounds = (round(cx-half), round(cy-half), round(cx+half), round(cy+half))
    else:
        x, y = row["xy"][0] * sx, row["xy"][1] * sy
        half = default_side / 2
        bounds = (round(x - half), round(y - half), round(x + half), round(y + half))
    x0, y0, x1, y1 = bounds
    result = [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("Feedback ROI is empty or outside image")
    return result


def roi_union(shape, boxes: list[list[int]]) -> np.ndarray:
    result = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        result[y0:y1, x0:x1] = True
    return result


def spread_interior_points(mask: np.ndarray, limit: int = 3) -> list[list[float]]:
    """Deterministic interior anchors, spatially separated in the observed mask."""
    if not mask.any() or limit <= 0:
        return []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    coords = np.column_stack(np.where(distance >= max(1, float(distance.max()) * 0.35)))
    if not len(coords):
        return []
    seed = int(np.argmax(distance[coords[:, 0], coords[:, 1]]))
    chosen = [coords[seed]]
    for _ in range(limit - 1):
        squared = np.min(np.sum((coords[:, None, :] - np.asarray(chosen)[None, :, :]) ** 2, axis=2), axis=1)
        index = int(np.argmax(squared))
        if squared[index] < 12**2:
            break
        chosen.append(coords[index])
    return [[float(x), float(y)] for y, x in chosen]


def identity_jobs(feedback: list[dict], image_size, source_size,
                  reference_ids: np.ndarray | None, limit: int,
                  default_side: int = 512, exploration: bool = False) -> list[dict]:
    """Build explicit identity hypotheses; proposed alternative points stay separate."""
    jobs = []
    sx, sy = image_size[0] / source_size[0], image_size[1] / source_size[1]
    seen = set()
    for index, row in enumerate(feedback):
        if row.get("status", "unresolved") != "unresolved":
            continue
        if row.get("reason") != "conflicting_instance_proposals" and not row.get("positive_points"):
            continue
        bounds = feedback_roi(row, image_size, source_size, default_side)
        x0, y0, x1, y1 = bounds
        explicit = [[float(x * sx), float(y * sy)] for x, y in row.get("positive_points", [])]
        negatives = [[float(x * sx), float(y * sy)] for x, y in row.get("negative_points", [])]
        target = row.get("target_group", (row.get("proposal_groups") or [None])[0])
        if explicit and reference_ids is not None:
            px,py=map(lambda v:int(round(v)),explicit[0])
            if 0<=py<reference_ids.shape[0] and 0<=px<reference_ids.shape[1]:
                target=int(reference_ids[py,px]) or None
        variants = [(target, explicit, negatives, "supplied")]
        if not explicit:
            variants = []
            if reference_ids is not None:
                local_ids = reference_ids[y0:y1, x0:x1]
                groups = row.get("proposal_groups", []) or [int(v) for v in np.unique(local_ids) if v]
                for group in groups:
                    positive = [[x + x0, y + y0] for x, y in spread_interior_points(local_ids == group)]
                    negative = [[x + x0, y + y0] for x, y in spread_interior_points((local_ids > 0) & (local_ids != group), 2)]
                    variants.append((int(group), positive, negative, "reference_id_interior"))
        if exploration and row.get("alternative_positive_points"):
            variants.append((None, [[float(x * sx), float(y * sy)] for x, y in row["alternative_positive_points"]],
                             negatives, "alternative"))
        for group, positive, negative, origin in variants:
            if len(jobs) >= limit:
                return jobs
            positive = [p for p in positive if x0 <= p[0] < x1 and y0 <= p[1] < y1][:3]
            negative = [p for p in negative if x0 <= p[0] < x1 and y0 <= p[1] < y1][:3]
            if not positive:
                continue
            key = (tuple(bounds), group, tuple(tuple(p) for p in positive))
            if key in seen:
                continue
            seen.add(key)
            jobs.append({"region_index": index, "roi_xyxy": bounds, "target_group": group,
                         "positive_points": positive, "negative_points": negative,
                         "hypothesis": row.get("hypothesis", "same_cable_through_conflict"),
                         "prompt_origin": origin, "status": "unresolved"})
    return jobs


def assess_identity_mask(proposed: np.ndarray, reference_ids: np.ndarray | None,
                         target_group: int | None, positive_local, negative_local,
                         min_anchor_retention: float = 0.90,
                         max_competitor_fraction: float = 0.05) -> dict:
    def hit(point):
        x, y = int(round(point[0])), int(round(point[1]))
        return bool(0 <= x < proposed.shape[1] and 0 <= y < proposed.shape[0] and proposed[y, x])

    audit = {"positive_hits": sum(hit(p) for p in positive_local),
             "positive_count": len(positive_local),
             "negative_hits": sum(hit(p) for p in negative_local),
             "new_supported_pixels": 0, "status": "rejected"}
    if not positive_local or audit["positive_hits"] != len(positive_local) or audit["negative_hits"]:
        return {**audit, "reason": "prompt_contradiction"}
    if reference_ids is None or not target_group or not np.any(reference_ids == target_group):
        return {**audit, "reason": "missing_reference_identity"}
    target = reference_ids == target_group
    competitor = (reference_ids > 0) & ~target
    retention = float(np.count_nonzero(proposed & target) / max(target.sum(), 1))
    intrusion = float(np.count_nonzero(proposed & competitor) / max(proposed.sum(), 1))
    new_supported = int(np.count_nonzero(proposed & (reference_ids == 0)))
    audit.update(anchor_retention=retention, competitor_fraction=intrusion,
                 new_supported_pixels=new_supported)
    if retention < min_anchor_retention:
        return {**audit, "reason": "reference_identity_lost"}
    if intrusion > max_competitor_fraction:
        return {**audit, "reason": "competing_identity_invaded"}
    return {**audit, "status": "accepted", "reason": "prompt_and_identity_consistent"}


def align_ids(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align arbitrary numeric IDs before testing stability, using overlap only."""
    old = [int(v) for v in np.unique(reference) if v]
    new = [int(v) for v in np.unique(candidate) if v]
    result = np.zeros(candidate.shape, np.uint16)
    if not new:
        return result
    overlap = np.zeros((len(old), len(new)), float)
    for i, old_id in enumerate(old):
        values, counts = np.unique(candidate[reference == old_id], return_counts=True)
        counts_by_id = dict(zip(values, counts))
        for j, new_id in enumerate(new):
            overlap[i, j] = counts_by_id.get(new_id, 0)
    mapping = {}
    if old:
        ii, jj = linear_sum_assignment(-overlap)
        mapping = {new[j]: old[i] for i, j in zip(ii, jj) if overlap[i, j] > 0}
    next_id = max(old, default=0) + 1
    for value in new:
        if value not in mapping:
            if next_id > np.iinfo(np.uint16).max:
                raise ValueError("Too many instance identities")
            mapping[value] = next_id
            next_id += 1
        result[candidate == value] = mapping[value]
    return result


def _fragment_count(ids: np.ndarray) -> int:
    count = 0
    for value in np.unique(ids):
        if value:
            n, _, stats, _ = cv2.connectedComponentsWithStats((ids == value).astype(np.uint8), 8)
            count += sum(int(stats[i, cv2.CC_STAT_AREA]) >= 16 for i in range(1, n))
    return int(count)


def commit_roi_update(previous_ids: np.ndarray, candidate_ids: np.ndarray,
                      previous_mask: np.ndarray, candidate_mask: np.ndarray,
                      boxes: list[list[int]], rollback: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    if not (previous_ids.shape == candidate_ids.shape == previous_mask.shape == candidate_mask.shape):
        raise ValueError("Feedback transaction coordinate spaces differ")
    roi = roi_union(previous_ids.shape, boxes)
    aligned = align_ids(previous_ids, candidate_ids)
    ids, mask = previous_ids.copy(), previous_mask.copy()
    ids[roi], mask[roi] = aligned[roi], candidate_mask[roi]
    ids[~mask] = 0
    old_assigned = roi & (previous_ids > 0)
    retained = float(np.count_nonzero(old_assigned & (ids == previous_ids)) / old_assigned.sum()) if old_assigned.any() else 1.0
    lost_visible = int(np.count_nonzero(previous_mask & ~mask))
    old_unassigned = int(np.count_nonzero(roi & previous_mask & (previous_ids == 0)))
    new_unassigned = int(np.count_nonzero(roi & mask & (ids == 0)))
    old_fragments, new_fragments = _fragment_count(previous_ids), _fragment_count(ids)
    improvements = old_unassigned - new_unassigned >= 16 or new_fragments < old_fragments
    changed = int(np.count_nonzero(ids != previous_ids))
    reason = "roi_update"
    accepted = bool(roi.any())
    if not roi.any():
        reason = "no_feedback_roi"
    elif rollback and (retained < 0.98 or lost_visible > 0):
        accepted, reason = False, "stable_evidence_changed"
    elif rollback and not improvements:
        accepted, reason = False, "no_measured_consistency_gain"
    if not accepted:
        ids, mask = previous_ids.copy(), previous_mask.copy()
    return ids, mask, {"accepted": accepted, "reason": reason,
                       "reference_label_retention": retained, "lost_visible_pixels": lost_visible,
                       "old_unassigned_pixels": old_unassigned, "new_unassigned_pixels": new_unassigned,
                       "old_fragments": old_fragments, "new_fragments": new_fragments,
                       "candidate_changed_pixels": changed,
                       "committed_changed_pixels": int(np.count_nonzero(ids != previous_ids)),
                       "committed_new_visible_pixels": int(np.count_nonzero(mask & ~previous_mask)),
                       "outside_roi_changed_pixels": int(np.count_nonzero((ids != previous_ids) & ~roi)),
                       "ground_truth_used": False,
                       "caveat": "Consistency proxy, not a guarantee of true segmentation accuracy."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-ids", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--previous-mask", type=Path, required=True)
    parser.add_argument("--candidate-mask", type=Path, required=True)
    parser.add_argument("--round-report", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    candidate_ids = np.asarray(Image.open(args.candidate_ids))
    size = (candidate_ids.shape[1], candidate_ids.shape[0])
    previous_ids = np.asarray(Image.open(args.previous_ids).resize(size, Image.Resampling.NEAREST))
    previous_mask = np.asarray(Image.open(args.previous_mask).convert("L").resize(size, Image.Resampling.NEAREST)) > 0
    candidate_mask = np.asarray(Image.open(args.candidate_mask).convert("L")) > 0
    report = json.loads(args.round_report.read_text(encoding="utf-8"))
    boxes = report.get("transaction_rois", [])
    ids, mask, audit = commit_roi_update(previous_ids, candidate_ids, previous_mask, candidate_mask, boxes, args.rollback)
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ids).save(args.output / "committed_instance_ids.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / "committed_mask.png")
    from cable_identity_graph import render
    rgb = np.asarray(Image.open(args.image).convert("RGB").resize(size, Image.Resampling.LANCZOS))
    render(rgb, ids, args.output / "committed_instances_overlay.png")
    (args.output / "feedback_transaction.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(audit, flush=True)


if __name__ == "__main__":
    main()
