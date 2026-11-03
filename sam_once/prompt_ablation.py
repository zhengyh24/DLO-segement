"""Frozen-pool prompt ablations and CPU-only validation of independent runs.

No planning, model calls, ground truth, or prediction-to-prompt feedback occurs
here. A group is one geometric path hypothesis, never a verified cable identity.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

import numpy as np

from .types import PromptPlan, PromptSpec


def _histogram(values):
    return {str(key): value for key, value in sorted(Counter(values).items())}


def _box(metadata, mode):
    canonical, alternate = "box_"+mode, mode+"_box"
    if canonical not in metadata and alternate not in metadata:
        raise ValueError(f"Frozen v2 metadata lacks {canonical}; do not re-plan during an ablation")
    value = metadata.get(canonical, metadata.get(alternate))
    if canonical in metadata and alternate in metadata:
        first = tuple(metadata[canonical]) if metadata[canonical] is not None else None
        second = tuple(metadata[alternate]) if metadata[alternate] is not None else None
        if first != second:
            raise ValueError(f"Conflicting frozen box aliases: {canonical}, {alternate}")
    return tuple(map(float, value)) if value is not None else None


def prompt_input_fingerprint(plan, *, prompt_slots=8):
    """Detect identical frozen conditions before running; excludes RGB/model."""
    value = {"crops": [(c.crop_id, c.xyxy) for c in plan.crops], "prompt_slots": prompt_slots,
             "prompts": [(p.prompt_id, p.crop_id, p.positive, p.negative, p.box) for p in plan.prompts]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def frozen_pool_fingerprint(plan):
    """Same digest for all policies derived from the same selected v2 pools."""
    rows = []
    for prompt in plan.prompts:
        metadata = prompt.metadata
        rows.append({"prompt_id": prompt.prompt_id, "crop_id": prompt.crop_id,
                     **{key: metadata.get(key) for key in (
                         "path_key", "candidate_id", "fragment_id", "positive_pool",
                         "positive_pool_arc_fractions", "negative_pools")},
                     "box_local": _box(metadata, "local"),
                     "box_expanded": _box(metadata, "expanded")})
    value = {"crops": [(c.crop_id, c.xyxy) for c in plan.crops], "groups": rows,
             "native_shape": plan.ridge.shape,
             "ridge_sha256": hashlib.sha256(np.ascontiguousarray(plan.ridge).tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def derive_plan(base_plan, pos_count, neg_count, negative_mode="neighbor", box_mode="none", *,
                prompt_slots=8):
    """Change prompt condition using precomputed pools; preserve every group.

    One positive is the pool sample nearest the path midpoint. Two positives
    are its endpoints; three use the entire (at most three-point) pool. Missing
    safe samples stay missing, with requested and actual counts both recorded.
    """
    if pos_count not in (1, 2, 3) or neg_count not in (0, 1, 2):
        raise ValueError("Expected pos_count in 1..3 and neg_count in 0..2")
    if negative_mode not in ("background", "neighbor", "none") or (negative_mode == "none" and neg_count):
        raise ValueError("Nonzero negatives require background or neighbor mode")
    if box_mode not in ("none", "local", "expanded"):
        raise ValueError("Expected box_mode none, local, or expanded")
    fingerprint = frozen_pool_fingerprint(base_plan)
    prompts = []
    for prompt in base_plan.prompts:
        metadata = deepcopy(prompt.metadata)
        required = ("path_key", "positive_pool", "positive_pool_arc_fractions", "negative_pools")
        if any(key not in metadata for key in required):
            raise ValueError("Frozen v2 pools are required; legacy v1 must remain an independent control")
        pool = [tuple(map(float, xy)) for xy in metadata["positive_pool"]]
        fractions = np.asarray(metadata["positive_pool_arc_fractions"], dtype=float)
        if (not 1 <= len(pool) <= 3 or len(set(pool)) != len(pool) or
            fractions.shape != (len(pool),) or not np.isfinite(fractions).all() or
            np.any((fractions < 0) | (fractions > 1)) or np.any(np.diff(fractions) < 0)):
            raise ValueError(f"Invalid frozen positive pool: {prompt.prompt_id}")
        if pos_count == 1:
            indices = [min(range(len(pool)), key=lambda i: (abs(fractions[i]-.5), i))]
        elif pos_count == 2 and len(pool) > 1:
            indices = [0, len(pool)-1]
        else:
            indices = list(range(len(pool)))
        positive = tuple(pool[i] for i in indices)
        if neg_count:
            if negative_mode not in metadata["negative_pools"]:
                raise ValueError(f"Frozen negative pool lacks mode {negative_mode}")
            negative_pool = metadata["negative_pools"][negative_mode]
            all_negatives = [tuple(map(float, xy)) for xy in negative_pool["points"]]
            if len(all_negatives) > 2 or len(set(all_negatives)) != len(all_negatives):
                raise ValueError("Invalid frozen negative pool; duplicates cannot fill point budgets")
            negative = tuple(all_negatives[:neg_count])
            evidence = deepcopy(negative_pool.get("evidence", [])[:len(negative)])
        else:
            negative, evidence = (), []
        box = None if box_mode == "none" else _box(metadata, box_mode)
        tokens = len(positive)+len(negative)+(2 if box is not None else 0)
        if tokens > prompt_slots:
            raise ValueError("Derived prompt exceeds the unchanged fixed token budget")
        metadata.update(negative_evidence=evidence, sparse_tokens=tokens,
                        requested_positive_count=pos_count, actual_positive_count=len(positive),
                        requested_negative_count=neg_count, actual_negative_count=len(negative),
                        negative_mode=negative_mode, box_mode=box_mode)
        metadata["prompt_ablation"] = {
            "frozen_pool_fingerprint": fingerprint,
            "requested_positive_count": pos_count, "actual_positive_count": len(positive),
            "requested_negative_count": neg_count, "actual_negative_count": len(negative),
            "negative_mode": negative_mode, "box_mode": box_mode,
            "positive_pool_indices": indices, "group_preserved": True,
            "sam_feedback_used": False, "ground_truth_used": False}
        prompts.append(PromptSpec(prompt.prompt_id, prompt.crop_id, positive, negative, box, metadata))
    audit = deepcopy(base_plan.audit)
    audit["prompt_ablation"] = {"frozen_pool_fingerprint": fingerprint,
        "requested_positive_count": pos_count, "requested_negative_count": neg_count,
        "negative_mode": negative_mode, "box_mode": box_mode, "group_selection_frozen": True,
        "sam_feedback_used": False, "ground_truth_used": False, "sam_calls": 0}
    result = PromptPlan(list(base_plan.crops), prompts, base_plan.ridge.copy(), audit)
    h, w = result.ridge.shape
    result.validate(np.empty((h, w, 3), np.uint8))
    audit["prompt_ablation"]["actual_summary"] = summarize_plan(result, prompt_slots=prompt_slots)
    return result


def summarize_plan(plan, *, prompt_slots=8):
    """Actual counts, including abstentions; nominal policy names are insufficient."""
    positive = [len(p.positive) for p in plan.prompts]
    negative = [len(p.negative) for p in plan.prompts]
    tokens = [len(p.positive)+len(p.negative)+(2 if p.box is not None else 0) for p in plan.prompts]
    return {"crop_count": len(plan.crops), "group_count": len(plan.prompts),
            "prompt_input_fingerprint_excludes_rgb_model": prompt_input_fingerprint(plan, prompt_slots=prompt_slots),
            "groups_per_crop": {c.crop_id: sum(p.crop_id == c.crop_id for p in plan.prompts)
                                for c in plan.crops},
            "positive_count_distribution": _histogram(positive),
            "negative_count_distribution": _histogram(negative),
            "joint_point_count_distribution": _histogram(
                f"{len(p.positive)}pos_{len(p.negative)}neg" for p in plan.prompts),
            "groups_with_box": sum(p.box is not None for p in plan.prompts),
            "sparse_token_distribution": _histogram(tokens),
            "adapter_padding_minus_one_distribution": _histogram(prompt_slots-n for n in tokens),
            "native_extra_padding_tokens_per_group": 1,
            "positive_count_shortfall_groups": sum(len(p.positive) < p.metadata.get(
                "prompt_ablation", {}).get("requested_positive_count", p.metadata.get("requested_positive_points",
                p.metadata.get("requested_positive_count", len(p.positive)))) for p in plan.prompts),
            "negative_count_shortfall_groups": sum(len(p.negative) < p.metadata.get(
                "prompt_ablation", {}).get("requested_negative_count", p.metadata.get("requested_negative_points",
                p.metadata.get("requested_negative_count", len(p.negative)))) for p in plan.prompts),
            "group_ids": [p.prompt_id for p in plan.prompts],
            "path_keys": [p.metadata.get("path_key") for p in plan.prompts],
            "candidate_ids": [p.metadata.get("candidate_id", p.metadata.get("fragment_id", p.metadata.get("path_key")))
                              for p in plan.prompts],
            "group_semantics": "independent geometric path hypotheses, not verified instance identities"}


def validate_call_audit(plan, sam_audit):
    """Return validation errors even for a failed attempt; never hide retries."""
    entered = 1 if plan.prompts else 0
    expected = {"backbone_forward_image": entered, "image_encoder": entered,
                "prompt_encoder": entered, "mask_decoder": entered,
                "detector": 0, "propagation_decoder": 0, "text_encoder": 0}
    errors = []
    if sam_audit.get("calls") != expected:
        errors.append("Entered module counts differ from the exact one-batch contract")
    for key, value in (("retries", 0), ("microbatches", entered),
                       ("requested_crops", len(plan.crops)), ("requested_prompts", len(plan.prompts))):
        if sam_audit.get(key) != value:
            errors.append(f"{key}: expected {value}, observed {sam_audit.get(key)}")
    if sam_audit.get("status") != ("ok" if plan.prompts else "empty_prompts"):
        errors.append("SAM attempt did not complete successfully")
    return {"passed": not errors, "errors": errors, "expected_calls": expected,
            "observed_calls": deepcopy(sam_audit.get("calls")),
            "retries": sam_audit.get("retries"), "status": sam_audit.get("status")}


def summarize_run(plan, records, sam_audit, *, require_unrefined=True):
    """Native raw-mask diagnostics, not accuracy metrics or new prompt evidence."""
    validation = validate_call_audit(plan, sam_audit)
    errors = validation["errors"]
    expected = {p.prompt_id: p for p in plan.prompts}
    crops = {c.crop_id: c for c in plan.crops}
    seen, rows, scores = [], [], []
    cover = np.zeros(plan.ridge.shape, np.uint32)
    for record in records:
        seen.append(record.prompt.prompt_id)
        try:
            record.validate()
            prompt = expected[record.prompt.prompt_id]
            if (record.crop != crops[prompt.crop_id] or record.prompt.positive != prompt.positive or
                record.prompt.negative != prompt.negative or record.prompt.box != prompt.box or
                any(record.prompt.metadata.get(key) != prompt.metadata.get(key)
                    for key in ("path_key", "candidate_id", "fragment_id"))):
                raise ValueError("Record coordinates/prompts differ from the frozen plan")
            if require_unrefined and (record.metadata.get("refined") or
                                     not np.array_equal(record.initial_logits, record.current_logits)):
                raise ValueError("Expected unchanged native SAM logits for the no-refiner experiment")
        except (ValueError, KeyError) as exc:
            errors.append(f"{record.prompt.prompt_id}: {exc}")
            continue
        mask = record.initial_logits > 0
        x0, y0, x1, y1 = record.crop.xyxy
        cover[y0:y1, x0:x1] += mask
        score = float(record.score)
        scores.append(score)
        def point_hits(points):
            return sum(bool(mask[int(y), int(x)]) for x, y in points)
        row = {"prompt_id": prompt.prompt_id, "crop_id": prompt.crop_id,
               "path_key": prompt.metadata.get("path_key"), "raw_iou_head_score": score,
               "backend_clipped_score_component": float(.2*np.clip(score, 0, 1)),
               "passes_backend_minimum_area": bool(mask.sum() >= 16),
               "foreground_pixels": int(mask.sum()), "crop_coverage_fraction": float(mask.mean()),
               "positive_count": len(prompt.positive), "positive_foreground_hits": point_hits(prompt.positive),
               "negative_count": len(prompt.negative), "negative_foreground_hits": point_hits(prompt.negative),
               "box_present": prompt.box is not None}
        if prompt.box is not None:
            bx0, by0, bx1, by1 = prompt.box
            inside = mask[int(np.floor(by0)):int(np.ceil(by1)), int(np.floor(bx0)):int(np.ceil(bx1))]
            row["foreground_outside_box_pixels"] = int(mask.sum()-inside.sum())
        rows.append(row)
    if len(seen) != len(set(seen)) or set(seen) != set(expected):
        errors.append("Record groups must match every frozen prompt exactly once")
    validation["passed"] = not errors
    positive_total = sum(row["positive_count"] for row in rows)
    negative_total = sum(row["negative_count"] for row in rows)
    return {"validation": validation, "plan": summarize_plan(plan), "per_group": rows,
            "score": {"count": len(scores), "negative_count": sum(x < 0 for x in scores),
                      "negative_fraction": sum(x < 0 for x in scores)/len(scores) if scores else None,
                      "minimum": min(scores) if scores else None, "maximum": max(scores) if scores else None,
                      "semantics": "unbounded native IoU-head output; fixed backend clips its 20% contribution"},
            "native_mask_coverage": {"foreground_pixels": int(np.count_nonzero(cover)),
                "image_fraction": float(np.mean(cover > 0)), "overlap_pixels": int(np.count_nonzero(cover > 1)),
                "positive_foreground_fraction": sum(r["positive_foreground_hits"] for r in rows)/positive_total
                                                if positive_total else None,
                "negative_foreground_fraction": sum(r["negative_foreground_hits"] for r in rows)/negative_total
                                                if negative_total else None,
                "point_sampling": "floor native crop XY; descriptive, not ground truth",
                "threshold": "initial_logits > 0", "native_size": list(plan.ridge.shape[::-1])},
            "sam_feedback_used": False, "ground_truth_used": False}
