"""Validation-only instance-metric checkpoint selection on cached SAM evidence.

No SAM forward is performed. A selected checkpoint is a validation candidate,
not a replacement for the historical coverage_color deployment baseline.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .io import load_case, new_directory, read_json, rgb_hash, sha256, write_json


def validation_records(obj, manifest_path):
    """Resolve only validation paths; inspect other splits' isolation metadata."""
    if obj.get("schema") != "sam_once_training_v1":
        raise ValueError("Expected sam_once_training_v1 manifest")
    records = obj.get("records", [])
    if not records:
        raise ValueError("Empty training manifest")
    names, clusters, hashes, selected = set(), {}, {}, []
    base = Path(manifest_path).resolve().parent
    for row in records:
        split = row.get("split")
        case, cluster, digest = row.get("case_id"), row.get("cluster_id"), row.get("rgb_sha256")
        if split not in ("train", "validation", "check"):
            raise ValueError("Every record must have a frozen known split")
        if (not isinstance(case, str) or not case or case in (".", "..")
                or any(c in case for c in "/\\:") or case != case.rstrip(" .")):
            raise ValueError("Case IDs must be safe single directory names")
        if case.casefold() in names:
            raise ValueError("Duplicate case ID (including Windows case aliases)")
        names.add(case.casefold())
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("Every record needs its frozen cluster_id")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Every record needs its frozen decoded RGB SHA256")
        for mapping, key, label in ((clusters, cluster, "cluster"), (hashes, digest, "RGB")):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Cross-split {label} leakage")
            mapping[key] = split
        if split != "validation":
            continue
        if not isinstance(row.get("dataset"), str) or not row["dataset"]:
            raise ValueError("Validation rows need a dataset/domain")
        if not isinstance(row.get("gt_masks"), list):
            raise ValueError("Validation rows need an explicit list of native GT masks")
        resolve = lambda p: str((base / p).resolve())
        selected.append({**row, "image_path": resolve(row["image_path"]),
                         "cache_dir": resolve(row["cache_dir"]),
                         "gt_masks": [resolve(p) for p in row["gt_masks"]]})
    if not selected:
        raise ValueError("No validation records; train/check cannot select checkpoints")
    return sorted(selected, key=lambda r: (r["dataset"], r["case_id"]))


def choose_checkpoint(baseline, candidates, baseline_failures=()):
    """Per-domain nonregression, then equal-domain PQ/IoU, then SHA ascending."""
    domains = sorted(baseline)
    if not domains:
        raise ValueError("Selection requires nonempty validation domains")
    for domain in domains:
        for metric in ("instance_iou", "pq"):
            value = baseline[domain].get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid baseline instance metric")
        if not isinstance(baseline[domain].get("n"), int) or baseline[domain]["n"] <= 0:
            raise ValueError("Validation domain must contain cases")
    decisions = []
    for candidate in candidates:
        reasons, deltas = [], {}
        summary = candidate.get("summaries", {})
        digest = candidate.get("checkpoint_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            reasons.append("invalid_checkpoint_sha256")
        if candidate.get("smoke_only", True) is not False:
            reasons.append("smoke_checkpoint_diagnostic_only")
        if candidate.get("status", "ok") != "ok":
            reasons.append("checkpoint_evaluation_failed")
        if baseline_failures:
            reasons.append("baseline_prediction_execution_failed")
        if candidate.get("case_failures"):
            reasons.append("candidate_prediction_execution_failed")
        if set(summary) != set(domains):
            reasons.append("validation_domain_mismatch")
        valid_values = True
        for domain in domains:
            row = summary.get(domain, {})
            if row.get("n") != baseline[domain]["n"]:
                reasons.append(f"case_count_mismatch:{domain}")
            deltas[domain] = {}
            for metric in ("instance_iou", "pq"):
                value = row.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    reasons.append(f"invalid_metric:{domain}:{metric}")
                    valid_values = False
                    continue
                delta = float(value - baseline[domain][metric])
                deltas[domain][metric] = delta
                if delta < 0:
                    reasons.append(f"domain_regression:{domain}:{metric}")
        mean_pq = sum(summary[d]["pq"] for d in domains) / len(domains) if valid_values else None
        mean_iou = sum(summary[d]["instance_iou"] for d in domains) / len(domains) if valid_values else None
        decisions.append({**candidate, "eligible": not reasons, "reasons": reasons,
                          "domain_deltas": deltas, "domain_mean_pq": mean_pq,
                          "domain_mean_instance_iou": mean_iou})
    eligible = [r for r in decisions if r["eligible"]]
    eligible.sort(key=lambda r: (-r["domain_mean_pq"], -r["domain_mean_instance_iou"], r["checkpoint_sha256"]))
    return {"selected": eligible[0] if eligible else None, "candidates": decisions,
            "rule": "Every validation domain IoU and PQ >= no_refinement; rank mean-domain PQ, then mean-domain IoU, then SHA256 ascending",
            "domain_weighting": "equal domains, not image-count weighted", "automatic_promotion": False}


def checkpoint_manifest_matches(checkpoint_audit, manifest_sha256):
    """A formal checkpoint must retain the exact frozen split manifest it used."""
    return (checkpoint_audit.get("smoke_only") is True
            or checkpoint_audit.get("training_manifest_sha256") == manifest_sha256)


def _validate_initial_cache(rgb, plan, records, audit, expected_rgb):
    if rgb_hash(rgb) != expected_rgb:
        raise ValueError("Validation RGB/cache mismatch")
    if audit.get("cache_kind") != "actual_single_batch_sam" or audit.get("ground_truth_used_for_inference") is not False:
        raise ValueError("Selection requires original GT-free single-batch SAM caches")
    if records:
        from .train import validate_native_cache
        validate_native_cache(audit, plan, records)
    else:
        sam = audit.get("sam", {})
        calls = sam.get("calls", {})
        expected = {"backbone_forward_image", "image_encoder", "prompt_encoder", "mask_decoder",
                    "detector", "propagation_decoder", "text_encoder"}
        if (plan.prompts or sam.get("status") != "empty_prompts" or sam.get("injected_test_model") is not False
                or set(calls) != expected or any(value != 0 for value in calls.values())):
            raise ValueError("Empty validation cases must retain a truthful zero-call audit")
    return records


def _snapshot(records):
    snapshot = {}
    for row in records:
        for filename in [row["image_path"], *row["gt_masks"], str(Path(row["cache_dir"]) / "case.json")]:
            snapshot[filename] = sha256(filename)
        with Image.open(row["image_path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb_hash(rgb) != row["rgb_sha256"]:
            raise ValueError("Validation source RGB changed after frozen manifest")
    return snapshot


def _check_snapshot(snapshot):
    for filename, expected in snapshot.items():
        if sha256(filename) != expected:
            raise ValueError(f"Validation source changed during selection: {filename}")


def _predict_cases(records, directory, model, config, checkpoint_audit=None):
    from .backend import assign_instances
    from .refiner import refine_proposals
    out = new_directory(directory)
    failures = []
    for row in records:
        case = new_directory(out / row["case_id"])
        started = time.perf_counter()
        audit = {"rgb_sha256": row["rgb_sha256"], "ground_truth_used_for_inference": False,
                 "sam_calls_this_command": 0, "source_cache": row["cache_dir"],
                 "timing_scope": "cached SAM evidence, optional refinement and fixed backend; excludes SAM forward and model loading; not RGB end-to-end",
                 "refinement": {"rounds": 0}, "checkpoint": checkpoint_audit}
        try:
            rgb, plan, proposals, parent = load_case(row["cache_dir"])
            _validate_initial_cache(rgb, plan, proposals, parent, row["rgb_sha256"])
            audit["parent_sam_audit"] = parent.get("sam")
            if model is not None:
                proposals, audit["refinement"] = refine_proposals(rgb, proposals, plan.ridge, model, config)
            ids, visible, unknown, backend = assign_instances(rgb, proposals, len(plan.crops))
            Image.fromarray(ids).save(case / "instance_ids.png")
            Image.fromarray(visible.astype(np.uint8) * 255).save(case / "visible_foreground.png")
            Image.fromarray(unknown.astype(np.uint8) * 255).save(case / "unknown_visible.png")
            audit.update(status=backend["status"], backend=backend)
        except Exception as exc:
            # Evaluation keeps failed cases as zero predictions in the denominator.
            audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failures.append({"case_id": row["case_id"], "error": audit["error"]})
        audit["cached_case_seconds"] = time.perf_counter() - started
        write_json(case / "inference.json", audit)
    return failures


def select_checkpoints(manifest, checkpoints, output, *, device="cpu", rounds=3,
                       max_visits=12, max_pixels=786432, roi_size=256, mode="iterative",
                       allow_smoke_refiner=False):
    from .evaluate import evaluate
    from .refiner import RefinerConfig, load_refiner
    manifest = Path(manifest).resolve()
    manifest_digest = sha256(manifest)
    source = read_json(manifest)
    records = validation_records(source, manifest)
    config = RefinerConfig(device=device, rounds=rounds, max_visits=max_visits,
                           max_pixels=max_pixels, roi_size=roi_size, mode=mode).validate()
    checkpoint_rows, seen = [], set()
    for checkpoint in checkpoints:
        path = Path(checkpoint).resolve()
        digest = sha256(path)
        if digest in seen:
            raise ValueError("Duplicate checkpoint content; supply each SHA256 only once")
        seen.add(digest)
        checkpoint_rows.append((path, digest))
    if not checkpoint_rows:
        raise ValueError("Supply at least one checkpoint")
    frozen = _snapshot(records)
    # Fail before model selection if the baseline cache provenance is invalid.
    for row in records:
        rgb, plan, proposals, audit = load_case(row["cache_dir"])
        _validate_initial_cache(rgb, plan, proposals, audit, row["rgb_sha256"])
    del rgb, plan, proposals, audit
    out = new_directory(output)
    validation_manifest = out / "validation_manifest.json"
    write_json(validation_manifest, {**source, "records": records,
                                     "selection_split": "validation", "parent_manifest_sha256": sha256(manifest)})
    write_json(out / "input_snapshot.json", {"files_sha256": frozen,
                                            "checkpoints_sha256": {str(p): h for p, h in checkpoint_rows}})
    _check_snapshot(frozen)
    baseline_path = out / "no_refinement"
    baseline_failures = _predict_cases(records, baseline_path, None, config)
    baseline = evaluate(validation_manifest, baseline_path, out / "no_refinement_evaluation", split="validation")
    candidates = []
    for index, (path, digest) in enumerate(checkpoint_rows):
        _check_snapshot(frozen)
        if sha256(path) != digest:
            raise ValueError("Checkpoint changed during selection")
        row = {"checkpoint": str(path), "checkpoint_sha256": digest, "status": "failed",
               "smoke_only": True, "summaries": {}}
        model = None
        try:
            model, model_audit = load_refiner(path, device)
            row["checkpoint_audit"] = model_audit
            row["smoke_only"] = model_audit["smoke_only"]
            if model_audit["sha256"] != digest:
                raise ValueError("Loaded checkpoint hash mismatch")
            if not checkpoint_manifest_matches(model_audit, manifest_digest):
                row.update(status="training_manifest_mismatch",
                           error="Non-smoke checkpoint must use this exact frozen training/validation manifest SHA256",
                           expected_training_manifest_sha256=manifest_digest,
                           observed_training_manifest_sha256=model_audit.get("training_manifest_sha256"))
                candidates.append(row)
                continue
            if row["smoke_only"] and not allow_smoke_refiner:
                row.update(status="smoke_refiner_rejected", error="Use --allow-smoke-refiner for diagnostic evaluation only")
                candidates.append(row)
                continue
            predictions = out / f"checkpoint_{index:03d}_{digest[:12]}"
            failures = _predict_cases(records, predictions, model, config, model_audit)
            _check_snapshot(frozen)
            report = evaluate(validation_manifest, predictions, out / f"evaluation_{index:03d}_{digest[:12]}",
                              split="validation", baseline=out / "no_refinement_evaluation" / "evaluation.json")
            row.update(status="ok", summaries=report["summaries"], case_failures=failures,
                       predictions=str(predictions.resolve()), evaluation=str((out / f"evaluation_{index:03d}_{digest[:12]}" / "evaluation.json").resolve()))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            del model
            gc.collect()
        candidates.append(row)
    _check_snapshot(frozen)
    if sha256(manifest) != manifest_digest:
        raise ValueError("Frozen split manifest changed during selection")
    result = choose_checkpoint(baseline["summaries"], candidates, baseline_failures)
    result.update(schema="sam_once_checkpoint_selection_v1", split="validation",
                  manifest_sha256=manifest_digest, validation_manifest_sha256=sha256(validation_manifest),
                  baseline={"name": "no_refinement", "summaries": baseline["summaries"],
                            "case_failures": baseline_failures,
                            "evaluation": str((out / "no_refinement_evaluation" / "evaluation.json").resolve())},
                  config=asdict(config), selection_case_ids=[r["case_id"] for r in records],
                  formal_baseline_replaced=False, instance_promotion=False,
                  scope="Validation checkpoint selection on cached one-batch SAM outputs; historical regression, not blind test or end-to-end speed evidence")
    write_json(out / "selection.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--max-visits", type=int, default=12)
    p.add_argument("--max-pixels", type=int, default=786432)
    p.add_argument("--roi-size", type=int, choices=(256, 512), default=256)
    p.add_argument("--mode", choices=("iterative", "fixedsingle"), default="iterative")
    p.add_argument("--allow-smoke-refiner", action="store_true")
    a = p.parse_args()
    result = select_checkpoints(**vars(a))
    print({"selected_sha256": result["selected"]["checkpoint_sha256"] if result["selected"] else None,
           "split": "validation", "automatic_promotion": False})


if __name__ == "__main__":
    main()
