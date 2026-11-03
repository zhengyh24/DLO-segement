"""Offline domain-wise instance evaluation, never imported by inference.

Missing/failed predictions remain in the denominator as zero predictions.
Morphological merge/fragment statistics are proxies, not physical identities.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .runtime import bootstrap
from .io import new_directory, read_json, write_json, sha256, rgb_hash
bootstrap()


def load_gt_mask(path, shape):
    """Same native per-instance PNG/NPY contract as training."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            mask = np.asarray(image)
    if mask.ndim != 2 or mask.shape != shape or not np.isfinite(mask).all():
        raise ValueError(f"GT/native RGB geometry or value mismatch: {path}")
    return mask > 0


def evaluate(manifest, predictions, output, split="check", baseline=None):
    from run_evidence_fusion_experiment import instance_scores
    from evaluate_full_comparison import binary
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    manifest = Path(manifest)
    rows = [r for r in read_json(manifest)["records"] if split == "all" or r["split"] == split]
    if not rows:
        raise ValueError("No requested evaluation cases")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation case IDs would reweight the metrics")
    out = new_directory(output)
    results = []
    for row in rows:
        resolve = lambda v: Path(v) if Path(v).is_absolute() else manifest.parent / v
        rgb = np.asarray(Image.open(resolve(row["image_path"])).convert("RGB"))
        if row.get("rgb_sha256", rgb_hash(rgb)) != rgb_hash(rgb):
            raise ValueError(f"Source RGB changed since frozen manifest: {row['case_id']}")
        gt = [load_gt_mask(resolve(p), rgb.shape[:2]) for p in row["gt_masks"]]
        union = np.logical_or.reduce(gt) if gt else np.zeros(rgb.shape[:2], bool)
        case = Path(predictions) / row["case_id"]
        audit = read_json(case / "inference.json") if (case / "inference.json").exists() else {"status": "missing"}
        ids = np.zeros(union.shape, np.uint16)
        visible = np.zeros(union.shape, bool)
        complete = all((case / p).is_file() for p in ("instance_ids.png", "visible_foreground.png", "unknown_visible.png"))
        status = audit.get("status", "unknown")
        if complete and status in ("ok", "empty", "geometry_failed"):
            if audit.get("rgb_sha256") != rgb_hash(rgb):
                raise ValueError(f"Prediction/RGB mismatch: {row['case_id']}")
            ids = np.asarray(Image.open(case / "instance_ids.png"))
            visible = np.asarray(Image.open(case / "visible_foreground.png")) > 0
            unknown = np.asarray(Image.open(case / "unknown_visible.png")) > 0
            if (ids.shape != union.shape or visible.shape != union.shape or unknown.shape != union.shape
                    or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
                    or np.any((ids > 0) & ~visible)
                    or not np.array_equal(unknown, visible & (ids == 0))):
                raise ValueError("Prediction violates native visible-coordinate contract")
        else:
            status = "missing_or_failed"
        metrics = instance_scores(ids, gt)
        gt_coverage = [float(np.count_nonzero(visible & g) / max(g.sum(), 1)) for g in gt]
        gt_fragment_counts = [len(np.unique(ids[g][ids[g] > 0])) for g in gt]
        merges = 0
        for ident in np.unique(ids[ids > 0]):
            # Count GT instances with >=10% area assigned to this prediction.
            if sum(np.count_nonzero((ids == ident) & g) >= max(1, .1*g.sum()) for g in gt) > 1:
                merges += 1
        thin = skeletonize(union) & (distance_transform_edt(union) <= 3)
        result = {"case_id": row["case_id"], "dataset": row.get("dataset", "unknown"), "status": status,
                  **metrics, **binary(visible, union),
                  "unknown_visible_pixels": int(np.count_nonzero(visible & (ids == 0))),
                  "visible_pixels": int(visible.sum()),
                  "gt_low_coverage_count_lt_10pct": sum(c < .1 for c in gt_coverage),
                  "fragment_proxy_excess": sum(max(0, n-1) for n in gt_fragment_counts),
                  "merge_proxy_count": merges,
                  "thin_centerline_recalled": int((visible & thin).sum()), "thin_centerline_pixels": int(thin.sum()),
                  "end_to_end_cold_seconds": audit.get("end_to_end_cold_seconds"),
                  "device_name": audit.get("sam", {}).get("device_name"),
                  "timing_scope": audit.get("timing_scope"),
                  "refinement_budget": audit.get("refinement", {})}
        results.append(result)
    summaries = {}
    for domain in sorted({r["dataset"] for r in results}):
        group = [r for r in results if r["dataset"] == domain]
        total = lambda key: sum(r[key] for r in group)
        tp, fp, fn = (total(k) for k in ("tp_instances", "fp_instances", "fn_instances"))
        times = [r["end_to_end_cold_seconds"] for r in group if r["end_to_end_cold_seconds"] is not None]
        summaries[domain] = {"n": len(group), "instance_iou": float(np.mean([r["instance_iou"] for r in group])),
            "pq": total("matched_iou_sum") / max(tp + .5*fp + .5*fn, 1),
            "foreground_iou": float(np.mean([r["foreground_iou"] for r in group])),
            "tp": tp, "fp": fp, "fn": fn, "failures": sum(r["status"] not in ("ok", "empty") for r in group),
            "unknown_visible_fraction": total("unknown_visible_pixels") / max(total("visible_pixels"), 1),
            "thin_centerline_recall_proxy": total("thin_centerline_recalled") / max(total("thin_centerline_pixels"), 1),
            "fragment_proxy_excess": total("fragment_proxy_excess"), "merge_proxy_count": total("merge_proxy_count"),
            "low_coverage_gt_count": total("gt_low_coverage_count_lt_10pct"),
            "timing_n": len(times), "cold_p50_seconds": float(np.median(times)) if times else None,
            "cold_p95_seconds": float(np.quantile(times, .95)) if times else None}
    report = {"schema": "sam_once_evaluation_v1", "manifest_sha256": sha256(manifest), "split": split,
              "scope": "historical regression unless source protocol establishes otherwise", "rows": results,
              "summaries": summaries, "automatic_promotion": False,
              "proxy_note": "thin width<=6px; fragments are IDs touching GT; merge is >=10% area of multiple GT. Not physical-error labels."}
    if baseline:
        old = read_json(baseline)
        same_cases = {(r["case_id"], r["dataset"]) for r in old["rows"]} == {(r["case_id"], r["dataset"]) for r in results}
        if not same_cases:
            raise ValueError("Baseline must contain exactly the same domain/case IDs")
        deltas = {d: {m: summaries[d][m] - old["summaries"][d][m] for m in ("instance_iou", "pq")} for d in summaries}
        signatures = lambda rr: {(r.get("device_name"), r.get("timing_scope")) for r in rr}
        same_timing = signatures(old["rows"]) == signatures(results) and all(r.get("device_name") for r in results)
        complete_timing = all(s["timing_n"] == s["n"] and old["summaries"][d]["timing_n"] == s["n"] for d,s in summaries.items())
        report["comparison"] = {"domain_deltas": deltas, "same_hardware_and_timing_scope": same_timing,
            "domain_quality_nonregression": all(v >= 0 for delta in deltas.values() for v in delta.values()),
            "latency_comparable": bool(same_timing and complete_timing),
            "faster_all_domains": bool(same_timing and complete_timing and all(summaries[d]["cold_p50_seconds"] < old["summaries"][d]["cold_p50_seconds"] for d in summaries))}
    write_json(out / "evaluation.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation", "check", "all"], default="check")
    p.add_argument("--baseline", type=Path)
    a = p.parse_args()
    print(evaluate(a.manifest, a.predictions, a.output, a.split, a.baseline)["summaries"])


if __name__ == "__main__":
    main()
