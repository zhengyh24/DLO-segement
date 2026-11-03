"""Bounded prompt study: freeze, infer without labels, then evaluate offline.

The study is a historical validation diagnostic, not a blind test or deployment
benchmark. Each nonempty configuration enters native SAM exactly once. Oracle
results use a separate format and can never enter the production cache loader.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .runtime import ROOT, bootstrap
from .io import new_directory, prompt_from_dict, read_json, rgb_hash, save_case, sha256, write_json
from .types import CropSpec, PromptPlan
bootstrap()

PAIRS = ((1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (2, 2))


def profiles():
    result = [{"id": "v1_control", "kind": "automatic", "planner": "v1"}]
    for pos, neg in PAIRS:
        result.append({"id": f"v2_p{pos}n{neg}_neighbor_none", "kind": "automatic", "planner": "v2",
                       "pos": pos, "neg": neg, "negative": "neighbor", "box": "none"})
    for negative, box in (("background", "none"), ("neighbor", "local"), ("neighbor", "expanded")):
        result.append({"id": f"v2_p2n1_{negative}_{box}", "kind": "automatic", "planner": "v2",
                       "pos": 2, "neg": 1, "negative": negative, "box": box})
    for box in ("none", "expanded"):
        result.append({"id": f"oracle_p3n1_other_gt_{box}", "kind": "oracle_diagnostic", "planner": "oracle",
                       "pos": 3, "neg": 1, "negative": "other_gt", "box": box})
    return result


def read_plan(path, ridge, rgb):
    obj = read_json(path)
    return PromptPlan([CropSpec(c["crop_id"], tuple(c["xyxy"])) for c in obj["crops"]],
                      [prompt_from_dict(p) for p in obj["prompts"]],
                      np.load(ridge, allow_pickle=False), obj["audit"]).validate(rgb)


def prompt_view(rgb, plan):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = ("#ff4040", "#00ff80", "#00bbff", "#ffff00", "#ff40ff", "#ffa020")
    crops = {c.crop_id: c for c in plan.crops}
    for i, p in enumerate(plan.prompts):
        dx, dy = crops[p.crop_id].xyxy[:2]
        color = colors[i % len(colors)]
        if p.box:
            x0, y0, x1, y1 = p.box
            draw.rectangle((x0+dx, y0+dy, x1+dx, y1+dy), outline=color, width=1)
        for x, y in p.positive:
            x, y = x+dx, y+dy
            draw.ellipse((x-4, y-4, x+4, y+4), fill=color, outline="black")
            draw.text((x+6, y-10), str(i+1), fill=color, stroke_width=1, stroke_fill="black")
        for x, y in p.negative:
            x, y = x+dx, y+dy
            draw.line((x-4, y-4, x+4, y+4), fill=color, width=2)
            draw.line((x-4, y+4, x+4, y-4), fill=color, width=2)
    return image


def prepare(args):
    from .planner import PlannerConfig, plan_prompts as plan_v1
    from .planner_v2 import V2Config, plan_prompts as plan_v2
    from .prompt_ablation import derive_plan, summarize_plan, prompt_input_fingerprint
    from .cli import sources
    out = new_directory(args.output)
    manifest = read_json(args.manifest)
    for record in manifest["records"]:
        for field in ("image", "masks"):
            record[field] = str((args.manifest.parent / record[field]).resolve())
    cases = []
    for domain in ("c135", "wire300"):
        eligible = [r for r in manifest["records"] if r["dataset"] == domain and r["split"] == "validation"
                    and max(r["shape"][:2]) <= 1008]
        chosen = sorted(eligible, key=lambda r: r["id"])[:2]
        if len(chosen) != 2:
            raise ValueError(f"Expected two native single-tile validation cases in {domain}")
        cases.extend(chosen)
    specs = profiles()
    queue, evaluation = [], []
    source_hashes = sources()
    for name in source_hashes:
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    protocol = {"schema": "prompt_study_v1", "status": "freezing", "profiles": specs,
                "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
                "checkpoint_sha256": sha256(args.checkpoint),
                "evaluation_source_sha256": {name: sha256(ROOT/"tools"/name) for name in
                    ("run_evidence_fusion_experiment.py", "evaluate_full_comparison.py")},
                "case_ids": [r["id"] for r in cases], "source_sha256": source_hashes,
                "selection": "First two lexicographic validation IDs per domain with native max(H,W)<=1008; no GT-count filter",
                "limits": {"max_crops": 1, "max_groups": 6, "refiner_rounds": 0, "max_run_seconds": 1800},
                "residency": "experimental_interactive_only", "sam_feedback": False,
                "oracle_warning": "GT-assisted diagnostic; unequal groups/points, not an accuracy upper bound or automatic method",
                "validation_scope": "4 historical validation images; no blind-test/generalization/adoption claim",
                "timing_scope": "Concurrent unrelated GPU job; shared model residency; no comparable end-to-end speedup"}
    write_json(out / "protocol.json", protocol)

    def freeze(case, spec, rgb, plan, planning_seconds):
        dest = new_directory(out / "plans" / case["id"] / spec["id"])
        write_json(dest / "plan.json", plan.metadata())
        np.save(dest / "ridge.npy", plan.ridge, allow_pickle=False)
        prompt_view(rgb, plan).save(dest / "prompts.png")
        queue.append({"case": case["id"], "domain": case["dataset"], "profile": spec["id"],
                      "kind": spec["kind"], "image": str(Path(case["image"]).resolve()),
                      "rgb_sha256": rgb_hash(rgb), "plan": str((dest/"plan.json").resolve()),
                      "plan_sha256": sha256(dest/"plan.json"), "ridge": str((dest/"ridge.npy").resolve()),
                      "ridge_sha256": sha256(dest/"ridge.npy"), "planning_seconds": planning_seconds,
                      "actual_input_fingerprint": prompt_input_fingerprint(plan),
                      "prompt_summary": summarize_plan(plan)})

    # All automatic planning precedes even loading the oracle module or labels.
    for case in cases:
        rgb = np.asarray(Image.open(case["image"]).convert("RGB")).copy()
        if hashlib.sha256(rgb.tobytes()).hexdigest() != case["rgb_sha256"]:
            raise ValueError("Historical manifest RGB hash mismatch")
        began = time.perf_counter()
        legacy = plan_v1(rgb, PlannerConfig())
        freeze(case, specs[0], rgb, legacy, time.perf_counter()-began)
        began = time.perf_counter()
        base = plan_v2(rgb, V2Config())
        elapsed = time.perf_counter()-began
        for spec in specs:
            if spec["planner"] == "v2":
                plan = derive_plan(base, spec["pos"], spec["neg"], spec["negative"], spec["box"])
                freeze(case, spec, rgb, plan.validate(rgb), elapsed)
        print(f"Frozen RGB-only plans: {case['id']}", flush=True)
    from .oracle_diagnostic import oracle_plan
    for case in cases:
        rgb = np.asarray(Image.open(case["image"]).convert("RGB")).copy()
        with np.load(case["masks"], allow_pickle=False) as data:
            gt = data["masks"].astype(bool)
        for spec in specs:
            if spec["planner"] == "oracle":
                began = time.perf_counter()
                plan = oracle_plan(rgb, gt, spec["pos"], spec["neg"], spec["negative"], spec["box"])
                freeze(case, spec, rgb, plan, time.perf_counter()-began)
        name = case["id"].removeprefix(case["dataset"]+"_")
        old = ROOT / "outputs/accuracy_optimization_20261012/selection_v2" / case["dataset"] / "coverage_color" / name
        old_visible = ROOT / "outputs/full_comparison_20261009" / case["dataset"] / "ours_hq" / name / "mask.png"
        evaluation.append({**case, "gt_count": len(gt), "gt_sha256": sha256(case["masks"]),
                           "old_ids": str(old/"instance_ids.png"), "old_ids_sha256": sha256(old/"instance_ids.png"),
                           "old_visible": str(old_visible), "old_visible_sha256": sha256(old_visible)})
    write_json(out / "inference_queue.json", queue)
    write_json(out / "evaluation_inputs.json", evaluation)
    protocol.update(status="frozen_before_sam", queue_sha256=sha256(out/"inference_queue.json"),
                    evaluation_inputs_sha256=sha256(out/"evaluation_inputs.json"), experiments=len(queue))
    protocol["identical_actual_inputs"] = [
        {"case": c["id"], "profiles": [q["profile"] for q in queue if q["case"] == c["id"] and q["actual_input_fingerprint"] == fingerprint]}
        for c in cases for fingerprint in sorted({q["actual_input_fingerprint"] for q in queue if q["case"] == c["id"]})
        if sum(q["case"] == c["id"] and q["actual_input_fingerprint"] == fingerprint for q in queue) > 1]
    write_json(out / "protocol.json", protocol)
    print(f"Frozen {len(queue)} independent configurations in {out}", flush=True)


def run(args):
    """Read only RGB, frozen plans and protocol; no GT paths/arrays are opened."""
    from .cli import export_prediction, sources
    from .sam_adapter import NativeSamAdapter
    from .prompt_ablation import summarize_run
    out = args.output
    protocol = read_json(out/"protocol.json")
    if protocol["source_sha256"] != sources() or protocol["queue_sha256"] != sha256(out/"inference_queue.json"):
        raise ValueError("Source or queue changed after freeze; prepare a new experiment")
    if protocol["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Checkpoint changed after freeze")
    predictions = new_directory(out/"predictions")
    queue = read_json(out/"inference_queue.json")
    adapter = NativeSamAdapter(checkpoint=args.checkpoint, repo=args.sam_repo, device=args.device,
                              max_crops=1, max_prompts=6, interactive_only_residency=True)
    started = time.perf_counter()
    completed = []
    for index, item in enumerate(queue):
        if time.perf_counter()-started > protocol["limits"]["max_run_seconds"]:
            write_json(out/"run_status.json", {"status": "time_limit_no_retry", "completed": completed, "unattempted": queue[index:]})
            return
        dest = new_directory(predictions/item["case"]/item["profile"])
        began = time.perf_counter()
        oracle = item["kind"] == "oracle_diagnostic"
        audit = {"status": "started", "ground_truth_used_for_inference": oracle,
                 "diagnostic_only": oracle, "deployment_allowed": not oracle, "item": item,
                 "refinement": {"rounds": 0}, "model_resident_from_prior_config": index > 0}
        write_json(dest/"inference.json", audit)
        try:
            rgb = np.asarray(Image.open(item["image"]).convert("RGB")).copy()
            if (rgb_hash(rgb) != item["rgb_sha256"] or sha256(item["plan"]) != item["plan_sha256"]
                    or sha256(item["ridge"]) != item["ridge_sha256"]):
                raise ValueError("Frozen inputs changed")
            plan = read_plan(item["plan"], item["ridge"], rgb)
            records, audit["sam"] = adapter.predict(rgb, plan)
            audit["diagnostics"] = summarize_run(plan, records, audit["sam"])
            audit["diagnostics"]["ground_truth_used"] = oracle
            if not audit["diagnostics"]["validation"]["passed"]:
                raise ValueError("Single-call or unchanged-logit contract failed")
            if oracle:
                # Deliberately not a sam_once_cache_v1; the training loader rejects it.
                np.savez_compressed(dest/"diagnostic_proposals.npz",
                                    **{f"logits_{i}": r.initial_logits for i, r in enumerate(records)})
                write_json(dest/"diagnostic_proposals.json", {"schema": "gt_assisted_diagnostic_only",
                           "ground_truth_used_for_inference": True, "deployment_allowed": False,
                           "scores": [r.score for r in records], "plan": plan.metadata()})
            else:
                save_case(dest/"sam_cache", rgb, plan, records, {**audit, "cache_kind": "actual_single_batch_sam"})
            export_prediction(dest, rgb, records, plan, audit)
            audit.update(status=audit["backend"]["status"], elapsed_seconds=time.perf_counter()-began)
            write_json(dest/"inference.json", audit)
            completed.append({"case": item["case"], "profile": item["profile"], "status": audit["status"]})
            print(f"[{index+1}/{len(queue)}] {item['case']} {item['profile']} {audit['status']} {audit['elapsed_seconds']:.2f}s", flush=True)
            write_json(out/"run_status.json", {"status": "running", "completed": completed, "remaining": len(queue)-index-1})
        except Exception as exc:
            audit.update(status="failed_no_retry", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                         sam_failure=getattr(exc, "audit", None), elapsed_seconds=time.perf_counter()-began)
            write_json(dest/"inference.json", audit)
            write_json(out/"run_status.json", {"status": "stopped_on_failure_no_retry", "completed": completed,
                       "failed": item, "unattempted": queue[index+1:]})
            raise
    write_json(out/"run_status.json", {"status": "complete", "completed": completed,
               "elapsed_seconds": time.perf_counter()-started, "configurations": len(queue)})


def metrics(ids, visible, gt):
    from run_evidence_fusion_experiment import instance_scores
    from evaluate_full_comparison import binary
    union = np.logical_or.reduce(gt) if len(gt) else np.zeros(ids.shape, bool)
    scores = instance_scores(ids, gt)
    denominator = scores["tp_instances"]+.5*(scores["fp_instances"]+scores["fn_instances"])
    values = np.unique(ids[ids > 0])
    # Descriptive proxies, not official merge/split event annotations.
    overlap = np.array([[np.count_nonzero((ids == v) & g) for v in values] for g in gt], dtype=float)
    if overlap.size:
        significant = overlap >= np.maximum(16, .1*np.asarray([g.sum() for g in gt])[:, None])
        merges = int(np.count_nonzero(significant.sum(0) >= 2))
        fragments = int(np.maximum(significant.sum(1)-1, 0).sum())
    else:
        merges = fragments = 0
    return {**scores, **binary(visible, union), "pq": scores["matched_iou_sum"]/max(denominator, 1),
            "merge_proxy": merges, "extra_fragment_proxy": fragments,
            "unknown_visible_pixels": int(np.count_nonzero(visible & (ids == 0))),
            "gt_visible_coverage": [float(np.count_nonzero(visible & g)/max(g.sum(), 1)) for g in gt]}


def aggregate(rows):
    sums = {k: sum(r[k] for r in rows) for k in ("tp_instances", "fp_instances", "fn_instances", "matched_iou_sum",
                                               "gt_count", "pred_count", "merge_proxy", "extra_fragment_proxy")}
    denominator = sums["tp_instances"]+.5*(sums["fp_instances"]+sums["fn_instances"])
    return {"images": len(rows), **sums, "pq": sums["matched_iou_sum"]/max(denominator, 1),
            **{k: float(np.mean([r[k] for r in rows])) for k in ("instance_iou", "foreground_iou", "recall")}}


def evaluate(args):
    from .cli import sources
    out = args.output
    protocol = read_json(out/"protocol.json")
    if protocol["source_sha256"] != sources() or protocol["queue_sha256"] != sha256(out/"inference_queue.json"):
        raise ValueError("Source or inference queue changed after freeze")
    for name, digest in protocol.get("evaluation_source_sha256", {}).items():
        if sha256(ROOT/"tools"/name) != digest:
            raise ValueError("Evaluation source changed after freeze")
    if not (out/"run_status.json").exists() or read_json(out/"run_status.json")["status"] == "running":
        raise ValueError("Finish or explicitly audit the terminated inference phase before evaluation")
    if protocol["evaluation_inputs_sha256"] != sha256(out/"evaluation_inputs.json"):
        raise ValueError("Evaluation inputs changed after freeze")
    rows = []
    cases = read_json(out/"evaluation_inputs.json")
    queue = read_json(out/"inference_queue.json")
    for case in cases:
        if sha256(case["masks"]) != case["gt_sha256"]:
            raise ValueError("GT changed after freeze")
        with np.load(case["masks"], allow_pickle=False) as data:
            gt = data["masks"].astype(bool)
        for item in [q for q in queue if q["case"] == case["id"]]+[
                {"case": case["id"], "domain": case["dataset"], "profile": "old_coverage_color", "kind": "historical_baseline"}]:
            old = item["kind"] == "historical_baseline"
            dest = out/"predictions"/case["id"]/item["profile"]
            if old:
                if sha256(case["old_ids"]) != case["old_ids_sha256"] or sha256(case["old_visible"]) != case["old_visible_sha256"]:
                    raise ValueError("Historical baseline changed")
                ids = np.asarray(Image.open(case["old_ids"])).copy()
                visible = np.asarray(Image.open(case["old_visible"])) > 0
                status = "historical_saved_prediction"
            else:
                audit = read_json(dest/"inference.json") if (dest/"inference.json").exists() else {"status": "unattempted"}
                status = audit["status"]
                if status in ("ok", "empty", "geometry_failed") and (dest/"instance_ids.png").exists():
                    ids = np.asarray(Image.open(dest/"instance_ids.png")).copy()
                    visible = np.asarray(Image.open(dest/"visible_foreground.png")) > 0
                else:
                    ids = np.zeros(gt.shape[1:], np.uint16)
                    visible = ids > 0
            if ids.shape != gt.shape[1:] or visible.shape != ids.shape:
                raise ValueError("Native coordinate mismatch")
            row = {"case": case["id"], "domain": case["dataset"], "profile": item["profile"],
                   "kind": item["kind"], "status": status, **metrics(ids, visible, gt)}
            if not old:
                row.update(prompt_summary=item["prompt_summary"], sam=audit.get("sam"),
                           inference_seconds=audit.get("elapsed_seconds"))
            rows.append(row)
    tables = {}
    for profile in [p["id"] for p in protocol["profiles"]]+["old_coverage_color"]:
        tables[profile] = {d: aggregate([r for r in rows if r["profile"] == profile and r["domain"] == d])
                           for d in ("c135", "wire300")}
    candidates = [p["id"] for p in protocol["profiles"] if p["planner"] == "v2"]
    def key(profile):
        return (sum(tables[profile][d]["pq"] for d in ("c135", "wire300")),
                sum(tables[profile][d]["instance_iou"] for d in ("c135", "wire300")))
    ranked = sorted(candidates, key=key, reverse=True)
    write_json(out/"results.json", {"rows": rows, "per_profile_domain": tables,
               "validation_ranking_equal_domain_pq_then_iou": ranked,
               "promotion": False, "scope": protocol["validation_scope"],
               "merge_fragment_proxy_definition": "Each predicted-ID/GT overlap >= max(16px,10% GT area); many GT per ID=merge; extra IDs per GT=fragments",
               "missing_policy": "Failed/unattempted configurations retained as empty predictions; not evidence of measured model accuracy"})
    print({p: tables[p] for p in ("v1_control", ranked[0], "old_coverage_color")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"outputs/sacm_autotune_20261014/real_adaptation/manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path("D:/sam31/weights/sam3.1_multiplex_fp16.pt"))
    parser.add_argument("--sam-repo", type=Path, default=Path("D:/sam31/repo"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
