"""Inference commands do not import supervision or evaluation code."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys
import time
import traceback
import numpy as np
from PIL import Image
from .runtime import ROOT, bootstrap
from .io import load_case, new_directory, read_json, rgb_hash, save_case, sha256, write_json
bootstrap()


def sources():
    paths = list((ROOT / "sam_once").glob("*.py")) + [ROOT / "tools" / name for name in
            ("cable_evidence_fusion.py", "cable_identity_selector.py", "mbest_backend.py", "refine_observed_ids.py")]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def export_prediction(out, rgb, records, plan, audit):
    from .backend import assign_instances
    ids, visible, unknown, backend_audit = assign_instances(rgb, records, len(plan.crops))
    Image.fromarray(ids).save(out / "instance_ids.png")
    Image.fromarray(visible.astype(np.uint8) * 255).save(out / "visible_foreground.png")
    Image.fromarray(unknown.astype(np.uint8) * 255).save(out / "unknown_visible.png")
    from cable_identity_graph import render
    render(rgb, ids, out / "instances_overlay.png")
    audit["backend"] = backend_audit
    return audit


def make_plan(rgb, args):
    if getattr(args, "planner", "v1") == "v2":
        from .planner_v2 import V2Config, plan_prompts
        from .prompt_ablation import derive_plan
        base = plan_prompts(rgb, V2Config(selection=args.selection,
                            max_prompts_per_crop=args.prompts_per_crop))
        return derive_plan(base, args.pos_count, args.neg_count, args.negative_mode, args.box_mode).validate(rgb)
    from .planner import PlannerConfig, plan_prompts
    return plan_prompts(rgb, PlannerConfig(selection=args.selection))


def infer(args):
    coarse_only = getattr(args, "coarse_only", False)
    if coarse_only and args.refiner:
        raise ValueError("--coarse-only exports the initial SAM evidence; omit --refiner")
    from .sam_adapter import NativeSamAdapter
    started = time.perf_counter()
    out = new_directory(args.output)
    audit = {"schema": "sam_once_inference_v1", "status": "started",
             "ground_truth_used_for_inference": False, "source_sha256": sources(),
             "native_resolution": True, "image": str(args.image.resolve()),
             "experimental_status": "research candidate; not promoted", "refinement": {"rounds": 0},
             "output_stage": "coarse_foreground" if coarse_only else "instances"}
    try:
        rgb = np.asarray(Image.open(args.image).convert("RGB")).copy()
        audit["rgb_sha256"] = rgb_hash(rgb)
        began = time.perf_counter()
        plan = make_plan(rgb, args)
        audit["planning_seconds"] = time.perf_counter() - began
        write_json(out / "prompt_plan.json", plan.metadata())
        began = time.perf_counter()
        adapter = NativeSamAdapter(checkpoint=args.checkpoint, repo=args.sam_repo, device=args.device,
                                   max_crops=args.max_crops, max_prompts=args.max_prompts,
                                   interactive_only_residency=getattr(args, "interactive_only_residency", False))
        audit["adapter_construction_seconds"] = time.perf_counter() - began
        proposals, audit["sam"] = adapter.predict(rgb, plan)
        del adapter
        # Release the large frozen model before the optional lightweight refiner.
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_case(out / "sam_cache", rgb, plan, proposals, {**audit, "cache_kind": "actual_single_batch_sam"})
        if args.refiner:
            from .refiner import RefinerConfig, load_refiner, refine_proposals
            model, checkpoint_audit = load_refiner(args.refiner, args.device)
            if checkpoint_audit["smoke_only"] and not args.allow_smoke_refiner:
                raise ValueError("Smoke weights are not trained candidates; use --allow-smoke-refiner for software validation only")
            proposals, audit["refinement"] = refine_proposals(rgb, proposals, plan.ridge, model,
                                                               RefinerConfig(roi_size=args.roi_size, rounds=args.rounds, device=args.device,
                                                                             max_visits=args.max_visits, max_pixels=args.max_pixels, mode=args.refinement_mode))
            audit["refiner_checkpoint"] = checkpoint_audit
            save_case(out / "refined_cache", rgb, plan, proposals, audit)
        if coarse_only:
            from .coarse import export_coarse
            export_coarse(out, rgb, proposals, plan, audit)
            audit["status"] = audit["coarse_stage"]["status"]
        else:
            export_prediction(out, rgb, proposals, plan, audit)
            audit["status"] = audit["backend"]["status"]
        audit["end_to_end_cold_seconds"] = time.perf_counter() - started
        audit["timing_scope"] = ("includes model load, native planning, SAM and coarse export; no identity backend"
                                 if coarse_only else "includes model load, native planning, SAM, optional refinement, backend and export")
        write_json(out / "inference.json", audit)
        print(json.dumps({"status": audit["status"], "output": str(out), "seconds": audit["end_to_end_cold_seconds"]}, ensure_ascii=False))
    except Exception as exc:
        if hasattr(exc, "audit"):
            audit["sam_failure"] = exc.audit
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                     elapsed_seconds=time.perf_counter() - started)
        write_json(out / "inference.json", audit)
        raise


def export_coarse_cache(args):
    """Reuse immutable initial evidence with zero additional SAM calls."""
    from .coarse import export_coarse
    out = new_directory(args.output)
    audit = {"status": "started", "output_stage": "coarse_foreground", "sam_calls_this_command": 0,
             "source_cache": str(args.cache.resolve()), "ground_truth_used_for_inference": False,
             "source_sha256": sources(), "refinement": {"rounds": 0}}
    try:
        rgb, plan, records, parent = load_case(args.cache)
        audit.update(parent_sam_audit=parent.get("sam"), source_case_sha256=sha256(args.cache/"case.json"),
                     rgb_sha256=rgb_hash(rgb))
        export_coarse(out, rgb, records, plan, audit)
        audit["status"] = audit["coarse_stage"]["status"]
        write_json(out/"inference.json", audit)
        print(json.dumps({"status": audit["status"], "sam_calls_this_command": 0, "output": str(out)}, ensure_ascii=False))
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(out/"inference.json", audit)
        raise


def refine_cache(args):
    from .refiner import RefinerConfig, load_refiner, refine_proposals
    out = new_directory(args.output)
    started = time.perf_counter()
    audit = {"schema": "sam_once_inference_v1", "status": "started",
             "source_cache": str(args.cache.resolve()), "sam_calls_this_command": 0,
             "source_sha256": sources(), "ground_truth_used_for_inference": False,
             "timing_scope": "cached refinement only; not full RGB end-to-end"}
    try:
        rgb, plan, records, parent = load_case(args.cache)
        audit.update(parent_sam_audit=parent.get("sam"), rgb_sha256=rgb_hash(rgb))
        model, model_audit = load_refiner(args.refiner, args.device)
        audit["refiner_checkpoint"] = model_audit
        if model_audit["smoke_only"] and not args.allow_smoke_refiner:
            raise ValueError("Smoke weights require --allow-smoke-refiner; no accuracy claim is allowed")
        records, audit["refinement"] = refine_proposals(rgb, records, plan.ridge, model,
            RefinerConfig(roi_size=args.roi_size, rounds=args.rounds, device=args.device,
                          max_visits=args.max_visits, max_pixels=args.max_pixels, mode=args.refinement_mode))
        save_case(out / "refined_cache", rgb, plan, records, audit)
        export_prediction(out, rgb, records, plan, audit)
        audit.update(status=audit["backend"]["status"], cached_refinement_and_backend_seconds=time.perf_counter() - started)
        write_json(out / "inference.json", audit)
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                     elapsed_seconds=time.perf_counter() - started)
        write_json(out / "inference.json", audit)
        raise


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:]
    if argv and argv[0] in ("train", "verify-native", "evaluate", "prepare-manifest", "select-checkpoint", "prompt-study"):
        module = {"train": "train", "verify-native": "verify_native", "evaluate": "evaluate", "prepare-manifest": "prepare", "select-checkpoint": "select_checkpoint", "prompt-study": "prompt_study"}[argv[0]]
        sys.argv = [f"sam_once.{module}", *argv[1:]]
        runpy.run_module(f"sam_once.{module}", run_name="__main__")
        return
    if argv and argv[0] == "test":
        import unittest
        pattern = argv[1] if len(argv) > 1 else "test_*.py"
        suite = unittest.defaultTestLoader.discover(str(ROOT / "sam_once"), pattern=pattern, top_level_dir=str(ROOT))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    parser = argparse.ArgumentParser(description="Single-batch SAM3.1 research pipeline. Commands: plan, predict, export-coarse, refine, train, verify-native, evaluate, prepare-manifest, select-checkpoint, test")
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="RGB-only geometry; no SAM and no GT")
    plan_parser.add_argument("--image", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--selection", choices=["iterative", "fixed"], default="iterative")
    pred = sub.add_parser("predict")
    pred.add_argument("--image", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--sam-repo", type=Path, default=Path("D:/sam31/repo"))
    pred.add_argument("--checkpoint", type=Path, default=Path("D:/sam31/weights/sam3.1_multiplex_fp16.pt"))
    pred.add_argument("--device", default="cuda")
    pred.add_argument("--max-crops", type=int, default=16)
    pred.add_argument("--max-prompts", type=int, default=96)
    pred.add_argument("--interactive-only-residency", action="store_true",
                      help="Experimental: keep only the native interactive branch on GPU; same weights and one batch")
    pred.add_argument("--selection", choices=["iterative", "fixed"], default="iterative")
    pred.add_argument("--refiner", type=Path)
    pred.add_argument("--coarse-only", action="store_true",
                      help="Export raw coarse foreground and independent soft proposals; skip refiner and identity backend")
    pred.add_argument("--allow-smoke-refiner", action="store_true")
    pred.add_argument("--rounds", type=int, choices=[1, 2, 3], default=3)
    for pp in (plan_parser, pred):
        pp.add_argument("--planner", choices=["v1", "v2"], default="v1",
                        help="v1 preserves the original control; v2 is an unpromoted RGB-only experiment")
        pp.add_argument("--prompts-per-crop", type=int, default=6, help="v2 group budget")
        pp.add_argument("--pos-count", type=int, choices=[1, 2, 3], default=3, help="v2 only")
        pp.add_argument("--neg-count", type=int, choices=[0, 1, 2], default=0, help="v2 only")
        pp.add_argument("--negative-mode", choices=["none", "background", "neighbor"], default="none", help="v2 only")
        pp.add_argument("--box-mode", choices=["none", "local", "expanded"], default="none", help="v2 only")
    cached_coarse = sub.add_parser("export-coarse", help="Export an existing initial SAM cache with zero new SAM calls")
    cached_coarse.add_argument("--cache", type=Path, required=True)
    cached_coarse.add_argument("--output", type=Path, required=True)
    ref = sub.add_parser("refine")
    ref.add_argument("--cache", type=Path, required=True)
    ref.add_argument("--refiner", type=Path, required=True)
    ref.add_argument("--allow-smoke-refiner", action="store_true")
    ref.add_argument("--output", type=Path, required=True)
    ref.add_argument("--device", default="cuda")
    ref.add_argument("--rounds", type=int, choices=[1, 2, 3], default=3)
    for pp in (pred, ref):
        pp.add_argument("--roi-size", type=int, choices=[256, 512], default=256)
        pp.add_argument("--max-visits", type=int, default=12)
        pp.add_argument("--max-pixels", type=int, default=786432)
        pp.add_argument("--refinement-mode", choices=["iterative", "fixedsingle"], default="iterative")
    args = parser.parse_args(argv)
    if args.command == "predict":
        infer(args)
    elif args.command == "export-coarse":
        export_coarse_cache(args)
    elif args.command == "refine":
        refine_cache(args)
    else:
        rgb = np.asarray(Image.open(args.image).convert("RGB")).copy()
        plan = make_plan(rgb, args)
        out = new_directory(args.output)
        write_json(out / "plan.json", plan.metadata())
        np.save(out / "ridge.npy", plan.ridge, allow_pickle=False)
        print(json.dumps(plan.audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
