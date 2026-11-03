"""RGB-only synchronized inference timing; no ground-truth scoring in timers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .checkpoint import atomic_json, build_from_checkpoint, new_directory, sha256
from .data import read_manifest, resolved_record_path
from .inference import decode_instances

CAPACITY_QUERIES = (8, 16, 24, 48, 80)
CAPACITY_REPETITIONS = 10


def failure_kind(exc):
    return "oom" if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower() else "exception"


def _profile_targets(model):
    base = getattr(model, "base", None)
    return (("encoder", getattr(base, "encoder_module", getattr(base, "trunk", getattr(base, "encoder", None)))),
            ("condition", getattr(model, "condition", None)),
            ("descriptor", getattr(model, "descriptor", None)),
            ("tracker", getattr(model, "tracker", None)),
            ("decoder", getattr(base, "decoder", None)))


def benchmark_forward(model, rgb, valid):
    # Encoder/native adapters may explicitly enable their own configured BF16
    # regions. No inherited training/global autocast is applied to other heads.
    with torch.autocast(rgb.device.type, enabled=False):
        return model(rgb, valid)


def runtime_precision(model, device):
    dtype_counts = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0)+parameter.numel()
    config = model.config
    native = getattr(config, "variant", None) == "S"
    encoder_amp = getattr(config, "encoder_amp", getattr(getattr(model, "base", None), "encoder_amp", None))
    return dict(outer_autocast_disabled=True, rgb_dtype="torch.float32", parameter_dtype_numel=dtype_counts,
        backend=getattr(config, "backend", "software_fixture"), encoder_amp_config=encoder_amp,
        internal_bf16_enabled=torch.device(device).type == "cuda" and encoder_amp == "bf16",
        internal_autocast_scope="native visual/text/fusion/detector forward" if native else "frozen visual encoder only",
        precision_policy="All main, module-profile and capacity model forwards disable outer autocast; adapters retain explicit internal precision. Training AMP is not the benchmark precision.")


def capacity_diagnostic_problems(scan):
    """Check every prespecified Q, retaining failed/OOM points in the report."""
    problems = []
    if not scan.get("available"):
        return ["capacity scan unavailable"]
    rows = scan.get("results", [])
    if sorted(r.get("queries", -1) for r in rows) != list(CAPACITY_QUERIES):
        problems.append("capacity points missing, duplicated or unexpected")
    for row in rows:
        stats = row.get("forward_seconds") or {}
        if row.get("error"):
            problems.append(f"capacity Q={row.get('queries')}: {row['error']}")
        elif (stats.get("n") != CAPACITY_REPETITIONS
              or any(not isinstance(stats.get(k), (float, int)) or not math.isfinite(stats[k]) or stats[k] < 0
                     for k in ("mean", "median", "p95", "maximum"))
              or row.get("trained_at_this_capacity") is not False or row.get("accuracy_evaluated") is not False):
            problems.append(f"capacity Q={row.get('queries')}: incomplete/nonfinite cost evidence")
    return problems


def benchmark_diagnostic_problems(report, variant=None):
    """Independently audit coverage, rather than trusting a top-level success flag."""
    problems = []
    profiles = report.get("module_profiles", [])
    expected_ids = [r.get("id") for r in report.get("per_observation", []) if r.get("repeat") == 0][:10]
    profile_ids = [r.get("id") for r in profiles]
    if (len(expected_ids) != min(10, report.get("images", 0)) or len(set(expected_ids)) != len(expected_ids)
            or profile_ids != expected_ids):
        problems.append("module profile image coverage missing, duplicated or inconsistent")
    required = set(report.get("module_profile_expected_modules", []))
    if variant:
        required.update(("encoder", "decoder"))
        if variant != "S": required.add("condition")
        if variant not in ("A", "S"): required.add("tracker")
        if variant not in ("A", "B", "S"): required.add("descriptor")
    for row in profiles:
        stages = row.get("modules_seconds", {})
        if row.get("error"):
            problems.append(f"module profile {row.get('id')}: {row['error']}")
        elif (not required.issubset(stages) or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in stages.values())):
            problems.append(f"module profile {row.get('id')}: missing/nonfinite module timings")
    if variant == "C" or "query_capacity_cost" in report:
        problems.extend(capacity_diagnostic_problems(report.get("query_capacity_cost", {})))
    return problems


def finalize_benchmark(report, variant=None):
    problems = benchmark_diagnostic_problems(report, variant)
    report.update(diagnostic_failures=len(problems), diagnostic_problems=problems,
                  diagnostics_complete=not problems)
    report["timing_complete"] = (report.get("observations") == report.get("images", 0)*report.get("repetitions", 0)
        and not report.get("failures") and not any(r.get("error") for r in report.get("per_observation", [])))
    report["complete"] = report["timing_complete"] and report["diagnostics_complete"]
    return report


def rgb_batch(path, canvas, device):
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), copy=True)
    h, w = rgb.shape[:2]
    if max(h, w) > canvas:
        raise ValueError("Native benchmark does not resize oversized images")
    tensor = torch.zeros(1, 3, canvas, canvas, dtype=torch.float32)
    tensor[0, :, :h, :w] = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
    valid = torch.zeros(1, 1, canvas, canvas, dtype=torch.bool)
    valid[..., :h, :w] = True
    return tensor.to(device), valid.to(device), (h, w)


def summary(values):
    a = np.asarray(values, dtype=float)
    return None if not len(a) else dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                                        p95=float(np.percentile(a, 95)), maximum=float(a.max()))


@torch.no_grad()
def profile_modules(model, rgb, valid, sync):
    """A separate instrumented pass, excluded from the end-to-end distribution."""
    stages, starts, handles = {}, {}, []
    for label, module in _profile_targets(model):
        if not isinstance(module, torch.nn.Module):
            continue
        def before(module, inputs, label=label):
            sync(); starts[label] = time.perf_counter()
        def after(module, inputs, outputs, label=label):
            sync(); stages[label] = stages.get(label, 0.)+time.perf_counter()-starts[label]
        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
    try:
        benchmark_forward(model, rgb, valid); sync()
    finally:
        for handle in handles:
            handle.remove()
    return stages


@torch.no_grad()
def run_benchmark(model, records, data_root, *, device="cuda", warmup=30, repetitions=3):
    if not records or warmup < 1 or repetitions < 1:
        raise ValueError("Nonempty records and positive warmup/repetitions required")
    model.eval()
    cuda = torch.device(device).type == "cuda"
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    canvas = model.config.canvas_size
    for i in range(warmup):
        row = records[i % len(records)]
        rgb, valid, _ = rgb_batch(resolved_record_path(data_root, row["image"]), canvas, device)
        benchmark_forward(model, rgb, valid)
    sync()
    rows = []
    for repeat in range(repetitions):
        for row in records:
            sync()
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            failure = None
            try:
                rgb, valid, size = rgb_batch(resolved_record_path(data_root, row["image"]), canvas, device)
                sync(); ready = time.perf_counter()
                outputs = benchmark_forward(model, rgb, valid)
                sync(); forwarded = time.perf_counter()
                prediction = decode_instances(outputs, (canvas, canvas), valid)[0]
                sync(); ended = time.perf_counter()
                record = dict(id=row["id"], repeat=repeat, original_size=list(size),
                              end_to_end_seconds=ended-start, preprocess_h2d_seconds=ready-start,
                              forward_seconds=forwarded-ready, postprocess_d2h_seconds=ended-forwarded,
                              n_pred=len(prediction["masks"]),
                              calls=outputs.get("calls"), encoded_views=1, processed_pixels=canvas*canvas)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                record = dict(id=row["id"], repeat=repeat, error=failure, failure_kind=failure_kind(exc))
            if cuda:
                record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            record["error"] = failure
            rows.append(record)
    profiled = []
    for row in records[:10]:
        try:
            rgb, valid, _ = rgb_batch(resolved_record_path(data_root, row["image"]), canvas, device)
            profiled.append(dict(id=row["id"], modules_seconds=profile_modules(model, rgb, valid, sync)))
        except Exception as exc:
            profiled.append(dict(id=row["id"], error=f"{type(exc).__name__}: {exc}", failure_kind=failure_kind(exc)))
    ok = [r for r in rows if r["error"] is None]
    return finalize_benchmark(dict(images=len(records), repetitions=repetitions, warmup=warmup, observations=len(rows),
                failures=len(rows)-len(ok), complete=len(ok)==len(rows),
                timing_scope="PNG read, native pad/normalize, H2D, synchronized model, exclusive instance outputs/D2H; no GT loading/scoring or disk output",
                module_profiling="Separate extra forward on first ten fixed images; excluded from end-to-end samples",
                module_profiles=profiled,
                module_profile_expected_modules=[name for name, module in _profile_targets(model) if isinstance(module, torch.nn.Module)],
                summaries={k: summary([r[k] for r in ok]) for k in (
                    "end_to_end_seconds", "forward_seconds", "preprocess_h2d_seconds", "postprocess_d2h_seconds")},
                parameters=model.parameter_counts(), device=str(device), runtime=runtime_precision(model, device),
                gpu=torch.cuda.get_device_name(device) if cuda else None,
                per_observation=rows))


@torch.no_grad()
def capacity_cost_scan(model, row, data_root, device):
    """Cost only: repeat trained query tensors, never claim predictions are trained at new Q."""
    condition = getattr(model, "condition", None)
    if condition is None:
        return dict(available=False, reason="native detector retains its official query count")
    original = condition.queries
    original_chunk = getattr(model.config, "decoder_query_chunk", 0)
    cuda = torch.device(device).type == "cuda"
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    results = []
    try:
        for q in CAPACITY_QUERIES:
            record = dict(queries=q, trained_at_this_capacity=False, accuracy_evaluated=False)
            try:
                condition.queries = torch.nn.Parameter(original.detach().repeat((q+len(original)-1)//len(original), 1)[:q].clone(), requires_grad=False)
                rgb, valid, _ = rgb_batch(resolved_record_path(data_root, row["image"]), model.config.canvas_size, device)
                for _ in range(5):
                    benchmark_forward(model, rgb, valid)
                sync()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                seconds = []
                for _ in range(CAPACITY_REPETITIONS):
                    sync(); start=time.perf_counter(); output=benchmark_forward(model,rgb,valid); sync()
                    seconds.append(time.perf_counter()-start)
                record.update(forward_seconds=summary(seconds), calls=output.get("calls"),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["failure_kind"] = failure_kind(exc)
                if cuda:
                    torch.cuda.empty_cache()
            results.append(record)
    finally:
        condition.queries = original
        if hasattr(model.config, "decoder_query_chunk"):
            model.config.decoder_query_chunk = original_chunk
    report = dict(available=True, interpretation="Query capacity cost microbenchmark using repeated query tensors; no segmentation accuracy claims", results=results)
    problems = capacity_diagnostic_problems(report)
    report.update(complete=not problems, diagnostic_failures=len(problems), diagnostic_problems=problems)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--images", type=int, default=100)
    p.add_argument("--warmup", type=int, default=30); p.add_argument("--repetitions", type=int, default=3)
    args = p.parse_args(argv)
    out = new_directory(args.output)
    manifest = read_manifest(args.manifest, allow_condition=True)
    records = sorted((r for r in manifest["records"] if r["split"] == "test"),
                     key=lambda r: hashlib.sha256(("efficiency_v1|"+r["rgb_sha256"]).encode()).hexdigest())[:args.images]
    if len(records) != args.images:
        raise ValueError("Insufficient test images for frozen benchmark count")
    model, payload = build_from_checkpoint(args.checkpoint, device=args.device)
    torch.set_num_threads(payload["train_config"].get("cpu_threads", 1))
    report = run_benchmark(model, records, args.data_root, device=args.device,
                           warmup=args.warmup, repetitions=args.repetitions)
    if payload["model_config"]["variant"] == "C":
        report["query_capacity_cost"] = capacity_cost_scan(model, records[0], args.data_root, args.device)
    report.update(checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest),
                  model_config=payload["model_config"])
    finalize_benchmark(report, payload["model_config"]["variant"])
    atomic_json(out / "benchmark.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="per_observation"}))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
