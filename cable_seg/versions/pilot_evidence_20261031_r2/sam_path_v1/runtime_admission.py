"""Fresh checkpoint reload and RGB-only runtime admission, never a test score."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .benchmark import benchmark_forward, profile_modules, rgb_batch, runtime_precision
from .checkpoint import atomic_json, build_from_checkpoint, sha256
from .data import read_manifest, resolved_record_path


def verify_loaded_state(model, payload):
    saved, loaded = payload["model_state"], model.checkpoint_state_dict()
    if set(saved) != set(loaded):
        raise ValueError("Reloaded checkpoint tensor keys changed")
    for name, tensor in saved.items():
        current = loaded[name].detach().cpu()
        if tensor.shape != current.shape or not torch.isfinite(current).all() or not torch.equal(tensor.cpu(), current):
            raise ValueError("Reloaded checkpoint tensor mismatch/nonfinite: "+name)
    native_b = {}
    if payload["model_config"]["variant"] == "S":
        native_b = {name: int(torch.count_nonzero(tensor)) for name,tensor in loaded.items()
                    if name.endswith((".q_b", ".v_b"))}
        if len(native_b) != 16 or not all(native_b.values()):
            raise ValueError("Native S must have 16 finite nonzero trained LoRA B matrices")
    return dict(tensors=len(loaded), exact_shape_and_value_reload=True, native_lora_b_nonzero_elements=native_b)


@torch.no_grad()
def runtime_admission(checkpoint, manifest, data_root, *, device="cuda"):
    # Building here (in a fresh subprocess) exercises the actual deployment path.
    model, payload = build_from_checkpoint(checkpoint, device=device)
    torch.set_num_threads(payload["train_config"].get("cpu_threads", 1))
    state = verify_loaded_state(model, payload)
    records = read_manifest(manifest, allow_condition=True)["records"]
    row = next((r for r in records if r["split"] == "train"), None)
    if row is None: raise ValueError("Runtime admission requires an original pilot train RGB")
    image_path = resolved_record_path(data_root, row["image"])
    config = payload["model_config"]
    rgb, valid, size = rgb_batch(image_path, config["canvas_size"], device)
    output = benchmark_forward(model, rgb, valid)
    q, chunk = config["num_queries"], config.get("decoder_query_chunk", 0)
    calls = dict(encoder=1, decoder=math.ceil(q/chunk) if chunk and config["variant"] != "S" else 1)
    if output.get("calls") != calls:
        raise ValueError("Reloaded model forward call contract failed")
    shapes = {}
    for key in ("mask_logits", "object_logits", "quality_logits"):
        tensor = output.get(key)
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("Reloaded model output missing/nonfinite: "+key)
        if (key == "mask_logits" and (tensor.ndim != 4 or tuple(tensor.shape[:2]) != (1,q) or min(tensor.shape[-2:]) < 1)
                or key != "mask_logits" and tuple(tensor.shape) != (1,q)):
            raise ValueError("Reloaded model output shape invalid: "+key)
        shapes[key] = list(tensor.shape)
    del output
    sync = (lambda: torch.cuda.synchronize(device)) if torch.device(device).type == "cuda" else (lambda: None)
    timings = profile_modules(model, rgb, valid, sync)
    required = {"encoder", "decoder"}
    variant = config["variant"]
    if variant != "S": required.add("condition")
    if variant not in ("A", "S"): required.add("tracker")
    if variant not in ("A", "B", "S"): required.add("descriptor")
    if not required.issubset(timings) or any(not math.isfinite(v) or v < 0 for v in timings.values()):
        raise ValueError("Runtime profile hooks missing/nonfinite: "+str(sorted(required-set(timings))))
    if model.last_calls != calls:
        raise ValueError("Profiled model forward call contract failed")
    return dict(complete=True, checkpoint_sha256=sha256(checkpoint), manifest_sha256=sha256(manifest),
        image_id=row["id"], image_sha256=sha256(image_path), split="train", original_size=list(size),
        model_config=config, reload=state, output_shapes=shapes, calls_per_forward=calls,
        model_forwards=2, required_profile_modules=sorted(required), modules_seconds=timings,
        runtime=runtime_precision(model, device), gt_loaded=False, gt_scored=False,
        interpretation="Admission only: two RGB-only forwards after a fresh reload; hook timings are not paper efficiency results.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "data-root", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    try:
        receipt = runtime_admission(args.checkpoint, args.manifest, args.data_root, device=args.device)
    except Exception as exc:
        atomic_json(out, dict(complete=False, error=f"{type(exc).__name__}: {exc}", gt_scored=False))
        raise
    atomic_json(out, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
