"""Software tests and optional REAL SAM native-size gradient admission check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import unittest

import torch

from .model import ModelConfig, SamPathModel, VARIANT_SPECS, variant_loss_metadata
from .losses import SetCriterion
from .train import backward_losses


def real_check(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Real SAM admission requires CUDA for this pilot")
    start = time.perf_counter()
    from .checkpoint import seed_everything
    seed_everything(42)
    model = SamPathModel(ModelConfig(variant=args.variant, backend=args.backend, sam_repo=args.sam_repo,
                         sam_checkpoint=args.sam_checkpoint, canvas_size=args.canvas_size,
                         sam2_repo=args.sam2_repo, sam2_checkpoint=args.sam2_checkpoint,
                         num_queries=args.queries, tracker_steps=args.steps)).to("cuda")
    model.train()
    side = args.canvas_size
    # Synthetic RGB/GT is for tensor/gradient admission only, never accuracy.
    generator = torch.Generator().manual_seed(17)
    image = torch.rand((1, 3, side, side), generator=generator).to("cuda")
    valid = torch.ones(1, 1, side, side, dtype=torch.bool, device="cuda")
    target = torch.zeros(1, side, side, device="cuda")
    target[:, side//3:side//3+max(2, side//80), 10:-10] = 1
    torch.cuda.reset_peak_memory_stats()
    outputs = model(image, valid)
    criterion = SetCriterion()
    losses = criterion(outputs, [{"masks": target, "id": "SOFTWARE_SMOKE"}], valid)
    backward_losses(losses, outputs, criterion, torch.amp.GradScaler("cuda", enabled=False),
                    variant_loss_metadata(args.variant)["gradient_policy"])
    grads = {}
    for name in ("condition", "tracker", "descriptor"):
        module = getattr(model, name, None)
        if module is None:
            continue
        parameters = list(module.parameters())
        grads[name] = {"nonzero": any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters),
                       "finite": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)}
    frozen_clean = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if not grads:
        parameters = [p for p in model.parameters() if p.requires_grad]
        grads["trainable_native"] = {"nonzero": any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters),
                                     "finite": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)}
    if not frozen_clean or not all(x["finite"] and x["nonzero"] for x in grads.values()):
        raise RuntimeError(f"Real SAM gradient admission failed: {grads}, frozen={frozen_clean}")
    torch.cuda.synchronize()
    result = {"scope": "real pretrained SAM tensor/gradient smoke ONLY; not trained accuracy",
              "counts": model.parameter_counts(), "calls": outputs["calls"],
              "output_shape": list(outputs["mask_logits"].shape),
              "loss": {k: float(v.detach()) for k, v in losses.items()},
              "gradient_checks": grads, "frozen_clean": frozen_clean,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "seconds": time.perf_counter()-start, "base_receipt": model.base.receipt}
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "base_receipt"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-sam", action="store_true")
    parser.add_argument("--variant", choices=tuple(VARIANT_SPECS), default="C")
    parser.add_argument("--backend", choices=("sam31", "sam21", "sam31_native"), default="sam31")
    parser.add_argument("--sam-repo", default="")
    parser.add_argument("--sam-checkpoint", default="")
    parser.add_argument("--sam2-repo", default="")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--canvas-size", type=int, default=1036)
    parser.add_argument("--queries", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.real_sam:
        real_check(args)
    else:
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
