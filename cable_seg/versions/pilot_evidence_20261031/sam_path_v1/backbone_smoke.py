"""Explicit real-asset smoke; no dataset, no optimizer updates, no accuracy claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sam31_native", "sam21"], required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canvas", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    start = time.perf_counter()
    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    if args.backend == "sam31_native":
        from .native_sam import NativeSam31LoRAModel
        config = SimpleNamespace(num_queries=200, sam_repo=args.repo, sam_checkpoint=args.checkpoint,
                                  canvas_size=args.canvas, init_seed=42, encoder_amp="bf16")
        model = NativeSam31LoRAModel(config).to(args.device).train()
        result = model(torch.rand(1, 3, args.canvas, args.canvas, device=args.device))
        loss = result["mask_logits"].square().mean()+result["object_logits"].square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert len(gradients) == 32 and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(bool(g.abs().sum()>0) for g in gradients) == 16  # A gradient starts after first B update
        receipt = dict(calls=result["calls"], mask_shape=list(result["mask_logits"].shape),
                       parameter_counts=model.parameter_counts(), base=model.base.receipt,
                       trainable_tensors_with_finite_grad=len(gradients),
                       nonzero_gradient_tensors=16)
    else:
        from .backbone_sam2 import Sam21Backbone
        model = Sam21Backbone(args.repo, args.checkpoint, args.canvas, "bf16").to(args.device).train()
        features = model.encode(torch.rand(1, 3, args.canvas, args.canvas, device=args.device))
        tokens = torch.randn(1, 2, 3, 256, device=args.device, requires_grad=True)
        dense = torch.randn(1, 2, 256, *features["low"].shape[-2:], device=args.device, requires_grad=True)
        result = model.decode(features, tokens, dense)
        sum(x.square().mean() for x in result).backward()
        assert tokens.grad is not None and tokens.grad.abs().sum()>0
        assert dense.grad is not None and dense.grad.abs().sum()>0
        assert all(p.grad is None for p in model.encoder.parameters())
        receipt = dict(base=model.receipt, low_shape=list(features["low"].shape),
                       high_shapes=[list(x.shape) for x in features["high"]],
                       output_shapes=[list(x.shape) for x in result],
                       encoder_frozen=True, prompt_tokens_and_guidance_grad=True)
    if args.device.startswith("cuda"): torch.cuda.synchronize()
    receipt.update(backend=args.backend, device=args.device, canvas=args.canvas,
                   passed=True, scope="software smoke with random RGB; no accuracy result or training updates",
                   elapsed_seconds=time.perf_counter()-start,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k != "base"}, indent=2))


if __name__ == "__main__":
    main()
