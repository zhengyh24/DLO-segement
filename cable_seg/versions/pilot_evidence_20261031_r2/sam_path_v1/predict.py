"""Deploy from one upright RGB image and a checkpoint; no annotation input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .checkpoint import atomic_json, build_from_checkpoint, new_directory, sha256
from .inference import autocast_context, decode_instances


def load_rgb(path, canvas_size):
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError("Input must be upright RGB: resolve EXIF orientation explicitly before deployment")
        rgb = np.array(source.convert("RGB"), copy=True)
    height, width = rgb.shape[:2]
    if max(height, width) > canvas_size:
        raise ValueError(f"Native image {height}x{width} exceeds canvas {canvas_size}; no implicit resize")
    tensor = torch.full((1, 3, canvas_size, canvas_size), .5)
    tensor[0, :, :height, :width] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.
    valid = torch.zeros((1, 1, canvas_size, canvas_size), dtype=torch.bool)
    valid[0, :, :height, :width] = True
    return rgb, tensor, valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", default="none", choices=("none", "bf16", "fp16"))
    parser.add_argument("--sam-repo"); parser.add_argument("--sam-checkpoint")
    args = parser.parse_args(argv)
    destination = new_directory(args.output)
    model, checkpoint = build_from_checkpoint(args.checkpoint, device=args.device,
                                             sam_repo=args.sam_repo, sam_checkpoint=args.sam_checkpoint)
    config = checkpoint["train_config"]
    rgb, images, valid = load_rgb(args.image, checkpoint["model_config"]["canvas_size"])
    images, valid = images.to(args.device), valid.to(args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), autocast_context(args.device, args.amp):
        outputs = model(images, valid)
    if outputs.get("calls") != {"encoder": 1, "decoder": 1}:
        raise RuntimeError(f"Single-forward call contract failed: {outputs.get('calls')}")
    result = decode_instances(outputs, images.shape[-2:], valid,
                              object_threshold=config["object_threshold"], mask_threshold=config["mask_threshold"])[0]
    forward_and_decode_seconds = time.perf_counter()-start
    height, width = rgb.shape[:2]
    ids = result["instances"][:height, :width]
    if ids.max() > 65535:
        raise ValueError("PNG ID format supports at most 65535 visible instances")
    Image.fromarray(ids.astype(np.uint16)).save(destination / "instances.png")
    Image.fromarray(result["union"][:height, :width].astype(np.uint8)*255).save(destination / "union.png")
    Image.fromarray(result["unknown"][:height, :width].astype(np.uint8)*255).save(destination / "unknown.png")
    np.savez_compressed(destination / "prediction.npz", instances=ids, union=result["union"][:height, :width],
                        unknown=result["unknown"][:height, :width])
    receipt = dict(image=str(Path(args.image).resolve()), image_sha256=sha256(args.image),
                   checkpoint_sha256=sha256(args.checkpoint), native_size=[height, width],
                   canvas_size=images.shape[-1], geometry_transform="right_bottom_pad_only",
                   backend=checkpoint["model_config"]["backend"], calls=outputs["calls"],
                   instances=int(ids.max()), queries=result["queries"], confidences=result["confidences"],
                   unknown_policy=result["unknown_policy"], seconds=time.perf_counter()-start,
                   forward_and_decode_seconds=forward_and_decode_seconds,
                   object_threshold=config["object_threshold"], mask_threshold=config["mask_threshold"])
    atomic_json(destination / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
