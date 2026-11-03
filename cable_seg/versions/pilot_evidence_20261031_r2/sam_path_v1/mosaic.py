"""Deterministic training-only native crop mosaics for query-capacity exposure.

This is a separately labelled augmentation experiment, not the original data
protocol. No resizing, hidden-wire completion or validation/test donors occur.
Density search is bounded and fails visibly rather than silently falling back.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .data import load_record


class MosaicDensityError(RuntimeError):
    pass


def validate_mosaic_config(config):
    result = dict(enabled=False, probability=.5, tiles=4, crop_size=256,
                  min_instances=25, max_instances=64, max_attempts=8, crop_candidates=8)
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f"Unknown mosaic fields: {sorted(unknown)}")
    result.update(config)
    if type(result["enabled"]) is not bool or not 0 <= result["probability"] <= 1:
        raise ValueError("Invalid mosaic enable/probability")
    for key in ("tiles", "crop_size", "min_instances", "max_instances", "max_attempts", "crop_candidates"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"Invalid positive mosaic integer: {key}")
    if result["max_instances"] < result["min_instances"]:
        raise ValueError("Mosaic density bounds are reversed")
    return result


class NativeMosaicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config, num_queries):
        self.base = dataset
        self.config = validate_mosaic_config(config)
        if dataset.split != "train" or any(row["split"] != "train" for row in dataset.rows):
            raise ValueError("Mosaic donors must exclusively belong to the train split")
        self.rows, self.protocol = dataset.rows, dataset.protocol
        self.root, self.split, self.manifest = dataset.root, dataset.split, dataset.manifest
        self.seed, self.canvas_size, self.epoch = dataset.seed, dataset.canvas_size, 0
        if self.config["max_instances"] > num_queries:
            raise ValueError("Mosaic upper density exceeds the trained query capacity")
        if self.config["tiles"]*self.config["crop_size"] > self.canvas_size:
            raise ValueError("Native crop mosaic exceeds the fixed training canvas")
        unique = {}
        for index, row in enumerate(self.rows):
            unique.setdefault(row["rgb_sha256"], index)
        self.donor_indices = np.array(list(unique.values()), dtype=np.int64)
        if len(self.donor_indices) < self.config["tiles"]**2:
            raise ValueError("Insufficient distinct train RGB donors for a native mosaic")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        key = f"native_mosaic_v1:{self.seed}:{self.epoch}:{index}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        if not self.config["enabled"] or rng.random() >= self.config["probability"]:
            self.base.epoch = self.epoch
            return self.base[index]
        cfg, best_count = self.config, 0
        crop, tiles = cfg["crop_size"], cfg["tiles"]
        for attempt in range(cfg["max_attempts"]):
            selected = rng.choice(self.donor_indices, tiles*tiles, replace=False)
            pieces, recipe, count = [], [], 0
            for donor in selected:
                row = self.rows[int(donor)]
                rgb, masks = load_record(self.root, row)
                h, w = rgb.shape[:2]
                if min(h, w) < crop:
                    raise ValueError(f"Training donor is smaller than native crop: {row['id']}")
                best = None
                for _ in range(cfg["crop_candidates"]):
                    y, x = int(rng.integers(h-crop+1)), int(rng.integers(w-crop+1))
                    region = masks[:, y:y+crop, x:x+crop]
                    overlap = region.sum(0) > 1
                    clean = region & ~overlap[None]
                    keep = np.flatnonzero(clean.reshape(len(clean), crop*crop).any(1))
                    if best is None or len(keep) > best[0]:
                        best = (len(keep), y, x, clean[keep].copy(), overlap.copy(), keep)
                n, y, x, clean, overlap, keep = best
                count += n
                pieces.append((rgb[y:y+crop, x:x+crop].copy(), clean, overlap))
                recipe.append(dict(id=row["id"], rgb_sha256=row["rgb_sha256"],
                    crop_xywh=[x, y, crop, crop], source_instance_indices=keep.tolist(), split="train"))
            best_count = max(best_count, count)
            if not cfg["min_instances"] <= count <= cfg["max_instances"]:
                continue
            side, canvas = tiles*crop, self.canvas_size
            image = torch.full((3, canvas, canvas), .5)
            targets = torch.zeros((count, canvas, canvas))
            valid = torch.zeros((1, canvas, canvas), dtype=torch.bool)
            cursor = 0
            for tile, (rgb, masks, overlap) in enumerate(pieces):
                y, x = (tile//tiles)*crop, (tile%tiles)*crop
                image[:, y:y+crop, x:x+crop] = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()/255.
                valid[:, y:y+crop, x:x+crop] = torch.from_numpy(~overlap)
                targets[cursor:cursor+len(masks), y:y+crop, x:x+crop] = torch.from_numpy(masks.astype(np.float32))
                cursor += len(masks)
            row = self.rows[index]
            return dict(image=image, masks=targets, valid=valid, id=row["id"],
                meta=dict(original_size=[side, side], original_shape=[side, side],
                    file_name=row["file_name"], split="train", augmentation="native_crop_mosaic_v1",
                    geometry_transform="native_crop_and_translation_only", source_num_instances=count,
                    source_condition_group="mixed_train_mosaic", domain="synthetic_mosaic",
                    mosaic=dict(seed=seed, attempt=attempt+1, instances=count, recipe=recipe),
                    independent_scene_split=self.protocol["independent_scene_split"]))
        raise MosaicDensityError(f"No legal native mosaic in {cfg['max_attempts']} attempts; "
                                 f"seed={seed}, maximum_observed={best_count}, range="
                                 f"[{cfg['min_instances']},{cfg['max_instances']}]; no low-density fallback")


def main(argv=None):
    """Bounded real-data capacity admission before training; no model needed."""
    import argparse
    import json
    from pathlib import Path
    from .data import NativeInstanceDataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--data-root")
    parser.add_argument("--output", required=True); parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.samples, args.epochs) < 1:
        raise ValueError("Preflight sample and epoch counts must be positive")
    base = NativeInstanceDataset(args.manifest, "train", 1036, args.seed, True, args.data_root)
    data = NativeMosaicDataset(base, dict(enabled=True, probability=1.), 80)
    indices = np.random.default_rng(args.seed).choice(len(data), min(args.samples, len(data)), replace=False)
    rows = []
    for epoch in range(args.epochs):
        data.epoch = epoch
        for index in indices:
            try:
                sample = data[int(index)]
                rows.append(dict(epoch=epoch, index=int(index), status="passed", **sample["meta"]["mosaic"]))
            except Exception as exc:
                rows.append(dict(epoch=epoch, index=int(index), status="failed", error=f"{type(exc).__name__}: {exc}"))
    failed = sum(row["status"] == "failed" for row in rows)
    result = dict(scope="bounded native mosaic donor/density preflight; no training accuracy", samples=len(rows),
                  failures=failed, status="passed" if not failed else "failed", config=data.config, rows=rows)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key:value for key,value in result.items() if key != "rows"}))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
