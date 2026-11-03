"""Portable native-pixel instance data; no old project package is imported.

This is a supervised data loader, not an RGB deployment input builder. Geometry
is never resized, cropped, transposed, or inferred from annotation quality.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

import numpy as np
from PIL import Image
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_mask_hash(masks):
    """Compatible with the existing frozen boolean NPZ export hash."""
    masks = np.ascontiguousarray(masks, dtype=bool)
    digest = hashlib.sha256()
    digest.update(str(tuple(masks.shape)).encode("ascii"))
    digest.update(b"|bool|")
    digest.update(masks.tobytes())
    return digest.hexdigest()


def safe_relative_path(value):
    """Reject Windows and POSIX traversal, even when running on the other OS."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Expected a nonempty relative path")
    win = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if win.drive or win.root or posix.is_absolute() or ".." in posix.parts or ":" in value:
        raise ValueError(f"Unsafe relative data path: {value}")
    if str(posix) in (".", ""):
        raise ValueError("A record path must name a file")
    return Path(*posix.parts)


def resolved_record_path(root, value):
    root = Path(root).resolve()
    result = (root / safe_relative_path(value)).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Record path escapes data_root, including via symlink")
    return result


def read_manifest(path, allow_condition=False):
    """Validate the entire frozen split before selecting any subset.

    RGB hash checks here use the declared hashes; load_record/prepare validation
    verify actual files. Near-duplicate or simulation-scene isolation is not
    inferred from exact hash/group isolation.
    """
    path = Path(path).resolve()
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    kind = obj.get("grouping_kind")
    if kind not in ("verified_scene", "condition_holdout_pilot"):
        raise ValueError("Expected verified_scene or condition_holdout_pilot grouping")
    if kind == "condition_holdout_pilot":
        if not allow_condition:
            raise ValueError("Condition pilot requires explicit allow_condition=True")
        if obj.get("independent_scene_split") is not False:
            raise ValueError("Condition pilot must set independent_scene_split=false")
    elif obj.get("independent_scene_split") is not True:
        raise ValueError("verified_scene requires independent_scene_split=true")
    records = obj.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest records must be a nonempty list")
    groups, hashes, ids, names, images, mask_paths = {}, {}, set(), set(), set(), set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Each manifest record must be an object")
        for key in ("id", "file_name", "split", "group_id", "rgb_sha256", "image", "masks", "shape"):
            if key not in row:
                raise ValueError(f"Manifest record is missing {key}")
        ident = row["id"]
        if not isinstance(ident, (str, int)) or isinstance(ident, bool) or not str(ident):
            raise ValueError("Invalid sample ID")
        split = row["split"]
        if split not in ("train", "validation", "test", "check"):
            raise ValueError(f"Unsupported split name: {split}")
        if not isinstance(row["group_id"], str) or not row["group_id"]:
            raise ValueError("Missing group identity")
        for key in ("rgb_sha256", "image_sha256", "masks_sha256", "decoded_masks_sha256"):
            if key in row and not re.fullmatch(r"[0-9a-f]{64}", row[key] if isinstance(row[key], str) else ""):
                raise ValueError(f"Invalid SHA256 field: {key}")
        shape = row["shape"]
        if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("shape must be native [height,width] positive integers")
        for seen, value in ((ids, str(ident)), (names, safe_relative_path(row["file_name"]).as_posix()),
                            (images, safe_relative_path(row["image"]).as_posix()),
                            (mask_paths, safe_relative_path(row["masks"]).as_posix())):
            if value in seen:
                raise ValueError(f"Repeated sample ID/source filename/data path: {value}")
            seen.add(value)
        for seen, value in ((groups, row["group_id"]), (hashes, row["rgb_sha256"])):
            if value in seen and seen[value] != split:
                raise ValueError("Frozen group or decoded RGB hash crosses splits")
            seen[value] = split
    actual = dict(Counter(row["split"] for row in records))
    if "counts" in obj and obj["counts"] != actual:
        raise ValueError(f"Declared split counts differ from records: {actual}")
    return obj


def manifest_root(path, obj, data_root=None):
    """Explicit relocation overrides a manifest's machine-specific root."""
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    value = obj.get("data_root", ".")
    if value is None:
        value = "."
    root = Path(value).expanduser()
    # Do not silently interpret a Windows absolute path as a Linux relative path.
    if PureWindowsPath(str(value)).drive and not root.is_absolute():
        raise ValueError("Manifest has a foreign absolute data_root; supply data_root explicitly")
    return root.resolve() if root.is_absolute() else (Path(path).resolve().parent / root).resolve()


def load_record(root, row):
    """Read upright PNG and [N,H,W] binary NPZ; validate bytes and coordinates."""
    image_path = resolved_record_path(root, row["image"])
    mask_path = resolved_record_path(root, row["masks"])
    if row.get("image_sha256") and sha256_file(image_path) != row["image_sha256"]:
        raise ValueError(f"Image file hash changed: {row['file_name']}")
    with Image.open(image_path) as image:
        if image.format != "PNG":
            raise ValueError("Decoded exports must use lossless PNG")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Non-upright EXIF is forbidden; no implicit orientation transform")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    if list(rgb.shape[:2]) != row["shape"]:
        raise ValueError("Native image shape differs from frozen manifest")
    if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError(f"Decoded RGB hash changed: {row['file_name']}")
    if row.get("masks_sha256") and sha256_file(mask_path) != row["masks_sha256"]:
        raise ValueError(f"Mask archive hash changed: {row['file_name']}")
    with np.load(mask_path, allow_pickle=False) as archive:
        if "masks" not in archive:
            raise ValueError("NPZ must contain the masks array")
        masks = archive["masks"]
    if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
        raise ValueError("Instance masks must have shape [N,native height,native width]")
    if masks.dtype.kind not in "bui f".replace(" ", "") or not np.isfinite(masks).all() or not np.isin(masks, [0, 1]).all():
        raise ValueError("Instance masks must contain only finite binary 0/1 values")
    masks = np.ascontiguousarray(masks, dtype=bool)
    if "num_instances" in row and row["num_instances"] != masks.shape[0]:
        raise ValueError("num_instances differs from NPZ (before empty-instance filtering)")
    if row.get("decoded_masks_sha256") and decoded_mask_hash(masks) != row["decoded_masks_sha256"]:
        raise ValueError("Decoded mask hash changed")
    return rgb, masks


class NativeInstanceDataset(torch.utils.data.Dataset):
    """Unaugmented paired data; right/bottom padding only, no instance cap."""
    def __init__(self, manifest, split="train", canvas_size=1036, seed=0,
                 allow_condition=False, data_root=None):
        self.manifest = Path(manifest).resolve()
        self.protocol = read_manifest(self.manifest, allow_condition)
        self.root = manifest_root(self.manifest, self.protocol, data_root)
        self.split = "validation" if split == "val" else split
        if self.split not in ("train", "validation", "test", "check"):
            raise ValueError("Unsupported requested split")
        if type(canvas_size) is not int or canvas_size <= 0:
            raise ValueError("canvas_size must be a positive integer")
        self.canvas_size, self.seed, self.epoch = canvas_size, int(seed), 0
        self.rows = [row for row in self.protocol["records"] if row["split"] == self.split]
        if not self.rows:
            raise ValueError(f"Empty requested split: {self.split}")
        for row in self.rows:
            if max(row["shape"]) > canvas_size:
                raise ValueError(f"Native image exceeds canvas; resizing/cropping is forbidden: {row['file_name']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rgb, source_masks = load_record(self.root, row)
        h, w = rgb.shape[:2]
        size = self.canvas_size
        if max(h, w) > size:
            raise ValueError("Native image exceeds canvas")
        overlap = source_masks.sum(axis=0, dtype=np.int32) > 1
        clean = source_masks & ~overlap[None]
        retained = np.flatnonzero(clean.reshape(len(clean), h * w).any(axis=1))
        masks = torch.zeros((len(retained), size, size), dtype=torch.float32)
        if len(retained):
            masks[:, :h, :w] = torch.from_numpy(clean[retained].astype(np.float32))
        image = torch.full((3, size, size), 0.5, dtype=torch.float32)
        image[:, :h, :w] = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
        valid = torch.zeros((1, size, size), dtype=torch.bool)
        valid[0, :h, :w] = torch.from_numpy(~overlap)
        meta = {"file_name": row["file_name"], "split": row["split"], "group_id": row["group_id"],
                "rgb_sha256": row["rgb_sha256"], "original_shape": [h, w], "original_size": [h, w], "shape": [h, w],
                "canvas_size": size, "padding": [0, 0, size - w, size - h],
                "source_instance_indices": retained.tolist(), "source_num_instances": len(source_masks),
                "ignored_overlap_pixels": int(overlap.sum()), "geometry_transform": "right_bottom_pad_only",
                "augmentation": "none", "independent_scene_split": self.protocol["independent_scene_split"]}
        if "domain" in row:
            meta["domain"] = row["domain"]
        return {"image": image, "valid": valid, "masks": masks, "id": row["id"], "meta": meta}


def collate_fn(samples):
    """Keep all variable-count instance masks in a list; do not truncate/pad N."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {"images": torch.stack([sample["image"] for sample in samples]),
            "valid": torch.stack([sample["valid"] for sample in samples]),
            "targets": [{"masks": sample["masks"], "id": sample["id"], "meta": sample["meta"]}
                        for sample in samples]}
