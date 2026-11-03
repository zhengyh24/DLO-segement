"""Validate an existing export or stream EXACT frozen WireSeg members to PNG/NPZ.

No download, split generation, resizing, source mutation, or implicit EXIF fix.
Export requires a new output directory. A failed partial directory is preserved;
use a different directory after fixing the cause. No automatic resume/overwrite.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_path_v1.data import (decoded_mask_hash, load_record, manifest_root, read_manifest,
                              resolved_record_path, safe_relative_path, sha256_file)


PILOT_COUNTS = {"train": 2000, "validation": 300}


def _write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _check_counts(protocol, require_pilot):
    counts = dict(Counter(row["split"] for row in protocol["records"]))
    if require_pilot and counts != PILOT_COUNTS:
        raise ValueError(f"Expected exact frozen 2000/300 pilot, found {counts}")
    return counts


def _check_frozen_membership(protocol, frozen):
    for key in ("grouping_kind", "independent_scene_split"):
        if protocol.get(key) != frozen.get(key):
            raise ValueError(f"Frozen protocol changed: {key}")
    actual = {row["file_name"]: row for row in protocol["records"]}
    expected = {row["file_name"]: row for row in frozen["records"]}
    if set(actual) != set(expected):
        raise ValueError("Source export must have identical membership to frozen manifest")
    for name, row in actual.items():
        for key in ("id", "split", "group_id", "rgb_sha256", "shape"):
            if row[key] != expected[name][key]:
                raise ValueError(f"Frozen {key} changed: {name}")


def validate_export(manifest, frozen_manifest=None, data_root=None, canvas_size=1036,
                    allow_condition=False, require_pilot=True, receipt=None, max_instances=24):
    """Read every decoded sample, without copying it or changing the manifest."""
    manifest = Path(manifest).resolve()
    protocol = read_manifest(manifest, allow_condition)
    counts = _check_counts(protocol, require_pilot)
    if type(canvas_size) is not int or canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if type(max_instances) is not int or max_instances <= 0:
        raise ValueError("max_instances must be a positive query capacity")
    if receipt is not None and Path(receipt).exists():
        raise FileExistsError("Validation receipt already exists; use a new path")
    if frozen_manifest is not None:
        _check_frozen_membership(protocol, read_manifest(frozen_manifest, allow_condition))
    root = manifest_root(manifest, protocol, data_root)
    overlaps = empty = total_instances = max_visible_instances = source_max = 0
    for row in protocol["records"]:
        if max(row["shape"]) > canvas_size:
            raise ValueError(f"Native image exceeds canvas: {row['file_name']}")
        _, masks = load_record(root, row)
        overlap = masks.sum(axis=0, dtype=np.int32) > 1
        overlaps += int(overlap.sum())
        clean = masks & ~overlap[None]
        total_instances += len(masks)
        visible_instances = int(clean.reshape(len(clean), np.prod(masks.shape[1:])).any(axis=1).sum())
        source_max = max(source_max, len(masks))
        max_visible_instances = max(max_visible_instances, visible_instances)
        empty += len(masks) - visible_instances
        if visible_instances > max_instances:
            raise ValueError(f"Visible instances exceed query capacity: {row['file_name']}: "
                             f"visible={visible_instances}, source={len(masks)}, max_instances={max_instances}; no truncation")
    result = {"schema": 1, "status": "validated", "complete": True,
              "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
              "data_root": str(root), "counts": counts, "records_checked": len(protocol["records"]),
              "source_instances": total_instances, "instances_empty_after_overlap_ignore": empty,
              "max_visible_instances": max_visible_instances, "source_max": source_max,
              "max_instances": max_instances,
              "overlap_pixels": overlaps, "canvas_size": canvas_size,
              "geometry_transform": "right_bottom_pad_only", "rgb_padding_value": 0.5,
              "independent_scene_split": protocol["independent_scene_split"],
              "pilot_2300_required": bool(require_pilot), "scope": "pilot2300" if require_pilot else "explicit_fixture",
              "frozen_membership_checked": frozen_manifest is not None,
              "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if frozen_manifest is not None:
        result["frozen_manifest_sha256"] = sha256_file(frozen_manifest)
    if receipt is not None:
        _write_json_new(receipt, result)
    return result


def decode_rles(value, shape):
    """Decode ordered COCO RLE; pycocotools is needed only for compressed RLE."""
    rles = json.loads(value) if isinstance(value, str) else value
    if not isinstance(rles, list):
        raise ValueError("masks_rle must be an ordered instance list")
    decoded = []
    height, width = map(int, shape)
    for item in rles:
        if not isinstance(item, dict) or list(item.get("size", [])) != [height, width]:
            raise ValueError("RLE shape does not match frozen native coordinates")
        counts = item.get("counts")
        if isinstance(counts, list):
            if any(type(v) is not int or v < 0 for v in counts) or sum(counts) != height * width:
                raise ValueError("Invalid uncompressed RLE counts")
            flat = np.zeros(height * width, dtype=bool)
            offset = 0
            for i, length in enumerate(counts):
                if i % 2:
                    flat[offset:offset + length] = True
                offset += length
            mask = flat.reshape((height, width), order="F")
        elif isinstance(counts, (str, bytes)):
            try:
                from pycocotools import mask as coco
            except ImportError as error:
                raise ImportError("Compressed WireSeg RLE requires pycocotools; install it in this environment") from error
            encoded = {"size": [height, width], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
            mask = coco.decode(encoded)
        else:
            raise ValueError("Invalid RLE counts type")
        if mask.shape != (height, width):
            raise ValueError("Decoded RLE dimensions differ from RGB")
        decoded.append(np.asarray(mask, dtype=bool))
    return np.stack(decoded) if decoded else np.zeros((0, height, width), dtype=bool)


def iter_parquet_rows(parquet_root, selected_names, batch_size=16):
    """Bound memory to one Arrow batch; decode selected RGBs only.

    Recursively accept data/*.parquet or a user-uploaded folder of shards. All
    shards are scanned for duplicate selected filenames; source row order does
    not determine the frozen sample order. No real-data shard is downloaded.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Parquet export requires pyarrow; install it in this environment") from error
    root = Path(parquet_root).resolve()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    shards = [root] if root.is_file() and root.suffix == ".parquet" else sorted(root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError("No local parquet shards found")
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        if not {"file_name", "image", "masks_rle"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Shard lacks official WireSeg columns: {shard}")
        offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["file_name", "image", "masks_rle"]):
            for index, name in enumerate(batch.column(0).to_pylist()):
                if name in selected_names:
                    yield shard, offset + index, {"file_name": name,
                          "image": batch.column(1)[index].as_py(), "masks_rle": batch.column(2)[index].as_py()}
            offset += len(batch)


def _decode_image(value, root):
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload is not None:
            source = io.BytesIO(payload)
        elif value.get("path"):
            source = resolved_record_path(root, value["path"])
        else:
            raise ValueError("Parquet image contains neither bytes nor a safe local path")
    elif isinstance(value, (bytes, bytearray)):
        source = io.BytesIO(value)
    else:
        raise ValueError("Expected Hugging Face image bytes/path structure")
    with Image.open(source) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Non-upright EXIF in source; export requires frozen upright pixels")
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def export_parquet(frozen_manifest, parquet_root, output, allow_condition=False,
                   require_pilot=True, canvas_size=1036, batch_size=16, max_instances=24):
    """Export all and only frozen records; never derive a new train/val split."""
    frozen_manifest, root, output = map(lambda p: Path(p).resolve(), (frozen_manifest, parquet_root, output))
    if output.exists():
        raise FileExistsError("Parquet export requires a new output directory")
    frozen = read_manifest(frozen_manifest, allow_condition)
    counts = _check_counts(frozen, require_pilot)
    if type(canvas_size) is not int or canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if type(max_instances) is not int or max_instances <= 0:
        raise ValueError("max_instances must be a positive query capacity")
    for row in frozen["records"]:
        if max(row["shape"]) > canvas_size:
            raise ValueError("Frozen native image is larger than canvas")
    indexed = {row["file_name"]: (index, row) for index, row in enumerate(frozen["records"])}
    output.mkdir(parents=True, exist_ok=False)
    (output / "rgb").mkdir()
    (output / "masks").mkdir()
    contract = {"schema": 1, "frozen_manifest_sha256": sha256_file(frozen_manifest),
                "counts": counts, "native_pixels": True, "resize": False,
                "require_pilot_2300": require_pilot, "source_root": str(root),
                "max_instances": max_instances,
                "scope": "pilot2300" if require_pilot else "explicit_fixture"}
    _write_json_new(output / "export_contract.json", contract)
    completed = {}
    try:
        for shard, source_row, raw in iter_parquet_rows(root, set(indexed), batch_size):
            name = raw["file_name"]
            if name in completed:
                raise ValueError(f"Selected filename occurs more than once in shards: {name}")
            index, frozen_row = indexed[name]
            rgb = _decode_image(raw["image"], root if root.is_dir() else root.parent)
            if list(rgb.shape[:2]) != frozen_row["shape"]:
                raise ValueError(f"Frozen shape mismatch: {name}")
            if hashlib.sha256(rgb.tobytes()).hexdigest() != frozen_row["rgb_sha256"]:
                raise ValueError(f"Frozen decoded RGB hash mismatch: {name}")
            masks = decode_rles(raw["masks_rle"], frozen_row["shape"])
            digest = decoded_mask_hash(masks)
            if frozen_row.get("decoded_masks_sha256") and digest != frozen_row["decoded_masks_sha256"]:
                raise ValueError(f"Frozen decoded mask hash mismatch: {name}")
            row = dict(frozen_row)
            row.update(image=f"rgb/{index:06d}.png", masks=f"masks/{index:06d}.npz")
            with (output / row["image"]).open("xb") as handle:
                Image.fromarray(rgb).save(handle, format="PNG")
            with (output / row["masks"]).open("xb") as handle:
                np.savez_compressed(handle, masks=masks)
            row.update(image_sha256=sha256_file(output / row["image"]),
                       masks_sha256=sha256_file(output / row["masks"]), decoded_masks_sha256=digest,
                       num_instances=len(masks), source_shard=str(shard), source_row=source_row)
            load_record(output, row)
            completed[name] = row
        missing = sorted(set(indexed) - set(completed))
        if missing:
            raise ValueError(f"Missing {len(missing)} frozen samples from uploaded shards; first={missing[:3]}")
        manifest = {key: value for key, value in frozen.items()
                    if key not in ("records", "data_root", "source", "receipt_note")}
        manifest.update(data_root=".", records=[completed[row["file_name"]] for row in frozen["records"]],
                        counts=counts, source_manifest_sha256=contract["frozen_manifest_sha256"],
                        scope=contract["scope"], source=str(root),
                        artifact_kind="native_instance_export", contains_decoded_data=True, membership_only=False,
                        label_contract="Native per-instance masks; original order and overlap retained in NPZ; loader ignores overlap.")
        _write_json_new(output / "manifest.json", manifest)
        result = validate_export(output / "manifest.json", frozen_manifest, canvas_size=canvas_size,
                                 allow_condition=allow_condition, require_pilot=require_pilot, max_instances=max_instances)
        result.update(status="complete", source_modified=False, training_launched=False)
        _write_json_new(output / "completion.json", result)
        return output / "manifest.json"
    except Exception as error:
        _write_json_new(output / "failure.json", {"status": "failed", "complete": False,
                        "processed": len(completed), "expected": len(indexed), "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Read and validate existing PNG/NPZ export, no copying")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--frozen-manifest", type=Path)
    validate.add_argument("--data-root", type=Path, help="Explicit relocated decoded data root")
    validate.add_argument("--receipt", type=Path, help="New receipt path; parent directory must exist")
    export = commands.add_parser("export-parquet", help="Stream exact frozen members from local uploaded parquet")
    export.add_argument("--frozen-manifest", required=True, type=Path)
    export.add_argument("--parquet-root", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--batch-size", type=int, default=16)
    for command in (validate, export):
        command.add_argument("--canvas-size", type=int, default=1036)
        command.add_argument("--max-instances", type=int, default=24,
                             help="Query capacity; reject excess nonempty visible instances after overlap ignore, never truncate")
        command.add_argument("--allow-condition-pilot", action="store_true")
        command.add_argument("--allow-small-fixture", action="store_true", help="Software checks only; disables exact 2000/300 check")
    args = parser.parse_args(argv)
    shared = dict(allow_condition=args.allow_condition_pilot, require_pilot=not args.allow_small_fixture,
                  canvas_size=args.canvas_size, max_instances=args.max_instances)
    if args.command == "validate":
        result = validate_export(args.manifest, args.frozen_manifest, args.data_root,
                                 receipt=args.receipt, **shared)
    else:
        path = export_parquet(args.frozen_manifest, args.parquet_root, args.output,
                              batch_size=args.batch_size, **shared)
        result = {"manifest": str(path), "status": "complete"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
