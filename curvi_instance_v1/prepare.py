"""Make a small SMOKE view/copy of an existing SACM NPZ export; never decode 36K.

Defaults use the existing local 2+2 export and check its membership against the
frozen 2000-train/300-validation receipt. Full data is supported by explicitly
passing its existing manifest (and --data-root when using a downloaded receipt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curvi_instance_v1.data import load_record, manifest_root, read_manifest

PROJECT = Path(__file__).resolve().parents[1]
LOCAL_SMOKE = PROJECT / "data/wireseg_demo/manifest.json"
FROZEN_2300 = PROJECT / "data/manifests/pilot2300.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source_manifest, output, frozen_manifest=None, per_split=2, mode="copy",
            allow_condition=False, data_root=None):
    source_manifest, output = Path(source_manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if per_split < 1:
        raise ValueError("SMOKE requires a positive bounded --per-split")
    if mode not in ("copy", "reference"):
        raise ValueError("mode must be copy or reference")
    source = read_manifest(source_manifest, allow_condition)
    root = Path(data_root).resolve() if data_root else manifest_root(source_manifest, source)
    frozen_manifest = Path(frozen_manifest).resolve() if frozen_manifest else source_manifest
    frozen = read_manifest(frozen_manifest, allow_condition)
    for key in ("grouping_kind", "independent_scene_split"):
        if source.get(key) != frozen.get(key):
            raise ValueError(f"Frozen protocol mismatch: {key}")
    frozen_rows = {r["file_name"]: r for r in frozen["records"]}
    counts, selected = {}, []
    for row in source["records"]:
        if row["file_name"] not in frozen_rows:
            raise ValueError("Source sample is absent from the frozen manifest")
        expected = frozen_rows[row["file_name"]]
        # Export IDs may differ in earlier 2+2 subset exports; original filename,
        # decoded RGB hash, split, group, and geometry define sample identity.
        for key in ("split", "group_id", "rgb_sha256", "shape"):
            if row.get(key) != expected.get(key):
                raise ValueError(f"Frozen {key} changed: {row['file_name']}")
        split = row["split"]
        if counts.get(split, 0) >= per_split:
            continue
        selected.append(dict(row)); counts[split] = counts.get(split, 0)+1
    if not selected:
        raise ValueError("No source records selected")
    # Validate selected pixels/masks before creating any output directory.
    for row in selected:
        load_record(root, row)
    output.mkdir(parents=True, exist_ok=False)
    if mode == "copy":
        (output / "rgb").mkdir(); (output / "masks").mkdir()
    for index, row in enumerate(selected):
        original_mask = root / row["masks"]
        row["masks_sha256"] = sha(original_mask)
        row["frozen_record_id"] = frozen_rows[row["file_name"]]["id"]
        if mode == "copy":
            image = f"rgb/{index:06d}{(root / row['image']).suffix.lower()}"
            masks = f"masks/{index:06d}.npz"
            shutil.copyfile(root / row["image"], output / image)
            shutil.copyfile(original_mask, output / masks)
            row["image"], row["masks"] = image, masks
    result = {key: source[key] for key in ("grouping_kind", "independent_scene_split", "grouping_basis") if key in source}
    result.update(schema=1, records=selected, counts=counts, data_root="." if mode == "copy" else str(root),
                  scope="SMOKE software subset; not a new accuracy experiment or independent scene test",
                  source_manifest=str(source_manifest), source_manifest_sha256=sha(source_manifest),
                  frozen_manifest=str(frozen_manifest), frozen_manifest_sha256=sha(frozen_manifest),
                  frozen_counts=frozen.get("counts", {}), selection="first N per frozen split in original order",
                  subset_limit=per_split, storage_mode=mode,
                  label_contract="Native visible per-instance masks. Overlap/padding ignored in loss; no hidden interpolation.")
    destination = output / "manifest.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    read_manifest(destination, allow_condition)
    for row in selected:
        load_record(output if mode == "copy" else root, row)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=LOCAL_SMOKE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, default=FROZEN_2300 if FROZEN_2300.exists() else None)
    parser.add_argument("--data-root", type=Path, help="Existing decoded export root; never downloads/exports data")
    parser.add_argument("--per-split", type=int, default=2)
    parser.add_argument("--mode", choices=("copy", "reference"), default="copy")
    parser.add_argument("--allow-condition-pilot", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.source_manifest, args.output, args.frozen_manifest, args.per_split,
                       args.mode, args.allow_condition_pilot, args.data_root)
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"manifest": str(manifest), "counts": obj["counts"], "scope": obj["scope"],
                      "independent_scene_split": obj["independent_scene_split"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
