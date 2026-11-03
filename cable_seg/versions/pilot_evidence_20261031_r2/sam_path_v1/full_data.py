"""Freeze a full WireSeg image holdout while preserving pilot membership.

No pixels are copied or changed. All synthetic images are assigned once; the
new test set contains no pilot image. This is not a held-out-condition or
independent-scene protocol. Near-duplicate scene identities are unavailable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_path_v1.data import NativeInstanceDataset, read_manifest, safe_relative_path, sha256_file


EXPECTED_CONDITIONS = {
    "condition:data_center/4_cable": 2000,
    "condition:data_center/8_cable": 2000,
    "condition:data_center/16_cable": 2000,
    **{f"condition:{family}/{count}_cable": 5000
       for family in ("flying_wire", "wire_on_plane") for count in (2, 4, 8)},
}


def read_sums(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        name = safe_relative_path(name.lstrip("*")).as_posix()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid SHA256SUMS digest")
        if name in entries:
            raise ValueError("Duplicate SHA256SUMS path")
        entries[name] = digest
    return entries


def build_protocol(source, pilot, *, seed=42, validation_fraction="0.05", test_fraction="0.05",
                   data_root=".", sums=None, expected_conditions=EXPECTED_CONDITIONS):
    """Deterministically allocate each condition; refuse impossible locked quotas.

    Existing pilot train and validation identities are pinned by filename and
    RGB hash, not by the different export's numeric IDs. Exact RGB duplicates
    in the source are rejected rather than silently splitting them.
    """
    val_fraction, test_fraction = Fraction(str(validation_fraction)), Fraction(str(test_fraction))
    if min(val_fraction, test_fraction) <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and leave training images")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    records = source.get("records", [])
    if not records:
        raise ValueError("Empty source manifest")
    if source.get("grouping_kind") != "condition_holdout_pilot_with_unassigned_extension":
        raise ValueError("Expected the preserved full synthetic export")
    if source.get("independent_scene_split") is not False:
        raise ValueError("Source does not establish independent scenes")
    if pilot.get("grouping_kind") != "condition_holdout_pilot" or pilot.get("independent_scene_split") is not False:
        raise ValueError("Expected the historical condition-holdout pilot")
    groups, by_name, by_hash = defaultdict(list), {}, {}
    for row in records:
        name = row["file_name"]
        if name in by_name or row["rgb_sha256"] in by_hash:
            raise ValueError("Duplicate source filename or exact decoded RGB hash")
        if row["split"] not in ("train", "validation", "unassigned"):
            raise ValueError("Unexpected pre-existing source split")
        condition = row["group_id"]
        if condition != "condition:" + "/".join(safe_relative_path(name).parts[:-1]):
            raise ValueError("Source condition does not match filename")
        groups[condition].append(row)
        by_name[name], by_hash[row["rgb_sha256"]] = row, name
    observed = {key: len(value) for key, value in groups.items()}
    if expected_conditions is not None and observed != expected_conditions:
        raise ValueError(f"Full synthetic condition inventory changed: {observed}")
    if dict(Counter(row["split"] for row in records)) != source.get("counts"):
        raise ValueError("Source declared counts do not match records")

    locked = {}
    for row in pilot["records"]:
        name = row["file_name"]
        if name in locked or name not in by_name or row["split"] not in ("train", "validation"):
            raise ValueError("Invalid, repeated or missing pilot identity")
        for key in ("split", "group_id", "shape", "rgb_sha256"):
            if row[key] != by_name[name][key]:
                raise ValueError(f"Pilot source membership/hash/geometry differs: {name}: {key}")
        locked[name] = row["split"]
    # Never silently lose historical supervised/validation membership.
    assigned = {row["file_name"]: row["split"] for row in records if row["split"] != "unassigned"}
    if assigned != locked:
        raise ValueError("Source assigned membership must equal the entire frozen pilot")

    assignment, condition_counts = dict(locked), {}
    for condition, rows in sorted(groups.items()):
        n = len(rows)
        exact_val, exact_test = n * val_fraction, n * test_fraction
        if exact_val.denominator != 1 or exact_test.denominator != 1:
            raise ValueError("Fractions must give integer counts in every condition")
        target = {"train": n - int(exact_val) - int(exact_test),
                  "validation": int(exact_val), "test": int(exact_test)}
        pinned = Counter(locked[row["file_name"]] for row in rows if row["file_name"] in locked)
        if any(pinned[split] > count for split, count in target.items()):
            raise ValueError(f"Requested quota smaller than pinned pilot members: {condition}")
        unassigned = [row for row in rows if row["file_name"] not in locked]
        # Stateless hash order is identical across Python/NumPy versions and input order.
        unassigned.sort(key=lambda row: (hashlib.sha256(
            f"wireseg-full-split-v1|{seed}|{condition}|{row['rgb_sha256']}".encode()).hexdigest(), row["file_name"]))
        cursor = 0
        for split in ("validation", "test", "train"):
            end = cursor + target[split] - pinned[split]
            for row in unassigned[cursor:end]:
                assignment[row["file_name"]] = split
            cursor = end
        if cursor != len(unassigned):
            raise AssertionError("Split allocation incomplete")
        condition_counts[condition] = target

    result_rows = []
    for row in sorted(records, key=lambda r: r["file_name"]):
        out = deepcopy(row)
        out.update(source_split=row["split"], source_condition_group=row["group_id"],
                   group_id="rgb_sha256:" + row["rgb_sha256"], split=assignment[row["file_name"]],
                   domain="wireseg_synthetic", pilot_member=row["file_name"] in locked)
        if sums is not None:
            for field, pathfield in (("image_sha256", "image"), ("masks_sha256", "masks")):
                key = safe_relative_path(row[pathfield]).as_posix()
                if key not in sums:
                    raise ValueError(f"File missing from SHA256SUMS: {key}")
                out[field] = sums[key]
        result_rows.append(out)
    return dict(schema=1, grouping_kind="stratified_image_holdout", independent_scene_split=False,
                independent_condition_split=False, split_unit="exact_decoded_rgb_hash", data_root=str(data_root),
                grouping_basis="Image holdout stratified over all nine family/count conditions; exact RGB hashes isolated. "
                               "Simulation scene identities and near-duplicate isolation are not established.",
                seed=seed, validation_fraction=str(val_fraction), test_fraction=str(test_fraction),
                counts=dict(Counter(row["split"] for row in result_rows)), condition_counts=condition_counts,
                historical_pilot_counts=dict(Counter(locked.values())),
                test_contract="No pilot image in test; test is used after validation-only model selection. "
                              "This set is newly held out for this B/C run, not a guarantee of no historical project exposure.",
                records=result_rows)


def real_compatibility(manifest, canvas_size=1036, num_queries=24):
    rows = manifest["records"]
    return dict(images=len(rows), shape_counts=dict(sorted(Counter("x".join(map(str, r["shape"])) for r in rows).items())),
                canvas_size=canvas_size, num_queries=num_queries,
                exceeds_canvas=sum(max(r["shape"]) > canvas_size for r in rows),
                exceeds_query_capacity=sum(r["num_instances"] > num_queries for r in rows),
                maximum_instances=max(r["num_instances"] for r in rows),
                maximum_native_side=max(max(r["shape"]) for r in rows),
                note="Diagnostic metadata only. Do not silently resize, crop, truncate instances or filter out incompatible test images.")


def validate_samples(manifest, *, data_root=None, samples_per_cell=2, canvas_size=1036, num_queries=24):
    """Verify decoded loader input in every condition/split cell, without training.

    All rows are checked for native shape and declared query count. Sampled file
    bytes, RGB pixels and NPZ contents are checked by the production loader.
    This is explicitly not a new full-pixel verification receipt.
    """
    if type(samples_per_cell) is not int or samples_per_cell < 1:
        raise ValueError("samples_per_cell must be positive")
    protocol = read_manifest(manifest, allow_condition=True)
    for row in protocol["records"]:
        if max(row["shape"]) > canvas_size or row["num_instances"] > num_queries:
            raise ValueError(f"Full inventory exceeds configured native canvas/query capacity: {row['file_name']}")
    checked = []
    for split in sorted(protocol["counts"]):
        dataset = NativeInstanceDataset(manifest, split, canvas_size, allow_condition=True, data_root=data_root)
        cells = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            cells[row["source_condition_group"]].append(index)
        for condition, indices in sorted(cells.items()):
            for index in sorted(indices, key=lambda i: dataset.rows[i]["file_name"])[:samples_per_cell]:
                sample = dataset[index]
                if len(sample["masks"]) > num_queries:
                    raise ValueError("Decoded visible instances exceed configured query capacity")
                checked.append(dict(id=sample["id"], file_name=sample["meta"]["file_name"], split=split,
                                    source_condition_group=condition, visible_instances=len(sample["masks"]),
                                    ignored_overlap_pixels=sample["meta"]["ignored_overlap_pixels"]))
    return dict(status="sampled_production_loader_passed", manifest_sha256=sha256_file(manifest),
                records_structurally_checked=len(protocol["records"]), pixels_checked=len(checked),
                samples_per_condition_split=samples_per_cell, full_pixels_rechecked=False,
                counts=protocol["counts"], canvas_size=canvas_size, num_queries=num_queries, samples=checked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--sha256sums", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", default="0.05")
    parser.add_argument("--test-fraction", default="0.05")
    parser.add_argument("--real-manifest")
    parser.add_argument("--sample-check", action="store_true")
    parser.add_argument("--samples-per-cell", type=int, default=2)
    parser.add_argument("--canvas-size", type=int, default=1036)
    parser.add_argument("--num-queries", type=int, default=24)
    args = parser.parse_args(argv)
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError("Use a new output directory; frozen manifests are never overwritten")
    source = json.loads(Path(args.source_manifest).read_text(encoding="utf-8-sig"))
    pilot = read_manifest(args.pilot_manifest, allow_condition=True)
    protocol = build_protocol(source, pilot, seed=args.seed, validation_fraction=args.validation_fraction,
                              test_fraction=args.test_fraction, data_root=args.data_root, sums=read_sums(args.sha256sums))
    protocol["provenance"] = {key + "_sha256": sha256_file(getattr(args, key))
                              for key in ("source_manifest", "pilot_manifest", "sha256sums")}
    dest.mkdir(parents=True)
    target = dest / "manifest.json"
    target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    read_manifest(target, allow_condition=True)
    report = {key: value for key, value in protocol.items() if key != "records"}
    report["manifest_sha256"] = sha256_file(target)
    report["status"] = "frozen_metadata_only_pixels_not_rechecked"
    if args.real_manifest:
        report["real_compatibility"] = real_compatibility(json.loads(Path(args.real_manifest).read_text(encoding="utf-8-sig")))
        report["real_source_manifest_sha256"] = sha256_file(args.real_manifest)
    (dest / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_check:
        sample_report = validate_samples(target, data_root=args.data_root, samples_per_cell=args.samples_per_cell,
                                         canvas_size=args.canvas_size, num_queries=args.num_queries)
        (dest / "sample_check.json").write_text(json.dumps(sample_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
