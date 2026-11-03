"""Offline manifest/GT export, preserving existing frozen image-level splits.

Does not run SAM or invent predictions. Cache paths refer to fresh SAM-once
outputs, never the historical multi-call SAM/HQ pipeline.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .io import new_directory, read_json, write_json, sha256, rgb_hash, load_case


def _source_isolation(sources):
    """RGB-only preflight; never invent a stable namespace across source files.

    Within-source split assignments are preserved. Across different source kinds
    use the same 9x8 dHash and 32x32 RGB MAD proxy as the historical project rule.
    This proves neither scene identity nor independent scene generalization.
    """
    loaded, kinds = [], set()
    for path in sources:
        source = Path(path).resolve()
        obj = read_json(source)
        kind = obj.get("grouping_kind")
        if kind not in ("historical_rgb_cluster_holdout", "condition_holdout_pilot", "verified_scene"):
            raise ValueError("Source must have an explicit frozen grouping protocol")
        if kind in kinds:
            raise ValueError("At most one source per grouping_kind: stable cross-file cluster namespace is not proven")
        kinds.add(kind)
        loaded.append((source, obj, sha256(source)[:12]))
    if not loaded:
        raise ValueError("At least one source manifest is required")
    entries, parent, group_anchor, hash_anchor = [], [], {}, {}
    matches = []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        parent[root(i)] = root(j)

    for source_index, (source, obj, key) in enumerate(loaded):
        for record in obj["records"]:
            split = {"test": "check"}.get(record["split"], record["split"])
            if split not in ("train", "validation", "check"):
                raise ValueError("Unknown frozen split")
            if record.get("group_id") is None or not str(record["group_id"]):
                raise ValueError("Every source record requires a frozen group_id")
            cluster = f"{key}:{record['group_id']}"
            image_path = (source.parent / record["image"]).resolve()
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                rgb = np.asarray(image)
                if record.get("rgb_sha256") != hashlib.sha256(rgb.tobytes()).hexdigest():
                    raise ValueError("Source RGB changed since frozen split")
                digest = rgb_hash(rgb)
                small = np.asarray(image.resize((32, 32)), dtype=np.float32)
                gray = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
                dhash = gray[:, 1:] > gray[:, :-1]
            index = len(entries)
            parent.append(index)
            item = {"source_index": source_index, "source_case": f"{key}_{record['id']}",
                    "cluster": cluster, "split": split, "rgb_sha256": digest,
                    "small": small, "dhash": dhash}
            for anchors, value in ((group_anchor, cluster), (hash_anchor, digest)):
                if value in anchors:
                    join(index, anchors[value])
                else:
                    anchors[value] = index
            for j, old in enumerate(entries):
                if old["source_index"] == source_index:
                    continue
                distance = int(np.count_nonzero(dhash != old["dhash"]))
                if digest != old["rgb_sha256"] and distance > 4:
                    continue
                mad = float(np.abs(small - old["small"]).mean())
                if digest == old["rgb_sha256"] or mad <= 12:
                    join(index, j)
                    matches.append({"first": old["source_case"], "second": item["source_case"],
                                    "dhash_hamming": distance, "rgb_thumbnail_mad": mad,
                                    "exact_decoded_rgb": digest == old["rgb_sha256"]})
            entries.append(item)
    components = {}
    for i, entry in enumerate(entries):
        components.setdefault(root(i), []).append(entry)
    overrides, joined = {}, []
    for group in components.values():
        if len({g["split"] for g in group}) > 1:
            cases = sorted(g["source_case"] for g in group)
            raise ValueError(f"Frozen cluster/RGB near-duplicate component crosses splits: {cases}")
        old_clusters = sorted({g["cluster"] for g in group})
        if len(old_clusters) <= 1:
            continue
        cluster = "rgb_verified_component:" + hashlib.sha256("\n".join(old_clusters).encode()).hexdigest()[:20]
        for old in old_clusters:
            overrides[old] = cluster
        joined.append({"cluster_id": cluster, "source_clusters": old_clusters,
                       "split": group[0]["split"], "case_count": len(group)})
    audit = {"group_namespace_policy": "at_most_one_source_per_grouping_kind",
             "source_grouping_kinds": sorted(kinds), "original_split_assignments_preserved": True,
             "source_records_checked": len(entries), "cross_source_pair_matches": matches,
             "merged_cluster_components": joined,
             "rule": "exact decoded RGB or dHash Hamming<=4 AND 32x32 RGB mean absolute difference<=12; connected components",
             "within_source_near_duplicate_rule": "preserve original frozen groups; no resplitting",
             "independent_scene_claim": False}
    return loaded, overrides, audit


def prepare(sources, cache_root, output, require_caches=False):
    loaded, cluster_overrides, isolation = _source_isolation(sources)
    out = new_directory(output)
    rows, clusters, hashes, names = [], {}, {}, set()
    for source, obj, source_key in loaded:
        for record in obj["records"]:
            ident = str(record["id"])
            if not ident or any(c in ident for c in "/\\:") or ident in (".", ".."):
                raise ValueError("Case IDs must be path-safe and unique")
            ident = f"{source_key}_{ident}"
            if ident in names:
                raise ValueError("Duplicate frozen sample")
            names.add(ident)
            split = {"test": "check"}.get(record["split"], record["split"])
            if split not in ("train", "validation", "check"):
                raise ValueError("Unknown frozen split")
            source_cluster = f"{source_key}:{record['group_id']}"
            cluster = cluster_overrides.get(source_cluster, source_cluster)
            image_path = (source.parent / record["image"]).resolve()
            rgb = np.asarray(Image.open(image_path).convert("RGB")).copy()
            if record.get("rgb_sha256") != hashlib.sha256(rgb.tobytes()).hexdigest():
                raise ValueError("Source RGB changed since frozen split")
            digest = rgb_hash(rgb)
            for seen, key in ((clusters, cluster), (hashes, digest)):
                if key in seen and seen[key] != split:
                    raise ValueError("Frozen cluster or decoded RGB crosses splits")
                seen[key] = split
            cache = (Path(cache_root) / ident / "sam_cache").resolve()
            if require_caches:
                cached_rgb, _, _, _ = load_case(cache)
                if rgb_hash(cached_rgb) != digest:
                    raise ValueError("SAM cache belongs to another RGB image")
            masks_path = source.parent / record["masks"]
            with np.load(masks_path, allow_pickle=False) as arrays:
                masks = arrays["masks"].astype(bool)
            if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
                raise ValueError("GT must already be in frozen native RGB coordinates")
            dest = out / "labels" / ident
            dest.mkdir(parents=True)
            paths = []
            for i, mask in enumerate(masks):
                target = dest / f"{i:04d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(target)
                paths.append(target.relative_to(out).as_posix())
            rows.append({"case_id": ident, "source_id": record["id"], "split": split,
                         "cluster_id": cluster, "source_cluster_id": source_cluster,
                         "dataset": record.get("dataset", "synthetic"),
                         "image_path": str(image_path), "cache_dir": str(cache), "gt_masks": paths,
                         "rgb_sha256": digest})
    result = {"schema": "sam_once_training_v1", "records": rows,
              "cross_source_isolation": isolation,
              "source_manifests_sha256": {str(Path(p).resolve()): sha256(p) for p in sources},
              "independent_scene_split": all(read_json(p).get("independent_scene_split") is True for p in sources),
              "cache_status": "verified" if require_caches else "paths_only_predictions_not_generated",
              "scope": "Preserved historical/condition splits; not a new blind test",
              "GT_use": "offline supervision only; labels never participate in prompt or ROI generation"}
    write_json(out / "manifest.json", result)
    # This input-only queue can be consumed without exposing GT to inference.
    write_json(out / "inference_inputs.json", {"records": [{k: r[k] for k in ("case_id", "image_path", "cache_dir")} for r in rows]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", nargs="+", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-caches", action="store_true")
    a = p.parse_args()
    obj = prepare(a.source, a.cache_root, a.output, a.require_caches)
    print({"records": len(obj["records"]), "manifest": str(a.output / "manifest.json"), "cache_status": obj["cache_status"]})


if __name__ == "__main__":
    main()
