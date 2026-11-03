"""Audit an immutable public HANDLOOM test release without arbitrary unpickling.

Exports verified RGB and numeric traces only. Traces are not mask ground truth,
and no scene IDs or visibility labels are invented. See local reference R07.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import pickle
import time
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

from .checkpoint import atomic_json, new_directory


REPOSITORY = "vainaviv/handloom"
API = "https://api.github.com/repos/"+REPOSITORY
REFERENCE = "DLO-seg/references/01_直接线缆_DLO/R07.md"
MAX_BYTES = 64 * 1024 * 1024


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """No arbitrary GLOBAL/import. Only NumPy's numeric serialization helpers."""
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np.core.multiarray, name)
        if module == "_codecs" and name == "encode":
            # Older NumPy Python-2 byte arrays use this codec helper. Restrict it.
            return _latin1_bytes
        raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")


def _latin1_bytes(value, encoding="latin1"):
    if not isinstance(value, str) or encoding not in ("latin1", "latin-1"):
        raise pickle.UnpicklingError("Only literal latin1 byte reconstruction is allowed")
    return value.encode("latin1")


def restricted_npy(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Source NPY exceeds the bounded audit input size")
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    if dtype.hasobject:
        if shape != ():
            raise ValueError("Only the official scalar dictionary NPY wrapper is admitted")
        value = RestrictedNumpyUnpickler(stream).load()
        if not isinstance(value, np.ndarray) or value.shape != ():
            raise ValueError("Expected scalar NumPy dictionary wrapper")
        value = value.item()
        if not isinstance(value, dict):
            raise ValueError("Expected official numeric/image field dictionary")
        return value
    return np.load(io.BytesIO(data), allow_pickle=False)


def fetch(url, *, max_bytes=MAX_BYTES):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "WireSeg-public-data-audit/1.0"}), timeout=60) as response:
                data = response.read(max_bytes+1)
            if len(data) > max_bytes:
                raise ValueError("Public resource exceeds maximum bytes")
            return data
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1+attempt)


def git_blob_sha(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def numeric_points(value):
    if isinstance(value, dict):
        keys = sorted(value, key=lambda k: int(k))
        array = np.asarray([value[key] for key in keys])
    else:
        array = np.asarray(value)
    if array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError("Trace must contain only finite numeric values")
    array = np.squeeze(array)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Trace shape is not [points,2]: {array.shape}")
    return array.astype(np.float32)


def describe(value):
    if isinstance(value, np.ndarray):
        return dict(type="ndarray", shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return dict(type="dict", length=len(value), first_keys=[str(k) for k in list(value)[:6]])
    if isinstance(value, (list, tuple)):
        return dict(type=type(value).__name__, length=len(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return dict(type=type(value).__name__, value=value)
    raise ValueError(f"Unexpected official data field type {type(value).__name__}")


def audit_handloom(output, *, limit=None, workers=4):
    out = new_directory(output)
    info = json.loads(fetch(API))
    branch = info["default_branch"]
    commit = json.loads(fetch(API+"/commits/"+branch))["sha"]
    tree = json.loads(fetch(API+"/git/trees/"+commit+"?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Incomplete official tree cannot establish an audit manifest")
    tree_rows = [r for r in tree["tree"] if r["type"] == "blob" and r["path"].startswith("data/real_data/")]
    test_rows = [r for r in tree_rows if r["path"].startswith("data/real_data/real_data_for_tracer/test/") and r["path"].endswith(".npy")]
    test_rows.sort(key=lambda r: r["path"])
    selected = test_rows if limit is None else test_rows[:limit]
    if not selected:
        raise ValueError("No official test rows selected")
    for name in ("raw", "rgb", "traces"):
        (out/name).mkdir()
    atomic_json(out/"source_manifest.json", dict(repository=REPOSITORY, commit=commit, reference=REFERENCE,
        license=info.get("license"), all_real_file_metadata=tree_rows, selected_test_files=selected,
        training_payload_downloaded=False, independent_scene_ids_available=False))

    def process(row):
        stem = Path(row["path"]).stem
        record = dict(id="handloom_test_"+stem, source_path=row["path"], git_blob_sha1=row["sha"],
                      source_commit=commit, scene_id=None, status="pending")
        try:
            data = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/"+row["path"])
            if len(data) != row["size"] or git_blob_sha(data) != row["sha"]:
                raise ValueError("Immutable Git blob size/hash mismatch")
            (out/"raw"/(stem+".npy")).write_bytes(data)
            value = restricted_npy(data)
            record.update(sha256=hashlib.sha256(data).hexdigest(), fields={k: describe(v) for k, v in value.items()})
            image = np.asarray(value["img"])
            if image.ndim != 3 or image.shape[2] < 3 or image.dtype.kind not in "buif" or not np.isfinite(image).all():
                raise ValueError("Image is not a finite HWC array")
            image = image[:, :, :3]
            if image.dtype.kind == "f" and image.max() <= 1:
                image = image*255
            if image.min() < 0 or image.max() > 255:
                raise ValueError("Image intensity outside [0,255]")
            image = image.astype(np.uint8)
            # Official viz_real.py uses cv2.imwrite(img), so its source img is
            # interpreted in BGR order. Verify against the corresponding PNG.
            position = test_rows.index(row)
            viz_path = f"data/real_data/real_data_for_tracer_viz/test/{position}.png"
            viz_row = next(r for r in tree_rows if r["path"] == viz_path)
            viz_data = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/"+viz_path)
            if git_blob_sha(viz_data) != viz_row["sha"]:
                raise ValueError("Official visualization hash mismatch")
            reference = np.asarray(Image.open(io.BytesIO(viz_data)).convert("RGB"))
            rgb = image[:, :, ::-1].copy()
            if rgb.shape != reference.shape or not np.array_equal(rgb, reference):
                raise ValueError("BGR->RGB interpretation does not exactly reproduce official visualization")
            Image.fromarray(rgb).save(out/"rgb"/(stem+".png"))
            traces = {"pixels_yx": numeric_points(value["pixels"])}
            if "dense_points" in value:
                indices = np.asarray(value["dense_points"])
                if indices.size == 0:
                    indices = np.empty(0, dtype=np.int64)
                if indices.ndim != 1 or indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(traces["pixels_yx"])).any():
                    raise ValueError("dense_points must be in-range trace indices, not spatial coordinates")
                traces["dense_point_indices"] = indices
            np.savez_compressed(out/"traces"/(stem+".npz"), **traces)
            record.update(status="rgb_and_numeric_trace_verified", rgb_path="rgb/"+stem+".png",
                trace_path="traces/"+stem+".npz", shape=list(rgb.shape[:2]),
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                rgb_file_sha256=hashlib.sha256((out/"rgb"/(stem+".png")).read_bytes()).hexdigest(),
                trace_file_sha256=hashlib.sha256((out/"traces"/(stem+".npz")).read_bytes()).hexdigest(),
                decoded_channel_contract="source BGR -> RGB; exact official PNG equality",
                trace_fields={key: list(array.shape) for key, array in traces.items()},
                trace_coordinate_order="row,column from official dataset.py get_crop_and_cond_pixels; no visibility inference",
                visible_instance_masks_available=False, trace_visibility_labels_available=False,
                admissible_evaluation="RGB-only inference, predicted-mask/trace overlays and non-accuracy output diagnostics",
                inadmissible_metrics=["mask_IoU", "PQ", "whole_visible_recovery", "scene_independent_CI", "natural_occlusion_identity_accuracy"])
        except Exception as exc:
            record.update(status="audit_failed", error=f"{type(exc).__name__}: {exc}")
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(process, selected))
    verified = [r for r in records if r["status"] == "rgb_and_numeric_trace_verified"]
    duplicate_groups = {}
    for row in verified:
        duplicate_groups.setdefault(row["rgb_sha256"], []).append(row["id"])
    result = dict(repository=REPOSITORY, commit=commit, reference=REFERENCE, official_test_files=len(test_rows),
        selected_files=len(selected), verified_files=len(verified), failures=len(records)-len(verified),
        full_official_test=len(selected) == len(test_rows),
        exact_rgb_duplicates=[group for group in duplicate_groups.values() if len(group) > 1],
        records=records, independent_scene_ids_available=False, visible_instance_masks_available=False,
        status="completed" if len(verified) == len(records) else "completed_with_audit_failures",
        dataset_license=info.get("license"), redistribution="not authorized by an explicit repository license; metadata receipt only for sharing",
        safe_reader="restricted NumPy dictionary unpickler; all other GLOBAL imports rejected")
    atomic_json(out/"audit.json", result)
    return result


def reaudit_cached(audit_root):
    """Recheck all immutable bytes after reader fixes; preserve earlier receipt."""
    root = Path(audit_root)
    previous_bytes = (root/"audit.json").read_bytes()
    previous_sha = hashlib.sha256(previous_bytes).hexdigest()
    previous_path = root/("audit_previous_"+previous_sha[:16]+".json")
    if not previous_path.exists():
        previous_path.write_bytes(previous_bytes)
    audit = json.loads(previous_bytes)
    records = []
    for old in audit["records"]:
        row = dict(old)
        stem = Path(row["source_path"]).stem
        try:
            raw = (root/"raw"/(stem+".npy")).read_bytes()
            if git_blob_sha(raw) != row["git_blob_sha1"]:
                raise ValueError("Immutable official raw blob changed")
            value = restricted_npy(raw)
            image = np.asarray(value["img"])
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("Cached release must have original uint8 BGR images")
            rgb = image[:, :, ::-1].copy()
            existing_rgb = np.asarray(Image.open(root/"rgb"/(stem+".png")).convert("RGB"))
            if not np.array_equal(rgb, existing_rgb):
                raise ValueError("Cached official-viz-verified RGB disagrees with source BGR")
            traces = {"pixels_yx": numeric_points(value["pixels"])}
            indices = np.asarray(value.get("dense_points", []))
            if indices.size == 0:
                indices = np.empty(0, dtype=np.int64)
            if indices.ndim != 1 or indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(traces["pixels_yx"])).any():
                raise ValueError("Invalid dense trace point indices")
            traces["dense_point_indices"] = indices
            np.savez_compressed(root/"traces"/(stem+".npz"), **traces)
            row.pop("error", None)
            row.update(status="rgb_and_numeric_trace_verified", sha256=hashlib.sha256(raw).hexdigest(),
                fields={k: describe(v) for k, v in value.items()},
                rgb_path="rgb/"+stem+".png", trace_path="traces/"+stem+".npz", shape=list(rgb.shape[:2]),
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                rgb_file_sha256=hashlib.sha256((root/"rgb"/(stem+".png")).read_bytes()).hexdigest(),
                trace_file_sha256=hashlib.sha256((root/"traces"/(stem+".npz")).read_bytes()).hexdigest(),
                decoded_channel_contract="source BGR -> RGB; cached official PNG verification reproduced",
                trace_fields={key: list(array.shape) for key, array in traces.items()},
                trace_coordinate_order="row,column from official dataset.py get_crop_and_cond_pixels; no visibility inference",
                visible_instance_masks_available=False, trace_visibility_labels_available=False,
                admissible_evaluation="RGB-only inference, predicted-mask/trace overlays and non-accuracy output diagnostics",
                inadmissible_metrics=["mask_IoU", "PQ", "whole_visible_recovery", "scene_independent_CI", "natural_occlusion_identity_accuracy"])
        except Exception as exc:
            row.update(status="audit_failed", error=f"{type(exc).__name__}: {exc}")
        records.append(row)
    groups = {}
    for row in records:
        if row["status"] == "rgb_and_numeric_trace_verified":
            groups.setdefault(row["rgb_sha256"], []).append(row["id"])
    failures = sum(r["status"] != "rgb_and_numeric_trace_verified" for r in records)
    audit.update(records=records, failures=failures, verified_files=len(records)-failures,
        status="completed" if not failures else "completed_with_audit_failures",
        exact_rgb_duplicates=[g for g in groups.values() if len(g) > 1], unique_verified_rgb_images=len(groups),
        previous_audit_sha256=previous_sha,
        reader_revision="empty dense_points are valid zero-length integer indices; all raw Git blob hashes reverified")
    atomic_json(root/"audit.json", audit)
    return audit


def rgb_only_diagnostics(model, audit_root, output, *, device="cpu", amp="none",
                         object_threshold=.5, mask_threshold=.5, sample_visuals=12):
    """Run every official row, preserving audit/inference failures in the receipt.

Reference-trace support is a descriptive proxy: visibility and full instance GT
are unavailable, so it is not instance accuracy or natural-occlusion accuracy.
"""
    import torch
    from .inference import autocast_context, decode_instances
    from .predict import load_rgb
    from .identity import attention_trace
    audit_root, out = Path(audit_root), new_directory(output)
    audit = json.loads((audit_root/"audit.json").read_text(encoding="utf-8"))
    if sample_visuals < 0:
        raise ValueError("sample_visuals must be nonnegative")
    # One representative per exact-RGB group, selected without predictions.
    representatives = {}
    for source in sorted(audit["records"], key=lambda r: r["id"]):
        if source["status"] == "rgb_and_numeric_trace_verified":
            representatives.setdefault(source["rgb_sha256"], source["id"])
    selected = {representatives[key] for key in sorted(representatives,
        key=lambda key: hashlib.sha256(("handloom_fixed_visuals_v1|"+key).encode()).hexdigest())[:sample_visuals]}
    (out/"predictions").mkdir()
    (out/"overlays").mkdir()
    records = []
    was_training = model.training
    model.eval()
    try:
        for source in audit["records"]:
            row = dict(id=source["id"], status="pending", scene_id=None)
            started = time.perf_counter()
            try:
                if source["status"] != "rgb_and_numeric_trace_verified":
                    raise ValueError("Source audit did not admit this RGB/trace")
                rgb, images, valid = load_rgb(audit_root/source["rgb_path"], model.config.canvas_size)
                if hashlib.sha256(rgb.tobytes()).hexdigest() != source["rgb_sha256"]:
                    raise ValueError("Audited RGB hash changed")
                images, valid = images.to(device), valid.to(device)
                with torch.no_grad(), autocast_context(device, amp):
                    outputs = model(images, valid)
                expected = model.expected_calls() if hasattr(model, "expected_calls") else {"encoder": 1, "decoder": 1}
                if outputs.get("calls") != expected:
                    raise ValueError("Inference call contract violated")
                result = decode_instances(outputs, images.shape[-2:], valid,
                    object_threshold=object_threshold, mask_threshold=mask_threshold)[0]
                h, w = rgb.shape[:2]
                ids = result["instances"][:h, :w]
                # Trace is first read only after the RGB-only forward and decode.
                if source.get("trace_file_sha256") and hashlib.sha256((audit_root/source["trace_path"]).read_bytes()).hexdigest() != source["trace_file_sha256"]:
                    raise ValueError("Audited numeric trace archive hash changed")
                with np.load(audit_root/source["trace_path"], allow_pickle=False) as trace_file:
                    points = trace_file["pixels_yx"]
                yy, xx = np.rint(points).astype(np.int64).T
                inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                observed = ids[yy[inside], xx[inside]]
                counts = np.bincount(observed, minlength=int(ids.max())+1)
                supported = int((observed > 0).sum())
                proxy = dict(reference_points=len(points), in_bounds_points=int(inside.sum()),
                    predicted_foreground_supported_points=supported,
                    support_fraction_in_bounds=supported/len(observed) if len(observed) else None,
                    dominant_predicted_id_fraction_among_supported=float(counts[1:].max()/supported) if supported else None,
                    warning="descriptive support only; reference trace may include hidden segments; other visible cables may lack annotation")
                if source["id"] in selected:
                    np.savez_compressed(out/"predictions"/(source["id"]+".npz"), instances=ids)
                    palette = np.random.default_rng(20261031).integers(40, 245, size=(max(int(ids.max())+1, 1), 3), dtype=np.uint8)
                    overlay = rgb.copy()
                    foreground = ids > 0
                    overlay[foreground] = (rgb[foreground].astype(float)*.45 + palette[ids[foreground]]*.55).astype(np.uint8)
                    view = Image.fromarray(overlay)
                    draw = ImageDraw.Draw(view)
                    for y, x in zip(yy[inside], xx[inside]):
                        draw.ellipse((int(x)-1, int(y)-1, int(x)+1, int(y)+1), fill=(255, 255, 255))
                    view.save(out/"overlays"/(source["id"]+".png"))
                row.update(status="predicted", native_shape=[h, w], predicted_instances=int(ids.max()),
                           calls=outputs["calls"], trace_support_proxy=proxy, attention=attention_trace(outputs),
                           rgb_sha256=source["rgb_sha256"])
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            row["seconds"] = time.perf_counter()-started
            records.append(row)
    finally:
        model.train(was_training)
    report = dict(status="completed" if all(r["status"] == "predicted" for r in records) else "completed_with_failures",
        source_commit=audit["commit"], images=len(records), failures=sum(r["status"] != "predicted" for r in records),
        forward_input="RGB and image padding validity only; traces read after forward",
        interpretation="External qualitative/trace-support diagnostic, not mask accuracy or independent-scene testing",
        unavailable_metrics=["mask_IoU", "PQ", "whole_visible_recovery", "natural_occlusion_identity_accuracy", "scene_independent_CI"],
        selected_visual_ids=sorted(selected), visual_selection="fixed hash of exact RGB; one lexicographic representative per group",
        saved_prediction_count=sum(r["status"] == "predicted" and r["id"] in selected for r in records),
        per_image=records)
    groups = {}
    for row in records:
        if row["status"] == "predicted":
            groups.setdefault(row["rgb_sha256"], []).append(row)
    group_rows = []
    for key, rows in sorted(groups.items()):
        supports = [r["trace_support_proxy"]["support_fraction_in_bounds"] for r in rows]
        group_rows.append(dict(rgb_sha256=key, records=[r["id"] for r in rows], scene_id=None,
            mean_reference_trace_support=float(np.mean(supports)) if all(v is not None for v in supports) else None))
    valid_supports = [row["mean_reference_trace_support"] for row in group_rows if row["mean_reference_trace_support"] is not None]
    report["exact_rgb_groups"] = group_rows
    report["descriptive_group_summary"] = dict(observed_unique_rgb=len(groups), expected_unique_rgb=len(representatives),
        complete=len(records) > 0 and all(r["status"] == "predicted" for r in records),
        macro_reference_trace_support=float(np.mean(valid_supports)) if valid_supports and len(valid_supports) == len(groups) and all(r["status"] == "predicted" for r in records) else None,
        interpretation="each exact-RGB group weighted equally; group is not a proven independent scene; trace visibility unknown")
    atomic_json(out/"diagnostics.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Format smoke only; omit for complete official test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-visuals", type=int, default=12)
    parser.add_argument("--audit-root", help="Run RGB-only diagnostics on an existing audited release")
    parser.add_argument("--reaudit-root", help="Recheck cached raw bytes after a reader update, preserving old receipt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", default="none", choices=("none", "bf16", "fp16"))
    parser.add_argument("--sam-repo"); parser.add_argument("--sam-checkpoint")
    args = parser.parse_args(argv)
    if args.reaudit_root:
        result = reaudit_cached(args.reaudit_root)
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))
        return 0 if not result["failures"] else 2
    if args.audit_root:
        if not args.checkpoint:
            parser.error("--audit-root requires --checkpoint")
        from .checkpoint import build_from_checkpoint, sha256
        model, checkpoint = build_from_checkpoint(args.checkpoint, device=args.device,
            sam_repo=args.sam_repo, sam_checkpoint=args.sam_checkpoint)
        config = checkpoint["train_config"]
        result = rgb_only_diagnostics(model, args.audit_root, args.output, device=args.device, amp=args.amp,
            object_threshold=config["object_threshold"], mask_threshold=config["mask_threshold"],
            sample_visuals=args.sample_visuals)
        result["checkpoint_sha256"] = sha256(args.checkpoint)
        atomic_json(Path(args.output)/"diagnostics.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, ensure_ascii=False))
        return 0 if not result["failures"] else 2
    result = audit_handloom(args.output, limit=args.limit, workers=args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))
    return 0 if not result["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
