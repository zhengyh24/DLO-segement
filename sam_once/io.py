"""Versioned, hash-checked native prediction caches, separate from supervision."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .types import CropSpec, PromptPlan, PromptSpec, ProposalRecord


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_hash(rgb):
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def prompt_from_dict(p):
    return PromptSpec(p["prompt_id"], p["crop_id"],
                      tuple(tuple(v) for v in p["positive"]),
                      tuple(tuple(v) for v in p.get("negative", [])),
                      tuple(p["box"]) if p.get("box") is not None else None,
                      p.get("metadata", {}))


def _reject_gt_metadata(value):
    """Reject explicit oracle/diagnostic provenance at every metadata depth.

    An outer GT-free flag cannot erase a nested source's contrary provenance.
    This is a cache boundary guard, not a detector for deliberately omitted GT.
    """
    stack, seen = [("metadata", value)], set()
    prohibited = {"ground_truth_used": True, "ground_truth_used_for_inference": True,
                  "diagnostic_only": True, "deployment_allowed": False}
    while stack:
        path, current = stack.pop()
        if not isinstance(current, (dict, list, tuple)) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, dict):
            for key, item in current.items():
                child = f"{path}.{key}"
                if key in prohibited and isinstance(item, (bool, np.bool_)) and bool(item) == prohibited[key]:
                    raise ValueError(f"GT-assisted/diagnostic metadata is forbidden in deployment caches: {child}")
                stack.append((child, item))
        else:
            stack.extend((f"{path}[{index}]", item) for index, item in enumerate(current))


def save_case(directory, rgb, plan, proposals, audit):
    _reject_gt_metadata({"plan_audit": plan.audit,
                         "prompt_metadata": [p.metadata for p in plan.prompts],
                         "proposal_metadata": [p.metadata for p in proposals], "audit": audit})
    plan.validate(rgb)
    known = {p.prompt_id: p for p in plan.prompts}
    crop_map = {c.crop_id: c for c in plan.crops}
    seen = set()
    for p in proposals:
        p.validate()
        if p.prompt.prompt_id not in known or p.prompt != known[p.prompt.prompt_id] or p.prompt.prompt_id in seen or p.crop != crop_map[p.prompt.crop_id]:
            raise ValueError("Proposal/prompt mapping mismatch")
        seen.add(p.prompt.prompt_id)
    out = new_directory(directory)
    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "ridge.npy", plan.ridge.astype(np.float32), allow_pickle=False)
    entries = []
    for i, p in enumerate(proposals):
        name = f"proposal_{i:04d}.npz"
        np.savez_compressed(out / name, initial=p.initial_logits.astype(np.float32), current=p.current_logits.astype(np.float32))
        entries.append({"prompt_id": p.prompt.prompt_id, "file": name, "score": float(p.score),
                        "metadata": p.metadata})
    paths = ["rgb.png", "ridge.npy"] + [p["file"] for p in entries]
    obj = {"schema": "sam_once_cache_v1", "rgb_sha256": rgb_hash(rgb),
           "plan": plan.metadata(), "proposals": entries, "audit": audit,
           "files_sha256": {p: sha256(out / p) for p in paths},
           "ground_truth_used_for_inference": False}
    write_json(out / "case.json", obj)
    return out


def load_case(directory):
    out = Path(directory)
    obj = read_json(out / "case.json")
    if obj.get("schema") != "sam_once_cache_v1" or obj.get("ground_truth_used_for_inference") is not False:
        raise ValueError("Expected a versioned GT-free SAM-once cache")
    _reject_gt_metadata(obj)
    if not {"rgb.png", "ridge.npy"}.issubset(obj["files_sha256"]):
        raise ValueError("Cache lacks required RGB/ridge integrity hashes")
    for name, digest in obj["files_sha256"].items():
        target = (out / name).resolve()
        if not target.is_relative_to(out.resolve()) or sha256(target) != digest:
            raise ValueError(f"Cache integrity failure: {name}")
    rgb = np.asarray(Image.open(out / "rgb.png").convert("RGB")).copy()
    if rgb_hash(rgb) != obj["rgb_sha256"]:
        raise ValueError("Native RGB hash mismatch")
    meta = obj["plan"]
    plan = PromptPlan([CropSpec(c["crop_id"], tuple(c["xyxy"])) for c in meta["crops"]],
                      [prompt_from_dict(p) for p in meta["prompts"]],
                      np.load(out / "ridge.npy", allow_pickle=False), meta["audit"]).validate(rgb)
    crops = {c.crop_id: c for c in plan.crops}
    prompts = {p.prompt_id: p for p in plan.prompts}
    records = []
    for row in obj["proposals"]:
        p = prompts[row["prompt_id"]]
        if row["file"] not in obj["files_sha256"]:
            raise ValueError("Unhashed proposal file")
        with np.load(out / row["file"], allow_pickle=False) as arrays:
            records.append(ProposalRecord(p, crops[p.crop_id], arrays["initial"].copy(),
                                          arrays["current"].copy(), row["score"], row["metadata"]).validate())
    if len({p.prompt.prompt_id for p in records}) != len(records):
        raise ValueError("Duplicated cached proposal")
    return rgb, plan, records, obj["audit"]
