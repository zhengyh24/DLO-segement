"""Export unmodified native proposal evidence without assigning instance IDs.

The max of overlapping logits is a foreground evidence union. It is neither a
calibrated probability nor an assertion that overlapping prompts share identity.
Independent proposals remain in the caller's initial proposal cache.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


UNOBSERVED_LOGIT = -1024.


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_raw_metadata(metadata):
    """A zero residual does not erase an explicit refinement marker."""
    stack, seen = [metadata], set()
    while stack:
        item = stack.pop()
        if not isinstance(item, (dict, list, tuple)) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, dict):
            if item.get("refined", False) or item.get("refinement_visits", 0):
                raise ValueError("Coarse export accepts only raw, unrefined proposals")
            stack.extend(item.values())
        else:
            stack.extend(item)


def export_coarse(out, rgb, records, plan, audit):
    """Write coarse evidence and populate ``audit['coarse_stage']``.

    Missing proposals are allowed and counted, including an empty result. Only
    pixels inside an actually returned proposal crop are observed. A zero pixel
    in the binary PNG outside that coverage is absence of evidence, not a model
    background prediction. Existing output files are never overwritten.
    """
    if not isinstance(audit, dict):
        raise ValueError("Coarse export audit must be a dictionary")
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected native uint8 RGB HxWx3")
    h, w = rgb.shape[:2]
    if min(h, w) <= 0:
        raise ValueError("Expected a nonempty native image")
    plan.validate(rgb)
    records = tuple(records)
    prompts = {prompt.prompt_id: prompt for prompt in plan.prompts}
    crops = {crop.crop_id: crop for crop in plan.crops}
    seen = set()
    for record in records:
        record.validate()
        prompt_id = record.prompt.prompt_id
        if (prompt_id not in prompts or record.prompt != prompts[prompt_id]
                or prompt_id in seen or record.crop != crops[record.prompt.crop_id]):
            raise ValueError("Coarse proposal/prompt/crop mapping mismatch")
        seen.add(prompt_id)
        _assert_raw_metadata(record.metadata)
        _assert_raw_metadata(record.prompt.metadata)
        if record.initial_logits.dtype.kind != "f" or record.current_logits.dtype.kind != "f":
            raise ValueError("Coarse logits must be real floating point arrays")
        if not np.array_equal(record.initial_logits, record.current_logits):
            raise ValueError("Coarse export requires initial_logits == current_logits")

    out = Path(out)
    paths = {"foreground": out / "coarse_foreground.png",
             "evidence": out / "coarse_evidence.npz",
             "overlay": out / "coarse_overlay.png"}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Coarse export will not overwrite {path}")
    out.mkdir(parents=True, exist_ok=True)

    # Start at -inf so genuine observed logits below the output sentinel remain
    # unchanged. The finite sentinel is assigned only to unobserved pixels.
    dtype = np.result_type(np.float32, *(record.current_logits.dtype for record in records))
    max_logits = np.full((h, w), -np.inf, dtype=dtype)
    observed = np.zeros((h, w), dtype=bool)
    for record in records:
        x0, y0, x1, y1 = record.crop.xyxy
        target = max_logits[y0:y1, x0:x1]
        np.maximum(target, record.current_logits, out=target)
        observed[y0:y1, x0:x1] = True
    max_logits[~observed] = UNOBSERVED_LOGIT
    foreground = observed & (max_logits > 0)

    with paths["foreground"].open("xb") as stream:
        Image.fromarray(foreground.astype(np.uint8) * 255).save(stream, format="PNG")
    with paths["evidence"].open("xb") as stream:
        np.savez_compressed(stream, max_logits=max_logits, observed_coverage=observed)
    overlay = rgb.copy()
    tint = np.array([40., 220., 100.], np.float32)
    overlay[foreground] = np.rint(rgb[foreground].astype(np.float32) * .55 + tint * .45).astype(np.uint8)
    with paths["overlay"].open("xb") as stream:
        Image.fromarray(overlay).save(stream, format="PNG")

    stage = {"schema": "sam_once_coarse_v1", "status": "ok" if records else "empty",
             "aggregation": "pixelwise_max_logits_in_native_crop_coordinates",
             "threshold": "max_logits > 0 within observed_coverage",
             "logit_semantics": "uncalibrated foreground evidence; not probability",
             "proposal_count": len(records), "planned_prompt_count": len(plan.prompts),
             "missing_prompt_count": len(plan.prompts) - len(records),
             "no_instance_assignment": True, "instance_assignment": "none",
             "raw_initial_equals_current": True, "refinement_applied": False,
             "foreground_pixels": int(foreground.sum()), "observed_pixels": int(observed.sum()),
             "unobserved_pixels": int((~observed).sum()), "unobserved_logit": UNOBSERVED_LOGIT,
             "coverage_semantics": "at least one returned proposal crop; unobserved is not predicted background",
             "native_size": [w, h], "evidence_paths": {key: path.name for key, path in paths.items()},
             "files_sha256": {path.name: _sha256(path) for path in paths.values()},
             "independent_proposals": "retained separately by the caller in the initial proposal cache"}
    audit["coarse_stage"] = stage
    return stage
