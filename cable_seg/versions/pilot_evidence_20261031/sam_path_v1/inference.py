"""Automatic RGB-only inference and deterministic exclusive pixel ownership."""
from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
from torch.nn import functional as F


def autocast_context(device, amp):
    if amp == "none":
        return nullcontext()
    if amp not in ("bf16", "fp16"):
        raise ValueError("amp must be none, bf16 or fp16")
    if torch.device(device).type != "cuda":
        raise ValueError("Outer AMP is supported on CUDA only; CPU smoke uses amp=none")
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def padding_valid(images, targets):
    """Construct input validity from RGB dimensions, never annotation overlaps."""
    valid = torch.zeros_like(images[:, :1], dtype=torch.bool)
    for index, target in enumerate(targets):
        meta = target.get("meta", {})
        size = meta.get("original_size", meta.get("original_shape"))
        if size is None:
            raise ValueError("Dataset must expose original_size to prevent GT-validity leakage")
        height, width = map(int, size[:2])
        if height > images.shape[-2] or width > images.shape[-1] or min(height, width) < 1:
            raise ValueError("Invalid native image dimensions")
        valid[index, :, :height, :width] = True
    return valid


def decode_instances(outputs, size, valid, *, object_threshold=.5, mask_threshold=.5):
    """One pixel belongs to at most one retained query; ties favor its lower index.

    Quality is reported by the network but deliberately not multiplied into
    confidence: it is a predicted IoU regression, not an object probability.
    """
    if not 0 <= object_threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("Thresholds must lie in [0,1]")
    logits = F.interpolate(outputs["mask_logits"].float(), size=size, mode="bilinear", align_corners=False)
    probabilities = logits.sigmoid()
    presence = outputs["object_logits"].float().sigmoid()
    if not torch.isfinite(logits).all() or not torch.isfinite(presence).all():
        raise ValueError("Non-finite automatic predictions")
    results = []
    for prob, confidence, canvas_valid in zip(probabilities, presence, valid):
        eligible = (confidence[:, None, None] >= object_threshold) & (prob >= mask_threshold)
        eligible &= canvas_valid.reshape(1, *size).bool()
        scores = (prob * confidence[:, None, None]).masked_fill(~eligible, -1)
        winning_score, winners = scores.max(0)
        ids = torch.zeros(size, dtype=torch.int32, device=prob.device)
        masks, queries, confidences = [], [], []
        for query in range(prob.shape[0]):
            mask = (winners == query) & (winning_score >= 0)
            if mask.any():
                masks.append(mask.cpu().numpy())
                ids[mask] = len(masks)
                queries.append(query); confidences.append(float(confidence[query]))
        array = np.stack(masks) if masks else np.zeros((0, *size), bool)
        results.append(dict(masks=array, instances=ids.cpu().numpy(), union=array.any(0),
                            unknown=np.zeros(size, dtype=bool), queries=queries, confidences=confidences,
                            unknown_policy="zero: no separate calibrated identity-abstention head"))
    return results
