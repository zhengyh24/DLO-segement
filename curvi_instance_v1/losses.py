"""Padding-aware supervised losses for the first direct-instance prototype.

``tracking_loss(out, batch)`` returns a differentiable scalar and a dictionary
of detached Python scalar diagnostics. Required batch keys: target, valid,
other, seed_target (Bx1xHxW), and exits (Bx8, N/NE/E/SE/S/SW/W/NW).
``other`` marks visible pixels belonging to other instances, not generic
background. Pixels also labeled as target are excluded from this extra negative
term, so ambiguous overlapping annotations are not contradictory supervision.

The fixed prototype objective is mask BCE + mask Dice + 0.5 other-instance
negative BCE + 0.5 (seed BCE + seed Dice) + 0.25 exit BCE. These coefficients
are explicit starting values, not tuned or validated research findings. Every
pixel loss, including both sides of Dice, excludes padding. An all-padding
sample contributes neither pixel loss nor exit loss. Empty targets and batches
with no valid pixels remain finite and differentiable.
"""

import torch
from torch.nn import functional as F


def _masked_bce(logits, target, valid):
    # torch.where before BCE also prevents NaN/Inf padding from poisoning the
    # reduction. Multiplication by zero only after BCE would not achieve that.
    keep = valid > 0
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    safe_target = torch.where(keep, target, torch.zeros_like(target))
    pixel_loss = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    return (pixel_loss * valid).sum() / valid.sum().clamp_min(1)


def _masked_dice(logits, target, valid):
    keep = valid > 0
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    safe_target = torch.where(keep, target, torch.zeros_like(target))
    prediction = safe_logits.sigmoid() * valid
    target = safe_target * valid
    axes = (1, 2, 3)
    numerator = 2 * (prediction * target).sum(axes) + 1.0
    denominator = prediction.sum(axes) + target.sum(axes) + 1.0
    per_image = 1 - numerator / denominator
    nonempty = (valid.sum(axes) > 0).to(logits.dtype)
    return (per_image * nonempty).sum() / nonempty.sum().clamp_min(1)


def tracking_loss(out, batch):
    """Return ``(loss, metrics)`` with strictly ignored invalid/padded pixels.

    Valid is interpreted as a binary mask. Target and seed_target may be soft
    [0, 1] labels; other is a binary support mask. Inputs are expected to be on
    the same device as model outputs. Metrics are local batch reductions.
    """
    logits = out["mask_logits"]
    seed_logits = out["seed_logits"]
    exit_logits = out["exit_logits"]
    if logits.ndim != 4 or logits.shape[1] != 1 or seed_logits.shape != logits.shape:
        raise ValueError("mask_logits and seed_logits must have matching Bx1xHxW shapes")
    for name in ("target", "valid", "other", "seed_target"):
        if batch[name].shape != logits.shape:
            raise ValueError(f"batch[{name!r}] must match mask logits")
        if batch[name].device != logits.device:
            raise ValueError(f"batch[{name!r}] must be on the model output device")
    if exit_logits.shape != (logits.shape[0], 8) or batch["exits"].shape != exit_logits.shape:
        raise ValueError("exit logits and targets must have shape Bx8")
    if batch["exits"].device != logits.device:
        raise ValueError("exit targets must be on the model output device")

    # Evaluate reductions in float32 even under automatic mixed precision.
    logits, seed_logits, exit_logits = logits.float(), seed_logits.float(), exit_logits.float()
    valid = (batch["valid"] > 0).to(logits.dtype)
    target = torch.where(valid.bool(), batch["target"].float(), torch.zeros_like(logits))
    seed_target = torch.where(valid.bool(), batch["seed_target"].float(), torch.zeros_like(logits))
    other = (batch["other"] > 0).to(logits.dtype) * valid * (target <= 0).to(logits.dtype)

    mask_bce = _masked_bce(logits, target, valid)
    mask_dice = _masked_dice(logits, target, valid)
    other_bce = _masked_bce(logits, torch.zeros_like(target), other)
    seed_bce = _masked_bce(seed_logits, seed_target, valid)
    seed_dice = _masked_dice(seed_logits, seed_target, valid)
    exit_valid = (valid.sum((1, 2, 3)) > 0).to(logits.dtype)[:, None].expand_as(exit_logits)
    exit_bce = _masked_bce(exit_logits, batch["exits"].float(), exit_valid)
    loss = mask_bce + mask_dice + 0.5 * other_bce + 0.5 * (seed_bce + seed_dice) + 0.25 * exit_bce
    pieces = {
        "loss": loss,
        "mask_bce": mask_bce,
        "mask_dice": mask_dice,
        "other_bce": other_bce,
        "seed_bce": seed_bce,
        "seed_dice": seed_dice,
        "exit_bce": exit_bce,
        "valid_pixels": valid.sum(),
        "other_pixels": other.sum(),
    }
    return loss, {name: float(value.detach().cpu()) for name, value in pieces.items()}
