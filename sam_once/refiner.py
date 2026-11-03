"""Proposal-conditioned native ROI refinement; never invokes SAM or uses labels.

R33/R34 inspired scheduling and residual refinement, not paper reproductions.
All proposal arrays retain their own crop coordinate system and prompt identity.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .types import ProposalRecord

FEATURE_NAMES = ("red", "green", "blue", "initial_logit", "current_logit",
                 "ridge", "other_support", "conflict", "positive", "negative", "box")


@dataclass(frozen=True)
class RefinerConfig:
    roi_size: int = 256
    rounds: int = 3
    max_visits: int = 12
    max_pixels: int = 12 * 256 * 256
    mode: str = "iterative"
    device: str = "cpu"

    def validate(self):
        if self.roi_size not in (256, 512):
            raise ValueError("Only native 256/512 ROI canvases are supported")
        if not 1 <= self.rounds <= 3 or self.max_visits < 0 or self.max_pixels < 0:
            raise ValueError("Invalid refinement budget (1-3 rounds required)")
        if self.mode not in ("iterative", "fixedsingle"):
            raise ValueError("mode must be iterative or fixedsingle")
        return self


@dataclass(frozen=True)
class ROI:
    proposal_index: int
    xyxy: tuple[int, int, int, int]  # proposal-local, unpadded valid pixels
    priority: float


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.GroupNorm(8, cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                 nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class ResidualRefiner(nn.Module):
    """Small shared 16/32/64/128 U-Net, initialized to exact identity.

    The gate scales a learned residual; competing proposals are input evidence,
    never a hard subtraction. Inputs and outputs remain at native ROI resolution.
    """
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = _Block(11, 16), _Block(16, 32)
        self.enc3, self.enc4 = _Block(32, 64), _Block(64, 128)
        self.dec3, self.dec2, self.dec1 = _Block(192, 64), _Block(96, 32), _Block(48, 16)
        self.delta, self.gate = nn.Conv2d(16, 1, 1), nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Expected Bx11xHxW proposal features")
        a = self.enc1(features)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        d = self.enc4(F.avg_pool2d(c, 2))
        up = lambda t, skip: torch.cat((F.interpolate(t, size=skip.shape[-2:],
                                                     mode="bilinear", align_corners=False), skip), 1)
        x = self.dec3(up(d, c))
        x = self.dec2(up(x, b))
        x = self.dec1(up(x, a))
        # The feature stores logits / 8; undo only this fixed numeric scale.
        return features[:, 4:5] * 8.0 + torch.sigmoid(self.gate(x)) * 4.0 * torch.tanh(self.delta(x))


def clone_proposals(proposals):
    return [ProposalRecord(p.prompt, p.crop, p.initial_logits.copy(), p.current_logits.copy(),
                           float(p.score), copy.deepcopy(p.metadata)) for p in proposals]


def _prob(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -40, 40)))


def _overlap_maps(proposals, index):
    """Unlabelled co-coverage evidence; not an assertion of same identity."""
    p = proposals[index]
    x0, y0, x1, y1 = p.crop.xyxy
    support = np.zeros_like(p.current_logits, dtype=np.float32)
    for j, other in enumerate(proposals):
        if j == index:
            continue
        a, b, c, d = other.crop.xyxy
        lx, ly, rx, ry = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
        if lx < rx and ly < ry:
            target = support[ly-y0:ry-y0, lx-x0:rx-x0]
            np.maximum(target, _prob(other.current_logits[ly-b:ry-b, lx-a:rx-a]), out=target)
    return support, support * _prob(p.current_logits)


def _points_channel(shape, points):
    result = np.zeros(shape, np.float32)
    h, w = shape
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        # Small native 5x5 marks, independent of any labels.
        if 0 <= ix < w and 0 <= iy < h:
            result[max(0, iy-2):min(h, iy+3), max(0, ix-2):min(w, ix+3)] = 1
    return result


def proposal_features(rgb, proposals, ridge, index):
    """Extract HxW evidence using only image, frozen prompts and predictions."""
    p = proposals[index]
    p.validate()
    x0, y0, x1, y1 = p.crop.xyxy
    h, w = rgb.shape[:2]
    if ridge.shape != (h, w) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Ridge/crop do not share native RGB coordinates")
    support, conflict = _overlap_maps(proposals, index)
    box = np.zeros_like(p.current_logits, dtype=np.float32)
    if p.prompt.box is not None:
        a, b, c, d = p.prompt.box
        box[max(0, int(b)):min(box.shape[0], math.ceil(d)),
            max(0, int(a)):min(box.shape[1], math.ceil(c))] = 1
    channels = [rgb[y0:y1, x0:x1].astype(np.float32).transpose(2, 0, 1) / 255.0,
                p.initial_logits[None].astype(np.float32) / 8.0,
                p.current_logits[None].astype(np.float32) / 8.0,
                ridge[y0:y1, x0:x1][None].astype(np.float32), support[None], conflict[None],
                _points_channel(box.shape, p.prompt.positive)[None],
                _points_channel(box.shape, p.prompt.negative)[None], box[None]]
    features = np.concatenate(channels)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite refinement evidence")
    return features


def _starts(length, size):
    if length <= size:
        return [0]
    return sorted(set([*range(0, length-size+1, size // 2), length-size]))


def schedule_rois(rgb, proposals, ridge, config, limit):
    """Rank a native overlapping grid without reading GT or model targets.

    Uncertainty, initial/current disagreement, uncovered ridges and co-coverage
    contribute. Within one pass avoid >50% duplicate coverage per proposal.
    Different proposals remain separate even for identical spatial boxes.
    """
    config.validate()
    if limit <= 0:
        return []
    candidates = []
    side = config.roi_size
    for i, p in enumerate(proposals):
        f = proposal_features(rgb, proposals, ridge, i)
        current, initial = _prob(p.current_logits), _prob(p.initial_logits)
        gray = f[:3].mean(0)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        priority = (4 * current * (1-current) + np.abs(current-initial)
                    + np.clip(f[5], 0, 1) * (1-current) + f[7]
                    + 0.25 * (gx+gy))
        for y in _starts(priority.shape[0], side):
            for x in _starts(priority.shape[1], side):
                x1, y1 = min(x+side, priority.shape[1]), min(y+side, priority.shape[0])
                patch = priority[y:y1, x:x1]
                # Mean plus upper tail retains fine difficult structures.
                score = float(patch.mean() + 0.5 * np.quantile(patch, 0.9))
                candidates.append(ROI(i, (x, y, x1, y1), score))
    candidates.sort(key=lambda r: (-r.priority, r.proposal_index, r.xyxy[1], r.xyxy[0]))
    chosen = []
    for roi in candidates:
        x0, y0, x1, y1 = roi.xyxy
        duplicate = False
        for old in chosen:
            if old.proposal_index != roi.proposal_index:
                continue
            a, b, c, d = old.xyxy
            overlap = max(0, min(c, x1)-max(a, x0)) * max(0, min(d, y1)-max(b, y0))
            if overlap > 0.5 * min((x1-x0)*(y1-y0), (c-a)*(d-b)):
                duplicate = True
                break
        if not duplicate:
            chosen.append(roi)
            if len(chosen) >= limit:
                break
    return chosen


def extract_roi(rgb, proposals, ridge, roi, roi_size):
    """Return unresized CHW padded features and 1HW valid mask."""
    x0, y0, x1, y1 = roi.xyxy
    features = proposal_features(rgb, proposals, ridge, roi.proposal_index)[:, y0:y1, x0:x1]
    h, w = features.shape[-2:]
    if h > roi_size or w > roi_size:
        raise ValueError("ROI exceeds canvas")
    padded = np.zeros((len(FEATURE_NAMES), roi_size, roi_size), np.float32)
    padded[:, :h, :w] = features
    valid = np.zeros((1, roi_size, roi_size), np.float32)
    valid[:, :h, :w] = 1
    return padded, valid


def visit_budget(config):
    config.validate()
    return min(config.max_visits, config.max_pixels // (config.roi_size ** 2))


def round_limits(config):
    total = visit_budget(config)
    rounds = 1 if config.mode == "fixedsingle" else config.rounds
    q, r = divmod(total, rounds)
    return [q + (i < r) for i in range(rounds)]


def commit_predictions(proposals, predictions):
    """Average only overlapping visited pixels; untouched values stay exact."""
    grouped = {}
    for roi, values in predictions:
        grouped.setdefault(roi.proposal_index, []).append((roi, values))
    for i, items in grouped.items():
        p = proposals[i]
        values = np.zeros_like(p.current_logits, np.float32)
        count = np.zeros_like(p.current_logits, np.float32)
        for roi, prediction in items:
            x0, y0, x1, y1 = roi.xyxy
            values[y0:y1, x0:x1] += prediction[:y1-y0, :x1-x0]
            count[y0:y1, x0:x1] += 1
        selected = count > 0
        p.current_logits[selected] = values[selected] / count[selected]
        p.metadata["refined"] = True
        p.metadata["refinement_visits"] = int(p.metadata.get("refinement_visits", 0)) + len(items)


@torch.inference_mode()
def refine_proposals(rgb, proposals, ridge, model, config=None):
    config = (config or RefinerConfig()).validate()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected native uint8 RGB")
    if ridge.shape != rgb.shape[:2] or not np.isfinite(ridge).all():
        raise ValueError("Invalid native ridge map")
    output = clone_proposals(proposals)
    for p in output:
        p.validate()
    was_training = model.training
    model.to(config.device).eval()
    visits = []
    try:
        for iteration, limit in enumerate(round_limits(config)):
            rois = schedule_rois(rgb, output, ridge, config, limit)
            predictions = []
            # All ROIs within a round use the same snapshot; only rounds feed back.
            for roi in rois:
                features, valid = extract_roi(rgb, output, ridge, roi, config.roi_size)
                tensor = torch.from_numpy(features[None]).to(config.device)
                predicted = model(tensor)[0, 0].float().cpu().numpy()
                if not np.isfinite(predicted).all():
                    raise ValueError("Non-finite refiner output")
                predictions.append((roi, predicted))
                visits.append({"round": iteration + 1, "proposal_index": roi.proposal_index,
                               "prompt_id": output[roi.proposal_index].prompt.prompt_id,
                               "xyxy": list(roi.xyxy), "priority": roi.priority,
                               "valid_pixels": int(valid.sum()), "model_pixels": config.roi_size ** 2})
            commit_predictions(output, predictions)
    finally:
        model.train(was_training)
    audit = {"schema": "sam_once_refinement_v1", "config": asdict(config),
             "sam_calls": 0, "visits": visits, "total_visits": len(visits),
             "total_model_pixels": len(visits) * config.roi_size ** 2,
             "total_valid_pixels": sum(v["valid_pixels"] for v in visits),
             "budget_visits": visit_budget(config), "feature_names": list(FEATURE_NAMES),
             "original_proposals_retained": True, "resized": False,
             "equal_budget_note": "Compare actual visits and model pixels; candidate exhaustion can reduce usage."}
    return output, audit


def load_refiner(path, device="cpu"):
    """Load this package's tensor/basic-type checkpoint without pickle objects."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "sam_once_refiner_checkpoint_v1":
        raise ValueError("Unsupported refiner checkpoint schema")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Refiner feature contract mismatch")
    model = ResidualRefiner()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    protocol = checkpoint.get("protocol", {})
    audit = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "schema": checkpoint["schema"], "smoke_only": protocol.get("smoke_only", True),
             "training_manifest_sha256": protocol.get("isolation", {}).get("manifest_sha256"),
             "actual_successful_updates": checkpoint.get("actual_successful_updates", 0),
             "checkpoint_selection": checkpoint.get("checkpoint_selection"),
             "instance_promotion": checkpoint.get("instance_promotion", False),
             "parameter_count": sum(p.numel() for p in model.parameters())}
    return model, audit
