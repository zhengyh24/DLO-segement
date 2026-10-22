"""RGB-only native-window instance tracking. No labels or legacy foreground input.

This is an immediate-commit prototype, not the proposed delayed joint decoder.
Run from the project root: python -m curvi_instance_v1.predict --help
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F


DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
DIRECTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass(frozen=True)
class InferenceConfig:
    tile: int = 256
    stride: int = 128
    max_seeds: int = 64
    max_steps: int = 32
    max_calls: int = 2048
    seed_threshold: float = 0.5
    mask_threshold: float = 0.5
    exit_threshold: float = 0.5
    conflict_margin: float = 0.1
    seed_sigma: float = 2.0
    peak_radius: int = 8
    duplicate_iou: float = 0.9

    def validate(self) -> None:
        for key in ("tile", "stride", "max_seeds", "max_steps", "max_calls"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.tile < 16 or self.stride > self.tile:
            raise ValueError("Require tile >= 16 and stride <= tile")
        if not np.isfinite(self.seed_sigma) or self.seed_sigma <= 0 or self.peak_radius < 1:
            raise ValueError("Require finite seed_sigma > 0 and peak_radius >= 1")
        for key in ("seed_threshold", "mask_threshold", "exit_threshold"):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f"{key} must lie strictly between 0 and 1")
        if not 0 <= self.conflict_margin <= 1:
            raise ValueError("conflict_margin must lie in [0, 1]")
        if not 0.5 < self.duplicate_iou <= 1:
            raise ValueError("duplicate_iou must lie in (0.5, 1]")


def _starts(length: int, tile: int, stride: int) -> list[int]:
    last = max(0, length - tile)
    values = list(range(0, last + 1, stride))
    if values[-1] != last:
        values.append(last)
    return values


def _origin(y: int, x: int, h: int, w: int, tile: int) -> tuple[int, int]:
    return min(max(0, y), max(0, h-tile)), min(max(0, x), max(0, w-tile))


def _crop(rgb: np.ndarray, y: int, x: int, tile: int,
          device: torch.device) -> tuple[torch.Tensor, int, int]:
    piece = rgb[y:y+tile, x:x+tile]
    hh, ww = piece.shape[:2]
    tensor = torch.from_numpy(piece.copy()).permute(2, 0, 1).float().div_(255)
    tensor = F.pad(tensor, (0, tile-ww, 0, tile-hh))
    return tensor[None].to(device), hh, ww


def _probability(value: torch.Tensor, expected: tuple[int, ...], name: str) -> np.ndarray:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name}: expected {expected}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"Nonfinite {name}")
    return value.float().sigmoid().detach().cpu().numpy()


def _local_peaks(heat: np.ndarray, observed: np.ndarray, cfg: InferenceConfig
                 ) -> tuple[list[tuple[float, int, int]], int]:
    """Deterministic max-pool peaks; suppress nearby plateau points, no GT."""
    value = torch.from_numpy(heat)[None, None]
    radius = cfg.peak_radius
    maximum = F.max_pool2d(value, 2*radius+1, stride=1, padding=radius)[0, 0].numpy()
    yy, xx = np.nonzero(observed & (heat >= cfg.seed_threshold) & (heat == maximum))
    if not len(yy):
        return [], 0
    order = np.lexsort((xx, yy, -heat[yy, xx]))
    blocked = np.zeros_like(observed, dtype=bool)
    selected: list[tuple[float, int, int]] = []
    # Need only max_seeds+1 to establish that the seed budget truncated discovery.
    for index in order:
        y, x = int(yy[index]), int(xx[index])
        if blocked[y, x]:
            continue
        selected.append((float(heat[y, x]), y, x))
        if len(selected) > cfg.max_seeds:
            break
        blocked[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = True
    return selected[:cfg.max_seeds], len(selected)


def _compete(best: np.ndarray, second: np.ndarray, owner: np.ndarray,
             scores: np.ndarray, instance_id: int) -> None:
    """Top two distinct instance scores; each completed track enters once."""
    wins = scores > best
    second[wins] = best[wins]
    best[wins] = scores[wins]
    owner[wins] = instance_id
    loses = ~wins
    second[loses] = np.maximum(second[loses], scores[loses])


@torch.inference_mode()
def predict_array(model: Any, rgb: np.ndarray, config: InferenceConfig | None = None,
                  device: str | torch.device = "cpu") -> dict[str, Any]:
    """Predict a native HWC uint8 RGB array using only model outputs as history.

    ``model.encode`` returns (feature_dict, seed_logits). Forward accepts RGB,
    seed, history, query, anchor and RGB spatial-validity, and returns mask/exit
    logits plus state. Spatial validity describes only image bounds, not labels.
    The caller owns model construction/loading; this function never reads a GT.
    """
    cfg = config or InferenceConfig()
    cfg.validate()
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a native uint8 HWC array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("rgb must have shape H,W,3 with nonempty axes")
    device = torch.device(device)
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    h, w = rgb.shape[:2]
    heat_sum = np.zeros((h, w), dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.uint32)
    best = np.zeros((h, w), dtype=np.float32)
    second = np.zeros_like(best)
    owner = np.zeros((h, w), dtype=np.uint32)
    crop_log: list[dict[str, Any]] = []
    track_log: list[dict[str, Any]] = []
    reasons: set[str] = set()
    calls = 0
    input_pixels = 0
    native_pixels = 0

    def charge(phase: str, y: int, x: int, hh: int, ww: int,
               track: int | None = None) -> dict[str, Any]:
        nonlocal calls, input_pixels, native_pixels
        calls += 1
        input_pixels += cfg.tile * cfg.tile
        native_pixels += hh * ww
        entry = {"call": calls, "phase": phase, "y": y, "x": x,
                 "valid_h": hh, "valid_w": ww, "input_h": cfg.tile,
                 "input_w": cfg.tile, "instance_attempt": track}
        crop_log.append(entry)
        return entry

    positions = [(y, x) for y in _starts(h, cfg.tile, cfg.stride)
                 for x in _starts(w, cfg.tile, cfg.stride)]
    discovery_done = 0
    for y, x in positions:
        if calls >= cfg.max_calls:
            reasons.add("network_call_budget_during_discovery")
            break
        tensor, hh, ww = _crop(rgb, y, x, cfg.tile, device)
        charge("discovery", y, x, hh, ww)
        encoded = model.encode(tensor)
        if not isinstance(encoded, tuple) or len(encoded) != 2:
            raise ValueError("encode must return (feature_dict, seed_logits)")
        prob = _probability(encoded[1], (1, 1, cfg.tile, cfg.tile), "seed_logits")[0, 0]
        heat_sum[y:y+hh, x:x+ww] += prob[:hh, :ww]
        coverage[y:y+hh, x:x+ww] += 1
        discovery_done += 1
        del encoded
    heat = heat_sum / np.maximum(coverage, 1)
    heat = heat.astype(np.float32)
    seeds, peak_count_lower_bound = _local_peaks(heat, coverage > 0, cfg)
    if peak_count_lower_bound > cfg.max_seeds:
        reasons.add("seed_budget")
    # Compact masks are used only for whole-mask duplicate rejection, not merging.
    accepted: list[tuple[int, np.ndarray, int]] = []
    attempted = 0
    suppressed = 0
    duplicates = 0
    processed_seeds = 0
    for seed_score, sy, sx in seeds:
        confident_owner = (best[sy, sx] >= cfg.mask_threshold and
                           (second[sy, sx] < cfg.mask_threshold or
                            (best[sy, sx] > second[sy, sx] and
                             best[sy, sx]-second[sy, sx] >= cfg.conflict_margin)))
        if confident_owner:
            processed_seeds += 1
            suppressed += 1
            track_log.append({"seed_y": sy, "seed_x": sx, "seed_score": seed_score,
                              "status": "seed_already_owned", "owner": int(owner[sy, sx])})
            continue
        if calls >= cfg.max_calls:
            reasons.add("network_call_budget_during_tracking")
            break
        processed_seeds += 1
        attempted += 1
        track_score = np.zeros((h, w), dtype=np.float32)
        query = anchor = None
        first = _origin(sy-cfg.tile//2, sx-cfg.tile//2, h, w, cfg.tile)
        frontier: list[tuple[float, int, int]] = [(-seed_score, *first)]
        queued = {first}
        visited: set[tuple[int, int]] = set()
        steps = 0
        while frontier and steps < cfg.max_steps and calls < cfg.max_calls:
            _, y, x = heapq.heappop(frontier)
            if (y, x) in visited:
                continue
            visited.add((y, x))
            tensor, hh, ww = _crop(rgb, y, x, cfg.tile, device)
            history = torch.zeros((1, 1, cfg.tile, cfg.tile), device=device)
            spatial_valid = torch.zeros_like(history)
            spatial_valid[0, 0, :hh, :ww] = 1
            history[0, 0, :hh, :ww] = torch.from_numpy(
                (track_score[y:y+hh, x:x+ww] >= cfg.mask_threshold).astype(np.float32)
            ).to(device)
            seed = torch.zeros_like(history)
            if steps == 0:
                cy, cx = sy-y, sx-x
                yy, xx = torch.meshgrid(torch.arange(hh, device=device),
                                        torch.arange(ww, device=device), indexing="ij")
                seed[0, 0, :hh, :ww] = torch.exp(
                    -((yy-cy).float()**2+(xx-cx).float()**2)/(2*cfg.seed_sigma**2))
            entry = charge("tracking", y, x, hh, ww, attempted)
            result = model(tensor, seed, history, query=query, anchor=anchor,
                           spatial_valid=spatial_valid)
            prob = _probability(result["mask_logits"],
                                (1, 1, cfg.tile, cfg.tile), "mask_logits")[0, 0]
            exits = _probability(result["exit_logits"], (1, 8), "exit_logits")[0]
            new_query, new_anchor = result["query"], result["anchor"]
            if new_query.ndim != 2 or new_query.shape[0] != 1 or new_anchor.shape != new_query.shape:
                raise ValueError("query and anchor must have matching B=1,D shapes")
            if not torch.isfinite(new_query).all() or not torch.isfinite(new_anchor).all():
                raise ValueError("Nonfinite query/anchor")
            if anchor is not None and not torch.equal(anchor, new_anchor):
                raise ValueError("Model changed the persistent anchor")
            query, anchor = new_query.detach(), new_anchor.detach()
            region = track_score[y:y+hh, x:x+ww]
            np.maximum(region, prob[:hh, :ww], out=region)
            entry["exit_probabilities"] = [float(v) for v in exits]
            entry["first_seed_prompt"] = steps == 0
            steps += 1
            for direction, (dy, dx) in enumerate(DIRECTIONS):
                if exits[direction] < cfg.exit_threshold:
                    continue
                # Do not clip an outward step into a different edge-parallel move.
                if ((dy < 0 and y == 0) or (dx < 0 and x == 0) or
                        (dy > 0 and y+cfg.tile >= h) or
                        (dx > 0 and x+cfg.tile >= w)):
                    continue
                nxt = _origin(y+dy*cfg.stride, x+dx*cfg.stride, h, w, cfg.tile)
                if nxt not in queued and nxt not in visited:
                    heapq.heappush(frontier, (-float(exits[direction]), *nxt))
                    queued.add(nxt)
        track_truncated = bool(frontier)
        if track_truncated:
            reasons.add("instance_step_budget" if steps >= cfg.max_steps else
                        "network_call_budget_during_tracking")
        mask = track_score >= cfg.mask_threshold
        area = int(mask.sum())
        duplicate_of = None
        duplicate_score = None
        if area:
            for prior_id, packed, prior_area in accepted:
                # The area ratio bounds IoU and avoids needless unpacking.
                if min(area, prior_area)/max(area, prior_area) < cfg.duplicate_iou:
                    continue
                old_mask = np.unpackbits(packed, count=h*w).reshape(h, w).astype(bool)
                intersection = int(np.logical_and(mask, old_mask).sum())
                iou = intersection / (area+prior_area-intersection)
                if iou >= cfg.duplicate_iou:
                    duplicate_of, duplicate_score = prior_id, iou
                    break
        instance_id = None
        if not area:
            status = "empty_prediction"
        elif duplicate_of is not None:
            status = "duplicate_mask_rejected"
            duplicates += 1
        else:
            instance_id = len(accepted)+1
            accepted.append((instance_id, np.packbits(mask), area))
            # Below-threshold evidence never competes for visible pixel ownership.
            track_score[~mask] = 0
            _compete(best, second, owner, track_score, instance_id)
            status = "accepted"
        track_log.append({"attempt": attempted, "seed_y": sy, "seed_x": sx,
                          "seed_score": seed_score, "status": status,
                          "instance_id": instance_id, "steps": steps, "area": area,
                          "truncated": track_truncated, "pending_origins": len(frontier),
                          "duplicate_of": duplicate_of, "duplicate_iou": duplicate_score})
    visible = best >= cfg.mask_threshold
    unknown = visible & (second >= cfg.mask_threshold) & ((best-second) < cfg.conflict_margin)
    # A zero margin still marks exact ties as ambiguous, rather than seed-order wins.
    unknown |= visible & (second >= cfg.mask_threshold) & (best == second)
    ids = owner.copy()
    ids[~visible | unknown] = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    audit = {
        "schema": 1, "method": "native_query_immediate_commit_v1",
        "native_shape_hw": [h, w], "config": asdict(cfg), "device": str(device),
        "network_calls": calls, "crop_calls": calls,
        "discovery_calls": discovery_done, "planned_discovery_calls": len(positions),
        "tracking_calls": calls-discovery_done, "input_pixels": input_pixels,
        "native_crop_pixels_with_revisits": native_pixels,
        "padding_input_pixels": input_pixels-native_pixels,
        "discovery_unique_pixels": int((coverage > 0).sum()),
        "seednum": len(seeds), "peak_count_lower_bound": peak_count_lower_bound,
        "processed_seed_count": processed_seeds, "instance_attempts": attempted,
        "seed_suppressed_count": suppressed, "duplicate_rejected_count": duplicates,
        "accepted_track_count": len(accepted),
        "output_nonempty_instance_count": len(np.unique(ids[ids > 0])),
        "visible_pixels": int(visible.sum()), "unknown_pixels": int(unknown.sum()),
        "truncated": bool(reasons), "truncation_reasons": sorted(reasons),
        "elapsed_seconds": time.perf_counter()-started,
        "direction_order": DIRECTION_NAMES, "crop_log": crop_log, "tracks": track_log,
        "semantics": {
            "id_zero": "background or ambiguous visible pixel; consult unknown.png",
            "unknown": "At least two distinct accepted tracks exceed mask threshold and their top probability gap is below conflict_margin, or exactly tied.",
            "visible": "Union of accepted instance masks, including ambiguous pixels.",
            "history": "Thresholded maximum of this instance's earlier predicted local probabilities; zero in padding.",
            "spatial_valid": "One on RGB pixels inside this native crop, zero only on padding; no label-derived validity enters the model.",
            "query": "One evolving query and one fixed initial anchor per track; shared across its queued windows.",
            "merge": "No contact-based ID merge; high whole-mask IoU rejects a duplicate track without adding its pixels.",
            "pixels": "Every discovery encode and tracking forward counts tile squared input pixels, including padding and repeat visits.",
            "thresholds": "Prototype defaults, not validation-calibrated confidence or proven identity uncertainty.",
        },
    }
    return {"instance_ids": ids, "visible": visible, "unknown": unknown,
            "seed_heatmap": heat, "discovery_coverage": coverage, "audit": audit}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device, allow_smoke: bool = False):
    from .model import InstanceTracker
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != 1:
        raise ValueError("Expected checkpoint schema=1")
    kind = checkpoint.get("kind")
    steps = checkpoint.get("optimizer_steps")
    if kind not in ("pilot", "smoke") or not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("Checkpoint requires kind='pilot'|'smoke' and nonnegative integer optimizer_steps")
    if (kind == "smoke" or steps == 0) and not allow_smoke:
        raise ValueError("Smoke/untrained checkpoint refused; use --allow-smoke only for software checks")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Checkpoint requires model_config and model state_dict")
    model = InstanceTracker(**model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {"kind": kind, "optimizer_steps": steps, "model_config": model_config,
                "smoke_override": bool(allow_smoke),
                "software_check_only": kind == "smoke" or steps == 0}
    return model.to(device).eval(), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--exif-policy", choices=("raw", "transpose"), default=None,
                        help="Required for nontrivial EXIF orientation; no guessing from labels")
    for name in ("tile", "stride", "max_seeds", "max_steps", "max_calls", "peak_radius"):
        parser.add_argument("--"+name.replace("_", "-"), type=int,
                            default=getattr(InferenceConfig(), name))
    for name in ("seed_threshold", "mask_threshold", "exit_threshold", "conflict_margin", "duplicate_iou", "seed_sigma"):
        parser.add_argument("--"+name.replace("_", "-"), type=float,
                            default=getattr(InferenceConfig(), name))
    args = parser.parse_args()
    cfg = InferenceConfig(**{name: getattr(args, name) for name in asdict(InferenceConfig())})
    cfg.validate()
    if args.output.exists():
        raise FileExistsError(f"Output directory must be new: {args.output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, checkpoint_meta = load_checkpoint(args.checkpoint, device, args.allow_smoke)
    with Image.open(args.image) as image:
        orientation = int(image.getexif().get(274, 1))
        source_size = list(image.size)
        source_mode = image.mode
        if orientation != 1 and args.exif_policy is None:
            raise ValueError(f"EXIF orientation={orientation}; explicitly choose --exif-policy raw or transpose")
        policy = args.exif_policy or "raw"
        if policy == "transpose":
            image = ImageOps.exif_transpose(image)
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = predict_array(model, rgb, cfg, device)
        ids = result["instance_ids"]
        np.save(args.output / "ID.npy", ids, allow_pickle=False)
        wrote_id_png = int(ids.max()) <= 65535
        if wrote_id_png:
            Image.fromarray(ids.astype(np.uint16)).save(args.output / "ID.png")
        for name in ("visible", "unknown"):
            Image.fromarray(result[name].astype(np.uint8)*255).save(args.output / f"{name}.png")
        np.save(args.output / "seed_heatmap.npy", result["seed_heatmap"], allow_pickle=False)
        Image.fromarray((result["discovery_coverage"] > 0).astype(np.uint8)*255).save(args.output / "discovery_coverage.png")
        audit = result["audit"]
        audit.update({"status": "completed", "checkpoint": str(args.checkpoint.resolve()),
                      "checkpoint_sha256": _sha(args.checkpoint), "checkpoint_metadata": checkpoint_meta,
                      "image": str(args.image.resolve()), "image_sha256": _sha(args.image),
                      "source_size_wh": source_size, "source_mode": source_mode,
                      "exif_orientation": orientation, "exif_policy": policy,
                      "rgb_resized": False, "id_png_written": wrote_id_png,
                      "id_dtype": str(ids.dtype),
                      "source_sha256": {name: _sha(Path(__file__).parent / name)
                                        for name in ("predict.py", "model.py")}})
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "network_calls": audit["network_calls"],
                          "instances": audit["output_nonempty_instance_count"],
                          "truncated": audit["truncated"], "software_check_only": checkpoint_meta["software_check_only"]}))
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "error": str(exc),
                    "config": asdict(cfg), "checkpoint_metadata": checkpoint_meta}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
