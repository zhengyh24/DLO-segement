"""Native, CPU-only geometric prompt planning (R11/R13/R15-inspired).

The ridge map and point groups are geometric hypotheses, not cable probabilities
or known instance identities. This module never imports SAM or label loaders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .types import CropSpec, PromptPlan, PromptSpec


@dataclass(frozen=True)
class PlannerConfig:
    canvas_size: int = 1008
    overlap: int = 256
    max_prompts_per_crop: int = 6
    seed: int = 20261022
    selection: str = "iterative"
    channels: tuple[str, ...] = ("gray", "lab_a", "lab_b")
    scales: tuple[float, ...] = (1., 4., 9., 25., 49., 100., 196., 324.)
    ridge_threshold: float = .15
    min_fragment_pixels: int = 8
    min_contrast: float = 5.
    junction_radius: int = 3
    max_negative_points: int = 2

    def validate(self):
        if self.canvas_size < 8 or not 0 <= self.overlap < self.canvas_size:
            raise ValueError("Expected canvas >= 8 and 0 <= overlap < canvas")
        if self.max_prompts_per_crop < 0 or self.min_fragment_pixels < 2:
            raise ValueError("Invalid prompt/fragment budget")
        if self.selection not in ("fixed", "iterative"):
            raise ValueError("selection must be fixed or iterative")
        if not self.channels or any(c not in ("gray", "lab_a", "lab_b") for c in self.channels):
            raise ValueError("Unknown geometric scalar channel")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Duplicate scalar channel")
        if len(self.scales) < 2 or any(not np.isfinite(v) or v <= 0 for v in self.scales):
            raise ValueError("GeoMPrompt requires at least two positive finite scales")
        if any(b <= a for a, b in zip(self.scales, self.scales[1:])):
            raise ValueError("Scales must be strictly increasing")
        if not 0 < self.ridge_threshold < 1 or self.junction_radius < 1:
            raise ValueError("Invalid ridge threshold/junction radius")
        if not 0 <= self.max_negative_points <= 2 or self.min_contrast < 0:
            raise ValueError("Invalid negative budget/contrast")
        return self


def _positions(length, size, stride):
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def native_crops(width: int, height: int, config=PlannerConfig()):
    """Full deterministic native coverage; padding belongs to the SAM adapter."""
    config.validate()
    if width <= 0 or height <= 0:
        raise ValueError("Image must be nonempty")
    stride = config.canvas_size - config.overlap
    return [CropSpec(f"r{r:03d}_c{c:03d}", (x, y, min(width, x + config.canvas_size),
                                                     min(height, y + config.canvas_size)))
            for r, y in enumerate(_positions(height, config.canvas_size, stride))
            for c, x in enumerate(_positions(width, config.canvas_size, stride))]


def _disk(radius):
    radius = int(max(1, math.ceil(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def _normalize_response(tensor):
    raw = np.maximum(np.asarray(tensor, np.float32), 0)
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite GeoMPrompt response")
    response = raw.max(axis=2)
    values = response[response > 0]
    scale = float(np.quantile(values, .95)) if values.size else 0.
    return (np.clip(response / scale, 0, 1).astype(np.float32) if scale > 0
            else np.zeros(response.shape, np.float32)), raw.argmax(axis=2)


def _scan_tile(rgb, config):
    """Use the installed GeoMPrompt implementation; never replace it with a stub."""
    try:
        from geomprompt.ridges.ridge_prompts import RidgePrompts
    except ImportError as exc:
        raise RuntimeError("GeoMPrompt is required in the active Python environment") from exc
    h, w = rgb.shape[:2]
    combined = np.zeros((h, w), np.float32)
    width_map = np.ones((h, w), np.float32)
    sources = np.zeros((h, w), np.int16)
    audit = {"channels": [], "failures": [], "warnings": [], "source_names": [],
             "boundary_policy": "GeoMPrompt native boundary suppression; no rescaling or padding"}
    if min(h, w) < 8:
        audit["failures"].append({"reason": "native_crop_too_small_for_ridge_detection"})
        return combined, width_map, sources, audit
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    scalar = {"gray": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
              "lab_a": lab[:, :, 1], "lab_b": lab[:, :, 2]}
    widths = np.asarray(RidgePrompts.scale_to_width(np.asarray(config.scales)), np.float32)
    for channel in config.channels:
        if int(np.ptp(scalar[channel])) < 2:
            audit["channels"].append({"channel": channel, "status": "constant"})
            continue
        try:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always", RuntimeWarning)
                detector = RidgePrompts(scalar[channel], scales=np.asarray(config.scales),
                                       method="sq-pcd", pool=False)
            for message in sorted({str(row.message) for row in observed}):
                audit["warnings"].append({"channel": channel, "message": message})
            for polarity in ("ridge", "valley"):
                response, scale_ids = _normalize_response(getattr(detector, polarity + "_tensor"))
                name = channel + ":" + polarity
                audit["source_names"].append(name)
                update = response > combined
                combined[update] = response[update]
                width_map[update] = widths[scale_ids[update]]
                sources[update] = len(audit["source_names"])
            audit["channels"].append({"channel": channel, "status": "ok"})
        except (ValueError, IndexError, FloatingPointError, ZeroDivisionError) as exc:
            audit["failures"].append({"channel": channel, "reason": type(exc).__name__,
                                       "message": str(exc)})
    return combined, width_map, sources, audit


def _ordered_fragment(coords):
    """Order an unbranched 8-neighbour fragment; never bridge disconnected pixels."""
    pixels = {tuple(map(int, p)) for p in coords}
    neighbors = {p: sorted((p[0] + dy, p[1] + dx) for dy in (-1, 0, 1)
                          for dx in (-1, 0, 1) if (dy or dx)
                          and (p[0] + dy, p[1] + dx) in pixels) for p in pixels}
    if any(len(n) > 2 for n in neighbors.values()):
        return None
    ends = sorted(p for p, n in neighbors.items() if len(n) <= 1)
    current = ends[0] if ends else min(pixels)
    ordered, visited = [], set()
    while current not in visited:
        ordered.append(current)
        visited.add(current)
        remaining = [n for n in neighbors[current] if n not in visited]
        if not remaining:
            break
        current = remaining[0]
    return np.asarray(ordered, np.int32) if len(ordered) == len(pixels) else None


def _fragment_candidates(rgb, ridge, width_map, source_map, source_names, config):
    active = ridge >= config.ridge_threshold
    skeleton = skeletonize(active)
    degree = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U,
                         np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - skeleton
    junctions = skeleton & (degree > 2)
    junction_zone = cv2.dilate(junctions.astype(np.uint8), _disk(config.junction_radius)) > 0
    pieces = skeleton & ~junction_zone
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pieces.astype(np.uint8), 8)
    candidates = []
    discarded = {"short": 0, "low_contrast": 0, "non_path": 0, "short_single_point_hypotheses": 0}
    color = rgb.astype(np.float32)
    for ident in range(1, count):
        length = int(stats[ident, cv2.CC_STAT_AREA])
        if length < 2:
            discarded["short"] += 1
            continue
        short = length < config.min_fragment_pixels
        bx, by, bw, bh = map(int, stats[ident, :4])
        coords = np.column_stack(np.where(labels[by:by + bh, bx:bx + bw] == ident))
        coords += np.array([by, bx], np.int32)
        ordered = _ordered_fragment(coords)
        if ordered is None:
            # Do not assert common identity through an ambiguous connected set.
            discarded["non_path"] += 1
            continue
        yy, xx = coords.T
        width = float(np.median(width_map[yy, xx]))
        width = float(np.clip(width, 1., min(rgb.shape[:2])))
        radius = int(math.ceil(min(width * 1.5 + 3, 96)))
        xa, ya = max(0, bx - radius), max(0, by - radius)
        xb, yb = min(rgb.shape[1], bx + bw + radius), min(rgb.shape[0], by + bh + radius)
        frag_mask = np.zeros((yb - ya, xb - xa), np.uint8)
        frag_mask[yy - ya, xx - xa] = 1
        outer = cv2.dilate(frag_mask, _disk(radius)) > 0
        inner = cv2.dilate(frag_mask, _disk(max(width * .6, 2))) > 0
        ring = outer & ~inner & (ridge[ya:yb, xa:xb] < .05)
        center_color = np.median(color[yy, xx], axis=0)
        contrast = float(np.linalg.norm(center_color - np.median(color[ya:yb, xa:xb][ring], axis=0))) if ring.any() else 0.
        if contrast < config.min_contrast * (2 if short else 1):
            discarded["low_contrast"] += 1
            continue
        near_junction = bool(np.any(cv2.dilate(frag_mask, _disk(config.junction_radius + 2))
                                    & junction_zone[ya:yb, xa:xb].astype(np.uint8)))
        sample_indices = [len(ordered) // 2] if near_junction or short else sorted(set(
            int(round(t * (len(ordered) - 1))) for t in (.2, .5, .8)))
        points = []
        point_colors = []
        for idx in sample_indices:
            y, x = map(int, ordered[idx])
            # Appearance consistency is evidence only; it never merges fragments.
            if np.linalg.norm(color[y, x] - center_color) <= max(12., .35 * contrast):
                points.append((float(x), float(y)))
                point_colors.append(color[y, x].tolist())
        if not points:
            y, x = map(int, ordered[len(ordered) // 2])
            points = [(float(x), float(y))]
        margin = max(8., 2 * width)
        box = (float(max(0, xx.min() - margin)), float(max(0, yy.min() - margin)),
               float(min(rgb.shape[1], xx.max() + margin + 1)),
               float(min(rgb.shape[0], yy.max() + margin + 1)))
        direction = ordered[-1].astype(float) - ordered[0]
        angle = float(math.atan2(direction[0], direction[1]) % math.pi)
        sources = sorted({source_names[i-1] for i in np.unique(source_map[yy, xx]) if i > 0})
        strength = float(np.mean(ridge[yy, xx]))
        score = strength * math.log1p(length) * min(contrast / 20., 3.) * (.5 if short else 1.)
        if short:
            discarded["short_single_point_hypotheses"] += 1
        candidates.append({"fragment_id": int(ident), "points": tuple(points), "box": box,
                           "coords": ordered, "score": score, "width": width, "angle": angle,
                           "contrast": contrast, "ridge_strength": strength,
                           "center_color": center_color, "hard_zone": near_junction,
                           "short_fragment": short,
                           "sources": sources, "length": length})
    return candidates, discarded


def _candidate_key(candidate, seed):
    # Geometric key, not list position: reordering candidates cannot change a plan.
    x, y = candidate["points"][0]
    return (float(y), float(x), float(candidate["angle"]), int(candidate["length"]), int(seed))


def _select_candidates(candidates, config):
    """Greedy geometric coverage rounds; fixed mode is the identical score pool."""
    remaining = sorted(candidates, key=lambda c: (-c["score"], _candidate_key(c, config.seed)))
    selected, occupied = [], set()
    while remaining and len(selected) < config.max_prompts_per_crop:
        ranked = []
        for item in remaining:
            # Spatial+orientation cells avoid penalizing a genuinely crossing line
            # as though it were the same hypothesis.
            orientation = int((item["angle"] + math.pi / 12) // (math.pi / 6)) % 6
            cells = {(int(y) // 24, int(x) // 24, orientation) for y, x in item["coords"]}
            novelty = len(cells - occupied) / max(len(cells), 1)
            utility = item["score"] * (.15 + .85 * novelty) if config.selection == "iterative" else item["score"]
            ranked.append((-utility, _candidate_key(item, config.seed), item, cells, novelty))
        _, _, winner, cells, novelty = min(ranked, key=lambda row: row[:2])
        remaining = [c for c in remaining if c is not winner]
        # Exact local rediscoveries do not consume another object prompt.
        if any(math.dist(winner["points"][0], old["points"][0]) < 2 for old in selected):
            continue
        chosen = dict(winner)
        chosen["selection_novelty"] = novelty
        selected.append(chosen)
        occupied.update(cells)
    return selected


def _safe_negatives(rgb, ridge, candidate, config):
    """Optional low-structure/color-separated patches, never fixed-width offsets.

    These remain background hypotheses. Abstain around junctions, in flat
    evidence-poor crops, and wherever other geometric ridges are nearby.
    """
    if config.max_negative_points == 0 or candidate["hard_zone"] or candidate.get("short_fragment", False):
        return (), []
    h, w = ridge.shape
    pixels = rgb.astype(np.float32)
    mean = cv2.boxFilter(pixels, -1, (5, 5), normalize=True)
    square = cv2.boxFilter(pixels * pixels, -1, (5, 5), normalize=True)
    variation = np.sqrt(np.maximum(square - mean * mean, 0)).max(axis=2)
    ridge_distance = cv2.distanceTransform((ridge < .03).astype(np.uint8), cv2.DIST_L2, 5)
    separation = np.linalg.norm(mean - candidate["center_color"], axis=2)
    clearance = max(5., candidate["width"])
    valid = ((ridge_distance >= clearance) & (variation <= 4.) & (separation >= 20.))
    valid[:3] = False; valid[-3:] = False; valid[:, :3] = False; valid[:, -3:] = False
    x0, y0, x1, y1 = candidate["box"]
    bbox = np.zeros((h, w), bool)
    bbox[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))] = True
    yy, xx = np.where(valid & bbox)
    if not len(xx):
        return (), []
    # Maximal measured ridge clearance, then color separation; stable coordinates.
    order = np.lexsort((xx, yy, -separation[yy, xx], -ridge_distance[yy, xx]))
    selected, evidence = [], []
    for k in order:
        x, y = int(xx[k]), int(yy[k])
        if any(math.dist((x, y), p) < max(12., 2 * clearance) for p in selected):
            continue
        selected.append((float(x), float(y)))
        evidence.append({"xy": [x, y], "ridge_clearance_px": float(ridge_distance[y, x]),
                         "patch_rgb_std_max": float(variation[y, x]),
                         "rgb_separation": float(separation[y, x]),
                         "status": "geometric_background_hypothesis"})
        if len(selected) == config.max_negative_points:
            break
    return tuple(selected), evidence


def plan_prompts(rgb: np.ndarray, config=PlannerConfig()) -> PromptPlan:
    """Scan every native tile, then freeze all independent point/box hypotheses."""
    config.validate()
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected native uint8 RGB HxWx3")
    h, w = rgb.shape[:2]
    crops = native_crops(w, h, config)
    full_ridge = np.zeros((h, w), np.float32)
    prompts, crop_audits = [], []
    for crop in crops:
        x0, y0, x1, y1 = crop.xyxy
        tile = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        ridge, widths, sources, scan = _scan_tile(tile, config)
        np.maximum(full_ridge[y0:y1, x0:x1], ridge, out=full_ridge[y0:y1, x0:x1])
        candidates, discarded = _fragment_candidates(tile, ridge, widths, sources,
                                                     scan["source_names"], config)
        selected = _select_candidates(candidates, config)
        for index, candidate in enumerate(selected):
            negative, negative_evidence = _safe_negatives(tile, ridge, candidate, config)
            metadata = {key: candidate[key] for key in ("fragment_id", "score", "width", "angle",
                         "contrast", "ridge_strength", "hard_zone", "short_fragment", "sources", "length", "selection_novelty")}
            metadata.update(hypothesis="locally_continuous_geometric_fragment",
                            instance_identity_verified=False, ground_truth_used=False,
                            positive_evidence="unbranched_connected_ridge_with_local_color_consistency",
                            negative_evidence=negative_evidence,
                            selection_round=index + 1, coordinate_system="native_crop_xy",
                            sparse_tokens=len(candidate["points"]) + len(negative) + 2)
            prompts.append(PromptSpec(f"{crop.crop_id}_p{index:02d}", crop.crop_id,
                                      candidate["points"], negative, candidate["box"], metadata))
        crop_audits.append({"crop_id": crop.crop_id, "scan": scan,
                            "candidate_count": len(candidates), "selected_count": len(selected),
                            "discarded_fragments": discarded,
                            "status": "ok" if selected else "no_prompt_candidates"})
    audit = {"method": "native_geomprompt_fragment_planner_v1", "references": ["R11", "R13", "R15"],
             "config": asdict(config), "ground_truth_used": False, "sam_calls": 0,
             "native_size": [w, h], "scanned_crop_count": len(crops),
             "empty_crop_count": sum(c["selected_count"] == 0 for c in crop_audits),
             "detector_failure_count": sum(len(c["scan"]["failures"]) for c in crop_audits),
             "candidate_count": sum(c["candidate_count"] for c in crop_audits),
             "prompt_count": len(prompts), "crop_audits": crop_audits,
             "ridge_semantics": "relative geometric response, not calibrated foreground probability",
             "status": "ok" if prompts else "empty_plan"}
    return PromptPlan(crops, prompts, full_ridge, audit).validate(rgb, config.canvas_size)
