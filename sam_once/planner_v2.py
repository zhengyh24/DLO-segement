"""RGB-only path prompt planning, independent of the frozen v1 implementation.

R11/R13/R15-inspired project adaptation. Each channel/polarity keeps its own
path graph. Neither connected geometry nor different color proves cable ID.
No SAM, supervision, label loader, or result-dependent feedback is imported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import warnings

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .planner import PlannerConfig, _disk, _normalize_response, _safe_negatives, native_crops
from .types import PromptPlan, PromptSpec


@dataclass(frozen=True)
class V2Config:
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
    pos_count: int = 3
    neg_count: int = 0
    box_mode: str = "none"
    negative_mode: str = "none"

    def validate(self):
        PlannerConfig(canvas_size=self.canvas_size, overlap=self.overlap,
                      max_prompts_per_crop=self.max_prompts_per_crop, seed=self.seed,
                      selection=self.selection, channels=self.channels, scales=self.scales,
                      ridge_threshold=self.ridge_threshold, min_fragment_pixels=self.min_fragment_pixels,
                      min_contrast=self.min_contrast, junction_radius=self.junction_radius).validate()
        if self.pos_count not in (1, 2, 3) or self.neg_count not in (0, 1, 2):
            raise ValueError("Use 1-3 positive and 0-2 negative requested points")
        if self.box_mode not in ("none", "local", "expanded"):
            raise ValueError("box_mode must be none/local/expanded")
        if self.negative_mode not in ("none", "background", "neighbor"):
            raise ValueError("negative_mode must be none/background/neighbor")
        return self


def path_graph(skeleton):
    """8-neighbour graph without a diagonal shortcut around an occupied corner.

    A staircase remains a path instead of producing three-way junctions at each
    corner. A true plus/X intersection still has degree >=3 and is cut below.
    """
    pixels = {tuple(map(int, p)) for p in np.argwhere(skeleton)}
    graph = {}
    diagonals_removed = 0
    for y, x in sorted(pixels):
        adjacent = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if not (dy or dx) or q not in pixels:
                    continue
                if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                    diagonals_removed += 1
                    continue
                adjacent.append(q)
        graph[(y, x)] = tuple(sorted(adjacent))
    return graph, diagonals_removed // 2


def split_paths(skeleton, junction_radius=3):
    graph, removed_diagonals = path_graph(skeleton)
    junctions = {p for p, adjacent in graph.items() if len(adjacent) > 2}
    cut, frontier = set(junctions), set(junctions)
    for _ in range(junction_radius):
        new = {q for p in frontier for q in graph[p]} - cut
        cut.update(new)
        frontier = new
    unseen = set(graph) - cut
    paths = []
    # This visits the same next minimum without rescanning every remaining pixel.
    for first in sorted(graph):
        if first not in unseen:
            continue
        component, queue = set(), [first]
        while queue:
            p = queue.pop()
            if p in component:
                continue
            component.add(p)
            queue.extend(q for q in graph[p] if q not in cut and q not in component)
        unseen -= component
        ends = sorted(p for p in component if sum(q in component for q in graph[p]) <= 1)
        cursor = ends[0] if ends else min(component)
        ordered, visited = [], set()
        while cursor not in visited:
            visited.add(cursor)
            ordered.append(cursor)
            next_pixels = [q for q in graph[cursor] if q in component and q not in visited]
            if not next_pixels:
                break
            cursor = next_pixels[0]
        if len(ordered) != len(component):
            continue  # Never manufacture a traversal across remaining ambiguity.
        near_junction = any(q in cut for p in component for q in graph[p])
        paths.append((np.asarray(ordered, np.int32), near_junction, not bool(ends)))
    return paths, {"skeleton_pixels": len(graph), "true_junction_pixels": len(junctions),
                   "junction_cut_pixels": len(cut), "diagonal_shortcuts_removed": removed_diagonals,
                   "path_components": len(paths)}


def arc_samples(coords, count=3):
    """Disperse points by observed path arc length, never Euclidean interpolation."""
    if not len(coords):
        return [], []
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(coords.astype(float), axis=0), axis=1))]
    total = float(arc[-1])
    fractions = (.5,) if count == 1 else ((.2, .8) if count == 2 else (.15, .5, .85))
    indices = sorted({int(np.argmin(np.abs(arc - fraction * total))) for fraction in fractions})
    points = [(float(coords[i, 1]), float(coords[i, 0])) for i in indices]
    return points, [float(arc[i] / total) if total else .5 for i in indices]


def _tangents(coords):
    a = coords.astype(np.float32)
    span = min(4, max(1, len(a) // 5))
    indices = np.arange(len(a))
    delta = a[np.minimum(indices + span, len(a)-1)] - a[np.maximum(indices-span, 0)]
    return delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6)


def _boxes(coords, width, shape):
    yy, xx = coords.T
    h, w = shape
    def make(margin):
        return [float(max(0, xx.min()-margin)), float(max(0, yy.min()-margin)),
                float(min(w, xx.max()+margin+1)), float(min(h, yy.max()+margin+1))]
    return make(max(8., 2*width)), make(max(24., 4*width))


def extract_branch(rgb, response, widths, source, config):
    paths, audit = split_paths(skeletonize(response >= config.ridge_threshold), config.junction_radius)
    candidates = []
    audit.update(short_discarded=0, low_contrast_discarded=0)
    rgb_float = rgb.astype(np.float32)
    for coords, near_junction, closed in paths:
        if len(coords) < config.min_fragment_pixels:
            audit["short_discarded"] += 1
            continue
        yy, xx = coords.T
        width = max(1., float(np.median(widths[yy, xx])))
        radius = int(math.ceil(min(96., max(6., width*1.5+3))))
        x0, x1 = max(0, int(xx.min())-radius), min(rgb.shape[1], int(xx.max())+radius+1)
        y0, y1 = max(0, int(yy.min())-radius), min(rgb.shape[0], int(yy.max())+radius+1)
        local = np.zeros((y1-y0, x1-x0), np.uint8)
        local[yy-y0, xx-x0] = 1
        outer = cv2.dilate(local, _disk(radius)) > 0
        inner = cv2.dilate(local, _disk(max(2., width*.6))) > 0
        ring = outer & ~inner & (response[y0:y1, x0:x1] < .03)
        center = np.median(rgb_float[yy, xx], axis=0)
        contrast = float(np.linalg.norm(center - np.median(rgb_float[y0:y1, x0:x1][ring], axis=0))) if ring.any() else 0.
        if contrast < config.min_contrast:
            audit["low_contrast_discarded"] += 1
            continue
        arc_length = float(np.linalg.norm(np.diff(coords.astype(float), axis=0), axis=1).sum())
        point_pool, arc_fractions = arc_samples(coords, 3)
        local_box, expanded_box = _boxes(coords, width, response.shape)
        strength = float(np.mean(response[yy, xx]))
        # Length saturates: one long strong path must not consume all selections.
        score = strength * math.log1p(arc_length) * min(3., contrast/20.)
        key = hashlib.sha256(source.encode() + np.ascontiguousarray(coords).tobytes()).hexdigest()[:20]
        candidates.append({"path_key": key, "coords": coords, "tangents": _tangents(coords),
                           "sources": [source], "score": score, "width": width, "contrast": contrast,
                           "ridge_strength": strength, "center_color": center, "hard_zone": near_junction,
                           "closed_path": closed, "path_length_px": arc_length,
                           "positive_pool": point_pool, "positive_pool_arc_fractions": arc_fractions,
                           "box_local": local_box, "box_expanded": expanded_box,
                           "rediscovered_sources": []})
    audit["candidate_count"] = len(candidates)
    return candidates, audit


def _scan_tile(rgb, config):
    from geomprompt.ridges.ridge_prompts import RidgePrompts
    ridge = np.zeros(rgb.shape[:2], np.float32)
    audit = {"branches": [], "failures": [], "warnings": [],
             "path_policy": "independent_channel_polarity_graphs; no merged-response skeleton"}
    candidates = []
    if min(rgb.shape[:2]) < 8:
        audit["failures"].append("native_crop_too_small")
        return ridge, candidates, audit
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    channels = {"gray": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), "lab_a": lab[:, :, 1], "lab_b": lab[:, :, 2]}
    scale_widths = np.asarray(RidgePrompts.scale_to_width(np.asarray(config.scales)), np.float32)
    for name in config.channels:
        if int(np.ptp(channels[name])) < 2:
            continue
        try:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always", RuntimeWarning)
                detector = RidgePrompts(channels[name], scales=np.asarray(config.scales), method="sq-pcd", pool=False)
            audit["warnings"].extend({"channel": name, "message": message}
                                     for message in sorted({str(w.message) for w in observed}))
            for polarity in ("ridge", "valley"):
                response, indices = _normalize_response(getattr(detector, polarity + "_tensor"))
                np.maximum(ridge, response, out=ridge)
                source = f"{name}:{polarity}"
                branch_candidates, branch_audit = extract_branch(rgb, response, scale_widths[indices], source, config)
                candidates.extend(branch_candidates)
                audit["branches"].append({"source": source, **branch_audit})
        except (ValueError, IndexError, FloatingPointError, ZeroDivisionError) as exc:
            audit["failures"].append({"channel": name, "error": f"{type(exc).__name__}: {exc}"})
    return ridge, candidates, audit


def deduplicate_paths(candidates):
    """Keep one observed path; never union points from separate hypotheses."""
    ordered = sorted(candidates, key=lambda c: (-c["score"], -c["path_length_px"], c["path_key"]))
    kept, dropped = [], 0
    # Coordinate arrays stay alive and immutable throughout this call. Shallow
    # copies in `kept` retain their identity, so reuse the exact original trees.
    trees, sample_indices = {}, {}
    for candidate in ordered:
        duplicate = None
        for old in kept:
            shorter, longer = (candidate, old) if len(candidate["coords"]) <= len(old["coords"]) else (old, candidate)
            shorter_key, longer_key = id(shorter["coords"]), id(longer["coords"])
            if shorter_key not in sample_indices:
                sample_indices[shorter_key] = np.linspace(0, len(shorter["coords"])-1, min(64, len(shorter["coords"]))).round().astype(int)
            if longer_key not in trees:
                trees[longer_key] = cKDTree(longer["coords"])
            sampled = sample_indices[shorter_key]
            distances, nearest = trees[longer_key].query(shorter["coords"][sampled])
            aligned = np.abs(np.sum(shorter["tangents"][sampled] * longer["tangents"][nearest], axis=1)) >= .85
            tolerance = max(1., min(2.5, .3*min(shorter["width"], longer["width"])))
            if float(np.mean((distances <= tolerance) & aligned)) >= .85:
                duplicate = old
                break
        if duplicate is None:
            kept.append(dict(candidate))
        else:
            duplicate["rediscovered_sources"] = sorted(set(duplicate["rediscovered_sources"] + candidate["sources"]))
            dropped += 1
    return kept, dropped


def select_paths(candidates, shape, config):
    """Balance geometric quality, uncovered spatial area and uncovered path cells."""
    if not candidates or not config.max_prompts_per_crop:
        return []
    h, w = shape
    budget = config.max_prompts_per_crop
    columns = max(1, math.ceil(math.sqrt(budget*w/max(h, 1))))
    rows = max(1, math.ceil(budget/columns))
    maximum = max(c["score"] for c in candidates)
    remaining = sorted(candidates, key=lambda c: c["path_key"])
    # Only covered sets change between rounds; geometry and quality do not.
    features = {}
    for candidate in remaining:
        coords = candidate["coords"]
        spatial = {(min(rows-1, int(y)*rows//h), min(columns-1, int(x)*columns//w)) for y, x in coords}
        angle = np.mod(np.arctan2(candidate["tangents"][:, 0], candidate["tangents"][:, 1]), math.pi)
        oriented = {(int(y)//24, int(x)//24, int(a/(math.pi/6)) % 6) for (y, x), a in zip(coords, angle)}
        quality = candidate["score"]/max(maximum, 1e-8)
        features[id(candidate)] = (spatial, oriented, quality)
    selected, spatial_covered, path_covered = [], set(), set()
    while remaining and len(selected) < budget:
        ranked = []
        for candidate in remaining:
            spatial, oriented, quality = features[id(candidate)]
            spatial_novelty = len(spatial-spatial_covered)/max(len(spatial), 1)
            path_novelty = len(oriented-path_covered)/max(len(oriented), 1)
            utility = quality if config.selection == "fixed" else quality * (.2 + .45*spatial_novelty + .35*path_novelty)
            ranked.append((-utility, -spatial_novelty, candidate["path_key"], candidate, spatial, oriented))
        winner = min(ranked, key=lambda row: row[:3])
        candidate = winner[3]
        selected.append({**candidate, "selection_utility": float(-winner[0]), "selection_spatial_novelty": float(-winner[1])})
        spatial_covered.update(winner[4]); path_covered.update(winner[5])
        remaining = [c for c in remaining if c is not candidate]
    return selected


def _neighbor_pool(rgb, ridge, candidate, others, component_labels=None):
    """Conservative different-appearance parallel-path negatives, never same-color.

    Separate paths can still be the same cable. Require distinct, locally stable
    colors AND a real non-ridge gap/parallel separation; abstain at junctions.
    Shared observed ridge connectivity vetoes the negative, including a visible
    U-bend whose two arms differ in appearance. Connectivity is used only as an
    abstention test and never as proof of identity or for joining positive paths.
    """
    if candidate["hard_zone"] or candidate["closed_path"]:
        return [], []
    candidate_colors = rgb[candidate["coords"][:, 0], candidate["coords"][:, 1]].astype(np.float32)
    candidate_mad = float(np.median(np.linalg.norm(candidate_colors-candidate["center_color"], axis=1)))
    if candidate_mad > 12.:
        return [], []
    if component_labels is None:
        _, component_labels = cv2.connectedComponents((ridge >= .15).astype(np.uint8), connectivity=8)
    own_components = set(map(int, component_labels[candidate["coords"][:, 0], candidate["coords"][:, 1]])) - {0}
    tree = cKDTree(candidate["coords"])
    choices = []
    for other in others:
        if other["path_key"] == candidate["path_key"] or other["hard_zone"]:
            continue
        other_components = set(map(int, component_labels[other["coords"][:, 0], other["coords"][:, 1]])) - {0}
        if own_components & other_components:
            continue
        color_distance = float(np.linalg.norm(candidate["center_color"] - other["center_color"]))
        # Different appearance is mandatory: never mark same-color U-bend arms negative.
        if color_distance < 45.:
            continue
        coords = other["coords"]
        colors = rgb[coords[:, 0], coords[:, 1]].astype(np.float32)
        color_mad = float(np.median(np.linalg.norm(colors-other["center_color"], axis=1)))
        if color_mad > 12.:
            continue
        distance, nearest = tree.query(coords)
        aligned = np.abs(np.sum(other["tangents"] * candidate["tangents"][nearest], axis=1))
        minimum = max(4., .65*(other["width"]+candidate["width"]))
        maximum = max(24., 3*(other["width"]+candidate["width"]))
        eligible = (distance >= minimum) & (distance <= maximum) & (aligned >= .94)
        valid = np.flatnonzero(eligible)
        if len(valid) < max(8, int(.4*len(coords))):
            continue
        for index in (int(valid[len(valid)//3]), int(valid[2*len(valid)//3])):
            y, x = map(int, coords[index])
            # Verify a background-like gap between two centerlines, not just distance.
            near = candidate["coords"][nearest[index]].astype(float)
            interior = np.linspace(near, coords[index], 9)[3:6].round().astype(int)
            gap_color = np.median(rgb[interior[:, 0], interior[:, 1]].astype(float), axis=0)
            gap_sep = min(float(np.linalg.norm(gap_color-candidate["center_color"])),
                          float(np.linalg.norm(gap_color-other["center_color"])))
            gap_ridge = float(np.max(ridge[interior[:, 0], interior[:, 1]]))
            if gap_sep < 20. or gap_ridge >= .05:
                continue
            evidence = {"other_path_key": other["path_key"], "rgb_center_distance": color_distance,
                        "candidate_color_mad": candidate_mad, "other_color_mad": color_mad,
                        "parallel_cosine": float(aligned[index]), "gap_ridge_response_max": gap_ridge,
                        "shared_observed_ridge_component": False,
                        "centerline_gap_px": float(distance[index]), "gap_rgb_separation": gap_sep,
                        "status": "different_path_hypothesis_not_verified_identity"}
            choices.append((float(distance[index]), y, x, evidence))
    pool, evidence = [], []
    for _, y, x, item in sorted(choices, key=lambda z: z[:3]):
        if any(math.dist((x, y), old) < 12 for old in pool):
            continue
        pool.append((float(x), float(y))); evidence.append(item)
        if len(pool) == 2:
            break
    return pool, evidence


def _chosen_positive(pool, fractions, count):
    if count == 1:
        return (pool[int(np.argmin(np.abs(np.asarray(fractions)-.5)))],)
    if count == 2 and len(pool) > 1:
        return (pool[0], pool[-1])
    return tuple(pool[:count])


def plan_prompts(rgb, config=V2Config()):
    config.validate()
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected native uint8 RGB HxWx3")
    h, w = rgb.shape[:2]
    crops = native_crops(w, h, config)
    ridge_map = np.zeros((h, w), np.float32)
    prompts, audits = [], []
    background_config = PlannerConfig(max_negative_points=2)
    for crop in crops:
        x0, y0, x1, y1 = crop.xyxy
        tile = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        ridge, candidates, scan = _scan_tile(tile, config)
        np.maximum(ridge_map[y0:y1, x0:x1], ridge, out=ridge_map[y0:y1, x0:x1])
        unique, duplicates = deduplicate_paths(candidates)
        selected = select_paths(unique, tile.shape[:2], config)
        _, component_labels = cv2.connectedComponents((ridge >= config.ridge_threshold).astype(np.uint8), connectivity=8)
        for index, candidate in enumerate(selected):
            background, bg_evidence = _safe_negatives(tile, ridge, {**candidate, "box": candidate["box_expanded"]}, background_config)
            neighbor, neighbor_evidence = _neighbor_pool(tile, ridge, candidate, unique, component_labels)
            pools = {"background": {"points": list(background), "evidence": bg_evidence},
                     "neighbor": {"points": neighbor, "evidence": neighbor_evidence}}
            positives = _chosen_positive(candidate["positive_pool"], candidate["positive_pool_arc_fractions"], config.pos_count)
            negatives = tuple(pools[config.negative_mode]["points"][:config.neg_count]) if config.negative_mode != "none" else ()
            box = None if config.box_mode == "none" else tuple(candidate["box_"+config.box_mode])
            metadata = {name: candidate[name] for name in ("path_key", "sources", "score", "width", "contrast",
                        "ridge_strength", "hard_zone", "closed_path", "path_length_px", "positive_pool",
                        "positive_pool_arc_fractions", "box_local", "box_expanded", "rediscovered_sources",
                        "selection_utility", "selection_spatial_novelty")}
            metadata.update(negative_pools=pools, local_box=candidate["box_local"], expanded_box=candidate["box_expanded"],
                            background_negative_pool=[{"xy": xy, "evidence": e} for xy, e in zip(background, bg_evidence)],
                            neighbor_negative_pool=[{"xy": xy, "evidence": e} for xy, e in zip(neighbor, neighbor_evidence)],
                            hypothesis="single_observed_branch_path", instance_identity_verified=False,
                            ground_truth_used=False, coordinate_system="native_crop_xy", selection_round=index+1,
                            requested_positive_count=config.pos_count, actual_positive_count=len(positives),
                            requested_negative_count=config.neg_count, actual_negative_count=len(negatives),
                            negative_mode=config.negative_mode, box_mode=config.box_mode,
                            sparse_tokens=len(positives)+len(negatives)+(2 if box else 0))
            prompts.append(PromptSpec(f"{crop.crop_id}_p{index:02d}", crop.crop_id, positives, negatives, box, metadata))
        audits.append({"crop_id": crop.crop_id, "scan": scan, "raw_candidate_count": len(candidates),
                       "unique_candidate_count": len(unique), "rediscoveries_removed": duplicates,
                       "selected_count": len(selected), "status": "ok" if selected else "no_prompt_candidates"})
    audit = {"method": "native_geomprompt_branch_path_planner_v2", "config": asdict(config),
             "references": ["R11", "R13", "R15"], "ground_truth_used": False, "sam_calls": 0,
             "native_size": [w, h], "scanned_crop_count": len(crops), "prompt_count": len(prompts),
             "empty_crop_count": sum(a["selected_count"] == 0 for a in audits),
             "detector_failure_count": sum(len(a["scan"]["failures"]) for a in audits), "crop_audits": audits,
             "status": "ok" if prompts else "empty_plan",
             "pool_policy": "same RGB-only path selection for every pos/neg/box configuration",
             "ridge_semantics": "geometric response, not a foreground probability",
             "negative_semantics": "conservative image-based hypotheses; no semantic identity guarantee"}
    return PromptPlan(crops, prompts, ridge_map, audit).validate(rgb, config.canvas_size)
