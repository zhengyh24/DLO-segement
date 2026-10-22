from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from geomprompt.ridges.ridge_prompts import RidgePrompts

from stage_a_geomprompt_sam31 import (
    Candidate,
    DEFAULT_DATASET,
    angular_distance,
    json_dump,
    load_truth,
    mark_hard_zones,
    prompt_metrics,
    regular_crops,
    safe_get_prompts,
)


DEFAULT_OUTPUT = Path(
    r"outputs/geomprompt_tuned_detector"
)


def scalar_channels(rgb: np.ndarray) -> dict[str, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "lab_l": lab[:, :, 0],
        "lab_a": lab[:, :, 1],
        "lab_b": lab[:, :, 2],
        "hsv_s": hsv[:, :, 1],
    }


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    xs = np.clip(xs.astype(np.float32), 0, w - 1)
    ys = np.clip(ys.astype(np.float32), 0, h - 1)
    sampled = cv2.remap(
        image.astype(np.float32, copy=False),
        xs.reshape(1, -1),
        ys.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return sampled.reshape(-1, image.shape[2])


def tube_profile(
    lab: np.ndarray,
    x: float,
    y: float,
    initial_angle_deg: float,
    candidate_widths: Sequence[float],
) -> dict:
    """Fit a locally smooth, two-sided color-separated stripe without labels."""
    best = {
        "score": 0.0,
        "separation": 0.0,
        "symmetry": 0.0,
        "longitudinal_std": 999.0,
        "width": float(candidate_widths[0]),
        "angle_deg": float(initial_angle_deg),
        "lab": [0.0, 0.0, 0.0],
    }
    for angle_offset in (-30.0, -15.0, 0.0, 15.0, 30.0):
        angle = (initial_angle_deg + angle_offset) % 180.0
        theta = math.radians(angle)
        tangent = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
        normal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=float)
        for diameter in candidate_widths:
            tangent_offsets = np.linspace(-1.2 * diameter, 1.2 * diameter, 9)
            inner_normal_offsets = np.asarray([-0.16, 0.0, 0.16]) * diameter
            center_coords = np.asarray([x, y])[None, None, :] + (
                tangent_offsets[:, None, None] * tangent[None, None, :]
                + inner_normal_offsets[None, :, None] * normal[None, None, :]
            )
            center_samples = bilinear_sample(
                lab, center_coords[..., 0].ravel(), center_coords[..., 1].ravel()
            ).reshape(len(tangent_offsets), len(inner_normal_offsets), 3)

            side_offsets = np.asarray([0.65, 0.82, 1.0]) * diameter
            side_colors = []
            for sign in (-1.0, 1.0):
                side_coords = np.asarray([x, y])[None, None, :] + (
                    tangent_offsets[:, None, None] * tangent[None, None, :]
                    + sign * side_offsets[None, :, None] * normal[None, None, :]
                )
                samples = bilinear_sample(
                    lab, side_coords[..., 0].ravel(), side_coords[..., 1].ravel()
                ).reshape(len(tangent_offsets), len(side_offsets), 3)
                side_colors.append(np.median(samples, axis=(0, 1)))

            center_track = np.median(center_samples, axis=1)
            center_color = np.median(center_track, axis=0)
            left_delta = float(np.linalg.norm(center_color - side_colors[0]))
            right_delta = float(np.linalg.norm(center_color - side_colors[1]))
            separation = min(left_delta, right_delta)
            symmetry = min(left_delta, right_delta) / max(max(left_delta, right_delta), 1e-6)
            longitudinal_std = float(np.mean(np.std(center_track, axis=0)))
            inner_std = float(np.mean(np.std(center_samples.reshape(-1, 3), axis=0)))
            flatness = math.exp(-longitudinal_std / 25.0) * math.exp(-inner_std / 38.0)
            score = separation * math.sqrt(max(symmetry, 0.0)) * flatness
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "separation": separation,
                    "symmetry": float(symmetry),
                    "longitudinal_std": longitudinal_std,
                    "inner_std": inner_std,
                    "width": float(diameter),
                    "angle_deg": float(angle),
                    "lab": [float(value) for value in center_color],
                }
    return best


def response_angle(tensor: np.ndarray, x: float, y: float, width: float) -> float:
    response = np.max(tensor, axis=2)
    ix, iy = int(round(x)), int(round(y))
    radius = max(25, int(round(width * 1.7)))
    xa, xb = max(0, ix - radius), min(response.shape[1], ix + radius + 1)
    ya, yb = max(0, iy - radius), min(response.shape[0], iy + radius + 1)
    local = response[ya:yb, xa:xb]
    nonzero = local[local > 0]
    if nonzero.size < 5:
        return 0.0
    threshold = float(np.quantile(nonzero, 0.55))
    coords = np.argwhere(local >= threshold).astype(np.float64)
    if len(coords) < 4:
        return 0.0
    coords -= coords.mean(axis=0, keepdims=True)
    eigvals, eigvecs = np.linalg.eigh(coords.T @ coords)
    direction_yx = eigvecs[:, int(np.argmax(eigvals))]
    return float(math.degrees(math.atan2(direction_yx[0], direction_yx[1])) % 180.0)


def channel_candidates(
    rgb_crop: np.ndarray,
    channel_name: str,
    channel_image: np.ndarray,
    scales: Sequence[float],
    count: int,
    quantile: float,
    min_component_size: int,
    seed: int,
) -> list[Candidate]:
    prompter = RidgePrompts(
        channel_image,
        scales=np.asarray(scales, dtype=float),
        method="sq-pcd",
        pool=False,
    )
    lab = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    raw: list[Candidate] = []
    for source, tensor in (("ridge", prompter.ridge_tensor), ("valley", prompter.valley_tensor)):
        points = safe_get_prompts(
            prompter,
            source,
            count,
            quantile,
            min_component_size,
            seed + (0 if source == "ridge" else 100019),
        )
        positive = tensor[tensor > 0]
        response_norm = float(np.quantile(positive, 0.95)) if positive.size else 1.0
        response_norm = max(response_norm, 1e-12)
        for x, y in points:
            ix = int(np.clip(round(x), 0, tensor.shape[1] - 1))
            iy = int(np.clip(round(y), 0, tensor.shape[0] - 1))
            values = tensor[iy, ix]
            scale_index = int(np.argmax(values))
            scale = float(scales[scale_index])
            response = float(values[scale_index])
            ridge_width = float(RidgePrompts.scale_to_width(scale))
            angle = response_angle(tensor, x, y, ridge_width)
            fit = tube_profile(
                lab,
                x,
                y,
                angle,
                candidate_widths=(16.0, 24.0, 32.0, 42.0, 54.0, 68.0),
            )
            response_score = min(response / response_norm, 3.0)
            combined_score = fit["score"] * (0.65 + 0.35 * response_score)
            raw.append(
                Candidate(
                    x=float(x),
                    y=float(y),
                    tile_id="",
                    source=source,
                    scale=scale,
                    width=fit["width"],
                    contrast=fit["separation"],
                    homogeneity=fit.get("inner_std", 999.0),
                    response=response,
                    score=float(combined_score),
                    angle_deg=fit["angle_deg"],
                    channel=channel_name,
                    tube_score=fit["score"],
                    tube_separation=fit["separation"],
                    tube_symmetry=fit["symmetry"],
                    tube_longitudinal_std=fit["longitudinal_std"],
                    lab_l=fit["lab"][0],
                    lab_a=fit["lab"][1],
                    lab_b=fit["lab"][2],
                )
            )
    return raw


def fuse_local_candidates(
    candidates: list[Candidate],
    merge_radius: float,
) -> list[Candidate]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    used = np.zeros(len(candidates), dtype=bool)
    coords = np.asarray([[item.x, item.y] for item in candidates], dtype=float)
    tree = cKDTree(coords)
    fused: list[Candidate] = []
    for index, item in enumerate(candidates):
        if used[index]:
            continue
        neighbours = [j for j in tree.query_ball_point(coords[index], merge_radius) if not used[j]]
        compatible = [
            j
            for j in neighbours
            if angular_distance(item.angle_deg, candidates[j].angle_deg) <= 38.0
        ]
        if not compatible:
            compatible = [index]
        weights = np.asarray([max(candidates[j].score, 1e-3) for j in compatible])
        item.x = float(np.average([candidates[j].x for j in compatible], weights=weights))
        item.y = float(np.average([candidates[j].y for j in compatible], weights=weights))
        channels = {candidates[j].channel for j in compatible}
        item.support = len(channels)
        item.channel = "+".join(sorted(channels))
        item.score *= 1.0 + 0.24 * (item.support - 1)
        fused.append(item)
        used[compatible] = True
    return fused


def continuity_components(candidates: list[Candidate], link_radius: float) -> list[list[int]]:
    if not candidates:
        return []
    coords = np.asarray([[item.x, item.y] for item in candidates], dtype=float)
    tree = cKDTree(coords)
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i, j in tree.query_pairs(link_radius):
        a, b = candidates[i], candidates[j]
        color_a = np.asarray([a.lab_l, a.lab_a, a.lab_b])
        color_b = np.asarray([b.lab_l, b.lab_a, b.lab_b])
        color_distance = float(np.linalg.norm(color_a - color_b))
        width_ratio = max(a.width, b.width) / max(min(a.width, b.width), 1e-6)
        dx, dy = b.x - a.x, b.y - a.y
        connecting_angle = math.degrees(math.atan2(dy, dx)) % 180.0
        alignment = min(
            angular_distance(a.angle_deg, connecting_angle),
            angular_distance(b.angle_deg, connecting_angle),
        )
        close_pair = math.hypot(dx, dy) <= 125.0
        if color_distance <= 34.0 and width_ratio <= 2.6 and (alignment <= 42.0 or close_pair):
            union(i, j)
    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def select_continuous_candidates(
    candidates: list[Candidate],
    min_tube_score: float,
    min_separation: float,
    min_symmetry: float,
    max_longitudinal_std: float,
    max_per_crop: int,
    min_distance: float,
) -> tuple[list[Candidate], dict]:
    locally_valid = [
        item
        for item in candidates
        if item.tube_score >= min_tube_score
        and item.tube_separation >= min_separation
        and item.tube_symmetry >= min_symmetry
        and item.tube_longitudinal_std <= max_longitudinal_std
    ]
    groups = continuity_components(locally_valid, 420.0)
    retained_indices: set[int] = set()
    component_stats = []
    for indices in groups:
        coords = np.asarray([[locally_valid[i].x, locally_valid[i].y] for i in indices])
        span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))) if len(coords) else 0.0
        support = max((locally_valid[i].support for i in indices), default=0)
        best_score = max((locally_valid[i].tube_score for i in indices), default=0.0)
        keep = (len(indices) >= 2 and span >= 115.0) or support >= 2 or best_score >= 24.0
        if keep:
            retained_indices.update(indices)
        component_stats.append(
            {
                "size": len(indices),
                "span": span,
                "support": support,
                "best_tube_score": best_score,
                "kept": keep,
            }
        )
    retained = [locally_valid[index] for index in sorted(retained_indices)]
    selected: list[Candidate] = []
    for item in sorted(retained, key=lambda candidate: candidate.score, reverse=True):
        if all(math.hypot(item.x - old.x, item.y - old.y) >= min_distance for old in selected):
            selected.append(item)
            if len(selected) >= max_per_crop:
                break
    diagnostics = {
        "raw_count": len(candidates),
        "locally_valid_count": len(locally_valid),
        "continuity_retained_count": len(retained),
        "selected_count": len(selected),
        "components": component_stats,
    }
    return selected, diagnostics


def detect_crop_tuned(rgb: np.ndarray, crop, args, crop_index: int):
    tile = rgb[crop.y0 : crop.y1, crop.x0 : crop.x1]
    channels = scalar_channels(tile)
    raw: list[Candidate] = []
    for channel_index, channel_name in enumerate(args.channels):
        raw.extend(
            channel_candidates(
                tile,
                channel_name,
                channels[channel_name],
                args.scales,
                args.prompts_per_channel_polarity,
                args.quantile,
                args.min_component_size,
                args.seed + crop_index * 1009 + channel_index * 97,
            )
        )
    fused = fuse_local_candidates(raw, args.channel_merge_radius)
    for item in fused:
        item.x += crop.x0
        item.y += crop.y0
        item.tile_id = crop.crop_id
    selected, diagnostics = select_continuous_candidates(
        fused,
        args.min_tube_score,
        args.min_separation,
        args.min_symmetry,
        args.max_longitudinal_std,
        args.max_prompts_per_crop,
        args.local_min_distance,
    )
    return selected, diagnostics


def global_deduplicate(candidates: list[Candidate], distance: float) -> list[Candidate]:
    selected: list[Candidate] = []
    for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True):
        duplicate = next(
            (
                old
                for old in selected
                if math.hypot(item.x - old.x, item.y - old.y) < distance
                and angular_distance(item.angle_deg, old.angle_deg) < 42.0
            ),
            None,
        )
        if duplicate is None:
            selected.append(item)
        else:
            duplicate.support = max(duplicate.support, item.support + 1)
            duplicate.score = max(duplicate.score, item.score) * 1.05
    return selected


def recenter_candidate(candidate: Candidate, lab: np.ndarray) -> Candidate:
    """Move an edge-biased ridge point toward the middle of its fitted stripe."""
    theta = math.radians(candidate.angle_deg)
    tangent = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
    normal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=float)
    diameter = candidate.width
    best_score = -1.0
    best_xy = np.asarray([candidate.x, candidate.y], dtype=float)
    best_color = np.asarray([candidate.lab_l, candidate.lab_a, candidate.lab_b])
    for fraction in (-0.38, -0.25, -0.12, 0.0, 0.12, 0.25, 0.38):
        center_xy = np.asarray([candidate.x, candidate.y]) + fraction * diameter * normal
        along = np.asarray([-0.55, 0.0, 0.55]) * diameter
        center_coords = center_xy[None, :] + along[:, None] * tangent[None, :]
        center_samples = bilinear_sample(lab, center_coords[:, 0], center_coords[:, 1])
        center_color = np.median(center_samples, axis=0)
        side_colors = []
        for sign in (-1.0, 1.0):
            side_xy = center_xy + sign * 0.72 * diameter * normal
            side_coords = side_xy[None, :] + along[:, None] * tangent[None, :]
            side_samples = bilinear_sample(lab, side_coords[:, 0], side_coords[:, 1])
            side_colors.append(np.median(side_samples, axis=0))
        deltas = [float(np.linalg.norm(center_color - side)) for side in side_colors]
        longitudinal_std = float(np.mean(np.std(center_samples, axis=0)))
        score = min(deltas) * math.sqrt(min(deltas) / max(max(deltas), 1e-6))
        score *= math.exp(-longitudinal_std / 22.0)
        if score > best_score:
            best_score = score
            best_xy = center_xy
            best_color = center_color
    candidate.x = float(np.clip(best_xy[0], 0, lab.shape[1] - 1))
    candidate.y = float(np.clip(best_xy[1], 0, lab.shape[0] - 1))
    candidate.lab_l, candidate.lab_a, candidate.lab_b = map(float, best_color)
    candidate.tube_score = max(candidate.tube_score, float(best_score))
    return candidate


def global_topology_filter(
    candidates: list[Candidate],
    link_radius: float,
    min_group_size: int,
    min_span: float,
    min_mean_tube_score: float,
    max_chromatic_longitudinal_std: float,
    max_achromatic_longitudinal_std: float,
    min_chroma: float,
) -> tuple[list[Candidate], list[dict]]:
    groups = continuity_components(candidates, link_radius)
    kept: list[Candidate] = []
    diagnostics: list[dict] = []
    for indices in groups:
        coords = np.asarray([[candidates[i].x, candidates[i].y] for i in indices])
        span = float(np.linalg.norm(np.ptp(coords, axis=0))) if len(coords) else 0.0
        mean_tube_score = float(np.mean([candidates[i].tube_score for i in indices]))
        mean_longitudinal_std = float(
            np.mean([candidates[i].tube_longitudinal_std for i in indices])
        )
        lab = np.asarray(
            [
                [candidates[i].lab_l, candidates[i].lab_a, candidates[i].lab_b]
                for i in indices
            ],
            dtype=float,
        )
        mean_chroma = float(np.mean(np.linalg.norm(lab[:, 1:] - 128.0, axis=1)))
        material_consistent = (
            mean_longitudinal_std <= max_chromatic_longitudinal_std
            if mean_chroma >= min_chroma
            else mean_longitudinal_std <= max_achromatic_longitudinal_std
        )
        keep = (
            len(indices) >= min_group_size
            and span >= min_span
            and mean_tube_score >= min_mean_tube_score
            and material_consistent
        )
        if keep:
            kept.extend(candidates[i] for i in indices)
        diagnostics.append(
            {
                "size": len(indices),
                "span": span,
                "mean_tube_score": mean_tube_score,
                "mean_longitudinal_std": mean_longitudinal_std,
                "mean_chroma": mean_chroma,
                "material_consistent": material_consistent,
                "kept": keep,
            }
        )
    return kept, diagnostics


def image_scaled_min_span(base_span: float, width: int, height: int) -> float:
    """Keep a topology span proportional to image size below a 1024px work crop.

    On the original 1024px-or-larger crops this is an identity transform.
    It uses image geometry only, never labels or prediction quality.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return base_span * min(1.0, math.sqrt(width * height) / 1024.0)


def draw_overlay(image: Image.Image, candidates: Sequence[Candidate], output: Path) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    channel_colors = {
        "lab_l": (255, 220, 30),
        "lab_a": (255, 70, 70),
        "lab_b": (70, 220, 255),
        "hsv_s": (180, 80, 255),
    }
    radius = 11
    for item in candidates:
        base_channel = item.channel.split("+")[0]
        color = channel_colors.get(base_channel, (40, 240, 80))
        if item.support >= 2:
            color = (40, 240, 80)
        draw.ellipse(
            (item.x - radius, item.y - radius, item.x + radius, item.y + radius),
            fill=color,
            outline="white",
            width=3,
        )
        length = max(32.0, item.width)
        theta = math.radians(item.angle_deg)
        draw.line(
            (
                item.x - length * math.cos(theta),
                item.y - length * math.sin(theta),
                item.x + length * math.cos(theta),
                item.y + length * math.sin(theta),
            ),
            fill=color,
            width=4,
        )
    canvas.thumbnail((1952, 1098), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93)


def detect_view(args, view_dir: Path) -> dict:
    started = time.perf_counter()
    image_pil = Image.open(view_dir / "rgb.png").convert("RGB")
    rgb = np.asarray(image_pil)
    crops = regular_crops(rgb.shape[1], rgb.shape[0], args.tile_size, args.overlap)
    all_candidates: list[Candidate] = []
    crop_diagnostics = []
    for index, crop in enumerate(crops):
        print(f"[{view_dir.name}] tuned crop {index + 1}/{len(crops)} {crop.crop_id}", flush=True)
        selected, diagnostics = detect_crop_tuned(rgb, crop, args, index)
        all_candidates.extend(selected)
        crop_diagnostics.append({"crop_id": crop.crop_id, **diagnostics})
    candidates = global_deduplicate(all_candidates, args.global_min_distance)
    hard_regions = mark_hard_zones(candidates)
    semantic, centerline, instances = load_truth(view_dir)
    metrics = prompt_metrics(candidates, semantic, centerline, instances, args.coverage_radius)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["raw_candidates_across_crops"] = int(sum(d["raw_count"] for d in crop_diagnostics))
    metrics["locally_valid_across_crops"] = int(
        sum(d["locally_valid_count"] for d in crop_diagnostics)
    )
    payload = {
        "image": str(view_dir / "rgb.png"),
        "image_size": list(image_pil.size),
        "config": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)},
        "crops": [asdict(crop) for crop in crops],
        "candidates": [asdict(item) for item in candidates],
        "hard_regions": hard_regions,
        "crop_diagnostics": crop_diagnostics,
        "metrics": metrics,
    }
    view_output = args.output / view_dir.name
    json_dump(view_output / "prompts_tuned.json", payload)
    draw_overlay(image_pil, candidates, view_output / "prompt_overlay_tuned.jpg")
    print(
        f"[{view_dir.name}] tuned prompts={len(candidates)} "
        f"precision={metrics['precision_on_semantic_mask']:.3f} "
        f"centerline@{args.coverage_radius:g}={metrics['centerline_coverage_at_radius']:.3f} "
        f"elapsed={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload


def postprocess_view(args, view_dir: Path) -> dict:
    source_path = args.output / view_dir.name / "prompts_tuned.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Tuned prompt file not found: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    image_pil = Image.open(view_dir / "rgb.png").convert("RGB")
    rgb = np.asarray(image_pil)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    candidates = [recenter_candidate(Candidate(**item), lab) for item in payload["candidates"]]
    candidates = global_deduplicate(candidates, args.global_min_distance)
    filtered, topology_diagnostics = global_topology_filter(
        candidates,
        args.global_link_radius,
        args.global_min_group_size,
        image_scaled_min_span(args.global_min_span, image_pil.width, image_pil.height),
        args.global_min_mean_tube_score,
        args.global_max_chromatic_longitudinal_std,
        args.global_max_achromatic_longitudinal_std,
        args.global_min_chroma,
    )
    hard_regions = mark_hard_zones(filtered)
    semantic, centerline, instances = load_truth(view_dir)
    metrics = prompt_metrics(filtered, semantic, centerline, instances, args.coverage_radius)
    result = {
        **payload,
        "postprocess_config": vars(args)
        | {"dataset": str(args.dataset), "output": str(args.output),
           "effective_global_min_span": image_scaled_min_span(
               args.global_min_span, image_pil.width, image_pil.height)},
        "candidates": [asdict(item) for item in filtered],
        "hard_regions": hard_regions,
        "pre_global_filter_count": len(candidates),
        "global_topology_diagnostics": topology_diagnostics,
        "metrics": metrics,
    }
    view_output = args.output / view_dir.name
    json_dump(view_output / "prompts_tuned_filtered.json", result)
    draw_overlay(image_pil, filtered, view_output / "prompt_overlay_tuned_filtered.jpg")
    print(
        f"[{view_dir.name}] filtered prompts={len(filtered)} "
        f"precision={metrics['precision_on_semantic_mask']:.3f} "
        f"centerline@{args.coverage_radius:g}={metrics['centerline_coverage_at_radius']:.3f}",
        flush=True,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Training-free multi-channel GeoMPrompt cable detector")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--postprocess-existing", action="store_true")
    parser.add_argument(
        "--skip-global-postprocess",
        action="store_true",
        help="Keep only raw tuned candidates; by default global material/topology filtering is applied",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--channels", nargs="+", default=["lab_l", "lab_a", "lab_b", "hsv_s"])
    parser.add_argument("--scales", type=float, nargs="+", default=[49, 100, 196, 324])
    parser.add_argument("--prompts-per-channel-polarity", type=int, default=6)
    parser.add_argument("--quantile", type=float, default=0.72)
    parser.add_argument("--min-component-size", type=int, default=5)
    parser.add_argument("--channel-merge-radius", type=float, default=30.0)
    parser.add_argument("--min-tube-score", type=float, default=8.5)
    parser.add_argument("--min-separation", type=float, default=11.0)
    parser.add_argument("--min-symmetry", type=float, default=0.20)
    parser.add_argument("--max-longitudinal-std", type=float, default=24.0)
    parser.add_argument("--max-prompts-per-crop", type=int, default=7)
    parser.add_argument("--local-min-distance", type=float, default=80.0)
    parser.add_argument("--global-min-distance", type=float, default=58.0)
    parser.add_argument("--global-link-radius", type=float, default=320.0)
    parser.add_argument("--global-min-group-size", type=int, default=2)
    parser.add_argument("--global-min-span", type=float, default=180.0)
    parser.add_argument("--global-min-mean-tube-score", type=float, default=40.0)
    parser.add_argument(
        "--global-max-chromatic-longitudinal-std", type=float, default=5.0
    )
    parser.add_argument(
        "--global-max-achromatic-longitudinal-std", type=float, default=2.8
    )
    parser.add_argument("--global-min-chroma", type=float, default=12.0)
    parser.add_argument("--coverage-radius", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def resolve_views(args) -> list[Path]:
    scene = args.dataset / "scene_000000"
    views = sorted(path for path in scene.glob("view_*") if (path / "rgb.png").is_file())
    if args.views:
        wanted = set(args.views)
        views = [path for path in views if path.name in wanted]
    if args.max_views is not None:
        views = views[: args.max_views]
    if not views:
        raise FileNotFoundError(f"No views found under {scene}")
    return views


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace", line_buffering=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.postprocess_existing:
        payloads = [postprocess_view(args, view_dir) for view_dir in resolve_views(args)]
    else:
        payloads = []
        for view_dir in resolve_views(args):
            raw_payload = detect_view(args, view_dir)
            payloads.append(
                raw_payload
                if args.skip_global_postprocess
                else postprocess_view(args, view_dir)
            )
    metric_names = ("precision_on_semantic_mask", "centerline_coverage_at_radius", "count")
    summary = {"views": len(payloads)}
    for name in metric_names:
        values = [float(payload["metrics"][name]) for payload in payloads]
        summary[f"mean_{name}"] = float(np.mean(values))
        summary[f"std_{name}"] = float(np.std(values))
    json_dump(args.output / "summary.json", summary)
    print(f"TUNED_SUMMARY {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
