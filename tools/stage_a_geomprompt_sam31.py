from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from model_paths import SAM31_REPO, SAM31_CHECKPOINT
from scipy.spatial import cKDTree

from geomprompt.ridges.ridge_prompts import RidgePrompts
from cable_prompt_geometry import (
    SquareTransform, component_evidence, normalize_points,
    select_prompted_components, width_normalized_limits,
)


DEFAULT_DATASET = Path(
    "datasets/scene_000000"
)
DEFAULT_OUTPUT = Path(
    r"outputs/stage_a_geomprompt_sam31"
)
DEFAULT_CHECKPOINT = SAM31_CHECKPOINT
DEFAULT_SAM_REPO = SAM31_REPO


@dataclass
class Candidate:
    x: float
    y: float
    tile_id: str
    source: str
    scale: float
    width: float
    contrast: float
    homogeneity: float
    response: float
    score: float
    angle_deg: float
    hard_zone: bool = False
    channel: str = "gray"
    tube_score: float = 0.0
    tube_separation: float = 0.0
    tube_symmetry: float = 0.0
    tube_longitudinal_std: float = 0.0
    support: int = 1
    lab_l: float = 0.0
    lab_a: float = 0.0
    lab_b: float = 0.0


@dataclass(frozen=True)
class Crop:
    crop_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    reason: str = "regular"

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def anchored_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def regular_crops(width: int, height: int, tile_size: int, overlap: int) -> list[Crop]:
    stride = tile_size - overlap
    xs = anchored_positions(width, tile_size, stride)
    ys = anchored_positions(height, tile_size, stride)
    return [
        Crop(
            crop_id=f"r{row:02d}_c{col:02d}",
            x0=x,
            y0=y,
            x1=min(x + tile_size, width),
            y1=min(y + tile_size, height),
        )
        for row, y in enumerate(ys)
        for col, x in enumerate(xs)
    ]


def safe_get_prompts(
    prompter: RidgePrompts,
    source: str,
    count: int,
    quantile: float,
    min_component_size: int,
    seed: int,
) -> list[list[float]]:
    try:
        kwargs = dict(
            ridge_prompt_count=count if source == "ridge" else None,
            valley_prompt_count=count if source == "valley" else None,
            test_quantile_filter=quantile,
            min_component_size=min_component_size,
            random_seed=seed,
        )
        result = prompter.get_prompts(**kwargs)
        return [list(map(float, item[0])) for item in result["batch_point_prompts"]]
    except (ValueError, IndexError, ZeroDivisionError):
        return []


def local_feature_score(
    rgb: np.ndarray,
    tensor: np.ndarray,
    x: float,
    y: float,
    scales: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    """Return scale, estimated width, contrast, homogeneity, response, angle."""
    h, w = rgb.shape[:2]
    ix = int(np.clip(round(x), 0, w - 1))
    iy = int(np.clip(round(y), 0, h - 1))
    values = tensor[iy, ix]
    scale_index = int(np.argmax(values))
    scale = float(scales[scale_index])
    response = float(values[scale_index])
    est_width = float(RidgePrompts.scale_to_width(scale))

    inner = max(2, int(round(est_width * 0.15)))
    outer = max(inner + 3, int(round(est_width * 0.85)))
    xa, xb = max(0, ix - outer), min(w, ix + outer + 1)
    ya, yb = max(0, iy - outer), min(h, iy + outer + 1)
    patch = rgb[ya:yb, xa:xb].astype(np.float32)
    yy, xx = np.ogrid[ya:yb, xa:xb]
    d2 = (xx - ix) ** 2 + (yy - iy) ** 2
    center_sel = d2 <= inner**2
    ring_sel = (d2 >= (outer * 0.60) ** 2) & (d2 <= outer**2)
    if not center_sel.any() or not ring_sel.any():
        return scale, est_width, 0.0, 999.0, response, 0.0
    center_pixels = patch[center_sel]
    ring_pixels = patch[ring_sel]
    center_color = np.median(center_pixels, axis=0)
    ring_color = np.median(ring_pixels, axis=0)
    contrast = float(np.linalg.norm(center_color - ring_color))
    homogeneity = float(np.mean(np.std(center_pixels, axis=0)))

    response_2d = np.max(tensor, axis=2)
    radius = max(24, int(round(est_width * 1.5)))
    xa2, xb2 = max(0, ix - radius), min(w, ix + radius + 1)
    ya2, yb2 = max(0, iy - radius), min(h, iy + radius + 1)
    local = response_2d[ya2:yb2, xa2:xb2]
    positive = local[local > 0]
    angle = 0.0
    if positive.size >= 6:
        threshold = float(np.quantile(positive, 0.60))
        coords = np.argwhere(local >= threshold)
        if len(coords) >= 4:
            coords = coords.astype(np.float64)
            coords -= coords.mean(axis=0, keepdims=True)
            covariance = coords.T @ coords
            eigvals, eigvecs = np.linalg.eigh(covariance)
            direction = eigvecs[:, int(np.argmax(eigvals))]
            angle = float(math.degrees(math.atan2(direction[0], direction[1])) % 180.0)
    return scale, est_width, contrast, homogeneity, response, angle


def angular_distance(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def select_spread(candidates: list[Candidate], limit: int, min_distance: float) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(math.hypot(candidate.x - old.x, candidate.y - old.y) >= min_distance for old in selected):
            selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected


def detect_crop(
    image: np.ndarray,
    crop: Crop,
    scales: Sequence[float],
    prompts_per_polarity: int,
    quantile: float,
    min_component_size: int,
    min_contrast: float,
    min_width: float,
    max_prompts: int,
    min_prompt_distance: float,
    seed: int,
) -> list[Candidate]:
    tile = image[crop.y0 : crop.y1, crop.x0 : crop.x1]
    prompter = RidgePrompts(
        tile,
        scales=np.asarray(scales, dtype=float),
        method="sq-pcd",
        pool=False,
    )
    raw: list[Candidate] = []
    for source, tensor in (("ridge", prompter.ridge_tensor), ("valley", prompter.valley_tensor)):
        points = safe_get_prompts(
            prompter,
            source,
            prompts_per_polarity,
            quantile,
            min_component_size,
            seed + (0 if source == "ridge" else 100003),
        )
        positive_values = tensor[tensor > 0]
        response_norm = float(np.quantile(positive_values, 0.95)) if positive_values.size else 1.0
        response_norm = max(response_norm, 1e-12)
        for x, y in points:
            scale, width, contrast, homogeneity, response, angle = local_feature_score(
                tile, tensor, x, y, scales
            )
            if contrast < min_contrast or width < min_width:
                continue
            response_score = min(response / response_norm, 3.0)
            homogeneity_score = 1.0 / (1.0 + homogeneity / 20.0)
            score = contrast * (0.35 + response_score) * homogeneity_score
            raw.append(
                Candidate(
                    x=x + crop.x0,
                    y=y + crop.y0,
                    tile_id=crop.crop_id,
                    source=source,
                    scale=scale,
                    width=width,
                    contrast=contrast,
                    homogeneity=homogeneity,
                    response=response,
                    score=float(score),
                    angle_deg=angle,
                )
            )
    return select_spread(raw, max_prompts, min_prompt_distance)


def deduplicate_candidates(candidates: list[Candidate], min_distance: float) -> list[Candidate]:
    return select_spread(candidates, len(candidates), min_distance)


def mark_hard_zones(
    candidates: list[Candidate],
    cluster_radius: float = 150.0,
    min_cluster_size: int = 3,
    min_angle_span: float = 32.0,
) -> list[dict]:
    if len(candidates) < min_cluster_size:
        return []
    coords = np.asarray([[item.x, item.y] for item in candidates], dtype=float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(cluster_radius)
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)

    hard_regions: list[dict] = []
    for indices in groups.values():
        if len(indices) < min_cluster_size:
            continue
        angles = [candidates[index].angle_deg for index in indices]
        max_difference = max(
            (angular_distance(a, b) for i, a in enumerate(angles) for b in angles[i + 1 :]),
            default=0.0,
        )
        sources = {candidates[index].source for index in indices}
        if max_difference < min_angle_span and len(sources) < 2:
            continue
        for index in indices:
            candidates[index].hard_zone = True
        hard_regions.append(
            {
                "center": [float(coords[indices, 0].mean()), float(coords[indices, 1].mean())],
                "candidate_indices": indices,
                "angle_span_deg": float(max_difference),
                "sources": sorted(sources),
            }
        )
    return hard_regions


def load_truth(view_dir: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    semantic = np.asarray(Image.open(view_dir / "mask.png")) > 0
    centerline = np.asarray(Image.open(view_dir / "centerline_visible.png")) > 0
    instances = [
        np.asarray(Image.open(path)) > 0
        for path in sorted((view_dir / "instances").glob("cable_*_mask.png"))
    ]
    return semantic, centerline, instances


def prompt_metrics(
    candidates: Sequence[Candidate],
    semantic: np.ndarray,
    centerline: np.ndarray,
    instances: Sequence[np.ndarray],
    radius: float,
) -> dict:
    if not candidates:
        return {
            "count": 0,
            "precision_on_semantic_mask": 0.0,
            "centerline_coverage_at_radius": 0.0,
            "instance_prompt_counts": [0 for _ in instances],
        }
    points = np.asarray([[item.x, item.y] for item in candidates], dtype=float)
    rounded = np.rint(points).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, semantic.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, semantic.shape[0] - 1)
    hits = semantic[rounded[:, 1], rounded[:, 0]]
    center_coords_yx = np.argwhere(centerline)
    if len(center_coords_yx):
        center_coords_xy = center_coords_yx[:, ::-1]
        nearest, _ = cKDTree(points).query(center_coords_xy, k=1)
        coverage = float(np.mean(nearest <= radius))
    else:
        coverage = 0.0
    instance_counts = [
        int(mask[rounded[:, 1], rounded[:, 0]].sum()) for mask in instances
    ]
    return {
        "count": len(candidates),
        "precision_on_semantic_mask": float(hits.mean()),
        "centerline_coverage_at_radius": coverage,
        "coverage_radius_px": float(radius),
        "instance_prompt_counts": instance_counts,
        "ridge_count": sum(item.source == "ridge" for item in candidates),
        "valley_count": sum(item.source == "valley" for item in candidates),
        "hard_zone_count": sum(item.hard_zone for item in candidates),
    }


def draw_prompt_overlay(
    image: Image.Image,
    crops: Sequence[Crop],
    candidates: Sequence[Candidate],
    hard_regions: Sequence[dict],
    output: Path,
) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for crop in crops:
        draw.rectangle((crop.x0, crop.y0, crop.x1 - 1, crop.y1 - 1), outline=(70, 150, 255), width=3)
    radius = max(7, int(round(min(image.size) / 260)))
    for item in candidates:
        color = (40, 235, 80) if item.source == "ridge" else (235, 55, 215)
        if item.hard_zone:
            color = (255, 190, 20)
        draw.ellipse(
            (item.x - radius, item.y - radius, item.x + radius, item.y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        length = 35
        theta = math.radians(item.angle_deg)
        draw.line(
            (
                item.x - length * math.cos(theta),
                item.y - length * math.sin(theta),
                item.x + length * math.cos(theta),
                item.y + length * math.sin(theta),
            ),
            fill=color,
            width=3,
        )
    for region in hard_regions:
        x, y = region["center"]
        draw.ellipse((x - 95, y - 95, x + 95, y + 95), outline=(255, 215, 0), width=7)
    canvas.thumbnail((1952, 1098), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def detect_view(args, view_dir: Path) -> dict:
    started = time.perf_counter()
    image_pil = Image.open(view_dir / "rgb.png").convert("RGB")
    image = np.asarray(image_pil)
    crops = regular_crops(image.shape[1], image.shape[0], args.tile_size, args.overlap)
    candidates: list[Candidate] = []
    for index, crop in enumerate(crops):
        print(
            f"[{view_dir.name}] ridge crop {index + 1}/{len(crops)} {crop.crop_id}",
            flush=True,
        )
        candidates.extend(
            detect_crop(
                image=image,
                crop=crop,
                scales=args.scales,
                prompts_per_polarity=args.prompts_per_polarity,
                quantile=args.quantile,
                min_component_size=args.min_component_size,
                min_contrast=args.min_contrast,
                min_width=args.min_width,
                max_prompts=args.max_prompts_per_crop,
                min_prompt_distance=args.local_min_distance,
                seed=args.seed + index,
            )
        )
    candidates = deduplicate_candidates(candidates, args.global_min_distance)
    hard_regions = mark_hard_zones(candidates)
    semantic, centerline, instances = load_truth(view_dir)
    metrics = prompt_metrics(candidates, semantic, centerline, instances, args.coverage_radius)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["regular_crop_count"] = len(crops)

    view_output = args.output / view_dir.name
    view_output.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(view_dir / "rgb.png"),
        "image_size": list(image_pil.size),
        "config": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "scales": args.scales,
            "prompts_per_polarity": args.prompts_per_polarity,
            "quantile": args.quantile,
            "min_component_size": args.min_component_size,
            "min_contrast": args.min_contrast,
            "min_width": args.min_width,
            "max_prompts_per_crop": args.max_prompts_per_crop,
        },
        "crops": [asdict(crop) for crop in crops],
        "candidates": [asdict(item) for item in candidates],
        "hard_regions": hard_regions,
        "metrics": metrics,
    }
    json_dump(view_output / "prompts.json", payload)
    draw_prompt_overlay(
        image_pil,
        crops,
        candidates,
        hard_regions,
        view_output / "prompt_overlay.jpg",
    )
    print(
        f"[{view_dir.name}] prompts={len(candidates)} precision={metrics['precision_on_semantic_mask']:.3f} "
        f"centerline@{args.coverage_radius:g}={metrics['centerline_coverage_at_radius']:.3f} "
        f"elapsed={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload


def binary_metrics(prediction: np.ndarray, truth: np.ndarray, centerline: np.ndarray) -> dict:
    prediction = prediction.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    component_count = max(0, cv2.connectedComponents(prediction.astype(np.uint8), 8)[0] - 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "centerline_recall": float(prediction[centerline].mean()) if centerline.any() else 0.0,
        "component_count": int(component_count),
        "predicted_pixels": int(prediction.sum()),
    }


def load_sam31(checkpoint: Path, repo: Path):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    print(f"Loading SAM3.1 checkpoint: {checkpoint}", flush=True)
    started = time.perf_counter()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint),
            max_num_objects=1,
            use_fa3=False,
            use_rope_real=False,
            async_loading_frames=False,
        )

    original_init_state = predictor.model.init_state

    def init_state_compat(**kwargs):
        kwargs.pop("offload_state_to_cpu", None)
        return original_init_state(**kwargs)

    predictor.model.init_state = init_state_compat
    original_build_output = predictor.model._build_sam2_output

    def build_output_compat(inference_state, frame_idx, refined_obj_id_to_mask=None):
        if frame_idx not in inference_state["cached_frame_outputs"]:
            return dict(refined_obj_id_to_mask or {})
        return original_build_output(inference_state, frame_idx, refined_obj_id_to_mask)

    predictor.model._build_sam2_output = build_output_compat
    print(
        f"SAM3.1 loaded in {time.perf_counter() - started:.1f}s on {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    return predictor


def component_at_point(mask: np.ndarray, point_xy: tuple[float, float]) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, 8)
    if count <= 1:
        return mask.astype(bool)
    x = int(np.clip(round(point_xy[0]), 0, mask.shape[1] - 1))
    y = int(np.clip(round(point_xy[1]), 0, mask.shape[0] - 1))
    label_id = int(labels[y, x])
    if label_id == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == label_id


def proposal_quality(
    mask: np.ndarray,
    positive_xy: tuple[float, float],
    candidate_width: float | None = None,
    width_area_factor: float | None = None,
    *,
    positive_points: Sequence[Sequence[float]] | None = None,
    negative_points: Sequence[Sequence[float]] = (),
    component_mode: str = "seed",
    width_normalized: bool = False,
) -> tuple[bool, dict]:
    min_area = 80
    if width_normalized:
        if candidate_width is None:
            raise ValueError("Width-normalized filtering requires an estimated cable width")
        min_area = width_normalized_limits(candidate_width)["point_min_area"]
    component_rows = None
    if component_mode == "seed":
        mask = component_at_point(mask, positive_xy)
    elif component_mode == "prompted":
        mask, component_rows = select_prompted_components(
            mask, positive_points or [positive_xy], negative_points, min_area=min_area,
        )
    else:
        raise ValueError("component_mode must be seed or prompted")
    area = int(mask.sum())
    crop_area = mask.size
    if area == 0:
        return False, {"area": 0, "reason": "positive_not_in_mask" if component_mode == "seed"
                       else "no_supported_components", "component_evidence": component_rows}
    ys, xs = np.nonzero(mask)
    bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    bbox_fill = area / max(bbox_area, 1)
    coords = np.column_stack((xs, ys)).astype(np.float64)
    if len(coords) >= 4:
        covariance = np.cov(coords, rowvar=False)
        eigvals = np.maximum(np.linalg.eigvalsh(covariance), 1e-6)
        elongation = float(math.sqrt(eigvals[-1] / eigvals[0]))
    else:
        elongation = 1.0
    area_fraction = area / crop_area
    max_area_from_width = (
        float(width_area_factor * candidate_width * max(mask.shape))
        if candidate_width is not None and width_area_factor is not None
        else None
    )
    accepted = (
        area >= min_area
        and area_fraction <= 0.16
        and (elongation >= 1.8 or bbox_fill <= 0.42)
        and (max_area_from_width is None or area <= max_area_from_width)
    )
    reason = "accepted" if accepted else (
        "width_area_filter" if max_area_from_width is not None and area > max_area_from_width
        else "shape_or_area_filter"
    )
    return accepted, {
        "area": area,
        "area_fraction": area_fraction,
        "bbox_fill": bbox_fill,
        "elongation": elongation,
        "max_area_from_width": max_area_from_width,
        "reason": reason,
        "component_mask": mask,
        "component_mode": component_mode,
        "min_area": min_area,
        "component_evidence": component_rows,
    }


def text_proposal_quality(mask: np.ndarray, candidate_width: float | None = None) -> tuple[bool, dict]:
    limits = width_normalized_limits(candidate_width) if candidate_width is not None else None
    min_area = limits["text_min_area"] if limits else 180
    min_long_side = limits["min_long_side"] if limits else 80
    area = int(np.count_nonzero(mask))
    if area < min_area:
        return False, {"area": area, "reason": "too_small"}
    ys, xs = np.nonzero(mask)
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    bbox_fill = area / max(bbox_width * bbox_height, 1)
    coords = np.column_stack((xs, ys)).astype(np.float64)
    eigenvalues = np.maximum(np.linalg.eigvalsh(np.cov(coords, rowvar=False)), 1e-6)
    elongation = float(math.sqrt(eigenvalues[-1] / eigenvalues[0]))
    accepted = (
        area / mask.size <= 0.18
        and max(bbox_width, bbox_height) >= min_long_side
        and (elongation >= 1.6 or bbox_fill <= 0.42)
    )
    return accepted, {
        "area": area,
        "bbox": [int(xs.min()), int(ys.min()), bbox_width, bbox_height],
        "bbox_fill": bbox_fill,
        "elongation": elongation,
        "reason": "accepted" if accepted else "shape_or_area_filter",
    }


def save_raw_proposal(mask: np.ndarray, crop: Crop, image_size, directory: Path | None,
                      name: str, accepted: bool, quality: dict, selected_mask=None,
                      points=(), labels=(), model_size=None, text_prompt=None) -> dict:
    """Opt-in audit cache, including rejected outputs and all disconnected pieces.

    The labels NPZ preserves every component without producing thousands of tiny
    files. Component IDs reconstruct exact masks; raw files never enter fusion
    through mask_path, which remains reserved for accepted selected proposals.
    """
    if directory is None:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    positives = [p for p, label in zip(points, labels) if label == 1]
    negatives = [p for p, label in zip(points, labels) if label == 0]
    component_labels, rows = component_evidence(mask, positives, negatives)
    for row in rows:
        row["selected_pixels"] = int(np.count_nonzero(
            (component_labels == row["component_id"]) & selected_mask)) if selected_mask is not None else 0
    mask_path = directory / f"{name}_raw.png"
    label_path = directory / f"{name}_components.npz"
    evidence_path = directory / f"{name}_evidence.json"
    Image.fromarray(np.asarray(mask, np.uint8) * 255).save(mask_path)
    np.savez_compressed(label_path, labels=component_labels)
    evidence = {"accepted": bool(accepted), "quality": quality, "components": rows,
                "crop_xyxy": [crop.x0, crop.y0, crop.x1, crop.y1],
                "image_size": list(image_size), "points_local": points, "point_labels": labels,
                "points_normalized": normalize_points(points, crop.width, crop.height, model_size),
                "model_canvas_size": list(model_size or (crop.width, crop.height)),
                "text_prompt": text_prompt, "raw_mask_path": str(mask_path.resolve()),
                "component_labels_path": str(label_path.resolve()),
                "ground_truth_used": False}
    json_dump(evidence_path, evidence)
    return {"raw_mask_path": str(mask_path.resolve()), "component_labels_path": str(label_path.resolve()),
            "raw_evidence_path": str(evidence_path.resolve()), "raw_component_count": len(rows)}


def run_crop_text(
    predictor,
    image: Image.Image,
    crop: Crop,
    temp_dir: Path,
    prompt: str,
    preserve_aspect: bool = False,
    proposal_dir: Path | None = None,
    raw_proposal_dir: Path | None = None,
    candidate_width: float | None = None,
) -> tuple[list[dict], np.ndarray]:
    crop_image = image.crop((crop.x0, crop.y0, crop.x1, crop.y1))
    # Optional for rectangular external images; square training crops are unchanged.
    if preserve_aspect and crop.width != crop.height:
        model_image = SquareTransform(crop.width, crop.height).pad(crop_image)
    else:
        model_image = crop_image
    temp_dir.mkdir(parents=True, exist_ok=True)
    crop_path = temp_dir / f"{crop.crop_id}_text.png"
    model_image.save(crop_path)
    session_id = predictor.handle_request(
        {"type": "start_session", "resource_path": str(crop_path)}
    )["session_id"]
    accepted_union = np.zeros((crop.height, crop.width), dtype=bool)
    records = []
    try:
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        })
        output = response.get("outputs") or {}
        for object_id, raw_mask in zip(
            output.get("out_obj_ids", []), output.get("out_binary_masks", [])
        ):
            mask = np.asarray(raw_mask, dtype=bool)[:crop.height, :crop.width]
            accepted, quality = text_proposal_quality(mask, candidate_width)
            if accepted:
                accepted_union |= mask
            saved_proposal = None
            if accepted and proposal_dir is not None:
                proposal_dir.mkdir(parents=True, exist_ok=True)
                saved_proposal = proposal_dir / f"{crop.crop_id}_text_{int(object_id):03d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(saved_proposal)
            records.append({
                "object_id": int(object_id),
                "accepted": accepted,
                "quality": quality,
                "mask_path": str(saved_proposal.resolve()) if saved_proposal else None,
                "crop_xyxy": [crop.x0, crop.y0, crop.x1, crop.y1],
                "image_size": list(image.size),
                **save_raw_proposal(mask, crop, image.size, raw_proposal_dir,
                                    f"{crop.crop_id}_text_{int(object_id):03d}", accepted, quality,
                                    selected_mask=mask if accepted else None,
                                    model_size=model_image.size, text_prompt=prompt),
            })
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id, "run_gc_collect": False}
        )
    return records, accepted_union


def point_prompt(candidate: Candidate, crop: Crop, use_negatives: bool,
                 width_normalized: bool = False) -> tuple[list[list[float]], list[int]]:
    px = candidate.x - crop.x0
    py = candidate.y - crop.y0
    points = [[px, py]]
    labels = [1]
    if use_negatives and not candidate.hard_zone:
        theta = math.radians(candidate.angle_deg)
        nx, ny = -math.sin(theta), math.cos(theta)
        offset = (width_normalized_limits(candidate.width)["negative_offset"] if width_normalized
                  else float(np.clip(candidate.width * 0.95, 18.0, 52.0)))
        for sign in (-1.0, 1.0):
            x = px + sign * offset * nx
            y = py + sign * offset * ny
            if 3 <= x < crop.width - 3 and 3 <= y < crop.height - 3:
                points.append([x, y])
                labels.append(0)
    return points, labels


def predict_points(predictor, session_id: str, crop: Crop, positive_points,
                   negative_points=(), model_size=None):
    """Independent 1–3-positive hypothesis; local (x,y), optional square canvas."""
    if not 1 <= len(positive_points) <= 3:
        raise ValueError("Use one to three explicitly verified positive points")
    points = [[float(x), float(y)] for x, y in [*positive_points, *negative_points]]
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    normalized = normalize_points(points, crop.width, crop.height, model_size)
    if set(map(tuple, positive_points)) & set(map(tuple, negative_points)):
        raise ValueError("The same point cannot be both positive and negative")
    predictor.handle_request({"type": "reset_session", "session_id": session_id})
    response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "obj_id": 1,
            "points": normalized,
            "point_labels": labels,
            "clear_old_points": True,
        }
    )
    output = response.get("outputs") or {}
    for obj_id, mask in zip(output.get("out_obj_ids", []), output.get("out_binary_masks", [])):
        if int(obj_id) == 1:
            array = np.asarray(mask, dtype=bool)
            mw, mh = model_size or (crop.width, crop.height)
            if array.shape != (mh, mw):
                raise ValueError("Predictor mask does not match the requested canvas")
            return array[:crop.height, :crop.width], points, labels
    return np.zeros((crop.height, crop.width), dtype=bool), points, labels


def predict_one(predictor, session_id: str, crop: Crop, candidate: Candidate, use_negatives: bool,
                extra_positive_points=(), extra_negative_points=(), model_size=None,
                width_normalized: bool = False):
    points, labels = point_prompt(candidate, crop, use_negatives, width_normalized)
    positives = [p for p, label in zip(points, labels) if label == 1] + list(extra_positive_points)
    negatives = [p for p, label in zip(points, labels) if label == 0] + list(extra_negative_points)
    return predict_points(predictor, session_id, crop, positives, negatives, model_size)


def boundary_seeds(mask: np.ndarray, crop: Crop, candidate: Candidate) -> list[dict]:
    """Offer a positive point on an accepted mask where it exits a crop."""
    height, width = mask.shape
    if not mask.any():
        return []
    result: list[dict] = []
    edge_specs = (
        ("left", mask[:, :3], 1, 0, -1, 0),
        ("right", mask[:, width - 3 :], width - 2, 0, 1, 0),
        ("top", mask[:3, :], 1, 1, 0, -1),
        ("bottom", mask[height - 3 :, :], height - 2, 1, 0, 1),
    )
    for edge_name, band, fixed, axis, dx, dy in edge_specs:
        positions = np.flatnonzero(np.any(band, axis=1 if axis == 0 else 0))
        if not len(positions):
            continue
        groups = np.split(positions, np.flatnonzero(np.diff(positions) > 32) + 1)
        for group in groups:
            if not len(group):
                continue
            across = int(np.median(group))
            edge_x = fixed if axis == 0 else across
            edge_y = across if axis == 0 else fixed
            target_x = int(np.clip(edge_x - dx * 32, 0, width - 1))
            target_y = int(np.clip(edge_y - dy * 32, 0, height - 1))
            ya, yb = max(0, target_y - 18), min(height, target_y + 19)
            xa, xb = max(0, target_x - 18), min(width, target_x + 19)
            patch_points = np.argwhere(mask[ya:yb, xa:xb])
            if not len(patch_points):
                continue
            nearest = patch_points[np.argmin(
                (patch_points[:, 1] + xa - target_x) ** 2
                + (patch_points[:, 0] + ya - target_y) ** 2
            )]
            result.append({
                "x": float(crop.x0 + xa + nearest[1]),
                "y": float(crop.y0 + ya + nearest[0]),
                "dx": dx,
                "dy": dy,
                "edge": edge_name,
                "source_crop": crop.crop_id,
                "candidate": asdict(candidate),
            })
    return result


def candidates_in_crop(
    candidates: Sequence[Candidate],
    crop: Crop,
    margin: int,
    limit: int,
    covered: np.ndarray | None = None,
) -> list[Candidate]:
    selected = []
    for item in candidates:
        if not (
            crop.x0 + margin <= item.x < crop.x1 - margin
            and crop.y0 + margin <= item.y < crop.y1 - margin
        ):
            continue
        if covered is not None:
            ix, iy = int(round(item.x)), int(round(item.y))
            if covered[iy, ix]:
                continue
        selected.append(item)
    return select_spread(selected, limit, 80.0)


def run_crop_sam(
    predictor,
    image: Image.Image,
    crop: Crop,
    candidates: Sequence[Candidate],
    temp_dir: Path,
    use_negatives: bool,
    width_area_factor: float | None = None,
    proposal_dir: Path | None = None,
    *,
    raw_proposal_dir: Path | None = None,
    positive_groups: Sequence[Sequence[Sequence[float]]] | None = None,
    negative_groups: Sequence[Sequence[Sequence[float]]] | None = None,
    component_mode: str = "seed",
    width_normalized: bool = False,
    preserve_aspect: bool = False,
) -> tuple[list[dict], np.ndarray, list[dict]]:
    crop_image = image.crop((crop.x0, crop.y0, crop.x1, crop.y1))
    for groups in (positive_groups, negative_groups):
        if groups is not None and len(groups) != len(candidates):
            raise ValueError("Prompt groups must have one entry per candidate")
    model_image = SquareTransform(crop.width, crop.height).pad(crop_image) if preserve_aspect else crop_image
    temp_dir.mkdir(parents=True, exist_ok=True)
    crop_path = temp_dir / f"{crop.crop_id}.png"
    model_image.save(crop_path)
    session_id = predictor.handle_request(
        {"type": "start_session", "resource_path": str(crop_path)}
    )["session_id"]
    accepted_union = np.zeros((crop.height, crop.width), dtype=bool)
    records: list[dict] = []
    exit_seeds: list[dict] = []
    try:
        for index, candidate in enumerate(candidates):
            mask, points, labels = predict_one(
                predictor, session_id, crop, candidate, use_negatives,
                extra_positive_points=positive_groups[index] if positive_groups is not None else (),
                extra_negative_points=negative_groups[index] if negative_groups is not None else (),
                model_size=model_image.size, width_normalized=width_normalized,
            )
            accepted, quality = proposal_quality(
                mask, (candidate.x - crop.x0, candidate.y - crop.y0),
                candidate_width=candidate.width,
                width_area_factor=width_area_factor,
                positive_points=[p for p, label in zip(points, labels) if label == 1],
                negative_points=[p for p, label in zip(points, labels) if label == 0],
                component_mode=component_mode, width_normalized=width_normalized,
            )
            component_mask = quality.pop("component_mask", np.zeros_like(mask, dtype=bool))
            if accepted:
                accepted_union |= component_mask
                exit_seeds.extend(boundary_seeds(component_mask, crop, candidate))
            saved_proposal = None
            if accepted and proposal_dir is not None:
                proposal_dir.mkdir(parents=True, exist_ok=True)
                saved_proposal = proposal_dir / f"{crop.crop_id}_{index:03d}.png"
                Image.fromarray(component_mask.astype(np.uint8) * 255).save(saved_proposal)
            records.append(
                {
                    "candidate": asdict(candidate),
                    "points_local": points,
                    "labels": labels,
                    "accepted": accepted,
                    "quality": quality,
                    "mask_path": str(saved_proposal.resolve()) if saved_proposal else None,
                    "crop_xyxy": [crop.x0, crop.y0, crop.x1, crop.y1],
                    "image_size": list(image.size),
                    **save_raw_proposal(mask, crop, image.size, raw_proposal_dir,
                                        f"{crop.crop_id}_{index:03d}", accepted, quality,
                                        selected_mask=component_mask, points=points, labels=labels,
                                        model_size=model_image.size),
                }
            )
            print(
                f"  {crop.crop_id} proposal {index + 1}/{len(candidates)} "
                f"accepted={accepted} area={quality.get('area', 0)}",
                flush=True,
            )
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id, "run_gc_collect": False}
        )
    return records, accepted_union, exit_seeds


def make_square_crop(
    center_x: float,
    center_y: float,
    size: int,
    image_width: int,
    image_height: int,
    crop_id: str,
    reason: str,
) -> Crop:
    size = min(size, image_width, image_height)
    x0 = int(round(center_x - size / 2))
    y0 = int(round(center_y - size / 2))
    x0 = int(np.clip(x0, 0, image_width - size))
    y0 = int(np.clip(y0, 0, image_height - size))
    return Crop(crop_id, x0, y0, x0 + size, y0 + size, reason)


def confirm_candidates_with_local_text(
    predictor,
    image: Image.Image,
    candidates: Sequence[Candidate],
    temp_dir: Path,
    crop_size: int = 256,
    prompt: str = "cable",
    prefix: str = "verify",
    raw_proposal_dir: Path | None = None,
    preserve_aspect: bool = False,
) -> tuple[list[Candidate], list[dict]]:
    """Keep points supported by a local semantic cable query, without labels."""
    accepted_candidates: list[Candidate] = []
    records: list[dict] = []
    for index, candidate in enumerate(candidates):
        crop = make_square_crop(
            candidate.x, candidate.y, crop_size, image.width, image.height,
            f"{prefix}_{index}", "local_cable_text_confirmation",
        )
        objects, local_mask = run_crop_text(predictor, image, crop, temp_dir, prompt,
                                           raw_proposal_dir=raw_proposal_dir,
                                           preserve_aspect=preserve_aspect)
        px = int(np.clip(round(candidate.x - crop.x0), 0, crop.width - 1))
        py = int(np.clip(round(candidate.y - crop.y0), 0, crop.height - 1))
        passed = bool(local_mask[py, px])
        records.append({
            "candidate": asdict(candidate),
            "crop": asdict(crop),
            "passed": passed,
            "accepted_text_objects": sum(int(item["accepted"]) for item in objects),
            "objects": objects,
        })
        if passed:
            accepted_candidates.append(candidate)
    return accepted_candidates, records


def adaptive_crops(
    candidates: Sequence[Candidate],
    hard_regions: Sequence[dict],
    initial_mask: np.ndarray,
    image_width: int,
    image_height: int,
    limit: int,
) -> list[Crop]:
    dilated = cv2.dilate(initial_mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    proposals: list[tuple[float, float, int, str, float]] = []
    for region in hard_regions:
        x, y = region["center"]
        proposals.append((x, y, 1280, "crossing_or_dense_geometry", 1e9))
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        ix = int(np.clip(round(candidate.x), 0, image_width - 1))
        iy = int(np.clip(round(candidate.y), 0, image_height - 1))
        if not dilated[iy, ix]:
            proposals.append((candidate.x, candidate.y, 768, "uncovered_ridge_prompt", candidate.score))

    selected: list[Crop] = []
    centers: list[tuple[float, float]] = []
    for x, y, size, reason, _ in proposals:
        if any(math.hypot(x - ox, y - oy) < 260 for ox, oy in centers):
            continue
        selected.append(
            make_square_crop(
                x,
                y,
                size,
                image_width,
                image_height,
                f"a{len(selected):02d}",
                reason,
            )
        )
        centers.append((x, y))
        if len(selected) >= limit:
            break
    return selected


def clean_semantic_mask(mask: np.ndarray, min_component_area: int = 180) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    cleaned = np.zeros_like(mask_u8)
    for label_id in range(1, count):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == label_id] = 1
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cleaned > 0


def select_feedback_seed(
    exit_seeds: list[dict],
    current_mask: np.ndarray,
    attempted: list[tuple[float, float, int, int]],
) -> dict | None:
    height, width = current_mask.shape
    for seed in exit_seeds:
        x, y = seed["x"], seed["y"]
        dx, dy = seed["dx"], seed["dy"]
        probe_x, probe_y = int(round(x + dx * 155)), int(round(y + dy * 155))
        if not (24 <= probe_x < width - 24 and 24 <= probe_y < height - 24):
            continue
        if any(
            old_dx == dx
            and old_dy == dy
            and math.hypot(x - old_x, y - old_y) < 220
            for old_x, old_y, old_dx, old_dy in attempted
        ):
            continue
        neighbourhood = current_mask[probe_y - 24 : probe_y + 25, probe_x - 24 : probe_x + 25]
        if np.any(neighbourhood):
            continue
        return seed
    return None


def save_mask_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    result = rgb.copy()
    color = np.asarray([20.0, 160.0, 255.0], dtype=np.float32)
    result[mask] = result[mask] * 0.45 + color * 0.55
    shown = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    shown.thumbnail((1952, 1098), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    shown.save(path, quality=92)


def run_sam_view(args, predictor, view_dir: Path, detector_payload: dict) -> dict:
    image = Image.open(view_dir / "rgb.png").convert("RGB")
    width, height = image.size
    candidates = [Candidate(**item) for item in detector_payload["candidates"]]
    hard_regions = detector_payload["hard_regions"]
    crops = [Crop(**item) for item in detector_payload["crops"]]
    semantic, centerline, _ = load_truth(view_dir)
    view_output = args.output / view_dir.name
    temp_dir = view_output / "_crop_cache"
    proposal_dir = view_output / "proposals" if getattr(args, "save_proposals", False) else None
    raw_dir = view_output / "raw_proposals" if getattr(args, "save_raw_proposals", False) else None
    text_options = {"raw_proposal_dir": raw_dir,
                    "preserve_aspect": getattr(args, "preserve_aspect", False)}
    point_options = {**text_options, "component_mode": getattr(args, "component_mode", "seed"),
                     "width_normalized": getattr(args, "width_normalized", False)}

    initial_mask = np.zeros((height, width), dtype=bool)
    records: list[dict] = []
    local_verification_records: list[dict] = []
    exit_seeds: list[dict] = []
    started = time.perf_counter()
    for index, crop in enumerate(crops):
        local_candidates = [item for item in candidates if item.tile_id == crop.crop_id]
        local_candidates = select_spread(
            local_candidates, args.sam_prompts_per_crop, args.local_min_distance
        )
        if args.local_text_gate and local_candidates:
            local_candidates, verification = confirm_candidates_with_local_text(
                predictor, image, local_candidates, temp_dir,
                crop_size=args.local_text_size, prompt=args.text_prompt,
                prefix=f"{crop.crop_id}_verify",
                **text_options,
            )
            local_verification_records.extend(verification)
        if not local_candidates:
            continue
        print(
            f"[{view_dir.name}] SAM regular crop {index + 1}/{len(crops)} "
            f"{crop.crop_id} prompts={len(local_candidates)}",
            flush=True,
        )
        crop_records, crop_union, crop_exits = run_crop_sam(
            predictor,
            image,
            crop,
            local_candidates,
            temp_dir,
            args.use_negatives,
            width_area_factor=args.width_area_factor,
            proposal_dir=proposal_dir,
            **point_options,
        )
        initial_mask[crop.y0 : crop.y1, crop.x0 : crop.x1] |= crop_union
        exit_seeds.extend(crop_exits)
        records.append({"crop": asdict(crop), "proposals": crop_records})
    initial_mask = clean_semantic_mask(initial_mask)
    initial_metrics = binary_metrics(initial_mask, semantic, centerline)
    Image.fromarray(initial_mask.astype(np.uint8) * 255).save(view_output / "mask_initial.png")
    save_mask_overlay(image, initial_mask, view_output / "overlay_initial.jpg")

    second_crops = adaptive_crops(
        candidates,
        hard_regions,
        initial_mask,
        width,
        height,
        args.max_adaptive_crops,
    )
    final_mask = initial_mask.copy()
    adaptive_records: list[dict] = []
    covered = cv2.dilate(initial_mask.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    for index, crop in enumerate(second_crops):
        local_candidates = candidates_in_crop(
            candidates,
            crop,
            margin=48,
            limit=args.sam_prompts_per_crop,
            covered=None if crop.reason == "crossing_or_dense_geometry" else covered,
        )
        if args.local_text_gate and local_candidates:
            local_candidates, verification = confirm_candidates_with_local_text(
                predictor, image, local_candidates, temp_dir,
                crop_size=args.local_text_size, prompt=args.text_prompt,
                prefix=f"{crop.crop_id}_verify",
                **text_options,
            )
            local_verification_records.extend(verification)
        if not local_candidates:
            continue
        print(
            f"[{view_dir.name}] SAM adaptive crop {index + 1}/{len(second_crops)} "
            f"{crop.crop_id} reason={crop.reason} prompts={len(local_candidates)}",
            flush=True,
        )
        crop_records, crop_union, crop_exits = run_crop_sam(
            predictor,
            image,
            crop,
            local_candidates,
            temp_dir,
            args.use_negatives,
            width_area_factor=args.width_area_factor,
            proposal_dir=proposal_dir,
            **point_options,
        )
        final_mask[crop.y0 : crop.y1, crop.x0 : crop.x1] |= crop_union
        exit_seeds.extend(crop_exits)
        adaptive_records.append({"crop": asdict(crop), "proposals": crop_records})
    final_mask = clean_semantic_mask(final_mask)
    final_metrics = binary_metrics(final_mask, semantic, centerline)
    Image.fromarray(final_mask.astype(np.uint8) * 255).save(view_output / "mask_adaptive.png")
    save_mask_overlay(image, final_mask, view_output / "overlay_adaptive.jpg")

    feedback_records: list[dict] = []
    attempted: list[tuple[float, float, int, int]] = []
    for feedback_index in range(args.max_feedback_crops):
        seed = select_feedback_seed(exit_seeds, final_mask, attempted)
        if seed is None:
            break
        x, y, dx, dy = seed["x"], seed["y"], seed["dx"], seed["dy"]
        attempted.append((x, y, dx, dy))
        crop = make_square_crop(
            x + dx * 420,
            y + dy * 420,
            args.feedback_crop_size,
            width,
            height,
            f"f{feedback_index:02d}",
            "sam_mask_boundary_continuation",
        )
        seed_candidate = replace(
            Candidate(**seed["candidate"]),
            x=x,
            y=y,
            tile_id=crop.crop_id,
            source="sam_feedback",
            score=1e6,
            hard_zone=True,
        )
        if not (crop.x0 + 8 <= x < crop.x1 - 8 and crop.y0 + 8 <= y < crop.y1 - 8):
            continue
        if args.local_text_gate:
            verified, verification = confirm_candidates_with_local_text(
                predictor, image, [seed_candidate], temp_dir,
                crop_size=args.local_text_size, prompt=args.text_prompt,
                prefix=f"{crop.crop_id}_verify",
                **text_options,
            )
            local_verification_records.extend(verification)
            if not verified:
                continue
        print(
            f"[{view_dir.name}] SAM feedback crop {feedback_index + 1} "
            f"{crop.crop_id} from={seed['source_crop']} edge={seed['edge']}",
            flush=True,
        )
        crop_records, crop_union, crop_exits = run_crop_sam(
            predictor, image, crop, [seed_candidate], temp_dir, use_negatives=False,
            width_area_factor=args.width_area_factor,
            proposal_dir=proposal_dir,
            **point_options,
        )
        existing = final_mask[crop.y0 : crop.y1, crop.x0 : crop.x1]
        gained = int(np.count_nonzero(crop_union & ~existing))
        if gained >= args.feedback_min_new_pixels:
            existing |= crop_union
            exit_seeds.extend(crop_exits)
        feedback_records.append({
            "crop": asdict(crop),
            "source_seed": {key: value for key, value in seed.items() if key != "candidate"},
            "new_pixels": gained,
            "accepted": gained >= args.feedback_min_new_pixels,
            "proposals": crop_records,
        })
    final_mask = clean_semantic_mask(final_mask)
    feedback_metrics = binary_metrics(final_mask, semantic, centerline)
    Image.fromarray(final_mask.astype(np.uint8) * 255).save(view_output / "mask_feedback.png")
    save_mask_overlay(image, final_mask, view_output / "overlay_feedback.jpg")

    text_records: list[dict] = []
    if args.use_text_rescue:
        for crop in crops:
            crop_records, crop_union = run_crop_text(
                predictor, image, crop, temp_dir, args.text_prompt, proposal_dir=proposal_dir,
                **text_options,
            )
            existing = final_mask[crop.y0 : crop.y1, crop.x0 : crop.x1]
            gained = int(np.count_nonzero(crop_union & ~existing))
            existing |= crop_union
            text_records.append({
                "crop": asdict(crop),
                "prompt": args.text_prompt,
                "new_pixels": gained,
                "objects": crop_records,
            })
            print(
                f"[{view_dir.name}] SAM text crop {crop.crop_id} "
                f"objects={len(crop_records)} gained={gained}",
                flush=True,
            )
    final_mask = clean_semantic_mask(final_mask)
    text_metrics = binary_metrics(final_mask, semantic, centerline)
    elapsed = time.perf_counter() - started
    text_metrics["elapsed_seconds"] = elapsed

    Image.fromarray(final_mask.astype(np.uint8) * 255).save(view_output / "mask_final.png")
    save_mask_overlay(image, final_mask, view_output / "overlay_final.jpg")
    result = {
        "image_size": [width, height],
        "initial_metrics": initial_metrics,
        "adaptive_metrics": final_metrics,
        "feedback_metrics": feedback_metrics,
        "text_metrics": text_metrics,
        "adaptive_crops": [asdict(crop) for crop in second_crops],
        "regular_crop_records": records,
        "adaptive_crop_records": adaptive_records,
        "feedback_crop_records": feedback_records,
        "text_crop_records": text_records,
        "local_text_verification_records": local_verification_records,
    }
    json_dump(view_output / "sam_results.json", result)
    if proposal_dir is not None:
        instance_proposals = []
        for batch in records + adaptive_records + [row for row in feedback_records if row["accepted"]]:
            instance_proposals.extend(row for row in batch["proposals"] if row.get("mask_path"))
        for batch in text_records:
            instance_proposals.extend(row for row in batch["objects"] if row.get("mask_path"))
        json_dump(view_output / "instance_proposals.json",
                  {"image_size": [width, height], "proposals": instance_proposals})
    print(
        f"[{view_dir.name}] initial IoU={initial_metrics['iou']:.3f} recall={initial_metrics['recall']:.3f}; "
        f"adaptive IoU={final_metrics['iou']:.3f} recall={final_metrics['recall']:.3f}; "
        f"feedback IoU={feedback_metrics['iou']:.3f} recall={feedback_metrics['recall']:.3f}; "
        f"text IoU={text_metrics['iou']:.3f} recall={text_metrics['recall']:.3f}; "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def summarize_detector(payloads: Sequence[dict]) -> dict:
    keys = ["precision_on_semantic_mask", "centerline_coverage_at_radius", "count", "elapsed_seconds"]
    summary = {"views": len(payloads)}
    for key in keys:
        values = [
            float(payload["metrics"][key])
            for payload in payloads
            if key in payload.get("metrics", {})
        ]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else 0.0
        summary[f"std_{key}"] = float(np.std(values)) if values else 0.0
    return summary


def summarize_sam(results: Sequence[dict]) -> dict:
    summary = {"views": len(results)}
    for stage in ("initial_metrics", "adaptive_metrics", "feedback_metrics", "text_metrics"):
        for key in ("precision", "recall", "iou", "f1", "centerline_recall", "component_count"):
            values = [float(result[stage][key]) for result in results]
            summary[f"{stage}_{key}_mean"] = float(np.mean(values)) if values else 0.0
            summary[f"{stage}_{key}_std"] = float(np.std(values)) if values else 0.0
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage A: tiled GeoMPrompt ridge prompting and adaptive SAM3.1 visible-cable masks"
    )
    parser.add_argument("--mode", choices=("prompts", "sam"), default="prompts")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam-repo", type=Path, default=DEFAULT_SAM_REPO)
    parser.add_argument("--views", nargs="*", default=None, help="e.g. view_000 view_005")
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--reuse-prompts", action="store_true")
    parser.add_argument("--save-proposals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-raw-proposals", action="store_true",
                        help="Audit all accepted/rejected raw outputs and component labels; no fusion change")
    parser.add_argument("--component-mode", choices=("seed", "prompted"), default="seed",
                        help="Experimental prompted mode preserves all explicitly positive-supported components")
    parser.add_argument("--width-normalized", action="store_true",
                        help="Experimental width-normalized point area gate and negative offsets")
    parser.add_argument("--preserve-aspect", action="store_true",
                        help="Pad rectangular point/text crops to square and transform prompt coordinates")
    parser.add_argument(
        "--prompt-source-output",
        type=Path,
        default=None,
        help="Optional detector output root to read prompts from without overwriting it",
    )
    parser.add_argument("--prompt-filename", default="prompts.json")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[36, 64, 100, 144, 196, 256],
        help="Lindeberg scale values t=sigma^2",
    )
    parser.add_argument("--prompts-per-polarity", type=int, default=10)
    parser.add_argument("--quantile", type=float, default=0.65)
    parser.add_argument("--min-component-size", type=int, default=5)
    parser.add_argument("--min-contrast", type=float, default=7.0)
    parser.add_argument("--min-width", type=float, default=11.0)
    parser.add_argument("--max-prompts-per-crop", type=int, default=8)
    parser.add_argument("--local-min-distance", type=float, default=72.0)
    parser.add_argument("--global-min-distance", type=float, default=56.0)
    parser.add_argument("--coverage-radius", type=float, default=64.0)
    parser.add_argument("--sam-prompts-per-crop", type=int, default=6)
    parser.add_argument("--max-adaptive-crops", type=int, default=6)
    parser.add_argument("--max-feedback-crops", type=int, default=0)
    parser.add_argument("--feedback-crop-size", type=int, default=1024)
    parser.add_argument("--feedback-min-new-pixels", type=int, default=400)
    parser.add_argument("--use-text-rescue", action="store_true")
    parser.add_argument("--text-prompt", default="cable")
    parser.add_argument("--local-text-gate", action="store_true",
                        help="Confirm ridge points using local SAM cable-text inference")
    parser.add_argument("--local-text-size", type=int, default=256)
    parser.add_argument("--width-area-factor", type=float, default=None,
                        help="Optional max point mask area = factor * ridge width * crop long side")
    parser.add_argument("--use-negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def resolve_views(args) -> list[Path]:
    scene = args.dataset / "scene_000000"
    views = sorted(path for path in scene.glob("view_*" ) if (path / "rgb.png").is_file())
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
    views = resolve_views(args)
    detector_payloads: list[dict] = []
    for view_dir in views:
        prompt_root = args.prompt_source_output or args.output
        cached = prompt_root / view_dir.name / args.prompt_filename
        if args.reuse_prompts and cached.is_file():
            payload = json.loads(cached.read_text(encoding="utf-8"))
        elif args.reuse_prompts:
            raise FileNotFoundError(f"Prompt file not found: {cached}")
        else:
            payload = detect_view(args, view_dir)
        detector_payloads.append(payload)
    detector_summary = summarize_detector(detector_payloads)
    json_dump(args.output / "detector_summary.json", detector_summary)
    print(f"DETECTOR_SUMMARY {json.dumps(detector_summary, ensure_ascii=False)}", flush=True)

    if args.mode == "prompts":
        return 0
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM3.1 checkpoint not found: {args.checkpoint}")
    if not args.sam_repo.is_dir():
        raise FileNotFoundError(f"SAM3.1 repository not found: {args.sam_repo}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SAM3.1 checkpoint")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = load_sam31(args.checkpoint, args.sam_repo)
    sam_results = [
        run_sam_view(args, predictor, view_dir, payload)
        for view_dir, payload in zip(views, detector_payloads)
    ]
    sam_summary = summarize_sam(sam_results)
    json_dump(args.output / "sam_summary.json", sam_summary)
    print(f"SAM_SUMMARY {json.dumps(sam_summary, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
