"""Training-free prompt geometry; no model calls, labels, or file side effects.

All point coordinates are (x, y) in the local crop unless stated otherwise.
Discovery seeds are hypotheses, not automatically trusted foreground prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class SquareTransform:
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Crop dimensions must be positive")

    @property
    def side(self) -> int:
        return max(self.width, self.height)

    def pad(self, image: Image.Image) -> Image.Image:
        if image.size != (self.width, self.height):
            raise ValueError("Image and transform dimensions disagree")
        fill = tuple(int(x) for x in np.median(np.asarray(image.convert("RGB")), axis=(0, 1)))
        canvas = Image.new("RGB", (self.side, self.side), fill)
        canvas.paste(image.convert("RGB"), (0, 0))
        return canvas

    def normalize(self, points) -> list[list[float]]:
        return normalize_points(points, self.width, self.height, (self.side, self.side))

    def restore_mask(self, mask: np.ndarray) -> np.ndarray:
        array = np.asarray(mask)
        if array.shape != (self.side, self.side):
            raise ValueError("Expected predictor mask at padded input resolution")
        return array[:self.height, :self.width].copy()


def normalize_points(points, width: int, height: int, model_size=None) -> list[list[float]]:
    """Reject outside-crop points instead of silently clamping them onto cables."""
    if width <= 0 or height <= 0:
        raise ValueError("Crop dimensions must be positive")
    mw, mh = model_size or (width, height)
    if mw < width or mh < height:
        raise ValueError("Model canvas must contain the crop")
    result = []
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError("Point lies outside the visible crop")
        result.append([float(x / mw), float(y / mh)])
    return result


def width_normalized_limits(width: float) -> dict:
    """Match legacy minimum point/text areas at a reference diameter of 12 px.

    Area scales quadratically and length linearly. These are experimental
    heuristics, not proof that a component is a cable.
    """
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Cable width must be positive and finite")
    scale = width / 12.0
    return {"point_min_area": max(4, int(round(80 * scale * scale))),
            "text_min_area": max(4, int(round(180 * scale * scale))),
            "min_long_side": max(4, int(round(80 * scale))),
            "negative_offset": max(2.0, 0.95 * width)}


def component_evidence(mask, positives=(), negatives=()) -> tuple[np.ndarray, list[dict]]:
    """Preserve every component and annotate explicit prompt support/contradiction."""
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("Mask must be two-dimensional")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, 8)
    height, width = array.shape
    normalize_points([*positives, *negatives], width, height)
    def ids(points):
        # Nearest-pixel lookup follows legacy prompting, with the last pixel bounded.
        return [int(labels[min(height-1, int(round(y))), min(width-1, int(round(x)))]) for x, y in points]
    pos_ids, neg_ids = ids(positives), ids(negatives)
    rows = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        rows.append({"component_id": label, "bbox_xywh": [x, y, w, h], "area": area,
                     "positive_count": pos_ids.count(label), "negative_count": neg_ids.count(label)})
    return labels, rows


def select_prompted_components(mask, positives, negatives=(), min_area: int = 1):
    """Keep disconnected visible pieces only when individually positive-supported.

    A negative-containing component is rejected conservatively. No gap filling
    or geometrical bridge is used, and unsupported pieces remain in the audit.
    """
    labels, rows = component_evidence(mask, positives, negatives)
    selected = np.zeros(labels.shape, bool)
    for row in rows:
        row["selected"] = bool(row["positive_count"] > 0 and row["negative_count"] == 0
                               and row["area"] >= min_area)
        if row["selected"]:
            selected |= labels == row["component_id"]
    return selected, rows


def endpoint_seeds(mask: np.ndarray, max_seeds: int = 24, min_separation: float = 12.0) -> list[dict]:
    """Observed endpoint and outward tangent; seeds stay on visible foreground."""
    if max_seeds <= 0:
        return []
    array = np.asarray(mask, bool)
    skeleton = skeletonize(array)
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    degree = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    coords = np.column_stack(np.where(skeleton))
    distance = cv2.distanceTransform(array.astype(np.uint8), cv2.DIST_L2, 5)
    seeds = []
    for y, x in np.column_stack(np.where(skeleton & (degree == 1))):
        nearby = coords[np.sum((coords - [y, x]) ** 2, axis=1) <= 24**2]
        if len(nearby) < 4:
            continue
        dy, dx = np.array([y, x]) - np.mean(nearby, axis=0)
        norm = math.hypot(dx, dy)
        if norm < 1 or any(math.hypot(x-r["x"], y-r["y"]) < min_separation for r in seeds):
            continue
        seeds.append({"x": float(x), "y": float(y), "dx": float(dx/norm), "dy": float(dy/norm),
                      "width": max(2.0, float(2*distance[y, x])), "source": "observed_endpoint",
                      "requires_image_verification": True})
        if len(seeds) >= max_seeds:
            break
    return seeds


def sparse_uncovered_seeds(mask: np.ndarray, spacing: int = 64, max_seeds: int = 64,
                           exclusion_radius: float = 0.0) -> list[dict]:
    """Deterministic raster-stratified discovery grid; never label it positive yet."""
    if spacing <= 0 or exclusion_radius < 0:
        raise ValueError("Spacing must be positive; exclusion radius nonnegative")
    if max_seeds <= 0:
        return []
    array = np.asarray(mask, bool)
    height, width = array.shape
    distance = cv2.distanceTransform((~array).astype(np.uint8), cv2.DIST_L2, 5)
    ys = range(min(spacing//2, height-1), height, spacing)
    xs = range(min(spacing//2, width-1), width, spacing)
    seeds = [{"x": float(x), "y": float(y), "source": "uncovered_grid",
              "requires_image_verification": True}
             for y in ys for x in xs if not array[y, x] and distance[y, x] >= exclusion_radius]
    if len(seeds) > max_seeds:
        indices = np.linspace(0, len(seeds)-1, max_seeds).round().astype(int)
        seeds = [seeds[i] for i in indices]
    return seeds
