"""Stage B prototype: visible skeleton fragments, conservative ID links, occlusion cues.

Hidden links are graph hypotheses only; they never add pixels to the visible mask.
Wide multi-colour bundles remain explicitly ambiguous, not invented instances.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def arc_endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    if len(coords) < 2:
        return []
    pixels = set(map(tuple, coords.tolist()))
    ends = []
    for y, x in coords:
        degree = sum((int(y + dy), int(x + dx)) in pixels
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        if degree <= 1:
            ends.append((int(y), int(x)))
    if len(ends) > 2:
        # Skeletons can have tiny spurs. Taking the first two raster-order tips
        # often chooses two tips at the same end of a long cable, which makes
        # downstream continuity and occlusion reasoning impossible.
        points = np.asarray(ends, dtype=np.float32)
        distances = np.sum((points[:, None] - points[None, :]) ** 2, axis=2)
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        ends = [ends[first], ends[second]]
    return ends


def endpoint_direction(coords: np.ndarray, endpoint: tuple[int, int]) -> np.ndarray:
    point = np.asarray(endpoint, dtype=float)
    distances = np.linalg.norm(coords.astype(float) - point, axis=1)
    near = coords[(distances > 0) & (distances <= 18)]
    if len(near) < 2:
        near = coords[np.argsort(distances)[1: min(len(coords), 10)]]
    if not len(near):
        return np.zeros(2)
    vector = point - np.median(near, axis=0)
    return vector / max(np.linalg.norm(vector), 1e-8)


def build_graph(rgb: np.ndarray, mask: np.ndarray, min_arc: int = 24,
                max_gap: int = 70) -> tuple[np.ndarray, dict]:
    skeleton = skeletonize(mask)
    neighbors = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skeleton
    junction = skeleton & (neighbors >= 4)
    # A small junction disk prevents all crossing arms from collapsing into one arc.
    break_zone = cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    arcs = skeleton & ~break_zone
    labels, count = ndi.label(arcs, structure=np.ones((3, 3), dtype=np.uint8))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    radius = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    arc_map = np.zeros(mask.shape, dtype=np.int32)
    nodes = []
    for original_id in range(1, count + 1):
        coords = np.column_stack(np.where(labels == original_id))
        if len(coords) < min_arc:
            continue
        arc_id = len(nodes) + 1
        arc_map[coords[:, 0], coords[:, 1]] = arc_id
        endpoints = arc_endpoints(coords)
        sampled = coords[::max(1, len(coords) // 150)]
        colors = lab[sampled[:, 0], sampled[:, 1]]
        widths = radius[sampled[:, 0], sampled[:, 1]] * 2
        nodes.append({
            "arc_id": arc_id, "length_pixels": int(len(coords)),
            "center": [float(coords[:, 1].mean()), float(coords[:, 0].mean())],
            "median_lab": np.median(colors, axis=0).round(2).tolist(),
            "median_width": float(np.median(widths)),
            "width_q75": float(np.percentile(widths, 75)),
            "endpoints": [{"xy": [x, y], "outward_yx": endpoint_direction(coords, (y, x)).tolist()}
                          for y, x in endpoints],
        })
    # Absolute width cannot distinguish a thick single cable from a merged
    # bundle. Use the scene's thinner visible arcs as a relative reference.
    reference_width = float(np.percentile([n["median_width"] for n in nodes], 25)) if nodes else 0.0
    for node in nodes:
        node["wide_bundle_candidate"] = bool(
            node["median_width"] > max(26.0, 1.8 * reference_width)
            and node["width_q75"] > max(30.0, 2.0 * reference_width)
        )
    parent = list(range(len(nodes) + 1))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    proposals = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            color_distance = float(np.linalg.norm(np.asarray(a["median_lab"]) - b["median_lab"]))
            width_ratio = max(a["median_width"], b["median_width"]) / max(min(a["median_width"], b["median_width"]), 1)
            if color_distance > 32 or width_ratio > 2.5:
                continue
            for ai, ea in enumerate(a["endpoints"]):
                for bi, eb in enumerate(b["endpoints"]):
                    pa = np.asarray(ea["xy"][::-1], dtype=float)
                    pb = np.asarray(eb["xy"][::-1], dtype=float)
                    delta = pb - pa
                    gap = float(np.linalg.norm(delta))
                    if not 1 <= gap <= max_gap:
                        continue
                    direction = delta / gap
                    ta = np.asarray(ea["outward_yx"], dtype=float)
                    tb = np.asarray(eb["outward_yx"], dtype=float)
                    alignment_a = float(ta @ direction)
                    alignment_b = float(tb @ -direction)
                    if gap > 12 and min(alignment_a, alignment_b) < 0.55:
                        continue
                    cost = gap / max_gap + color_distance / 32 + (2 - alignment_a - alignment_b) * 0.25
                    proposals.append({"a": a["arc_id"], "b": b["arc_id"],
                                      "a_end": ai, "b_end": bi, "gap": gap,
                                      "color_distance": color_distance, "cost": cost})
    proposals.sort(key=lambda item: item["cost"])
    used = set()
    edges = []
    for edge in proposals:
        left, right = (edge["a"], edge["a_end"]), (edge["b"], edge["b_end"])
        if left in used or right in used:
            continue
        # Avoid forced identity decisions at a near-tie crossing.
        competitors = [item for item in proposals if item is not edge
                       and ((item["a"], item["a_end"]) in (left, right)
                            or (item["b"], item["b_end"]) in (left, right))]
        if competitors and min(x["cost"] for x in competitors) - edge["cost"] < 0.12:
            continue
        used.update((left, right))
        ra, rb = find(edge["a"]), find(edge["b"])
        if ra != rb:
            parent[rb] = ra
        edges.append({**edge, "kind": "visible_continuity" if edge["gap"] <= 12 else "occlusion_hypothesis"})
    roots = {root: new_id for new_id, root in enumerate(sorted({find(n["arc_id"]) for n in nodes}), 1)}
    arc_ids = np.zeros(len(nodes) + 1, dtype=np.uint16)
    for node in nodes:
        arc_ids[node["arc_id"]] = roots[find(node["arc_id"])]
        node["provisional_instance_id"] = int(arc_ids[node["arc_id"]])
    if np.any(arc_map):
        nearest = ndi.distance_transform_edt(arc_map == 0, return_distances=False, return_indices=True)
        nearest_arcs = arc_map[nearest[0], nearest[1]]
        instance_map = arc_ids[nearest_arcs] * mask.astype(np.uint16)
    else:
        instance_map = np.zeros(mask.shape, dtype=np.uint16)
    accepted_keys = {(e["a"], e["a_end"], e["b"], e["b_end"]) for e in edges}
    unresolved = [item for item in proposals
                  if (item["a"], item["a_end"], item["b"], item["b_end"]) not in accepted_keys
                  and item["gap"] > 12]
    feedback = []
    for edge in unresolved[:30]:
        first = nodes[edge["a"] - 1]["endpoints"][edge["a_end"]]["xy"]
        second = nodes[edge["b"] - 1]["endpoints"][edge["b_end"]]["xy"]
        feedback.append({"xy": [(first[0] + second[0]) / 2, (first[1] + second[1]) / 2],
                         "reason": "ambiguous_identity_gap", "gap": edge["gap"]})
    junction_labels, junction_count = ndi.label(junction)
    for junction_id in range(1, junction_count + 1):
        coords = np.column_stack(np.where(junction_labels == junction_id))
        if len(coords) >= 1:
            feedback.append({"xy": [float(coords[:, 1].mean()), float(coords[:, 0].mean())],
                             "reason": "junction_review", "junction_pixels": len(coords)})
    for node in nodes:
        if node["wide_bundle_candidate"]:
            feedback.append({"xy": node["center"], "reason": "merged_bundle_review",
                             "arc_id": node["arc_id"]})
    # A single long visible arc can conceal a missed intersecting cable. This is
    # a lower-priority context probe, not an assertion that the arc is a bundle.
    for node in nodes:
        if not node["wide_bundle_candidate"] and node["length_pixels"] >= 200:
            feedback.append({"xy": node["center"], "reason": "long_arc_context_review",
                             "arc_id": node["arc_id"]})
    filtered_feedback = []
    for item in feedback:
        if all(math.dist(item["xy"], old["xy"]) >= 90 for old in filtered_feedback):
            filtered_feedback.append(item)
    graph = {
        "visible_arc_count": len(nodes), "provisional_instance_count": len(roots),
        "nodes": nodes, "accepted_edges": edges,
        "occlusion_hypothesis_count": sum(e["kind"] == "occlusion_hypothesis" for e in edges),
        "unresolved_pair_count": len(unresolved),
        "feedback_regions": filtered_feedback[:40],
        "junction_pixels": int(junction.sum()),
        "wide_bundle_arc_count": sum(n["wide_bundle_candidate"] for n in nodes),
        "reference_single_cable_width": reference_width,
        "caveat": "IDs are provisional graph associations; broad merged bundles cannot be separated from a union mask alone. Hidden links are not painted into the visible mask.",
    }
    return instance_map, graph


def render(rgb: np.ndarray, instances: np.ndarray, output: Path) -> None:
    canvas = rgb.astype(np.float32).copy()
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        color = np.asarray([(53 * int(instance_id) + 45) % 210 + 30,
                            (97 * int(instance_id) + 65) % 210 + 30,
                            (149 * int(instance_id) + 90) % 210 + 30], dtype=np.float32)
        canvas[instances == instance_id] = 0.38 * canvas[instances == instance_id] + 0.62 * color
    Image.fromarray(canvas.astype(np.uint8)).save(output)


def appearance_instance_proposal(rgb: np.ndarray, mask: np.ndarray,
                                 graph: dict, merge_distance: float = 22.0,
                                 max_groups: int = 8) -> tuple[np.ndarray | None, dict]:
    """Optional color-supported IDs, without treating a color as a proven cable ID.

    Arc medians provide image-specific color prototypes. The proposal can split
    a skeleton arc that accidentally traverses two differently colored cables.
    Equal-colored distinct cables remain an explicit unresolved case.
    """
    nodes = sorted(graph["nodes"], key=lambda row: -row["length_pixels"])
    groups: list[dict] = []
    for node in nodes:
        color = np.asarray(node["median_lab"], dtype=np.float32)
        weight = math.sqrt(node["length_pixels"])
        nearest = min(range(len(groups)),
                      key=lambda i: float(np.linalg.norm(color - groups[i]["center"])),
                      default=None)
        if nearest is not None and np.linalg.norm(color - groups[nearest]["center"]) <= merge_distance:
            group = groups[nearest]
            group["center"] = (group["center"] * group["weight"] + color * weight) / (group["weight"] + weight)
            group["weight"] += weight
            group["arc_ids"].append(node["arc_id"])
        else:
            groups.append({"center": color.copy(), "weight": weight, "arc_ids": [node["arc_id"]]})
    audit = {"group_count": len(groups), "merge_distance": merge_distance,
             "groups": [{"center_lab": group["center"].round(2).tolist(),
                         "arc_ids": group["arc_ids"]} for group in groups],
             "caveat": "Appearance groups are hypotheses, not verified cable IDs; repeated colors can merge distinct cables."}
    if not 2 <= len(groups) <= max_groups:
        audit["status"] = "skipped_group_count"
        return None, audit
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    centers = np.asarray([group["center"] for group in groups], dtype=np.float32)
    result = np.zeros(mask.shape, dtype=np.uint16)
    ys, xs = np.where(mask)
    # Chunk foreground pixels to avoid a K x H x W float allocation at 4K.
    for start in range(0, len(ys), 250_000):
        yy = ys[start:start + 250_000]
        xx = xs[start:start + 250_000]
        colors = lab[yy, xx]
        distances = np.sum((colors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        result[yy, xx] = np.argmin(distances, axis=1).astype(np.uint16) + 1
    audit["status"] = "proposed"
    audit["pixel_count_by_group"] = [int(np.count_nonzero(result == i + 1)) for i in range(len(groups))]
    return result, audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-arc", type=int, default=24)
    p.add_argument("--max-gap", type=int, default=70)
    p.add_argument("--appearance-refine", action="store_true",
                   help="Write experimental color-supported instance proposals separately from graph IDs")
    p.add_argument("--appearance-merge-distance", type=float, default=22.0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mask_image = Image.open(args.mask).convert("L")
    image = Image.open(args.image).convert("RGB")
    if image.size != mask_image.size:
        image = image.resize(mask_image.size, Image.Resampling.LANCZOS)
    rgb = np.asarray(image)
    mask = np.asarray(mask_image) > 0
    instances, graph = build_graph(rgb, mask, args.min_arc, args.max_gap)
    graph["image_size"] = [image.width, image.height]
    Image.fromarray(instances).save(args.output / "provisional_instance_ids.png")
    render(rgb, instances, args.output / "provisional_instances_overlay.png")
    if args.appearance_refine:
        appearance, audit = appearance_instance_proposal(
            rgb, mask, graph, merge_distance=args.appearance_merge_distance)
        if appearance is not None:
            Image.fromarray(appearance).save(args.output / "appearance_instance_ids.png")
            render(rgb, appearance, args.output / "appearance_instances_overlay.png")
        (args.output / "appearance_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "identity_graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    print({key: graph[key] for key in ("visible_arc_count", "provisional_instance_count",
          "occlusion_hypothesis_count", "wide_bundle_arc_count")}, flush=True)


if __name__ == "__main__":
    main()
