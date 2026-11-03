"""Deterministic opaque-occlusion stress tests, separate from natural occlusion.

GT is used to construct this artificial benchmark and to score saved outputs,
never as a model forward input. All instance masks lose the occluder pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .checkpoint import atomic_json, build_from_checkpoint, new_directory, sha256
from .data import NativeInstanceDataset
from .evaluate import evaluate_model


PROTOCOL = "controlled_opaque_occlusion_v1"
SEVERITIES = (0, 1, 2, 4, 8)


def stable_digest(value):
    return hashlib.sha256((PROTOCOL+"|"+str(value)).encode("utf-8")).hexdigest()


def fixed_indices(rows, count=300):
    if count < 1 or count > len(rows):
        raise ValueError("Requested fixed subset exceeds available images")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    return sorted(range(len(rows)), key=lambda i: (stable_digest(ids[i]), ids[i]))[:count]


def rectangle(shape, center, tangent, width, severity):
    y, x = np.ogrid[:shape[0], :shape[1]]
    dx, dy = x-center[0], y-center[1]
    along = dx*tangent[0]+dy*tangent[1]
    across = -dx*tangent[1]+dy*tangent[0]
    return (np.abs(along) <= severity*width/2) & (np.abs(across) <= 3*width)


def _roi(mask, center, limit=128):
    ys, xs = np.nonzero(mask)
    order = np.lexsort((xs, ys, (xs-center[0])**2+(ys-center[1])**2))[:limit]
    return {"pixels_xy": np.column_stack((xs[order], ys[order])).tolist()}


def build_occlusion_plan(masks, sample_id, native_size):
    masks = np.asarray(masks, bool)
    h, w = map(int, native_size)
    masks = masks[:, :h, :w]
    rng = np.random.default_rng(int(stable_digest(sample_id)[:16], 16))
    order = rng.permutation(len(masks))
    for instance in order:
        mask = masks[instance]
        if mask.sum() < 64:
            continue
        distance = ndimage.distance_transform_edt(mask)
        candidates = np.argwhere(mask & (distance >= ndimage.maximum_filter(distance, size=3)))
        if not len(candidates):
            continue
        for index in rng.permutation(len(candidates))[:32]:
            cy, cx = candidates[index]
            width = max(2., 2*float(distance[cy, cx]))
            radius = int(np.ceil(5*width))
            ya, yb, xa, xb = max(0, cy-radius), min(h, cy+radius+1), max(0, cx-radius), min(w, cx+radius+1)
            yy, xx = np.nonzero(mask[ya:yb, xa:xb])
            points = np.column_stack((xx+xa-cx, yy+ya-cy)).astype(float)
            if len(points) < 8:
                continue
            covariance = points.T@points / len(points)
            _, vectors = np.linalg.eigh(covariance)
            tangent = vectors[:, -1]
            if tangent[0] < 0 or (tangent[0] == 0 and tangent[1] < 0):
                tangent = -tangent
            center = [int(cx), int(cy)]
            occluder = rectangle((h, w), center, tangent, width, 8)
            cc, count = ndimage.label(mask & ~occluder, np.ones((3, 3), bool))
            if count < 2:
                continue
            areas = np.bincount(cc.ravel())
            substantial = np.flatnonzero(areas >= 16)
            substantial = substantial[substantial != 0]
            if len(substantial) < 2:
                continue
            # Stable two largest visible pieces; no prediction/model result used.
            pieces = sorted(substantial.tolist(), key=lambda n: (-areas[n], n))[:2]
            regions = [_roi(cc == component, center) for component in pieces]
            events = [dict(id=f"{sample_id}:controlled-positive", same_instance=True,
                           regions=regions, source_instance_indices=[int(instance), int(instance)],
                           kind="artificial_occlusion", ambiguous=False)]
            negative = []
            for j, other in enumerate(masks):
                visible = other & ~occluder
                if j == instance or visible.sum() < 16:
                    continue
                ys, xs = np.nonzero(visible)
                negative.append((float(np.min((xs-cx)**2+(ys-cy)**2)), j, visible))
            if negative:
                _, j, visible = min(negative, key=lambda row: (row[0], row[1]))
                events.append(dict(id=f"{sample_id}:controlled-negative", same_instance=False,
                                   regions=[regions[0], _roi(visible, center)],
                                   source_instance_indices=[int(instance), int(j)],
                                   kind="artificial_nearby_negative", ambiguous=False))
            return dict(applicable=True, center_xy=center, tangent_xy=tangent.tolist(),
                        local_width_pixels=width, source_instance=int(instance), events=events,
                        selection="hash-seeded GT geometry only; event ROIs fixed outside maximum severity")
    return dict(applicable=False, reason="no_eligible_two_visible_piece_cut_found_in_fixed_32_candidates_per_instance",
                events=[])


def apply_occlusion(sample, plan, severity):
    if severity not in SEVERITIES:
        raise ValueError("Severity must be one of 0,1,2,4,8 local widths")
    result = {**sample, "meta": dict(sample["meta"])}
    image, masks = sample["image"].clone(), sample["masks"].clone()
    h, w = sample["meta"]["original_size"]
    occluder = np.zeros(image.shape[-2:], bool)
    if plan["applicable"] and severity:
        occluder[:h, :w] = rectangle((h, w), plan["center_xy"], plan["tangent_xy"],
                                    plan["local_width_pixels"], severity)
        cover = torch.from_numpy(occluder)
        image[:, cover] = .5
        masks[:, cover] = 0
    retained = torch.nonzero(masks.flatten(1).any(1), as_tuple=False).flatten()
    masks = masks[retained]
    result.update(image=image, masks=masks)
    original_indices = sample["meta"].get("source_instance_indices", list(range(len(sample["masks"]))))
    result["meta"].update(source_instance_indices=[original_indices[i] for i in retained.tolist()],
        augmentation=PROTOCOL, occlusion=dict(applicable=plan["applicable"], severity_widths=severity,
        occluder_pixels=int(occluder.sum()), color_rgb=[.5, .5, .5],
        plan={k: v for k, v in plan.items() if k != "events"}), identity_events=plan["events"])
    return result


class OcclusionDataset:
    def __init__(self, dataset, severity, count=300):
        self.dataset, self.severity = dataset, severity
        self.indices = fixed_indices(dataset.rows, count)
        self.rows = [dataset.rows[i] for i in self.indices]
        self.split = dataset.split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.dataset[self.indices[index]]
        plan = build_occlusion_plan(sample["masks"].numpy(), sample["id"], sample["meta"]["original_size"])
        return apply_occlusion(sample, plan, self.severity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", default="none", choices=("none", "bf16", "fp16"))
    parser.add_argument("--sam-repo"); parser.add_argument("--sam-checkpoint")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--losses", action="store_true")
    args = parser.parse_args(argv)
    out = new_directory(args.output)
    model, ckpt = build_from_checkpoint(args.checkpoint, device=args.device,
        sam_repo=args.sam_repo, sam_checkpoint=args.sam_checkpoint)
    config = ckpt["train_config"]
    dataset = NativeInstanceDataset(args.manifest, "test", ckpt["model_config"]["canvas_size"],
                                    config["seed"], True, args.data_root)
    from .losses import SetCriterion
    criterion = SetCriterion(**config["loss"]).to(args.device) if args.losses else None
    rows = []
    for severity in SEVERITIES:
        stress = OcclusionDataset(dataset, severity, args.count)
        report = evaluate_model(model, stress, device=args.device, amp=args.amp,
            object_threshold=config["object_threshold"], mask_threshold=config["mask_threshold"],
            criterion=criterion, identity_diagnostics=True, attention_diagnostics=False,
            diagnostic_output=out, attention_samples=12,
            diagnostic_subdir=f"diagnostic_samples/severity_{severity}")
        report.update(protocol=PROTOCOL, severity_widths=severity,
                      selected_ids=[row["id"] for row in stress.rows],
                      applicable_images=sum(bool(row.get("metadata", {}).get("occlusion", {}).get("applicable")) for row in report["per_image"]),
                      no_cut_policy="fixed selected images retained unchanged when no eligible cut; never replaced by easier images",
                      interpretation="Artificial opaque occlusion; no natural-occlusion or scene-independence claim")
        atomic_json(out/f"severity_{severity}.json", report)
        rows.append({k: v for k, v in report.items() if k != "per_image"})
    atomic_json(out/"completion.json", dict(protocol=PROTOCOL, checkpoint_sha256=sha256(args.checkpoint),
        manifest_sha256=sha256(args.manifest), reports=rows,
        status="completed" if all(r["failures"] == 0 and r["complete"] and not r.get("diagnostic_failures") for r in rows) else "completed_with_failures",
        resampling_unit="original image; all severities of one image stay paired"))
    return 0 if all(r["failures"] == 0 and r["complete"] and not r.get("diagnostic_failures") for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
