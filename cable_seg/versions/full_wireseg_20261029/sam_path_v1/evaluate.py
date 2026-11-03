"""Whole-split RGB-only evaluation; failures stay in the image denominator."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import atomic_json, build_from_checkpoint, new_directory, sha256
from .data import NativeInstanceDataset, collate_fn
from .inference import autocast_context, decode_instances, padding_valid
from .metrics import score_image, summarize


def scalar_losses(losses, weights):
    """Detached logging only; reject nonfinite components rather than plotting them."""
    raw = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
           for key, value in losses.items()}
    if set(raw) != {*weights, "total"} or not all(math.isfinite(v) for v in raw.values()):
        raise FloatingPointError("Loss components are missing or nonfinite")
    weighted = {key: raw[key] * float(weight) for key, weight in weights.items()}
    if not math.isclose(sum(weighted.values()), raw["total"], rel_tol=2e-5, abs_tol=2e-6):
        raise ValueError("Weighted loss components do not sum to total")
    return raw, weighted


def loss_summary(rows, weights):
    """Do not present a successful-only mean as a complete split loss."""
    observed = [row for row in rows if row.get("loss") is not None]
    complete = bool(rows) and len(observed) == len(rows)
    means = {key: sum(row["loss"][key] for row in observed)/len(observed)
             for key in (*weights, "total")} if observed else None
    weighted = {key: means[key]*weight for key, weight in weights.items()} if means else None
    return dict(aggregation="arithmetic mean of per-image criterion values", images=len(rows),
                observed_images=len(observed), failed_images=len(rows)-len(observed), complete=complete,
                weights=weights, mean=means if complete else None,
                weighted_mean=weighted if complete else None,
                observed_mean=means, observed_weighted_mean=weighted)


@torch.no_grad()
def evaluate_model(model, dataset, *, device="cpu", amp="none", object_threshold=.5, mask_threshold=.5,
                   criterion=None):
    was_training = model.training
    model.eval()
    rows = []
    try:
        for index in range(len(dataset)):
            sample, error, calls = None, None, None
            raw_loss, weighted_loss, loss_error = None, None, None
            start = time.perf_counter()
            try:
                sample = dataset[index]
                batch = collate_fn([sample])
                images = batch["images"].to(device)
                input_valid = padding_valid(images, batch["targets"])
                with autocast_context(device, amp):
                    outputs = model(images, input_valid)
                calls = outputs.get("calls")
                if calls != {"encoder": 1, "decoder": 1}:
                    raise RuntimeError(f"Single-forward call contract failed: {calls}")
                prediction = decode_instances(outputs, images.shape[-2:], input_valid,
                                              object_threshold=object_threshold,
                                              mask_threshold=mask_threshold)[0]
                score = score_image(prediction["masks"], sample["masks"].numpy(), sample["valid"].numpy())
                if criterion is not None:
                    # Supervision is consumed only after the one RGB-only forward.
                    try:
                        targets = [{**target, "masks": target["masks"].to(device)}
                                   for target in batch["targets"]]
                        with autocast_context(device, amp):
                            losses = criterion(outputs, targets, batch["valid"].to(device))
                        raw_loss, weighted_loss = scalar_losses(losses, criterion.weights)
                    except Exception as exc:
                        loss_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if sample is not None:
                    gt = sample["masks"].numpy()
                    score = score_image(np.zeros((0, *gt.shape[1:]), bool), gt,
                                        sample["valid"].numpy(), failed=True)
                else:
                    # Unknown GT cardinality cannot be honestly added to global PQ.
                    score = dict(instance_iou=0., union_iou=0., tp=0, fp=0, fn=0, matched_iou=0.,
                                 n_pred=0, n_gt=None, empty_prediction=True, failed=True, gt_available=False)
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            record = dataset.rows[index] if hasattr(dataset, "rows") else {"id": index}
            score.update(id=sample["id"] if sample is not None else record.get("id", index),
                         domain=(sample.get("meta", {}).get("domain", "unspecified") if sample is not None
                                 else record.get("domain", "unspecified")),
                         condition=(sample.get("meta", {}).get("source_condition_group", "unspecified")
                                    if sample is not None else record.get("source_condition_group", "unspecified")),
                         seconds=time.perf_counter()-start, calls=calls, error=error)
            rows.append(score)
            if criterion is not None:
                score.update(loss=raw_loss, weighted_loss=weighted_loss, loss_error=loss_error or error)
    finally:
        model.train(was_training)
    report = summarize(rows)
    report["by_domain"] = {domain: summarize([row for row in rows if row["domain"] == domain])
                           for domain in sorted({row["domain"] for row in rows})}
    report["by_condition"] = {condition: summarize([row for row in rows if row["condition"] == condition])
                             for condition in sorted({row["condition"] for row in rows})}
    report.update(split=dataset.split, object_threshold=object_threshold, mask_threshold=mask_threshold,
                  forward_input="RGB + padding-only validity; no GT prompts or overlap validity",
                  per_image=rows, total_seconds=sum(row["seconds"] for row in rows))
    if criterion is not None:
        report["losses"] = loss_summary(rows, criterion.weights)
        for condition, metrics in report["by_condition"].items():
            metrics["losses"] = loss_summary([row for row in rows if row["condition"] == condition], criterion.weights)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation", choices=("train", "val", "validation", "test", "check"))
    parser.add_argument("--data-root")
    parser.add_argument("--allow-condition", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", default="none", choices=("none", "bf16", "fp16"))
    parser.add_argument("--sam-repo"); parser.add_argument("--sam-checkpoint")
    parser.add_argument("--losses", action="store_true", help="Also report supervised split losses from the same forward")
    args = parser.parse_args(argv)
    destination = new_directory(args.output)
    model, checkpoint = build_from_checkpoint(args.checkpoint, device=args.device,
                                             sam_repo=args.sam_repo, sam_checkpoint=args.sam_checkpoint)
    config = checkpoint["train_config"]
    dataset = NativeInstanceDataset(args.manifest, args.split, checkpoint["model_config"]["canvas_size"],
                                    config["seed"], args.allow_condition, args.data_root)
    # Thresholds travel with the trained checkpoint. Test/check CLIs cannot tune them.
    from .losses import SetCriterion
    criterion = SetCriterion(**config["loss"]).to(args.device) if args.losses else None
    report = evaluate_model(model, dataset, device=args.device, amp=args.amp,
                            object_threshold=config["object_threshold"], mask_threshold=config["mask_threshold"],
                            criterion=criterion)
    report.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
                  manifest_sha256=sha256(args.manifest), backend=checkpoint["model_config"]["backend"],
                  checkpoint_progress=checkpoint["progress"])
    atomic_json(destination / "metrics.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, ensure_ascii=False))
    return 0 if report["complete"] and not report["failures"] and report.get("losses", {}).get("complete", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
