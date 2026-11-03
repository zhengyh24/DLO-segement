"""Whole-split RGB-only evaluation; failures stay in the image denominator."""
from __future__ import annotations

import argparse
import hashlib
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
                   criterion=None, identity_diagnostics=False, event_annotations=None,
                   attention_diagnostics=False, diagnostic_output=None, attention_samples=12,
                   diagnostic_subdir="diagnostic_samples"):
    was_training = model.training
    model.eval()
    rows = []
    if type(attention_samples) is not int or attention_samples < 0:
        raise ValueError("attention_samples must be a nonnegative integer")
    ids = [str(row["id"]) for row in dataset.rows] if hasattr(dataset, "rows") else [str(i) for i in range(len(dataset))]
    selected = set(sorted(ids, key=lambda value: hashlib.sha256(("fixed_attention_v1|"+value).encode()).hexdigest())[:attention_samples]) if identity_diagnostics or attention_diagnostics else set()
    try:
        for index in range(len(dataset)):
            sample, error, calls = None, None, None
            raw_loss, weighted_loss, loss_error = None, None, None
            components, weighted_components = None, None
            identity, events, attention, diagnostic_error = None, None, None, None
            difficulty, artifacts = None, None
            start = time.perf_counter()
            try:
                sample = dataset[index]
                batch = collate_fn([sample])
                images = batch["images"].to(device)
                input_valid = padding_valid(images, batch["targets"])
                with autocast_context(device, amp):
                    outputs = model(images, input_valid)
                calls = outputs.get("calls")
                expected_calls = model.expected_calls() if hasattr(model, "expected_calls") else {"encoder": 1, "decoder": 1}
                if calls != expected_calls:
                    raise RuntimeError(f"Single-forward call contract failed: {calls}")
                prediction = decode_instances(outputs, images.shape[-2:], input_valid,
                                              object_threshold=object_threshold,
                                              mask_threshold=mask_threshold)[0]
                score = score_image(prediction["masks"], sample["masks"].numpy(), sample["valid"].numpy())
                # Diagnostics cannot affect predictions, matching, model selection,
                # or the frozen metric score. Their failures are separate receipts.
                if identity_diagnostics or attention_diagnostics or event_annotations:
                    try:
                        from .identity import visible_identity, identity_events, attention_trace, difficulty_metadata, attention_components, save_diagnostic_sample
                        if identity_diagnostics:
                            identity = visible_identity(prediction["masks"], sample["masks"].numpy(), sample["valid"].numpy())
                            h, w = sample["meta"]["original_size"]
                            difficulty = difficulty_metadata(sample["image"].numpy()[:, :h, :w].transpose(1, 2, 0),
                                sample["masks"].numpy()[:, :h, :w], sample["valid"].numpy()[..., :h, :w])
                        annotation = (event_annotations or {}).get(str(sample["id"]), sample.get("meta", {}).get("identity_events", []))
                        if annotation:
                            events = identity_events(prediction["masks"], annotation, sample["valid"].numpy())
                        if attention_diagnostics:
                            attention = attention_trace(outputs)
                        if str(sample["id"]) in selected:
                            attention, arrays = attention_components(outputs, sample["masks"].numpy(), sample["valid"].numpy(), prediction)
                            if diagnostic_output is not None:
                                folder = Path(diagnostic_output)/diagnostic_subdir
                                artifacts = save_diagnostic_sample(folder, sample, prediction, arrays)
                                artifacts["directory"] = diagnostic_subdir
                                artifacts["sha256"] = {key: sha256(folder/artifacts[key]) for key in ("panel", "prediction", "attention") if key in artifacts}
                    except Exception as exc:
                        diagnostic_error = f"{type(exc).__name__}: {exc}"
                if criterion is not None:
                    # Supervision is consumed only after the one RGB-only forward.
                    try:
                        targets = [{**target, "masks": target["masks"].to(device)}
                                   for target in batch["targets"]]
                        with autocast_context(device, amp):
                            losses = criterion(outputs, targets, batch["valid"].to(device))
                        raw_loss, weighted_loss = scalar_losses(losses, criterion.weights)
                        if hasattr(criterion, "last_components"):
                            components = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                                          for key, value in criterion.last_components.items()}
                            if not all(math.isfinite(v) for v in components.values()):
                                raise FloatingPointError("Nonfinite descriptor component")
                            weighted_components = {key: value*criterion.component_weights[key] for key, value in components.items()}
                            if not math.isclose(sum(components.values()), raw_loss["descriptor"], rel_tol=2e-5, abs_tol=2e-6):
                                raise ValueError("Descriptor components do not sum to descriptor loss")
                    except Exception as exc:
                        loss_error = f"{type(exc).__name__}: {exc}"
                        raw_loss, weighted_loss, components, weighted_components = None, None, None, None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if sample is not None:
                    gt = sample["masks"].numpy()
                    score = score_image(np.zeros((0, *gt.shape[1:]), bool), gt,
                                        sample["valid"].numpy(), failed=True)
                    if identity_diagnostics or event_annotations:
                        try:
                            from .identity import visible_identity, identity_events, difficulty_metadata
                            empty = np.zeros((0, *gt.shape[1:]), bool)
                            if identity_diagnostics:
                                identity = visible_identity(empty, gt, sample["valid"].numpy(), failed=True)
                                h, w = sample["meta"]["original_size"]
                                difficulty = difficulty_metadata(sample["image"].numpy()[:, :h, :w].transpose(1, 2, 0),
                                    gt[:, :h, :w], sample["valid"].numpy()[..., :h, :w])
                            annotation = (event_annotations or {}).get(str(sample["id"]), sample.get("meta", {}).get("identity_events", []))
                            if annotation:
                                events = identity_events(empty, annotation, sample["valid"].numpy(), failed=True)
                        except Exception as exc:
                            diagnostic_error = f"{type(exc).__name__}: {exc}"
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
            if identity_diagnostics or attention_diagnostics or event_annotations:
                score.update(identity=identity, identity_events=events, attention=attention,
                             difficulty=difficulty, diagnostic_artifacts=artifacts,
                             diagnostic_error=diagnostic_error,
                             metadata=sample.get("meta", {}) if sample is not None else record)
            rows.append(score)
            if criterion is not None:
                score.update(loss=raw_loss, weighted_loss=weighted_loss, loss_error=loss_error or error)
                if hasattr(criterion, "last_components"):
                    score.update(loss_components=components, weighted_loss_components=weighted_components)
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
        if hasattr(criterion, "component_weights"):
            observed = [r for r in rows if r.get("loss_components") is not None and r.get("loss_error") is None]
            complete = len(observed) == len(rows)
            means = {key: sum(r["loss_components"][key] for r in observed)/len(observed)
                     for key in criterion.component_weights} if observed else None
            report["loss_components"] = dict(images=len(rows), observed_images=len(observed), complete=complete,
                weights=criterion.component_weights, mean=means if complete else None, observed_mean=means,
                weighted_mean={key: value*criterion.component_weights[key] for key, value in means.items()} if complete and means else None,
                interpretation="descriptor loss decomposition; already included in descriptor and total")
    if identity_diagnostics:
        from .identity import difficulty_strata, DIFFICULTY_PROTOCOL
        report["difficulty_strata"] = difficulty_strata(rows)
        report["difficulty_protocol"] = DIFFICULTY_PROTOCOL
    if identity_diagnostics or attention_diagnostics:
        report["diagnostic_samples"] = dict(selected_ids=sorted(selected), maximum_images=attention_samples,
            saved_images=sum(r.get("diagnostic_artifacts") is not None for r in rows),
            selection="first IDs by SHA256('fixed_attention_v1|'+id), independent of scores and seeds",
            scope="soft attention mass assigned offline to visible GT components on fixed samples; no natural event annotation claim")
    if identity_diagnostics or event_annotations:
        from .identity import summarize_identity
        report["identity_diagnostics"] = summarize_identity(rows)
    if identity_diagnostics or attention_diagnostics or event_annotations:
        report["diagnostic_failures"] = sum(row.get("diagnostic_error") is not None for row in rows)
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
    parser.add_argument("--identity-diagnostics", action="store_true")
    parser.add_argument("--attention-diagnostics", action="store_true")
    parser.add_argument("--attention-samples", type=int, default=12, help="Fixed-hash diagnostic artifacts; identity flag alone enables this subset")
    parser.add_argument("--events", help="JSON object mapping image IDs to pre-frozen offline identity events")
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
                            criterion=criterion, identity_diagnostics=args.identity_diagnostics,
                            attention_diagnostics=args.attention_diagnostics,
                            diagnostic_output=destination, attention_samples=args.attention_samples,
                            event_annotations=json.loads(Path(args.events).read_text(encoding="utf-8")) if args.events else None)
    report.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
                  manifest_sha256=sha256(args.manifest), backend=checkpoint["model_config"]["backend"],
                  checkpoint_progress=checkpoint["progress"])
    atomic_json(destination / "metrics.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, ensure_ascii=False))
    return 0 if report["complete"] and not report["failures"] and not report.get("diagnostic_failures") and report.get("losses", {}).get("complete", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
