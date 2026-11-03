"""Bounded refiner training on frozen one-pass SAM caches; no SAM imports.

Formal training requires independently recorded native parity and 24GB capacity
gates. The --smoke path performs exactly two CPU updates on synthetic fixtures.
Validation loss selects a diagnostic checkpoint, never claims instance gains.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .types import CropSpec, PromptSpec, ProposalRecord
from .refiner import (ResidualRefiner, RefinerConfig, FEATURE_NAMES, clone_proposals,
                      schedule_rois, extract_roi, round_limits, commit_predictions)


def rgb_sha(rgb):
    return hashlib.sha256(str(rgb.shape).encode() + np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _resolve(base, value):
    p = Path(value)
    return p if p.is_absolute() else base / p


def _file_sha(path):
    from .io import sha256
    return sha256(path)


def validate_native_cache(audit, plan, proposals, gate=None):
    """Fail closed on mock, refined, truncated or uninstrumented SAM caches.

    These are integrity/provenance checks, not authentication against a malicious
    writer able to replace both arrays and their receipts.
    """
    sam = audit.get("sam", {})
    empty = not plan.prompts and not proposals
    expected = {"backbone_forward_image": 1, "image_encoder": 1,
                "prompt_encoder": 1, "mask_decoder": 1,
                "detector": 0, "propagation_decoder": 0, "text_encoder": 0}
    if empty:
        expected = {key: 0 for key in expected}
    if (audit.get("cache_kind") != "actual_single_batch_sam"
            or audit.get("ground_truth_used_for_inference") is not False
            or sam.get("status") != ("empty_prompts" if empty else "ok")
            or sam.get("injected_test_model") is not False
            or sam.get("sam_version") != "3.1_multiplex_interactive"
            or sam.get("calls") != expected or sam.get("microbatches") != (0 if empty else 1)
            or sam.get("retries") != 0 or sam.get("input_resize") is not False
            or sam.get("canvas_size") != 1008
            or sam.get("residency_policy") not in ("native_full_predictor", "experimental_interactive_only")):
        raise ValueError("Formal training requires a non-injected, instrumented native single-batch SAM cache")
    digest = sam.get("checkpoint_sha256")
    if not (empty and digest is None) and (not isinstance(digest, str) or len(digest) != 64
                                         or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("SAM cache lacks a checkpoint SHA256 fingerprint")
    sources = sam.get("source_files_sha256")
    if not isinstance(sources, dict) or not sources or any(
            not isinstance(value, str) or len(value) != 64 for value in sources.values()):
        raise ValueError("SAM cache lacks complete source file fingerprints")
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if sam.get("source_fingerprint") != fingerprint:
        raise ValueError("SAM cache source fingerprint does not match its file hashes")
    coverage = sam.get("weight_coverage") or {}
    if not empty and (not str(sam.get("device", "")).startswith("cuda") or coverage.get("missing") != []
            or not isinstance(coverage.get("used_parameters_found"), int)
            or coverage["used_parameters_found"] <= 0):
        raise ValueError("SAM cache lacks real CUDA execution/loaded-weight coverage evidence")
    h, w = plan.ridge.shape
    for crop in plan.crops:
        x0, y0, x1, y1 = crop.xyxy
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
                and max(x1-x0, y1-y0) <= 1008):
            raise ValueError("SAM cache has invalid native crop coordinates")
    if (sam.get("requested_crops") != len(plan.crops) or sam.get("requested_prompts") != len(plan.prompts)
            or {p.prompt.prompt_id for p in proposals} != {p.prompt_id for p in plan.prompts}
            or len(proposals) != len(plan.prompts)):
        raise ValueError("SAM cache proposal/call cardinality mismatch")
    if any(not np.array_equal(p.initial_logits, p.current_logits) for p in proposals):
        raise ValueError("Training requires untouched initial SAM proposals")
    if gate is not None:
        expected_binding = gate["binding"]
        if not (empty and digest is None) and digest != expected_binding["checkpoint_sha256"]:
            raise ValueError("SAM cache checkpoint differs from verified gate")
        if sam.get("source_fingerprint") != expected_binding["source_fingerprint"]:
            raise ValueError("SAM cache source fingerprint differs from verified gate")
        for field in ("prompt_slots", "canvas_size", "dtype", "encoder_dtype", "decoder_dtype",
                      "precision_policy", "residency_policy"):
            if sam.get(field) != expected_binding[field]:
                raise ValueError(f"SAM cache {field} differs from verified gate")
        capacity = gate["capacity"]
        if not empty and (len(plan.crops) > capacity["tested_K"] or len(plan.prompts) > capacity["tested_Q"]):
            raise ValueError("SAM cache exceeds verified 24GB batch capacity")
    return sam


class LazyCaseReader:
    """At most two decoded RGB/logit/target cases are retained in host RAM."""
    def __init__(self, max_cached=2):
        if max_cached not in (1, 2):
            raise ValueError("Case reader cache is bounded to one or two cases")
        self.max_cached = max_cached
        self._cache = OrderedDict()

    def load(self, record):
        if "rgb" in record:  # Only the explicitly synthetic smoke fixtures.
            return record
        key = record["case_id"]
        for name, digest in record["frozen_files"].items():
            if _file_sha(name) != digest:
                raise ValueError(f"Training source changed after isolation scan: {name}")
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        from .io import load_case
        rgb, plan, proposals, audit = load_case(record["cache_dir"])
        if rgb_sha(rgb) != record["rgb_sha256"]:
            raise ValueError("Cached RGB changed after isolation scan")
        masks = []
        for filename in record["gt_masks"]:
            if Path(filename).suffix.lower() == ".npy":
                mask = np.load(filename, allow_pickle=False)
            else:
                with Image.open(filename) as im:
                    mask = np.asarray(im)
            if mask.ndim != 2 or mask.shape != rgb.shape[:2]:
                raise ValueError(f"GT must be a native HxW per-instance mask: {filename}")
            masks.append(mask > 0)
        loaded = {**record, "rgb": rgb, "ridge": plan.ridge, "proposals": proposals,
                  "gt_masks": masks, "cache_audit": audit}
        self._cache[key] = loaded
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return loaded


def load_training_manifest(path, gate=None):
    """Check all splits for cluster and decoded-RGB leakage before updates."""
    from .io import load_case
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sam_once_training_v1":
        raise ValueError("Expected sam_once_training_v1 manifest")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("Empty training manifest")
    seen_case, seen_cluster, seen_rgb = set(), {}, {}
    loaded = []
    empty_cases = {split: [] for split in ("train", "validation", "check")}
    for record in records:
        split, case, cluster = record["split"], str(record["case_id"]), str(record["cluster_id"])
        if split not in ("train", "validation", "check") or not case or not cluster or record["cluster_id"] is None:
            raise ValueError("Every record needs case_id, cluster_id and a known split")
        if case in seen_case:
            raise ValueError(f"Duplicate case_id: {case}")
        seen_case.add(case)
        if cluster in seen_cluster and seen_cluster[cluster] != split:
            raise ValueError(f"Cross-split cluster leakage: {cluster}")
        seen_cluster[cluster] = split
        image_path = _resolve(path.parent, record["image_path"])
        if not image_path.is_file():
            raise ValueError(f"Missing source image: {image_path}")
        cache_dir = _resolve(path.parent, record["cache_dir"]).resolve()
        rgb, plan, proposals, audit = load_case(cache_dir)
        with Image.open(image_path) as image:
            source_rgb = np.asarray(image.convert("RGB"))
        if not np.array_equal(source_rgb, rgb):
            raise ValueError(f"Source RGB does not match cached native RGB: {case}")
        sha = rgb_sha(rgb)
        if sha in seen_rgb and seen_rgb[sha] != split:
            raise ValueError(f"Cross-split decoded RGB leakage: {case}")
        seen_rgb[sha] = split
        validate_native_cache(audit, plan, proposals, gate)
        empty = not proposals
        if empty:
            empty_cases[split].append(case)
        # Check images participate only in isolation auditing; their labels are not read.
        if split == "check":
            del rgb, source_rgb, plan, proposals, audit
            continue
        if not isinstance(record.get("gt_masks"), list):
            raise ValueError("gt_masks must be an explicit list (possibly empty)")
        masks = []
        frozen = {str(image_path.resolve()): _file_sha(image_path),
                  str(cache_dir / "case.json"): _file_sha(cache_dir / "case.json")}
        for value in record["gt_masks"]:
            filename = _resolve(path.parent, value).resolve()
            frozen[str(filename)] = _file_sha(filename)
            masks.append(str(filename))
        loaded.append({"case_id": case, "cluster_id": cluster, "split": split,
                       "cache_dir": str(cache_dir), "gt_masks": masks,
                       "rgb_sha256": sha, "frozen_files": frozen, "empty_proposals": empty})
        del rgb, source_rgb, plan, proposals, audit
    if not any(c["split"] == "train" for c in loaded) or not any(c["split"] == "validation" for c in loaded):
        raise ValueError("Training and validation cases are both required")
    return loaded, {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "cases_checked": len(records), "clusters_checked": len(seen_cluster),
                    "check_labels_read": False, "source_rgb_matches_cache": True,
                    "empty_prompt_case_ids_by_split": empty_cases,
                    "empty_prompt_zero_call_cases": sum(len(v) for v in empty_cases.values()),
                    "sam_executed_cases": len(records)-sum(len(v) for v in empty_cases.values()),
                    "host_memory": "streaming scan; lazy reader caches at most two cases",
                    "isolation": "cluster_id and decoded RGB SHA; not certified scene isolation"}


def match_targets(proposals, gt_masks, image_shape):
    """Associate frozen positive anchors, never use GT to select inference ROIs.

    No hit is an empty negative proposal. Hits on >1 GT instance are ambiguous
    and excluded from supervised loss (but still participate in the rollout).
    """
    h, w = image_shape
    for mask in gt_masks:
        if mask.shape != (h, w):
            raise ValueError("Target masks must share native RGB coordinates")
    targets, counts = [], {"positive": 0, "empty": 0, "ambiguous": 0}
    for p in proposals:
        x0, y0, x1, y1 = p.crop.xyxy
        hits = set()
        for x, y in p.prompt.positive:
            ix = min(x1-1, max(x0, x0 + int(round(x))))
            iy = min(y1-1, max(y0, y0 + int(round(y))))
            for j, mask in enumerate(gt_masks):
                if mask[iy, ix]:
                    hits.add(j)
        if len(hits) > 1:
            targets.append(None)
            counts["ambiguous"] += 1
        elif not hits:
            targets.append(np.zeros_like(p.initial_logits, np.float32))
            counts["empty"] += 1
        else:
            targets.append(gt_masks[next(iter(hits))][y0:y1, x0:x1].astype(np.float32))
            counts["positive"] += 1
    return targets, counts


def valid_bce_dice(logits, target, valid):
    """Only native valid pixels contribute; padding has identically zero grad."""
    denom = valid.sum().clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / denom
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + 0.4 * dice


def rollout_case(case, model, config, optimizer=None):
    """Detached self-rollout, one successful update per supervised ROI.

    Scheduling precedes target access. A round uses a frozen proposal snapshot;
    predicted crops are committed together. Training updates the model after each
    ROI, so no multi-round autograd graph or SAM call is retained.
    """
    proposals = clone_proposals(case["proposals"])
    targets, counts = match_targets(proposals, case["gt_masks"], case["rgb"].shape[:2])
    losses, updates, visits = [], 0, 0
    for limit in round_limits(config):
        rois = schedule_rois(case["rgb"], proposals, case["ridge"], config, limit)
        predictions = []
        for roi in rois:
            features, valid = extract_roi(case["rgb"], proposals, case["ridge"], roi, config.roi_size)
            tensor = torch.from_numpy(features[None]).to(config.device)
            prediction = model(tensor)
            predictions.append((roi, prediction.detach()[0, 0].float().cpu().numpy()))
            visits += 1
            matched = targets[roi.proposal_index]
            if matched is None:
                continue
            x0, y0, x1, y1 = roi.xyxy
            target = np.zeros((1, 1, config.roi_size, config.roi_size), np.float32)
            target[0, 0, :y1-y0, :x1-x0] = matched[y0:y1, x0:x1]
            loss = valid_bce_dice(prediction, torch.from_numpy(target).to(config.device),
                                  torch.from_numpy(valid[None]).to(config.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite refinement loss")
            losses.append(float(loss.detach()))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
        commit_predictions(proposals, predictions)
    return {"loss": float(np.mean(losses)) if losses else None, "updates": updates,
            "visits": visits, "supervised_rois": len(losses), "target_counts": counts,
            "empty_proposals": not proposals, "sam_calls": 0}


def _fixture(split):
    size = 256
    rgb = np.full((size, size, 3), 36 if split == "train" else 44, np.uint8)
    rgb[:, 120:136] = (190, 70, 50)
    mask = np.zeros((size, size), bool)
    mask[:, 120:136] = True
    logits = np.full((size, size), -3, np.float32)
    logits[:, 116:140] = 2
    crop = CropSpec("c0", (0, 0, size, size))
    prompt = PromptSpec("p0", "c0", ((128., 128.),), ((30., 30.),), (108., 1., 148., 255.))
    proposal = ProposalRecord(prompt, crop, logits.copy(), logits.copy(), 0.8)
    return {"case_id": "smoke_"+split, "cluster_id": "smoke_"+split, "split": split,
            "rgb": rgb, "ridge": mask.astype(np.float32), "proposals": [proposal],
            "gt_masks": [mask], "rgb_sha256": rgb_sha(rgb)}


def validate_gate_receipt(path):
    if path is None:
        raise ValueError("Formal training requires --gate-receipt")
    path = Path(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("native_parity_pass") is not True or receipt.get("capacity_24gb_pass") is not True:
        raise ValueError("Formal updates blocked: native parity and 24GB capacity must both pass")
    referenced = {}
    for name, mode in (("native_parity_receipt", "parity"), ("capacity_24gb_receipt", "capacity")):
        ref = receipt.get(name)
        if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
            raise ValueError(f"Formal gate requires hash-bound {name}")
        reference = _resolve(path.parent, ref["path"])
        if _file_sha(reference) != ref["sha256"]:
            raise ValueError(f"Verification receipt hash mismatch: {name}")
        data = json.loads(reference.read_text(encoding="utf-8"))
        if data.get("mode") != mode or data.get("status") != "passed" or data.get("offline_verification_only") is not True:
            raise ValueError(f"Native verification did not pass: {name}")
        batch = data.get("batch_audit", {})
        if (batch.get("injected_test_model") is not False or batch.get("status") != "ok"
                or batch.get("checkpoint_sha256") is None or not batch.get("source_fingerprint")):
            raise ValueError(f"Verification lacks native checkpoint/source provenance: {name}")
        referenced[mode] = data
    parity, capacity = referenced["parity"], referenced["capacity"]
    for name in ("permutation_isolation", "independent_native_reference"):
        rows = parity.get(name)
        if not isinstance(rows, list) or not rows or not all(r.get("passed") is True for r in rows):
            raise ValueError(f"Native parity receipt lacks passing {name}")
    fields = ("checkpoint_sha256", "source_fingerprint", "prompt_slots", "canvas_size", "dtype",
              "encoder_dtype", "decoder_dtype", "precision_policy", "residency_policy")
    binding = {field: parity["batch_audit"].get(field) for field in fields}
    if any(value is None for value in binding.values()):
        raise ValueError("Verification binding fields are incomplete")
    if binding["residency_policy"] not in ("native_full_predictor", "experimental_interactive_only"):
        raise ValueError("Verification uses an unknown residency policy")
    if any(capacity["batch_audit"].get(field) != value for field, value in binding.items()):
        raise ValueError("Parity and capacity receipts use different checkpoints, sources or inference settings")
    cap = capacity.get("capacity", {})
    if (cap.get("verified_24gb") is not True or cap.get("passed") is not True
            or cap.get("not_a_24gb_capacity_claim") is not False
            or not isinstance(cap.get("device_total_memory_bytes"), int)
            or not 23*1024**3 <= cap["device_total_memory_bytes"] <= 25*1024**3
            or not isinstance(cap.get("tested_K"), int) or not isinstance(cap.get("tested_Q"), int)
            or min(cap["tested_K"], cap["tested_Q"]) < 1):
        raise ValueError("Receipt does not certify a real 24GB single-batch capacity test")
    return {"path": str(path.resolve()), "sha256": _file_sha(path),
            "receipt": receipt, "binding": binding, "capacity": cap}


def _rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"generator": state[0], "keys": state[1].tolist(), "position": state[2],
                      "has_gauss": state[3], "cached_gaussian": state[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _checkpoint(model, optimizer, protocol, steps, best, ledger):
    return {"schema": "sam_once_refiner_checkpoint_v1", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "actual_successful_updates": steps,
            "best_validation_loss": best, "checkpoint_selection": "diagnostic_validation_loss_only",
            "instance_promotion": False, "protocol": protocol, "rng": _rng_state(),
            "ledger": ledger, "feature_names": FEATURE_NAMES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--gate-receipt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--roi-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-visits", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=3*256*256)
    parser.add_argument("--mode", choices=("iterative", "fixedsingle"), default="iterative")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.validate_every <= 0:
        parser.error("Steps and validation interval must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite any run directory: {output}")
    gate = None
    if args.smoke:
        args.device, args.steps, args.validate_every = "cpu", 2, 1
        args.roi_size, args.max_visits, args.max_pixels = 256, 1, 256*256
        args.rounds = 1
        cases, isolation = [_fixture("train"), _fixture("validation")], {"synthetic_fixture_only": True}
    else:
        gate = validate_gate_receipt(args.gate_receipt)
        if not args.manifest:
            parser.error("Formal training requires --manifest")
        cases, isolation = load_training_manifest(args.manifest, gate)
    config = RefinerConfig(args.roi_size, args.rounds, args.max_visits,
                           args.max_pixels, args.mode, args.device).validate()
    if sum(round_limits(config)) == 0:
        raise ValueError("Training pixel/visit budget is empty")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = ResidualRefiner().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    trains = [c for c in cases if c["split"] == "train"]
    validations = [c for c in cases if c["split"] == "validation"]
    reader = LazyCaseReader(max_cached=2)
    protocol = {"schema": "sam_once_refiner_training_protocol_v1", "smoke_only": args.smoke,
                "production_claim_allowed": False, "seed": args.seed, "config": asdict(config),
                "planned_successful_updates": args.steps, "gate": gate, "isolation": isolation,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "sam_calls": 0, "sam_weights_updated": False, "amp": False,
                "loss": "BCE+0.4Dice, valid native pixels only", "detached_self_rollout": True,
                "selection": "best.pt is minimum diagnostic validation loss; instance IoU/PQ evaluation required before promotion",
                "train_cases": [c["case_id"] for c in trains],
                "validation_cases": [c["case_id"] for c in validations]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    steps, attempts, best, last_validation = 0, 0, float("inf"), -1
    consecutive_no_updates = 0
    ledger = []
    start = time.perf_counter()
    while steps < args.steps:
        # Clamp ROI visits to the remaining update count; ambiguous ROI may yield no update.
        remaining = args.steps - steps
        step_config = RefinerConfig(config.roi_size, config.rounds, min(config.max_visits, remaining),
                                    min(config.max_pixels, remaining*config.roi_size**2), config.mode, config.device)
        case = reader.load(trains[attempts % len(trains)])
        model.train()
        stats = rollout_case(case, model, step_config, optimizer)
        steps += stats["updates"]
        attempts += 1
        consecutive_no_updates = consecutive_no_updates + 1 if stats["updates"] == 0 else 0
        ledger.append({"attempt": attempts, "case_id": case["case_id"], "actual_updates": steps, **stats})
        if consecutive_no_updates >= len(trains) or attempts > args.steps * max(10, len(trains)*3):
            (output / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            (output / "failure.json").write_text(json.dumps({"reason": "no_supervised_updates_over_training_cycle",
                "actual_successful_updates": steps, "attempts": attempts, "sam_calls": 0,
                "empty_prompt_cases": isolation.get("empty_prompt_case_ids_by_split", {})}, indent=2), encoding="utf-8")
            raise RuntimeError("Bounded training stopped: insufficient supervised scheduled proposals")
        should_validate = (steps == args.steps or steps // args.validate_every > last_validation // args.validate_every)
        if stats["updates"] and should_validate:
            model.eval()
            with torch.inference_mode():
                val_stats = [rollout_case(reader.load(c), model, config) for c in validations]
            losses = [s["loss"] for s in val_stats if s["loss"] is not None]
            if not losses:
                ledger.append({"actual_updates": steps, "validation_loss": None,
                               "validation_cases": val_stats,
                               "validation_empty_cases": sum(s["empty_proposals"] for s in val_stats),
                               "validation_cases_without_loss": len(val_stats)})
                (output / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
                (output / "failure.json").write_text(json.dumps({"reason": "no_supervised_validation_loss",
                    "actual_successful_updates": steps, "validation_cases_without_loss": len(val_stats)}, indent=2), encoding="utf-8")
                raise ValueError("No unambiguous validation targets among independently scheduled ROIs")
            loss = float(np.mean(losses))
            ledger.append({"actual_updates": steps, "validation_loss": loss, "validation_cases": val_stats,
                           "validation_empty_cases": sum(s["empty_proposals"] for s in val_stats),
                           "validation_cases_without_loss": sum(s["loss"] is None for s in val_stats)})
            if loss < best:
                best = loss
                torch.save(_checkpoint(model, optimizer, protocol, steps, best, ledger), output / "best.pt")
            last_validation = steps
            torch.save(_checkpoint(model, optimizer, protocol, steps, best, ledger), output / "last.pt")
            (output / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    summary = {"smoke_only": args.smoke, "actual_successful_updates": steps, "planned_updates": args.steps,
               "amp_skips": 0, "sam_calls": 0, "wall_seconds": time.perf_counter()-start,
               "best_validation_loss_diagnostic": best, "instance_promotion": False,
               "empty_prompt_cases": isolation.get("empty_prompt_case_ids_by_split", {}),
               "output": str(output.resolve())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
