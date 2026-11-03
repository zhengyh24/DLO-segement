"""Train A/B/C/D into a fresh run directory; exact resume uses a new directory."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import torch

from .checkpoint import (atomic_json, atomic_torch_save, checkpoint_payload, new_directory,
                         read_checkpoint, restore_rng, seed_everything, sha256, verify_base_receipt,
                         engine_source_sha256, verify_engine_source)
from .data import NativeInstanceDataset, collate_fn
from .evaluate import evaluate_model
from .inference import autocast_context, padding_valid
from .model import ModelConfig, SamPathModel
from .losses import SetCriterion


@dataclass
class TrainConfig:
    manifest: str
    model: dict = field(default_factory=dict)
    data_root: str | None = None
    allow_condition: bool = False
    seed: int = 42
    epochs: int = 5
    batch_size: int = 1
    lr: float = 1.e-4
    decoder_lr: float = 1.e-5
    weight_decay: float = .01
    device: str = "cpu"
    amp: str = "none"
    workers: int = 0
    save_every: int = 100
    validate_every: int = 1
    object_threshold: float = .5
    mask_threshold: float = .5
    train_split: str = "train"
    val_split: str = "validation"
    loss: dict = field(default_factory=dict)
    max_batches: int | None = None
    grad_clip: float = 1.
    cpu_threads: int = 1

    def __post_init__(self):
        if self.workers != 0:
            raise ValueError("Exact batch-cursor resume currently requires workers=0")
        for key in ("epochs", "batch_size", "save_every", "validate_every", "cpu_threads"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_batches is not None and (type(self.max_batches) is not int or self.max_batches < 1):
            raise ValueError("max_batches must be a positive cumulative opportunity budget")
        if self.train_split != "train" or self.val_split not in ("val", "validation"):
            raise ValueError("Checkpoint selection must use the frozen validation split")
        if self.lr <= 0 or self.decoder_lr <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("Invalid optimizer settings")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp must be none, bf16 or fp16")
        if self.amp != "none" and torch.device(self.device).type != "cuda":
            raise ValueError("CPU smoke requires amp=none")
        if not all(0 <= value <= 1 for value in (self.object_threshold, self.mask_threshold)):
            raise ValueError("Thresholds must lie in [0,1]")
        self.model = dict(self.model)
        if self.model.get("init_seed", self.seed) != self.seed:
            raise ValueError("model.init_seed must equal training seed for the paired A-D protocol")
        self.model["init_seed"] = self.seed
        self.model = asdict(ModelConfig(**self.model))


def _log(path, event):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
        stream.flush()


def _epoch_indices(length, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch * 1000003)
    return torch.randperm(length, generator=generator).tolist()


def _to_device(batch, device):
    targets = [{**item, "masks": item["masks"].to(device)} for item in batch["targets"]]
    return batch["images"].to(device), batch["valid"].to(device), targets


def _resume_contract(config, saved, provenance):
    if saved["provenance"]["manifest_sha256"] != provenance["manifest_sha256"]:
        raise ValueError("Resume manifest differs from the frozen parent manifest")
    if saved["torch_version"] != str(torch.__version__):
        raise ValueError("Exact resume requires the same PyTorch version")
    verify_engine_source(saved)
    if saved["provenance"].get("runtime") != provenance["runtime"]:
        raise ValueError("Exact resume runtime changed; restore the original software, visible CUDA "
                         "devices, precision and determinism settings. Deployment inference may use "
                         "a different runtime, but is not an exact training resume.")
    current, old = asdict(config), dict(saved["train_config"])
    # Directory relocation and extending a cumulative budget do not change training math.
    for values in (current, old):
        for key in ("manifest", "data_root", "epochs", "max_batches", "save_every"):
            values.pop(key, None)
        values["model"] = dict(values["model"])
        for key in ("sam_repo", "sam_checkpoint"):
            values["model"].pop(key, None)
    if current != old:
        differing = [key for key in current if current[key] != old.get(key)]
        raise ValueError(f"Resume changes training protocol: {differing}")


def _runtime_receipt(device):
    device = torch.device(device)
    receipt = dict(python=sys.version.split()[0], platform=platform.system(),
                   machine=platform.machine(), numpy=np.__version__, scipy=scipy.__version__,
                   torch=str(torch.__version__), cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), device_type=device.type,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if device.type == "cuda":
        receipt.update(device_name=torch.cuda.get_device_name(device),
                       device_capability=list(torch.cuda.get_device_capability(device)),
                       device_count=torch.cuda.device_count())
    return receipt


def train(config, output, resume=None):
    output = new_directory(output)
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed)
    model_config = ModelConfig(**config.model)
    train_data = NativeInstanceDataset(config.manifest, config.train_split, model_config.canvas_size,
                                       config.seed, config.allow_condition, config.data_root)
    val_data = NativeInstanceDataset(config.manifest, config.val_split, model_config.canvas_size,
                                     config.seed, config.allow_condition, config.data_root)
    provenance = dict(manifest_sha256=sha256(config.manifest), train_images=len(train_data),
                      validation_images=len(val_data), native_resize=False,
                      independent_scene_split=train_data.protocol["independent_scene_split"],
                      resume_from=str(Path(resume).resolve()) if resume else None,
                      parent_checkpoint_sha256=sha256(resume) if resume else None)
    provenance["runtime"] = _runtime_receipt(config.device)
    provenance["engine_source_sha256"] = engine_source_sha256()
    saved = read_checkpoint(resume) if resume else None
    if saved is not None:
        _resume_contract(config, saved, provenance)
    model = SamPathModel(model_config).to(config.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Model contains no trainable parameters")
    decoder = [parameter for name, parameter in model.named_parameters()
               if parameter.requires_grad and name.startswith("base.decoder.")]
    added = [parameter for name, parameter in model.named_parameters()
             if parameter.requires_grad and not name.startswith("base.decoder.")]
    optimizer = torch.optim.AdamW([{"params": decoder, "lr": config.decoder_lr, "name": "decoder"},
                                   {"params": added, "lr": config.lr, "name": "new_modules"}],
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp == "fp16")
    criterion = SetCriterion(**config.loss).to(config.device)
    progress = dict(epoch=0, batch_cursor=0, opportunities=0, successful_updates=0, amp_skips=0,
                    samples=0, native_pixels=0, supervised_pixels=0, elapsed_seconds=0.,
                    best_instance_iou=None, best_pq=None, best_epoch=None)
    if resume:
        verify_base_receipt(model, saved)
        model.load_trainable_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        progress.update(saved["progress"])
        restore_rng(saved["rng"])
    atomic_json(output / "config.json", asdict(config))
    atomic_json(output / "provenance.json", {**provenance, "base": model.base.receipt,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "parameter_counts": model.parameter_counts(),
                "checkpoint_state_elements": sum(v.numel() for v in model.checkpoint_state_dict().values())})
    origin_elapsed = progress["elapsed_seconds"]
    run_start = time.perf_counter()

    def save(name="last.pt"):
        progress["elapsed_seconds"] = origin_elapsed + time.perf_counter()-run_start
        payload = checkpoint_payload(model, model_config, asdict(config), optimizer, scaler, progress, provenance)
        atomic_torch_save(payload, output / name)

    save()
    if resume and progress["best_instance_iou"] is not None:
        # Only copy a verified parent best checkpoint; never mislabel parent last as best.
        parent_best = Path(resume).parent / "best.pt"
        if parent_best.is_file():
            best = read_checkpoint(parent_best)
            if (best["progress"]["best_instance_iou"] == progress["best_instance_iou"] and
                    best["progress"]["best_epoch"] == progress["best_epoch"] and
                    best["provenance"]["manifest_sha256"] == provenance["manifest_sha256"]):
                atomic_torch_save(best, output / "best.pt")
    _log(output / "logs.jsonl", dict(event="start", resumed=bool(resume), progress=dict(progress)))
    batches_per_epoch = math.ceil(len(train_data)/config.batch_size)
    while progress["epoch"] < config.epochs:
        if config.max_batches is not None and progress["opportunities"] >= config.max_batches:
            break
        epoch = progress["epoch"]
        train_data.epoch = epoch
        order = _epoch_indices(len(train_data), config.seed, epoch)
        model.train()
        for batch_cursor in range(progress["batch_cursor"], batches_per_epoch):
            if config.max_batches is not None and progress["opportunities"] >= config.max_batches:
                break
            indices = order[batch_cursor*config.batch_size:(batch_cursor+1)*config.batch_size]
            batch = collate_fn([train_data[index] for index in indices])
            images, loss_valid, targets = _to_device(batch, config.device)
            input_valid = padding_valid(images, targets)
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            with autocast_context(config.device, config.amp):
                outputs = model(images, input_valid)
                if outputs.get("calls") != {"encoder": 1, "decoder": 1}:
                    raise RuntimeError("Model violated one encoder / one batched decoder call")
                losses = criterion(outputs, targets, loss_valid)
                total = losses["total"]
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite loss; training stops without a successful update")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip or float("inf"))
            if not scaler.is_enabled() and not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradients; optimizer update was not executed")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            if torch.device(config.device).type == "cuda":
                torch.cuda.synchronize()
            progress["opportunities"] += 1
            progress["successful_updates"] += int(not skipped)
            progress["amp_skips"] += int(skipped)
            progress["samples"] += len(indices)
            progress["native_pixels"] += int(input_valid.sum())
            progress["supervised_pixels"] += int(loss_valid.sum())
            progress["batch_cursor"] = batch_cursor + 1
            progress["elapsed_seconds"] = origin_elapsed + time.perf_counter()-run_start
            _log(output / "logs.jsonl", dict(event="update", epoch_index=epoch, batch_index=batch_cursor,
                 ids=[target["id"] for target in targets], amp_skipped=bool(skipped),
                 grad_norm=float(grad_norm) if torch.isfinite(grad_norm) else None,
                 loss={key: float(value.detach()) for key, value in losses.items()},
                 seconds=time.perf_counter()-start, calls=outputs["calls"], **progress))
            if progress["opportunities"] % config.save_every == 0:
                save()
        if progress["batch_cursor"] != batches_per_epoch:
            break
        progress["epoch"] = epoch + 1
        progress["batch_cursor"] = 0
        if progress["epoch"] % config.validate_every == 0 or progress["epoch"] == config.epochs:
            val_data.epoch = epoch
            report = evaluate_model(model, val_data, device=config.device, amp=config.amp,
                                    object_threshold=config.object_threshold, mask_threshold=config.mask_threshold)
            atomic_json(output / f"validation_epoch_{progress['epoch']:04d}.json", report)
            _log(output / "logs.jsonl", dict(event="validation", epoch=progress["epoch"],
                 metrics={key: value for key, value in report.items() if key != "per_image"}))
            current = (report["instance_iou_macro"], report["pq_global"] or 0.)
            previous = (progress["best_instance_iou"] if progress["best_instance_iou"] is not None else -1.,
                        progress["best_pq"] if progress["best_pq"] is not None else -1.)
            if report["complete"] and current > previous:
                progress.update(best_instance_iou=current[0], best_pq=current[1], best_epoch=progress["epoch"])
                save("best.pt")
        save()
    save()
    status = "completed" if progress["epoch"] >= config.epochs else "budget_stopped"
    final = dict(status=status, progress=progress, output=str(output.resolve()),
                 selection="validation instance_iou_macro, then validation pq_global; thresholds frozen in config",
                 best_checkpoint_available=(output / "best.pt").is_file())
    atomic_json(output / "completion.json", final)
    _log(output / "logs.jsonl", dict(event="finish", **final))
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config = TrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    print(json.dumps(train(config, args.output, args.resume), ensure_ascii=False))


if __name__ == "__main__":
    main()
