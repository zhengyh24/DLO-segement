"""Short-sequence supervised training. Validation here is conditional, NOT PQ.

The sampled window trajectory and initial seed are supervised. History may be
model-predicted; recurrent queries are never replaced by GT instance IDs.
"""
import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrackerDataset, exit_targets
from .model import InstanceTracker
from .losses import tracking_loss


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def project_history(past, origins, tile):
    """Place accepted native crops in the new coordinates; no resampling.

    past = [(B1HW mask, B2 xy origins), ...]. Predictions are detached by the
    caller: truncated spatial feedback, while query BPTT remains enabled.
    """
    if not past:
        raise ValueError("Use zeros for the first observation")
    history = past[0][0].new_zeros((len(origins), 1, tile, tile))
    for mask, old_origins in past:
        for b in range(len(origins)):
            x, y = map(int, origins[b].tolist())
            ox, oy = map(int, old_origins[b].tolist())
            left, top = max(x, ox), max(y, oy)
            right, bottom = min(x + tile, ox + tile), min(y + tile, oy + tile)
            if right <= left or bottom <= top:
                continue
            src = mask[b, :, top-oy:bottom-oy, left-ox:right-ox]
            dst = history[b, :, top-y:bottom-y, left-x:right-x]
            history[b, :, top-y:bottom-y, left-x:right-x] = torch.maximum(src, dst)
    return history


def sequence_loss(model, batch, device, teacher_prob=0.0):
    rgb = batch["rgb"].to(device)
    bsz, steps, _, height, width = rgb.shape
    if height != width:
        raise ValueError("Training sampler contract requires square native crops")
    active = batch.get("step_valid", torch.ones(bsz, steps)).to(device)
    query = anchor = None
    past, totals, losses = [], {}, []
    intersection = union = 0.0
    observations = 0
    teacher_used = 0
    for t in range(steps):
        current = {key: batch[key][:, t].to(device) for key in
                   ("seed", "target", "other", "valid", "seed_target")}
        is_active = active[:, t].float().view(bsz, 1, 1, 1)
        current["valid"] = current["valid"].float() * is_active
        origins = batch["origins"][:, t]  # integers on CPU; preserve native xy
        spatial_valid = batch["spatial_valid"][:, t].to(device)
        history = (project_history(past, origins, height) if past
                   else torch.zeros_like(current["target"]))
        history = history * spatial_valid
        current["exits"] = exit_targets(current["target"], history, current["valid"])
        out = model(rgb[:, t], current["seed"], history, query=query, anchor=anchor,
                    spatial_valid=spatial_valid)
        query, anchor = out["query"], out["anchor"]
        keep = active[:, t].bool()
        if keep.any():
            selected_out = {k: v[keep] for k, v in out.items() if torch.is_tensor(v)}
            selected_batch = {k: v[keep] for k, v in current.items()}
            loss, stats = tracking_loss(selected_out, selected_batch)
            losses.append(loss * keep.sum())
            count = int(keep.sum())
            observations += count
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value) * count
            valid = current["valid"].bool()
            prediction = (out["mask_logits"].detach() > 0) & valid
            target = current["target"].bool() & valid
            intersection += float((prediction & target).sum())
            union += float((prediction | target).sum())
        # Same hard mask feedback used at deployment; no GT query or future seed.
        feedback = (out["mask_logits"].detach() > 0).float()
        if teacher_prob > 0:
            teacher = torch.rand(bsz, 1, 1, 1, device=device) < teacher_prob
            feedback = torch.where(teacher, current["target"].float(), feedback)
            if t < steps - 1:
                teacher_used += int((teacher.flatten() & keep).sum())
        feedback = feedback * spatial_valid * is_active
        past.append((feedback, origins))
    if not losses:
        raise ValueError("No active observations in sequence")
    stats = {k: v / observations for k, v in totals.items()}
    stats.update(observations=observations, teacher_feedbacks=teacher_used,
                 conditional_intersection=intersection, conditional_union=union)
    return torch.stack(losses).sum() / observations, stats


@torch.no_grad()
def validate(model, dataset, batch_size, device, limit):
    model.eval()
    loss_sum = n = inter = union = 0.0
    crop_forwards = input_pixels = 0
    for index, batch in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)):
        if limit and index >= limit:
            break
        loss, stats = sequence_loss(model, batch, device, teacher_prob=0.0)
        crop_forwards += int(batch["rgb"].shape[0]*batch["rgb"].shape[1])
        input_pixels += int(batch["rgb"].numel()//3)
        count = stats["observations"]
        n += count
        loss_sum += float(loss) * count
        inter += stats["conditional_intersection"]
        union += stats["conditional_union"]
    if not n:
        raise ValueError("Empty validation")
    model.train()
    return {"conditional_loss": loss_sum/n,
            "conditional_target_iou": inter/union if union else 1.0,
            "observations": int(n), "gt_initial_seed": True,
            "crop_forwards": crop_forwards, "input_pixels": input_pixels,
            "gt_window_trajectory": True, "predicted_history": True,
            "deployment_metric": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=("smoke", "pilot"), default="smoke")
    p.add_argument("--allow-condition-pilot", action="store_true")
    p.add_argument("--updates", type=int, default=12)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val-every", type=int, default=6)
    p.add_argument("--val-batches", type=int, default=0)
    p.add_argument("--teacher-start", type=float, default=0.5)
    p.add_argument("--teacher-end", type=float, default=0.0)
    args = p.parse_args()
    if min(args.updates, args.batch, args.tile, args.steps, args.val_every) < 1:
        p.error("All budgets and sizes must be positive")
    if not 0 <= args.teacher_start <= 1 or not 0 <= args.teacher_end <= 1:
        p.error("Teacher probabilities must be in [0,1]")
    if args.amp and not args.device.startswith("cuda"):
        p.error("--amp requires CUDA in this prototype")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    training = TrackerDataset(args.manifest, "train", tile=args.tile, steps=args.steps,
                              seed=args.seed, allow_condition=args.allow_condition_pilot)
    validation = TrackerDataset(args.manifest, "validation", tile=args.tile, steps=args.steps,
                                seed=args.seed+1, allow_condition=args.allow_condition_pilot)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=False)
    model_config = {"query_dim": 128, "widths": [32, 64, 128, 192]}
    model = InstanceTracker(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    manifest_path = Path(args.manifest).resolve()
    metadata = {"schema": 1, "kind": args.kind, "model_config": model_config,
                "args": vars(args), "manifest": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_sha256": source_hashes, "torch": str(torch.__version__),
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "parameters": sum(v.numel() for v in model.parameters()),
                "seed_sigma": training.seed_sigma,
                "train_images": len(training), "validation_images": len(validation),
                "gt_initial_seed": True, "gt_window_trajectory": True,
                "independent_scene_split": json.loads(manifest_path.read_text(encoding="utf-8")).get("independent_scene_split", False),
                "scientific_result": False, "resumable_exactly": False}
    write_json(outdir/"config.json", metadata)
    snapshot = outdir/"source"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot/source.name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    began = time.perf_counter()
    successful = attempts = skipped = inputs = pixels = 0
    validation_forwards = validation_pixels = 0
    best = float("inf")
    epoch, loader, iterator = -1, None, None

    def save(name):
        payload = dict(metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scaler=scaler.state_dict(), optimizer_steps=successful, attempts=attempts,
                       skipped_updates=skipped, validation=last_validation)
        temporary = outdir/(name+".tmp")
        torch.save(payload, temporary)
        temporary.replace(outdir/name)

    last_validation = None
    with (outdir/"train.jsonl").open("w", encoding="utf-8") as log:
        while successful < args.updates:
            if attempts >= args.updates*3:
                raise RuntimeError("Too many non-finite updates; inspect retained logs")
            if iterator is None:
                epoch += 1
                training.epoch = epoch
                loader = DataLoader(training, batch_size=args.batch, shuffle=True, num_workers=0,
                                    generator=torch.Generator().manual_seed(args.seed+epoch))
                iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = None
                continue
            frac = successful / max(1, args.updates-1)
            teacher_prob = args.teacher_start+(args.teacher_end-args.teacher_start)*frac
            attempts += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                loss, stats = sequence_loss(model, batch, device, teacher_prob)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            query_grad = sum(float(v.grad.detach().float().square().sum()) for v in
                             model.query_update.parameters() if v.grad is not None) ** 0.5
            finite = bool(torch.isfinite(loss) & torch.isfinite(norm))
            if finite:
                scaler.step(optimizer)
                successful += 1
            else:
                skipped += 1
            scaler.update()
            inputs += int(batch["rgb"].shape[0]*args.steps)  # includes padded sequence slots
            pixels += int(batch["rgb"].shape[0]*args.steps*args.tile*args.tile)
            record = dict(stats, loss=float(loss.detach()), step=successful, attempt=attempts,
                          finite=finite, teacher_probability=teacher_prob,
                          query_update_gradient_norm=query_grad,
                          elapsed_seconds=time.perf_counter()-began)
            if finite and (successful % args.val_every == 0 or successful == args.updates):
                last_validation = validate(model, validation, args.batch, device, args.val_batches)
                validation_forwards += last_validation["crop_forwards"]
                validation_pixels += last_validation["input_pixels"]
                record["validation"] = last_validation
                if last_validation["conditional_loss"] < best:
                    best = last_validation["conditional_loss"]
                    save("best.pt")
                save("last.pt")
            log.write(json.dumps(record)+"\n")
            log.flush()
            print(json.dumps(record), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    completion = dict(metadata, status="completed", optimizer_steps=successful,
                      update_opportunities=attempts, skipped_updates=skipped,
                      training_crop_forwards=inputs, training_input_pixels=pixels,
                      validation_crop_forwards=validation_forwards, validation_input_pixels=validation_pixels,
                      total_crop_forwards=inputs+validation_forwards, total_input_pixels=pixels+validation_pixels,
                      elapsed_seconds=time.perf_counter()-began,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                      best_conditional_loss=best, last_validation=last_validation,
                      limitation="Functional/pilot training; GT seeds/windows in conditional validation are not automatic-instance accuracy.")
    write_json(outdir/"completion.json", completion)
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
