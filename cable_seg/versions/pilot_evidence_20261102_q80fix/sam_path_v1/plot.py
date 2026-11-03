"""Render auditable loss curves from run logs. Never invent missing observations."""
from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


LOSS_NAMES = ("total", "mask_bce", "mask_dice", "object", "quality", "path", "descriptor", "geometry")


def events(path):
    """Ignore only an unfinished final line while another process writes it."""
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                return
            try:
                yield json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Malformed complete JSON line {number}: {path}") from exc


def history(run, seen=None):
    """Stitch exact-resume parents, excluding parent updates beyond saved cursor."""
    run = Path(run).resolve()
    seen = set() if seen is None else set(seen)
    if run in seen:
        raise ValueError("Resume history contains a cycle")
    seen.add(run)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8-sig"))
    first = next(events(run / "logs.jsonl"), None)
    if not first or first.get("event") != "start":
        raise ValueError(f"Missing start event: {run}")
    parent = provenance.get("resume_from")
    if parent:
        cutoff = first["progress"]["opportunities"]
        epoch = first["progress"]["epoch"]
        for event in history(Path(parent).parent, seen):
            if event.get("event") == "update" and event["opportunities"] <= cutoff:
                yield event
            elif event.get("event") in ("epoch_train", "validation") and event["epoch"] <= epoch:
                yield event
    yield from events(run / "logs.jsonl")


def finite_mapping(values):
    if not isinstance(values, dict) or set(values) != set(LOSS_NAMES):
        raise ValueError("Every observation must contain all raw loss components")
    result = {key: float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite loss observation; refusing misleading curves")
    return result


def ema(values, span=200):
    """Causal EMA with the first observation as initialization; no interpolation."""
    values = np.asarray(values, dtype=float)
    if span < 1:
        raise ValueError("EMA span must be positive")
    result = np.empty_like(values)
    if len(values):
        result[0] = values[0]
        alpha = 2/(span+1)
        for index in range(1, len(values)):
            result[index] = alpha*values[index]+(1-alpha)*result[index-1]
    return result


def collect(run, destination):
    """CSV contains every observation; figures retain every original step vertex."""
    run, destination = Path(run), Path(destination)
    spec = json.loads((run / "loss_spec.json").read_text(encoding="utf-8-sig"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8-sig"))
    weights = spec["weights"]
    if set(weights) != set(LOSS_NAMES)-{"total"}:
        raise ValueError("Unrecognized loss specification")
    columns = ["opportunity", "successful_updates", "epoch", "batch_index", "amp_skipped", "samples"]
    columns += [f"raw_{key}" for key in LOSS_NAMES]+[f"weighted_{key}" for key in weights]
    steps, series = array("d"), {key: array("d") for key in LOSS_NAMES}
    train_epochs, val_epochs, metric_epochs = {}, {}, {}
    last, skipped, observations = 0, 0, 0
    csv_path = destination / "steps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for event in history(run):
            kind = event.get("event")
            if kind == "update":
                step = event["opportunities"]
                if step != last+1:
                    raise ValueError(f"Missing or duplicated update: expected {last+1}, got {step}")
                last = step
                raw = finite_mapping(event["loss"])
                weighted = {key: raw[key]*float(value) for key, value in weights.items()}
                if not math.isclose(sum(weighted.values()), raw["total"], rel_tol=2e-5, abs_tol=2e-6):
                    raise ValueError("Loss specification does not reconstruct logged total")
                if "weighted_loss" in event and any(not math.isclose(event["weighted_loss"][key], value,
                        rel_tol=2e-5, abs_tol=2e-6) for key, value in weighted.items()):
                    raise ValueError("Logged weighted losses disagree with the specification")
                skipped += int(event["amp_skipped"])
                observations += 1
                steps.append(step)
                for key in LOSS_NAMES:
                    series[key].append(raw[key])
                writer.writerow(dict(opportunity=step, successful_updates=event["successful_updates"],
                    epoch=event["epoch_index"]+1, batch_index=event["batch_index"],
                    amp_skipped=int(event["amp_skipped"]), samples=len(event["ids"]),
                    **{f"raw_{key}": value for key, value in raw.items()},
                    **{f"weighted_{key}": value for key, value in weighted.items()}))
            elif kind == "epoch_train":
                train_epochs[event["epoch"]] = dict(loss=finite_mapping(event["loss"]), samples=event["samples"])
            elif kind == "validation":
                metrics = event["metrics"]
                metric_epochs[event["epoch"]] = metrics
                summary = metrics.get("losses")
                if summary and summary["complete"]:
                    val_epochs[event["epoch"]] = dict(loss=finite_mapping(summary["mean"]), samples=summary["images"])
    with (destination / "epochs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["epoch", "split", "samples"]+[f"raw_{key}" for key in LOSS_NAMES]+[f"weighted_{key}" for key in weights]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, rows in (("train", train_epochs), ("validation", val_epochs)):
            for epoch, record in sorted(rows.items()):
                writer.writerow(dict(epoch=epoch, split=split, samples=record["samples"],
                    **{f"raw_{key}": value for key, value in record["loss"].items()},
                    **{f"weighted_{key}": record["loss"][key]*weight for key, weight in weights.items()}))
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["epoch", "images", "failures", "complete", "instance_iou_macro", "pq_global", "union_iou_macro"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch, record in sorted(metric_epochs.items()):
            writer.writerow({"epoch": epoch, **{key: record.get(key) for key in fields[1:]}})
    return dict(run=str(run.resolve()), spec=spec, backend=config["model"]["backend"], steps=np.asarray(steps),
                series={key: np.asarray(values) for key, values in series.items()},
                train_epochs=train_epochs, val_epochs=val_epochs, metric_epochs=metric_epochs,
                observations=observations, amp_skips=skipped,
                incomplete_validation_epochs=[epoch for epoch in metric_epochs if epoch not in val_epochs],
                step_csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())


def render(runs, output, ema_span=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"path.simplify": False, "font.size": 9, "axes.spines.top": False,
                               "axes.spines.right": False, "agg.path.chunksize": 10000})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    collected = {}
    for label, run in runs.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Run labels must contain only letters, numbers, underscore or hyphen")
        folder = output / label
        folder.mkdir()
        collected[label] = collect(run, folder)
    colors = dict(zip(collected, plt.get_cmap("tab10").colors))
    fixture_note = "SOFTWARE FIXTURE ONLY | " if any(rec["backend"] == "toy" for rec in collected.values()) else ""
    def draw_step(ax, key, weighted=False):
        for label, record in collected.items():
            if key in record["spec"]["inactive"]:
                continue
            factor = record["spec"]["weights"].get(key, 1.) if weighted else 1.
            values = record["series"][key]*factor
            ax.plot(record["steps"], values, color=colors[label], linewidth=.4, alpha=.14, rasterized=True)
            ax.plot(record["steps"], ema(values, ema_span), color=colors[label], linewidth=1.1,
                    label=f"{label}: EMA {ema_span}", rasterized=True)
        ax.set_xlabel("Optimizer opportunity (AMP skips retained)")
        ax.set_ylabel("Weighted contribution" if weighted else "Raw loss")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
    def draw_epoch(ax, key, weighted=False):
        for label, record in collected.items():
            if key in record["spec"]["inactive"]:
                continue
            factor = record["spec"]["weights"].get(key, 1.) if weighted else 1.
            for split, style, marker in (("train", "-", "o"), ("val", "--", "s")):
                rows = record[f"{split}_epochs"]
                if rows:
                    ax.plot(sorted(rows), [rows[e]["loss"][key]*factor for e in sorted(rows)],
                            color=colors[label], linestyle=style, marker=marker, markersize=3,
                            label=f"{label} {split}")
        ax.set_xlabel("Completed epoch")
        ax.set_ylabel("Weighted contribution" if weighted else "Raw loss")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
    for key in LOSS_NAMES:
        rows = 1 if key == "total" else 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4*rows), squeeze=False)
        draw_step(axes[0, 0], key)
        draw_epoch(axes[0, 1], key)
        if rows == 2:
            draw_step(axes[1, 0], key, True)
            draw_epoch(axes[1, 1], key, True)
        inactive = [label for label, rec in collected.items() if key in rec["spec"]["inactive"]]
        subtitle = (" | inactive: "+", ".join(inactive)) if inactive else ""
        if key == "total":
            subtitle += " | B/C objectives differ; total is not a quality comparison"
        fig.suptitle(fixture_note+key.replace("_", " ")+subtitle)
        fig.tight_layout()
        fig.savefig(output / f"loss_{key}.png", dpi=170)
        fig.savefig(output / f"loss_{key}.pdf", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for key, ax in zip(LOSS_NAMES, axes.flat):
        draw_epoch(ax, key)
        ax.set_title(key.replace("_", " "))
    fig.suptitle(fixture_note+"Epoch loss curves | train: solid, validation: dashed | total objectives differ")
    fig.tight_layout()
    fig.savefig(output / "epoch_overview.png", dpi=170)
    fig.savefig(output / "epoch_overview.pdf", dpi=170)
    plt.close(fig)
    receipt = dict(ema_span=ema_span, raw_step_data="Every complete update retained in CSV and figure vertices",
        partial_final_json_line="ignored; any malformed complete line is an error",
        note="No missing validation losses are imputed. Train curves contain all optimizer opportunities, including AMP skips."
             " Training uses sample-weighted batch means; validation uses per-image means."
             " B/C total objectives differ; compare common components plus held-out segmentation metrics.",
        runs={label: {key: rec[key] for key in ("run", "spec", "backend", "observations", "amp_skips",
                  "incomplete_validation_epochs", "step_csv_sha256")} for label, rec in collected.items()})
    (output / "curve_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RUN_DIRECTORY")
    parser.add_argument("--output", required=True, help="New output directory; previous figures are preserved")
    parser.add_argument("--ema-span", type=int, default=200)
    args = parser.parse_args(argv)
    runs = {}
    for item in args.runs:
        label, separator, path = item.partition("=")
        if not separator or not path or label in runs:
            parser.error("--runs requires unique LABEL=RUN_DIRECTORY entries")
        runs[label] = path
    if args.ema_span < 1:
        parser.error("--ema-span must be positive")
    print(json.dumps(render(runs, args.output, args.ema_span), ensure_ascii=False))


if __name__ == "__main__":
    main()
