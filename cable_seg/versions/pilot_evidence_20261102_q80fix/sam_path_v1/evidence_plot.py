"""Full legacy loss plots plus the contrast/compact decomposition, without double counting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .plot import render, history, ema


def render_evidence(runs, output):
    receipt = render(runs, output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    components = ("descriptor_contrast", "descriptor_compact")
    series = {}
    for label, run in runs.items():
        spec = json.loads((Path(run)/"loss_spec.json").read_text())
        series[label] = {"steps": [], "spec": spec, "train": {}, "validation": {}, **{k:[] for k in components}}
        records = []
        for event in history(run):
            values = event.get("loss_components")
            if event["event"] == "update" and values is not None:
                series[label]["steps"].append(event["opportunities"])
                for key in components:
                    series[label][key].append(values[key])
                records.append(dict(step=event["opportunities"], epoch=event["epoch_index"]+1,
                                    **values, **{"weighted_"+k:event["weighted_loss_components"][k] for k in components}))
            elif event["event"] == "epoch_train" and values is not None:
                series[label]["train"][event["epoch"]] = values
            elif event["event"] == "validation":
                summary = event["metrics"].get("loss_components", {})
                if summary.get("complete"):
                    series[label]["validation"][event["epoch"]] = summary["mean"]
        with (output/label/"descriptor_components_steps.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "epoch", *components, *["weighted_"+k for k in components]])
            writer.writeheader(); writer.writerows(records)
        with (output/label/"descriptor_components_epochs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "split", *components])
            writer.writeheader()
            for split in ("train", "validation"):
                for epoch, values in series[label][split].items():
                    writer.writerow(dict(epoch=epoch, split=split, **values))
    for key in components:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for label, data in series.items():
            if not data["spec"].get(key+"_active", False):
                continue
            axes[0].plot(data["steps"], data[key], alpha=.12, linewidth=.5)
            axes[0].plot(data["steps"], ema(data[key]), label=label)
            for split, style in (("train", "-"), ("validation", "--")):
                rows = data[split]
                axes[1].plot(sorted(rows), [rows[e][key] for e in sorted(rows)], style, label=label+" "+split)
        for ax in axes:
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
        axes[0].set_xlabel("Optimizer opportunity"); axes[1].set_xlabel("Completed epoch")
        fig.suptitle(key+" | decomposition already included in descriptor loss")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output/("loss_"+key+"."+extension), dpi=170)
        plt.close(fig)
    receipt["descriptor_decomposition"] = "Raw/weighted step CSV, epoch CSV, PNG/PDF; not added twice to total"
    (output/"curve_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    runs = {}
    for item in args.runs:
        label, sep, path = item.partition("=")
        if not sep or label in runs:
            raise ValueError("Unique LABEL=PATH required")
        runs[label] = path
    print(json.dumps(render_evidence(runs, args.output)))


if __name__ == "__main__":
    main()
