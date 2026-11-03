"""Frozen B/C full-data queue: train, best/last held-out test, loss plots, receipts.

Run as a script or module. This supervisor never resumes or retries an experiment,
and never terminates another process. Launch its foreground under tmux on Linux.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_path_v1.checkpoint import atomic_json, read_checkpoint, seed_everything, sha256, verify_engine_source
from sam_path_v1.data import NativeInstanceDataset
from sam_path_v1.train import TrainConfig, _runtime_receipt


ARMS = ("B", "C")
OFFICIAL_FILES = ("sam3/model/vitdet.py", "sam3/model/necks.py", "sam3/sam/mask_decoder.py",
                  "sam3/sam/prompt_encoder.py", "sam3/sam/transformer.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def paired(config, smoke=False):
    value = json.loads(json.dumps(config))
    value["model"].pop("variant")
    if smoke:
        for key in ("max_batches", "save_every"):
            value.pop(key, None)
    return value


def check_disk(*paths):
    free = {str(p): shutil.disk_usage(p).free for p in dict.fromkeys(paths)}
    if any(value < 3 * 1024**3 for value in free.values()):
        raise RuntimeError(f"At least 3 GiB free is required on each relevant volume: {free}")
    return free


def source_files():
    root = Path(__file__).resolve().parent
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".sh"))


def validate_configs(config_dir, counts):
    configs = {arm: asdict(TrainConfig(**read(config_dir / f"{arm}.json"))) for arm in ARMS}
    for arm, config in configs.items():
        if config["model"]["variant"] != arm or config["model"]["backend"] != "sam31":
            raise ValueError(f"{arm}: expected matching SAM3.1 arm")
        if not config["device"].startswith("cuda") or config["max_batches"] is not None:
            raise ValueError("Full training requires CUDA and unlimited full configured epochs")
        if config["validate_every"] != 1:
            raise ValueError("Full experiment requires validation and loss logging every epoch")
        if paired(config) != paired(configs["B"]):
            raise ValueError("B/C configurations must differ only by model.variant")
    reference = configs["B"]
    for split, count in counts.items():
        dataset = NativeInstanceDataset(reference["manifest"], split, reference["model"]["canvas_size"],
                                        reference["seed"], reference["allow_condition"], reference["data_root"])
        if len(dataset) != count:
            raise ValueError(f"Expected {count} {split} images; manifest has {len(dataset)}")
    return configs


def validate_smoke(smoke, arm, config):
    saved, completion = read_checkpoint(smoke / "last.pt"), read(smoke / "completion.json")
    verify_engine_source(saved)
    actual = asdict(TrainConfig(**saved["train_config"]))
    if actual["model"]["variant"] != arm or actual["max_batches"] != 2:
        raise ValueError(f"{arm} admission requires its explicit two-update smoke")
    if paired(actual, True) != paired(config, True) or read(smoke / "config.json") != saved["train_config"]:
        raise ValueError(f"{arm} smoke configuration mismatch")
    if saved["provenance"].get("resume_from") is not None:
        raise ValueError("Smoke must initialize independently from the base SAM checkpoint")
    if completion["status"] not in ("budget_stopped", "completed"):
        raise ValueError("Smoke did not complete its bounded budget")
    for key, value in (("opportunities", 2), ("successful_updates", 2), ("amp_skips", 0)):
        if saved["progress"][key] != value or completion["progress"][key] != value:
            raise ValueError(f"{arm} smoke must have {key}={value}")
    steps = [float(state["step"]) for state in saved["optimizer"]["state"].values() if "step" in state]
    if not steps or any(step != 2 for step in steps):
        raise ValueError("Smoke optimizer state does not confirm two successful updates")
    events = [json.loads(line) for line in (smoke / "logs.jsonl").read_text(encoding="utf-8").splitlines()]
    updates = [event for event in events if event.get("event") == "update"]
    if len(updates) != 2 or any(event["amp_skipped"] or event["calls"] != {"encoder": 1, "decoder": 1}
                              or not all(math.isfinite(v) for v in event["loss"].values()) for event in updates):
        raise ValueError("Smoke logs are incomplete, nonfinite, skipped or violate one-pass inference")
    seed_everything(config["seed"])
    if saved["provenance"]["runtime"] != _runtime_receipt(config["device"]):
        raise ValueError("Smoke and full training must use the same server/runtime/visible devices")
    if saved["provenance"]["manifest_sha256"] != sha256(config["manifest"]):
        raise ValueError("Smoke manifest changed")
    base = saved["base_receipt"]
    if base["checkpoint_sha256"] != sha256(config["model"]["sam_checkpoint"]):
        raise ValueError("SAM base checkpoint changed")
    if set(base["source_sha256"]) != set(OFFICIAL_FILES):
        raise ValueError("Incomplete official SAM source receipt")
    for name in OFFICIAL_FILES:
        if sha256(Path(config["model"]["sam_repo"]) / name) != base["source_sha256"][name]:
            raise ValueError(f"Official SAM code changed: {name}")
    return dict(path=str(smoke), checkpoint_sha256=sha256(smoke / "last.pt"),
                base=base, runtime=saved["provenance"]["runtime"], passed=True)


def freeze(config_dir, configs, counts, smoke_dir):
    smokes = {arm: validate_smoke(smoke_dir / arm, arm, configs[arm]) for arm in ARMS}
    files = source_files() + [config_dir / f"{arm}.json" for arm in ARMS]
    files += [Path(configs["B"]["manifest"]), Path(configs["B"]["model"]["sam_checkpoint"])]
    files += [Path(configs["B"]["model"]["sam_repo"]) / name for name in OFFICIAL_FILES]
    return dict(passed=True, created_utc=now(), counts=counts, configs=configs, smokes=smokes,
                frozen_files={str(p): sha256(p) for p in files},
                file_stats={str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files},
                initialization="same original SAM checkpoint, component-specific deterministic random seeds; no pilot resume",
                test_policy="test used only after each arm finishes; fixed best/last checkpoints and frozen thresholds",
                order=list(ARMS))


def verify_frozen(admission, full=True):
    for name, expected in admission["frozen_files"].items():
        path = Path(name)
        if full:
            valid = path.is_file() and sha256(path) == expected
        else:
            valid = path.is_file() and [path.stat().st_size, path.stat().st_mtime_ns] == admission["file_stats"][name]
        if not valid:
            raise ValueError(f"Frozen experiment input changed: {name}")


def verify_training(root, config, counts):
    completion = read(root / "completion.json")
    progress = completion["progress"]
    expected_updates = math.ceil(counts["train"] / config["batch_size"]) * config["epochs"]
    expected = dict(epoch=config["epochs"], batch_cursor=0, opportunities=expected_updates,
                    successful_updates=expected_updates, amp_skips=0, samples=counts["train"] * config["epochs"])
    if completion["status"] != "completed" or any(progress.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Incomplete training budget; expected {expected}, got {progress}")
    if not all((root / name).is_file() for name in ("best.pt", "last.pt")):
        raise ValueError("Missing best or last checkpoint")
    provenance = read(root / "provenance.json")
    if provenance.get("resume_from") is not None:
        raise ValueError("Full arm unexpectedly resumed a previous experiment")
    for epoch in range(1, config["epochs"] + 1):
        validation = read(root / f"validation_epoch_{epoch:04d}.json")
        if (validation["images"] != counts["validation"] or not validation["complete"] or validation["failures"]
                or len(validation["per_image"]) != counts["validation"]):
            raise ValueError(f"Epoch {epoch} validation failed completeness admission")
        losses = validation.get("losses", {})
        if (losses.get("complete") is not True or losses.get("observed_images") != counts["validation"]
                or losses.get("failed_images") != 0 or not losses.get("mean")
                or not losses.get("weighted_mean")):
            raise ValueError(f"Epoch {epoch} validation loss coverage is incomplete")
        if any(not math.isfinite(value) for key in ("mean", "weighted_mean") for value in losses[key].values()):
            raise ValueError(f"Epoch {epoch} validation loss is nonfinite")
    return completion


def latest_update(path):
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 131072))
        lines = handle.read().splitlines(keepends=True)
    for line in reversed(lines):
        if not line.endswith(b"\n"):
            continue
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if event.get("event") == "update":
            return {key: event.get(key) for key in ("epoch", "batch_cursor", "opportunities", "successful_updates",
                                                  "amp_skips", "samples", "elapsed_seconds", "loss")}
    return None


def logged_validation_epochs(path):
    """Recent completed validation log events; file creation precedes log append."""
    if not path.is_file():
        return set()
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 2 * 1024**2))
        lines = handle.read().splitlines(keepends=True)
    epochs = set()
    for line in lines:
        if line.endswith(b"\n"):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if event.get("event") == "validation":
                epochs.add(event["epoch"])
    return epochs


class Supervisor:
    def __init__(self, args, admission, lock_fd=None):
        self.args, self.admission = args, admission
        self.root, self.variant, self.phase = Path(args.output), None, "admitted"
        self.child_pid, self.summaries = None, {}
        self.plot_index = 0
        self.integrity_failure = None
        self.lock_fd = lock_fd

    def status(self, **extra):
        row = dict(status=self.phase, variant=self.variant, launcher_pid=os.getpid(), child_pid=self.child_pid,
                   updated_utc=now(), completed_variants=list(self.summaries), **extra)
        if self.variant:
            row["progress"] = latest_update(self.root / self.variant / "logs.jsonl")
        atomic_json(self.root / "status.json", row)
        return row

    def event(self, **extra):
        row = self.status(**extra)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        # ASCII console receipts also work in Windows embedded-Python test runs.
        print(json.dumps(row, ensure_ascii=True, allow_nan=False), flush=True)

    def plots(self, label):
        runs = [f"{arm}={self.root / arm}" for arm in ARMS if (self.root / arm / "loss_spec.json").is_file()]
        if not runs:
            return
        self.plot_index += 1
        destination = self.root / "curves" / f"{self.plot_index:03d}_{label}"
        command = [sys.executable, str(Path(__file__).with_name("plot.py")), "--runs", *runs,
                   "--output", str(destination), "--ema-span", "200"]
        with (self.root / "plot.stdout.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Loss plotting failed (exit={result.returncode}); see plot.stdout.log")
        atomic_json(self.root / "latest_curves.json", dict(path=str(destination), updated_utc=now()))

    def command(self, command, log_name, allowed=(0,), watch_epochs=False):
        verify_frozen(self.admission)
        check_disk(self.root, Path(__file__).resolve().parent)
        start, observed = time.monotonic(), set()
        run = self.root / self.variant if self.variant else self.root
        with (self.root / log_name).open("x", encoding="utf-8") as log:
            # A surviving child retains the shared lock even if its supervisor
            # is interrupted. No second GPU queue can enter during that interval.
            inherited = {"pass_fds": (self.lock_fd,)} if self.lock_fd is not None else {}
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=Path(__file__).resolve().parents[1], **inherited)
            self.child_pid = process.pid
            self.event(command=command, started_utc=now())
            while process.poll() is None:
                time.sleep(self.args.poll_seconds)
                # Integrity changes stop the queue after this owned child exits. We
                # do not orphan it or silently kill a running training experiment.
                try:
                    verify_frozen(self.admission, full=False)
                except Exception as exc:
                    self.integrity_failure = str(exc)
                reports = set(run.glob("validation_epoch_*.json")) if watch_epochs else set()
                if reports - observed:
                    recorded_epochs = logged_validation_epochs(run / "logs.jsonl")
                    reports = observed | {path for path in reports if int(path.stem.rsplit("_", 1)[1]) in recorded_epochs}
                if reports - observed and self.integrity_failure is None:
                    observed = reports
                    try:
                        self.plots(f"{self.variant}_epoch_{len(reports):04d}")
                    except Exception as exc:
                        self.integrity_failure = f"Curve generation failed: {exc}"
                elapsed = time.monotonic() - start
                self.status(process_elapsed_seconds=elapsed, process_alive=process.poll() is None,
                            elapsed_warning=elapsed > self.args.warning_hours * 3600,
                            pending_queue_failure=self.integrity_failure)
            code = process.wait()
        self.child_pid = None
        self.event(exit_code=code, process_elapsed_seconds=time.monotonic() - start)
        if self.integrity_failure:
            raise RuntimeError(self.integrity_failure)
        verify_frozen(self.admission)
        if code not in allowed:
            raise RuntimeError(f"Command failed with exit={code}; see {log_name}")
        return code

    def run(self):
        for arm in ARMS:
            self.variant, self.phase = arm, "training"
            config = self.admission["configs"][arm]
            arm_root = self.root / arm
            self.command([sys.executable, "-u", "-m", "sam_path_v1.train", "--config",
                          str(Path(self.args.config_dir) / f"{arm}.json"), "--output", str(arm_root)],
                         f"{arm}.train.stdout.log", watch_epochs=True)
            completion = verify_training(arm_root, config, self.admission["counts"])
            self.phase = "testing"
            tests = {}
            for checkpoint in ("best", "last"):
                destination = self.root / "test" / arm / checkpoint
                command = [sys.executable, "-u", "-m", "sam_path_v1.evaluate", "--checkpoint",
                           str(arm_root / f"{checkpoint}.pt"), "--manifest", config["manifest"],
                           "--split", "test", "--output", str(destination), "--device", config["device"],
                           "--amp", config["amp"]]
                if config["data_root"]:
                    command += ["--data-root", config["data_root"]]
                if config["allow_condition"]:
                    command += ["--allow-condition"]
                code = self.command(command, f"{arm}.{checkpoint}.test.stdout.log", allowed=(0, 2))
                report = read(destination / "metrics.json")
                if report["images"] != self.admission["counts"]["test"] or len(report["per_image"]) != report["images"]:
                    raise ValueError("Test report omitted images")
                if (report.get("split") != "test"
                        or report.get("checkpoint_sha256") != sha256(arm_root / f"{checkpoint}.pt")
                        or report.get("manifest_sha256") != sha256(config["manifest"])
                        or report.get("object_threshold") != config["object_threshold"]
                        or report.get("mask_threshold") != config["mask_threshold"]):
                    raise ValueError("Test report checkpoint, split, manifest or thresholds do not match the frozen run")
                tests[checkpoint] = {k: v for k, v in report.items() if k != "per_image"}
                tests[checkpoint].update(exit_code=code, report=str(destination / "metrics.json"))
            self.phase = "plotting"
            self.plots(f"{arm}_complete")
            summary = dict(variant=arm, completion=completion, tests=tests,
                           best_checkpoint=str(arm_root / "best.pt"), last_checkpoint=str(arm_root / "last.pt"))
            atomic_json(arm_root / "full_arm_summary.json", summary)
            self.summaries[arm] = summary
            write_summary(self.root, self.summaries, final=False)
            self.event()
        self.variant, self.phase = None, "plotting"
        self.plots("BC_final")
        failure = any(not t["complete"] or t["failures"] for s in self.summaries.values() for t in s["tests"].values())
        self.phase = "completed_with_test_failures" if failure else "completed"
        write_summary(self.root, self.summaries, final=True)
        result = dict(status=self.phase, finished_utc=now(), variants=self.summaries,
                      all_training_completed=True, test_failures_included=failure, exit_code=2 if failure else 0)
        atomic_json(self.root / "full_completion.json", result)
        self.event()
        return result["exit_code"]


def write_summary(root, summaries, final):
    columns = ("variant", "checkpoint", "epoch", "images", "instance_iou_macro", "pq_global",
               "union_iou_macro", "tp", "fp", "fn", "failures", "complete", "successful_updates", "amp_skips")
    rows = []
    for arm, summary in summaries.items():
        for name, report in summary["tests"].items():
            row = {key: report.get(key) for key in columns}
            row.update(variant=arm, checkpoint=name, epoch=report["checkpoint_progress"]["epoch"],
                       successful_updates=report["checkpoint_progress"]["successful_updates"],
                       amp_skips=report["checkpoint_progress"]["amp_skips"])
            rows.append(row)
    with (root / "test_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    atomic_json(root / "test_results.json", dict(final=final, rows=rows, variants=summaries))
    lines = ["# Full WireSeg B/C results", "", f"Final: {final}. Test thresholds were frozen before training.", "",
             "| Variant | Checkpoint | Epoch | Images | Instance IoU | PQ | Foreground IoU | Failures |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "unavailable" if value is None else f"{value * 100:.3f}%"
    for row in rows:
        lines.append(f"| {row['variant']} | {row['checkpoint']} | {row['epoch']} | {row['images']} | "
                     f"{percent(row['instance_iou_macro'])} | {percent(row['pq_global'])} | "
                     f"{percent(row['union_iou_macro'])} | {row['failures']} |")
    lines += ["", "Best checkpoint selection uses validation instance IoU then PQ only. Test results do not select models.",
              "Native RGB, no GT prompts, one encoder and one batched decoder call per image.",
              "Synthetic image holdout is not verified scene isolation. Real Wire300 is not evaluated by this queue.",
              "B/C total loss has different auxiliary terms; compare common components and held-out metrics.",
              "Per-image failures remain in reports. See test/{B,C}/{best,last}/metrics.json and latest_curves.json."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-dir", required=True, help="Contains matching two-step B/ and C/ runs")
    parser.add_argument("--lock-file", required=True, help="Existing shared project flock; never replaced or deleted")
    parser.add_argument("--train-images", type=int, default=32400)
    parser.add_argument("--val-images", type=int, default=1800)
    parser.add_argument("--test-images", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--warning-hours", type=float, default=24, help="Advisory only; never auto-kills a child")
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("The full queue requires Linux flock; unit tests are platform independent")
    if not 1 <= args.poll_seconds <= 120 or args.warning_hours <= 0:
        parser.error("poll-seconds must be 1..120 and warning-hours must be positive")
    counts = dict(train=args.train_images, validation=args.val_images, test=args.test_images)
    if any(count < 1 for count in counts.values()):
        parser.error("All split counts must be positive")
    for key in ("config_dir", "output", "smoke_dir", "lock_file"):
        setattr(args, key, str(Path(getattr(args, key)).expanduser().resolve()))
    import fcntl
    with open(args.lock_file, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=False)
        (root / "launcher.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
        atomic_json(root / "status.json", dict(status="admission", launcher_pid=os.getpid(), updated_utc=now()))
        supervisor = None
        try:
            check_disk(root, Path(__file__).resolve().parent)
            configs = validate_configs(Path(args.config_dir), counts)
            admission = freeze(Path(args.config_dir), configs, counts, Path(args.smoke_dir))
            atomic_json(root / "admission.json", admission)
            supervisor = Supervisor(args, admission, lock.fileno())
            return supervisor.run()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            atomic_json(root / "full_completion.json", dict(status="failed", error=error, finished_utc=now(),
                        variant=supervisor.variant if supervisor else None,
                        variants=supervisor.summaries if supervisor else {},
                        child_pid=supervisor.child_pid if supervisor else None,
                        note="No automatic retry or process termination was performed", exit_code=1))
            atomic_json(root / "status.json", dict(status="failed", error=error, updated_utc=now(),
                        launcher_pid=os.getpid(), child_pid=supervisor.child_pid if supervisor else None))
            raise


if __name__ == "__main__":
    raise SystemExit(main())
