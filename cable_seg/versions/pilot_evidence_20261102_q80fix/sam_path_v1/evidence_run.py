"""Finite, frozen pilot evidence queue; no retries, deletion or notifications."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .checkpoint import atomic_json, read_checkpoint, sha256, verify_engine_source
from .data import NativeInstanceDataset
from .full_run import latest_update, check_disk, verify_training
from .train import TrainConfig


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot():
    package = Path(__file__).resolve().parent
    return {str(p): sha256(p) for p in sorted(package.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".sh", ".json") and "__pycache__" not in p.parts}


def verify_snapshot(snapshot):
    for name, digest in snapshot.items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise RuntimeError(f"Frozen file changed: {name}")


def expected_calls(config):
    model = config["model"]
    chunk = model.get("decoder_query_chunk", 0)
    return {"encoder": 1, "decoder": math.ceil(model["num_queries"]/chunk)
            if chunk and model["backend"] != "sam31_native" else 1}


def exact_members(rows, expected, context):
    ids = [row.get("id") for row in rows]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"{context}: missing, duplicate or unexpected IDs")


def verify_smoke(dest, config):
    """Two actual finite updates, correct calls and exact admitted base receipt."""
    saved, completion = read_checkpoint(dest/"last.pt"), read(dest/"completion.json")
    verify_engine_source(saved)
    if not smoke_compatible(saved["train_config"], config):
        raise ValueError("Smoke does not represent its experiment structure")
    if completion["status"] != "budget_stopped":
        raise ValueError("Smoke did not stop at its two-update budget")
    for key, expected in (("successful_updates", 2), ("opportunities", 2), ("amp_skips", 0), ("samples", 2)):
        if saved["progress"][key] != expected or completion["progress"][key] != expected:
            raise ValueError(f"Smoke progress failed: {key}")
    updates = [row for row in map(json.loads, (dest/"logs.jsonl").read_text().splitlines()) if row["event"] == "update"]
    if len(updates) != 2:
        raise ValueError("Smoke needs exactly two logged optimizer updates")
    for index, row in enumerate(updates, 1):
        if (row["successful_updates"] != index or row["opportunities"] != index or row["amp_skipped"]
                or row["calls"] != expected_calls(config)
                or not all(math.isfinite(v) for v in row["loss"].values())):
            raise ValueError("Smoke update is nonfinite, skipped or violates configured call count")
    base = saved["base_receipt"]
    model = config["model"]
    checkpoint = model["sam2_checkpoint"] if model["backend"] == "sam21" else model["sam_checkpoint"]
    if base.get("checkpoint_sha256") != sha256(checkpoint) or not base.get("strict_retained_state"):
        raise ValueError("Smoke base checkpoint admission failed")
    sources = base.get("source_sha256", {})
    if not sources:
        raise ValueError("Smoke has no official source receipt")
    for name, digest in sources.items():
        if sha256(Path(base["repo"])/name) != digest:
            raise ValueError("Smoke official source changed")
    return saved


def verify_log_membership(dest, config, train_ids, validation_ids):
    """Validate observations, rather than trusting a final progress counter."""
    members = [set() for _ in range(config["epochs"])]
    updates = 0
    validation_epochs, training_epochs = [], []
    with (dest/"logs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Training log ends with a partial line")
            row = json.loads(line)
            if row["event"] == "update":
                updates += 1
                epoch = row["epoch_index"]
                if (epoch != (updates-1)//len(train_ids) or row["successful_updates"] != updates
                        or row["opportunities"] != updates or row["amp_skips"] != 0
                        or row["amp_skipped"] or row["calls"] != expected_calls(config)
                        or len(row["ids"]) != 1 or row["ids"][0] not in train_ids
                        or row["ids"][0] in members[epoch]
                        or not all(math.isfinite(v) for v in row["loss"].values())):
                    raise ValueError("Training update evidence mismatch")
                members[epoch].add(row["ids"][0])
            elif row["event"] == "validation": validation_epochs.append(row["epoch"])
            elif row["event"] == "epoch_train": training_epochs.append(row["epoch"])
    sequence = list(range(1, config["epochs"]+1))
    if updates != len(train_ids)*config["epochs"] or any(m != set(train_ids) for m in members):
        raise ValueError("Each epoch must contain every frozen training member exactly once")
    if validation_epochs != sequence or training_epochs != sequence:
        raise ValueError("Training/validation epoch log coverage is incomplete")
    for epoch in sequence:
        report = read(dest/f"validation_epoch_{epoch:04d}.json")
        exact_members(report["per_image"], validation_ids, "validation")


def verify_test_report(report, expected_ids, manifest_hash, checkpoint_hash, config, tag):
    exact_members(report["per_image"], expected_ids, "fixed test")
    if report["images"] != len(expected_ids):
        raise ValueError("Test image count differs from member count")
    if report.get("manifest_sha256") != manifest_hash or report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Test provenance does not match the fixed manifest/checkpoint")
    progress = report["checkpoint_progress"]
    epochs = config["epochs"]
    if tag == "last" and progress["epoch"] != epochs:
        raise ValueError("Last checkpoint did not complete the training budget")
    if not 1 <= progress["epoch"] <= epochs or progress["successful_updates"] != progress["epoch"]*2000 or progress["amp_skips"]:
        raise ValueError("Test checkpoint progress mismatch")
    for key in ("object_threshold", "mask_threshold"):
        if report.get(key) != config[key]: raise ValueError("Test thresholds changed")
    # An unreadable GT is retained as a failure, but cannot count as completed
    # evaluation or support a full-dataset PQ claim.
    if report.get("unreadable_ground_truth", 0) or any(not r.get("gt_available", True) or r.get("n_gt") is None for r in report["per_image"]):
        return False
    return bool(report.get("complete"))


def structure_key(config):
    model = dict(config["model"])
    model.pop("init_seed", None)
    return json.dumps(dict(model=model, loss=config["loss"], mosaic=config.get("mosaic")), sort_keys=True)


def smoke_compatible(smoke, config):
    def stripped(c):
        c = json.loads(json.dumps(c))
        for name in ("max_batches", "save_every", "seed"):
            c.pop(name, None)
        c["model"].pop("init_seed", None)
        return c
    return stripped(smoke) == stripped(config)


class EvidenceQueue:
    def __init__(self, protocol, output, lock_fd=None, poll_seconds=30, protocol_path=None):
        self.protocol, self.root = protocol, Path(output)
        self.lock_fd, self.poll_seconds = lock_fd, poll_seconds
        self.phase, self.job, self.child = "preflight", None, None
        self.finished, self.evaluations, self.integrity_failure = [], [], None
        self.snapshot = {}
        self.protocol_path = Path(protocol_path).resolve() if protocol_path else None
        self.task_receipts = []
        self.expected_evaluations = self.expected_tasks()

    def expected_tasks(self):
        result = []
        for job in self.protocol["jobs"]:
            ident = job["id"]
            for tag in ("best", "last"):
                result.append(dict(id="test_"+ident+"_"+tag, phase="synthetic_test", kind="synthetic",
                                   job=ident, tag=tag, report=str(self.root/"test"/ident/tag/"metrics.json"), expected_images=1800))
            if job["stage"] == "main":
                result.append(dict(id="occlusion_"+ident, phase="occlusion_test", kind="robustness", job=ident,
                                   report=str(self.root/"robustness"/ident/"completion.json"), expected_images=300))
            if job["seed"] == 42 and job["stage"] != "native80":
                result.append(dict(id="benchmark_"+ident, phase="efficiency", kind="benchmark", job=ident,
                                   report=str(self.root/"benchmark"/ident/"benchmark.json"), expected_images=100))
        for item in self.protocol.get("additional_evaluations", []):
            if not item.get("report"):
                raise ValueError("Every additional evaluation requires a fixed report path")
            report = Path(item["report"])
            result.append(dict(id=item["id"], phase=item["phase"], kind=item.get("kind", item["phase"]),
                               job=item.get("job", item["id"]), tag=item.get("tag"),
                               report=str(report if report.is_absolute() else self.root/report),
                               expected_images=item.get("expected_images")))
        if len({x["id"] for x in result}) != len(result):
            raise ValueError("Duplicate expected evaluation ID")
        return result

    def freeze(self, configs):
        """Freeze before the first smoke, including original protocol and assets."""
        self.snapshot = source_snapshot()
        paths = [self.root/"protocol.json", Path(self.protocol["test_manifest"])]
        if self.protocol_path is not None: paths.append(self.protocol_path)
        paths += [Path(p) for p in self.protocol.get("frozen_inputs", [])]
        for job, config in zip(self.protocol["jobs"], configs):
            paths += [Path(job["config_path"]), Path(config["manifest"])]
            model = config["model"]
            key = "sam2" if model["backend"] == "sam21" else "sam"
            paths.append(Path(model[key+"_checkpoint"]))
            repo = Path(model[key+"_repo"])
            for path in repo.rglob("*"):
                if path.is_file() and path.suffix in (".py", ".json", ".yaml", ".yml", ".gz") and not any(
                        component in (".git", "__pycache__", ".pytest_cache") for component in path.parts):
                    paths.append(path)
        known_flags = {"--manifest", "--config", "--events", "--annotations", "--split-manifest", "--data-manifest"}
        for item in self.protocol.get("additional_evaluations", []):
            paths += [Path(p) for p in item.get("inputs", [])]
            arguments = item.get("arguments", [])
            paths += [Path(arguments[i+1]) for i, token in enumerate(arguments[:-1]) if token in known_flags]
        for path in sorted(set(p.resolve() for p in paths)):
            if not path.is_file(): raise FileNotFoundError(f"Frozen input missing: {path}")
            self.snapshot[str(path)] = sha256(path)
        atomic_json(self.root/"expected_evaluations.json", self.expected_evaluations)
        self.snapshot[str((self.root/"expected_evaluations.json").resolve())] = sha256(self.root/"expected_evaluations.json")
        atomic_json(self.root/"freeze.json", dict(created_utc=utc(), frozen_files=self.snapshot,
                                                  before_first_smoke=True))

    def record_task(self, ident, code, complete_override=None):
        expected = next(x for x in self.expected_evaluations if x["id"] == ident)
        path = Path(expected["report"])
        receipt = {**expected, "exit_code": code, "missing": not path.is_file()}
        if not path.is_file():
            receipt.update(complete=False, failures=1, diagnostic_failures=0, sha256=None)
        else:
            report = read(path)
            if expected["kind"] == "robustness":
                from .robustness import SEVERITIES
                reports = report.get("reports", [])
                coverage = {r.get("severity_widths") for r in reports} == set(SEVERITIES) and len(reports) == len(SEVERITIES)
                coverage &= all(r.get("images") == expected["expected_images"] for r in reports)
                if coverage:
                    from .robustness import fixed_indices
                    records = [r for r in read(self.protocol["test_manifest"])["records"] if r["split"] == "test"]
                    selected = {records[i]["id"] for i in fixed_indices(records, expected["expected_images"])}
                    for severity in SEVERITIES:
                        details = read(path.parent/f"severity_{severity}.json")
                        exact_members(details["per_image"], selected, "robustness severity")
                        coverage &= (details.get("complete") is True and details.get("failures", 0) == 0
                                     and details.get("diagnostic_failures", 0) == 0)
                failures = sum(r.get("failures", 0) for r in reports)
                diagnostic = sum(r.get("diagnostic_failures", 0) for r in reports)
                complete = coverage and report.get("status") == "completed" and all(r.get("complete") for r in reports)
            else:
                failures, diagnostic = report.get("failures", 0), report.get("diagnostic_failures", 0)
                complete = report.get("complete", report.get("status") == "completed")
                if expected["expected_images"] is not None:
                    complete = complete and report.get("images") == expected["expected_images"]
            if complete_override is not None: complete = complete and complete_override
            receipt.update(complete=bool(complete and code == 0 and not failures and not diagnostic),
                           failures=failures, diagnostic_failures=diagnostic, sha256=sha256(path))
        if any(r["id"] == ident for r in self.task_receipts): raise ValueError("Duplicate task receipt")
        self.task_receipts.append(receipt)
        self.write_task_receipts()
        return receipt

    def write_task_receipts(self):
        observed = {x["id"] for x in self.task_receipts}
        missing = [x["id"] for x in self.expected_evaluations if x["id"] not in observed]
        result = dict(expected=self.expected_evaluations, tasks=self.task_receipts,
                      missing_expected=missing,
                      complete=not missing and all(r["complete"] for r in self.task_receipts))
        atomic_json(self.root/"task_receipts.json", result)
        return result

    def status(self, **extra):
        item = dict(updated_utc=utc(), phase=self.phase, job_id=self.job,
                    launcher_pid=os.getpid(), child_pid=self.child,
                    finished_training=self.finished, total_training=len(self.protocol["jobs"]), **extra)
        if self.job:
            item["progress"] = latest_update(self.root / "train" / self.job / "logs.jsonl")
        atomic_json(self.root / "status.json", item)
        return item

    def event(self, **extra):
        item = self.status(**extra)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False)+"\n")
        print(json.dumps(item), flush=True)

    def command(self, name, module, arguments, allowed=(0,)):
        if self.snapshot:
            verify_snapshot(self.snapshot)
        check_disk(self.root, Path(__file__).parent)
        command = [sys.executable, "-m", "sam_path_v1."+module, *map(str, arguments)]
        logpath = self.root / "stdout" / (name+".log")
        started = time.monotonic()
        with logpath.open("x", encoding="utf-8") as log:
            inherited = dict(pass_fds=(self.lock_fd,)) if self.lock_fd is not None else {}
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       stdout=log, stderr=subprocess.STDOUT, **inherited)
            self.child = process.pid
            self.event(command=command, stdout=str(logpath))
            while process.poll() is None:
                time.sleep(self.poll_seconds)
                self.status(process_alive=process.poll() is None,
                            process_elapsed_seconds=time.monotonic()-started)
            code = process.wait()
        self.child = None
        self.event(exit_code=code, process_elapsed_seconds=time.monotonic()-started)
        if self.snapshot:
            verify_snapshot(self.snapshot)
        if code not in allowed:
            raise RuntimeError(f"{name} exited {code}; evidence retained in {logpath}")
        return code

    def admit(self):
        configs = []
        structures = {}
        for job in self.protocol["jobs"]:
            config = asdict(TrainConfig(**read(job["config_path"])))
            expected = asdict(TrainConfig(**job["config"]))
            if config != expected:
                raise ValueError(f"Protocol/config disagree: {job['id']}")
            if (config["epochs"] != 5 or config["batch_size"] != 1 or config["max_batches"] is not None
                    or config["validate_every"] != 1 or config["seed"] != job["seed"]):
                raise ValueError("Unapproved training budget or validation schedule")
            if config["model"]["backend"] == "toy":
                raise ValueError("Real queue refuses toy backbones")
            configs.append(config)
            structures.setdefault(structure_key(config), (job, config))
        first = configs[0]
        if any(c["manifest"] != first["manifest"] or c["data_root"] != first["data_root"] for c in configs):
            raise ValueError("All arms must use the same original pilot manifest/data root")
        if sha256(first["manifest"]) != self.protocol["pilot_manifest_sha256"]:
            raise ValueError("Original pilot manifest changed")
        if sha256(self.protocol["test_manifest"]) != self.protocol["test_manifest_sha256"]:
            raise ValueError("Fixed test manifest changed")
        for split, expected in (("train", 2000), ("validation", 300)):
            data = NativeInstanceDataset(first["manifest"], split, first["model"]["canvas_size"],
                                         first["seed"], True, first["data_root"])
            if len(data) != expected:
                raise ValueError(f"Original {split} members/count changed")
        pilot = read(first["manifest"])
        self.train_ids = {r["id"] for r in pilot["records"] if r["split"] == "train"}
        self.validation_ids = {r["id"] for r in pilot["records"] if r["split"] == "validation"}
        self.test_ids = {r["id"] for r in read(self.protocol["test_manifest"])["records"] if r["split"] == "test"}
        if (len(self.train_ids), len(self.validation_ids), len(self.test_ids)) != (2000, 300, 1800):
            raise ValueError("Frozen IDs are duplicated or missing")
        self.freeze(configs)
        self.command("unit_tests", "check", [])
        admissions = []
        for index, (job, config) in enumerate(structures.values()):
            self.job = job["id"]
            smoke_config = {**config, "max_batches": 2, "save_every": 1}
            path = self.root / "smoke_configs" / (job["id"]+".json")
            atomic_json(path, smoke_config)
            dest = self.root / "smoke" / job["id"]
            self.command("smoke_"+job["id"], "train", ["--config", path, "--output", dest])
            saved = verify_smoke(dest, config)
            runtime_path = dest / "runtime_admission.json"
            self.command("runtime_admission_"+job["id"], "runtime_admission", [
                "--checkpoint", dest/"last.pt", "--manifest", first["manifest"],
                "--data-root", first["data_root"], "--device", config["device"], "--output", runtime_path])
            runtime = read(runtime_path)
            if (not runtime.get("complete") or runtime.get("checkpoint_sha256") != sha256(dest/"last.pt")
                    or runtime.get("manifest_sha256") != self.protocol["pilot_manifest_sha256"]
                    or runtime.get("image_id") not in self.train_ids or runtime.get("split") != "train"
                    or runtime.get("gt_loaded") is not False or runtime.get("gt_scored") is not False
                    or runtime.get("model_forwards") != 2 or runtime.get("calls_per_forward") != expected_calls(config)):
                raise ValueError("Fresh-reload RGB runtime admission receipt failed")
            admissions.append(dict(job=job["id"], structure=structure_key(config),
                                   checkpoint_sha256=sha256(dest / "last.pt"),
                                   base=saved["base_receipt"], runtime=saved["provenance"]["runtime"],
                                   runtime_admission=runtime, runtime_admission_sha256=sha256(runtime_path)))
        verify_snapshot(self.snapshot)
        atomic_json(self.root / "admission.json", dict(passed=True, created_utc=utc(), structures=admissions,
                    frozen_files=self.snapshot, main_runs=30, sam21_runs=9, native80_runs=3,
                    no_test_scoring_during_admission=True))
        self.job = None
        self.event(admitted_structures=len(admissions))

    def train_all(self):
        for job in self.protocol["jobs"]:
            self.job, self.phase = job["id"], "training"
            dest = self.root / "train" / self.job
            self.command("train_"+self.job, "train", ["--config", job["config_path"], "--output", dest])
            config = asdict(TrainConfig(**read(job["config_path"])))
            completion = verify_training(dest, config, {"train": 2000, "validation": 300})
            verify_log_membership(dest, config, self.train_ids, self.validation_ids)
            self.finished.append(dict(id=self.job, stage=job["stage"], variant=job["variant"],
                                      seed=job["seed"], completion=completion))
            self.command("curves_"+self.job, "evidence_plot", ["--runs", self.job+"="+str(dest),
                         "--output", self.root / "curves" / self.job])
            self.event(training_verified=True)

    def evaluate_all(self):
        # No test results exist until every predeclared training run has finished.
        for job in self.protocol["jobs"]:
            self.job, self.phase = job["id"], "synthetic_test"
            for tag in ("best", "last"):
                checkpoint = self.root / "train" / self.job / (tag+".pt")
                destination = self.root / "test" / self.job / tag
                task_id = "test_"+self.job+"_"+tag
                code = self.command(task_id, "evaluate", ["--checkpoint", checkpoint,
                    "--manifest", self.protocol["test_manifest"], "--data-root", self.protocol["test_data_root"],
                    "--split", "test", "--output", destination, "--device", "cuda", "--allow-condition",
                    "--identity-diagnostics"], allowed=(0, 2))
                report = read(destination / "metrics.json")
                config = asdict(TrainConfig(**read(job["config_path"])))
                complete = verify_test_report(report, self.test_ids, self.protocol["test_manifest_sha256"],
                                              sha256(checkpoint), config, tag)
                self.record_task(task_id, code, complete)
                self.evaluations.append(dict(job=self.job, tag=tag, report=str(destination / "metrics.json"),
                                            failures=report["failures"], complete=report["complete"]))
            if job["stage"] == "main":
                self.phase = "occlusion_test"
                code = self.command("occlusion_"+self.job, "robustness", ["--checkpoint", self.root / "train" / self.job / "best.pt",
                    "--manifest", self.protocol["test_manifest"], "--data-root", self.protocol["test_data_root"],
                    "--output", self.root / "robustness" / self.job, "--device", "cuda", "--count", "300"], allowed=(0, 2))
                self.record_task("occlusion_"+self.job, code)
            if job["seed"] == 42 and job["stage"] != "native80":
                self.phase = "efficiency"
                code = self.command("benchmark_"+self.job, "benchmark", ["--checkpoint", self.root / "train" / self.job / "best.pt",
                    "--manifest", self.protocol["test_manifest"], "--data-root", self.protocol["test_data_root"],
                    "--output", self.root / "benchmark" / self.job], allowed=(0, 2))
                self.record_task("benchmark_"+self.job, code)

    def run(self):
        self.admit()
        self.train_all()
        self.evaluate_all()
        # Real/external evaluation is appended from explicitly frozen job commands.
        for item in self.protocol.get("additional_evaluations", []):
            self.job, self.phase = item["id"], item["phase"]
            code = self.command(item["id"], item["module"], item["arguments"], allowed=(0, 2))
            self.record_task(item["id"], code)
        tasks = self.write_task_receipts()
        self.job, self.phase = None, "reporting"
        self.command("final_report", "evidence_report", ["--protocol", self.root / "protocol.json",
                     "--runs", self.root, "--output", self.root / "final_results"])
        analysis = read(self.root/"final_results"/"analysis.json")
        complete = tasks["complete"] and analysis.get("complete") is True
        self.phase = "completed" if complete else "completed_with_evaluation_failures"
        atomic_json(self.root / "completion.json", dict(status=self.phase, finished_utc=utc(),
                    training=self.finished, tests=self.evaluations, task_receipts=tasks,
                    report_complete=analysis.get("complete"), retried=False, restarted=False))
        self.event()
        self.command("package_results", "evidence_report", ["--protocol", self.root/"protocol.json",
                     "--runs", self.root, "--output", self.root/"final_results", "--package-only"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True); p.add_argument("--output", required=True)
    p.add_argument("--lock", required=True); p.add_argument("--poll-seconds", type=float, default=30)
    args = p.parse_args(argv)
    if os.name != "posix":
        raise RuntimeError("Real training supervisor requires Linux advisory locking")
    import fcntl
    lock = Path(args.lock).open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("stdout", "smoke_configs", "smoke", "train", "curves", "test", "robustness", "benchmark"):
        (root / name).mkdir()
    protocol = read(args.protocol)
    atomic_json(root / "protocol.json", protocol)
    queue = EvidenceQueue(protocol, root, lock.fileno(), args.poll_seconds, args.protocol)
    try:
        queue.run()
    except BaseException as exc:
        queue.phase = "failed"
        queue.event(error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "completion.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}",
                    training=queue.finished, tests=queue.evaluations, task_receipts=queue.write_task_receipts(),
                    finished_utc=utc(), retried=False))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
