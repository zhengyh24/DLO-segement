"""Queue integrity checks; tiny fixtures never claim model accuracy."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sam_path_v1.checkpoint import sha256
from sam_path_v1.full_run import (Supervisor, latest_update, logged_validation_epochs, paired, validate_configs,
                                  verify_frozen, verify_training, write_summary)
from sam_path_v1.train import TrainConfig


def write(path, obj):
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


class FullQueueTests(unittest.TestCase):
    def test_pairing_only_ignores_variant(self):
        b = asdict(TrainConfig(manifest="manifest.json", device="cuda", model={"variant": "B"}))
        c = asdict(TrainConfig(manifest="manifest.json", device="cuda", model={"variant": "C"}))
        self.assertEqual(paired(b), paired(c))
        c["epochs"] += 1
        self.assertNotEqual(paired(b), paired(c))
        c["epochs"] -= 1
        c["max_batches"] = 2
        self.assertEqual(paired(b, True), paired(c, True))
        self.assertNotEqual(paired(b), paired(c))

    def test_config_admission_rejects_hidden_budget_difference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for arm in "BC":
                write(root / f"{arm}.json", asdict(TrainConfig(manifest="m.json", device="cuda",
                                      epochs=5 if arm == "B" else 10, model={"variant": arm})))
            with self.assertRaisesRegex(ValueError, "only by model.variant"):
                validate_configs(root, dict(train=3, validation=1, test=1))

    def test_hash_fence_detects_equal_size_content_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "frozen.json"
            path.write_text("AAAA", encoding="utf-8")
            receipt = dict(frozen_files={str(path): sha256(path)},
                           file_stats={str(path): [path.stat().st_size, path.stat().st_mtime_ns]})
            verify_frozen(receipt)
            path.write_text("BBBB", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen experiment input changed"):
                verify_frozen(receipt)

    def test_completed_training_requires_actual_updates_samples_and_all_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = dict(epochs=2, batch_size=2)
            counts = dict(train=3, validation=2)
            progress = dict(epoch=2, batch_cursor=0, opportunities=4, successful_updates=4,
                            amp_skips=0, samples=6)
            write(root / "completion.json", dict(status="completed", progress=progress))
            write(root / "provenance.json", dict(resume_from=None))
            for name in ("best.pt", "last.pt"):
                (root / name).touch()
            for epoch in (1, 2):
                write(root / f"validation_epoch_{epoch:04d}.json",
                      dict(images=2, complete=True, failures=0, per_image=[{}, {}],
                           losses=dict(complete=True, observed_images=2, failed_images=0,
                                       mean={"total": .1}, weighted_mean={"mask_dice": .1})))
            self.assertEqual(verify_training(root, config, counts)["status"], "completed")
            report = read_json(root / "validation_epoch_0001.json")
            report["losses"]["complete"] = False
            write(root / "validation_epoch_0001.json", report)
            with self.assertRaisesRegex(ValueError, "validation loss coverage"):
                verify_training(root, config, counts)
            report["losses"]["complete"] = True
            write(root / "validation_epoch_0001.json", report)
            progress["successful_updates"] = 3
            write(root / "completion.json", dict(status="completed", progress=progress))
            with self.assertRaisesRegex(ValueError, "Incomplete training budget"):
                verify_training(root, config, counts)
            progress["successful_updates"] = 4
            write(root / "completion.json", dict(status="budget_stopped", progress=progress))
            with self.assertRaisesRegex(ValueError, "Incomplete training budget"):
                verify_training(root, config, counts)

    def test_missing_validation_rows_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            progress = dict(epoch=1, batch_cursor=0, opportunities=3, successful_updates=3,
                            amp_skips=0, samples=3)
            write(root / "completion.json", dict(status="completed", progress=progress))
            write(root / "provenance.json", dict(resume_from=None))
            for name in ("best.pt", "last.pt"):
                (root / name).touch()
            write(root / "validation_epoch_0001.json", dict(images=2, complete=True, failures=0, per_image=[{}]))
            with self.assertRaisesRegex(ValueError, "validation failed"):
                verify_training(root, dict(epochs=1, batch_size=1), dict(train=3, validation=2))

    def test_status_reader_does_not_read_partial_update(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "logs.jsonl"
            path.write_text('{"event":"update","successful_updates":12}\n'
                            '{"event":"update","successful_updates":13}', encoding="utf-8")
            self.assertEqual(latest_update(path)["successful_updates"], 12)

    def test_plot_epoch_waits_for_complete_validation_log_event(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "logs.jsonl"
            path.write_text('{"event":"validation","epoch":1}\n'
                            '{"event":"validation","epoch":2}', encoding="utf-8")
            self.assertEqual(logged_validation_epochs(path), {1})

    def test_test_failures_and_null_pq_are_retained_in_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = dict(images=2, instance_iou_macro=.2, pq_global=None, union_iou_macro=.3,
                          tp=1, fp=1, fn=2, failures=1, complete=False,
                          checkpoint_progress=dict(epoch=2, successful_updates=6, amp_skips=0))
            write_summary(root, {"B": {"tests": {"best": report, "last": report}}}, final=False)
            self.assertEqual(read_json(root / "test_results.json")["rows"][0]["failures"], 1)
            self.assertIn("unavailable", (root / "RESULTS.md").read_text(encoding="utf-8"))

    def test_subprocess_failure_is_recorded_and_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = SimpleNamespace(output=str(root), poll_seconds=.01, warning_hours=24)
            supervisor = Supervisor(args, dict(frozen_files={}, file_stats={}))
            supervisor.phase = "testing"
            with patch("sam_path_v1.full_run.check_disk", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "exit=7"):
                    supervisor.command([sys.executable, "-c", "raise SystemExit(7)"], "failed.stdout.log")
            status = read_json(root / "status.json")
            self.assertEqual(status["exit_code"], 7)
            self.assertIsNone(status["child_pid"])
            self.assertEqual(len((root / "events.jsonl").read_text().splitlines()), 2)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
