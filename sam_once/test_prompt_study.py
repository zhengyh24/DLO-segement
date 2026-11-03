"""Experiment integrity tests: phase separation, audit tampering and failure accounting."""
import ast
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from . import prompt_study as study
from .io import write_json, sha256
from .oracle_diagnostic import oracle_plan
from .prompt_ablation import summarize_plan


class StudyTests(unittest.TestCase):
    def test_inference_phase_does_not_open_supervision(self):
        source = inspect.getsource(study.run)
        tree = ast.parse(source)
        literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertNotIn("evaluation_inputs.json", literals)
        self.assertNotIn("masks", literals)
        self.assertNotIn(".oracle_diagnostic", source)
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "predict" for n in ast.walk(tree)), 1)

    def test_missing_run_kept_and_tampered_queue_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gt = np.ones((1, 12, 12), bool)
            np.savez(out/"gt.npz", masks=gt)
            Image.fromarray(np.ones((12, 12), np.uint16)).save(out/"old.png")
            Image.fromarray(np.full((12, 12), 255, np.uint8)).save(out/"visible.png")
            cases, queue = [], []
            for domain in ("c135", "wire300"):
                cases.append({"id": domain, "dataset": domain, "masks": str(out/"gt.npz"),
                              "gt_sha256": sha256(out/"gt.npz"), "old_ids": str(out/"old.png"),
                              "old_ids_sha256": sha256(out/"old.png"), "old_visible": str(out/"visible.png"),
                              "old_visible_sha256": sha256(out/"visible.png")})
                for profile in ("v1_control", "candidate"):
                    queue.append({"case": domain, "domain": domain, "profile": profile, "kind": "automatic", "prompt_summary": {}})
            write_json(out/"evaluation_inputs.json", cases)
            write_json(out/"inference_queue.json", queue)
            write_json(out/"run_status.json", {"status": "stopped_on_failure_no_retry"})
            write_json(out/"protocol.json", {"source_sha256": {}, "queue_sha256": sha256(out/"inference_queue.json"),
                       "evaluation_inputs_sha256": sha256(out/"evaluation_inputs.json"), "validation_scope": "test fixture",
                       "profiles": [{"id": "v1_control", "planner": "v1"}, {"id": "candidate", "planner": "v2"}]})
            with patch("sam_once.cli.sources", return_value={}):
                study.evaluate(SimpleNamespace(output=out))
                result = study.read_json(out/"results.json")
                missing = [r for r in result["rows"] if r["status"] == "unattempted"]
                self.assertEqual(len(missing), 4)
                self.assertTrue(all(r["fn_instances"] == 1 and r["pq"] == 0 for r in missing))
                write_json(out/"inference_queue.json", queue[:-1])
                with self.assertRaisesRegex(ValueError, "queue changed"):
                    study.evaluate(SimpleNamespace(output=out))

    def test_instance_identity_metrics_do_not_reward_foreground_merge(self):
        gt = np.zeros((2, 20, 30), bool)
        gt[0, 2:6, :] = True
        gt[1, 12:16, :] = True
        visible = gt.any(0)
        merged = visible.astype(np.uint16)
        result = study.metrics(merged, visible, gt)
        self.assertEqual(result["foreground_iou"], 1)
        self.assertEqual(result["pq"], 0)  # Strict IoU > .5, not >=.
        self.assertEqual(result["merge_proxy"], 1)
        self.assertEqual(result["fn_instances"], 2)

    def test_oracle_negative_abstention_reported(self):
        rgb = np.zeros((40, 80, 3), np.uint8)
        gt = np.zeros((1, 40, 80), bool)
        gt[0, 15:24, 8:72] = True
        plan = oracle_plan(rgb, gt, 3, 1, "other_gt", "none")
        summary = summarize_plan(plan)
        self.assertEqual(summary["group_count"], 1)
        self.assertEqual(summary["negative_count_shortfall_groups"], 1)


if __name__ == "__main__":
    unittest.main()
