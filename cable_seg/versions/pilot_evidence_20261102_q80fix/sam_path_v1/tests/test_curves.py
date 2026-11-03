"""Loss logging and plotting contracts, using CPU fixtures rather than accuracy claims."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sam_path_v1.data import NativeInstanceDataset
from sam_path_v1.evaluate import evaluate_model, loss_summary, scalar_losses
from sam_path_v1.losses import SetCriterion
from sam_path_v1.model import ModelConfig, SamPathModel
from sam_path_v1.plot import collect, ema, events, finite_mapping, render
from sam_path_v1.tests.test_engine import fixture_manifest
from sam_path_v1.train import TrainConfig, train


class LossSummaryTests(unittest.TestCase):
    def test_weighting_and_nonfinite_rejection(self):
        raw, weighted = scalar_losses({"mask": torch.tensor(3.), "total": torch.tensor(6.)}, {"mask": 2.})
        self.assertEqual(weighted, {"mask": 6.})
        report = loss_summary([{"loss": raw}, {"loss": {"mask": 1., "total": 2.}}], {"mask": 2.})
        self.assertEqual(report["mean"]["mask"], 2.)
        self.assertEqual(report["weighted_mean"]["mask"], 4.)
        report = loss_summary([{"loss": raw}, {"loss": None}], {"mask": 2.})
        self.assertFalse(report["complete"])
        self.assertIsNone(report["mean"])
        self.assertEqual(report["observed_mean"]["mask"], 3.)
        with self.assertRaises(FloatingPointError):
            scalar_losses({"mask": float("nan"), "total": 3.}, {"mask": 1.})
        with self.assertRaises(ValueError):
            scalar_losses({"mask": 3., "total": 1.}, {"mask": 1.})

    def test_validation_same_forward_and_loss_coverage(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture_manifest(root)
            protocol = json.loads(manifest.read_text(encoding="utf-8"))
            for row in protocol["records"]:
                row["source_condition_group"] = "fixture_condition_"+row["id"]
            manifest.write_text(json.dumps(protocol), encoding="utf-8")
            dataset = NativeInstanceDataset(manifest, "validation", 16)
            model = SamPathModel(ModelConfig(backend="toy", variant="C", canvas_size=16,
                                            num_queries=2, tracker_steps=2, encoder_amp="none"))
            calls = []
            hook = model.register_forward_hook(lambda *args: calls.append(1))
            report = evaluate_model(model, dataset, criterion=SetCriterion())
            hook.remove()
            self.assertEqual(len(calls), len(dataset))
            self.assertTrue(report["losses"]["complete"])
            self.assertEqual(set(report["by_condition"]), {"fixture_condition_3", "fixture_condition_4"})
            self.assertEqual(report["by_condition"]["fixture_condition_3"]["losses"]["images"], 1)
            for key, mean in report["losses"]["mean"].items():
                self.assertAlmostEqual(mean, np.mean([row["loss"][key] for row in report["per_image"]]))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            (root/"mask3.npz").unlink()
            broken = evaluate_model(model, dataset, criterion=SetCriterion())
            self.assertFalse(broken["losses"]["complete"])
            self.assertIsNone(broken["losses"]["mean"])
            self.assertEqual(broken["losses"]["failed_images"], 1)


class CurveTests(unittest.TestCase):
    def test_complete_lines_only_and_causal_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"logs.jsonl"
            path.write_bytes(b'{"value": 1}\n{"partial":')
            self.assertEqual(list(events(path)), [{"value": 1}])
            path.write_bytes(b'{"value": 1}\nnot-json\n')
            with self.assertRaisesRegex(ValueError, "Malformed complete"):
                list(events(path))
        np.testing.assert_array_equal(ema([1., 2., 3.], 1), [1., 2., 3.])
        np.testing.assert_allclose(ema([1., 3., 2.], 3), [1., 2., 2.])
        with self.assertRaises(ValueError):
            finite_mapping({"total": float("nan")})

    def test_logs_csv_resume_and_render(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture_manifest(root)
            base = dict(manifest=str(manifest), model=dict(backend="toy", variant="B", canvas_size=16,
                        num_queries=2, tracker_steps=2, encoder_amp="none"), seed=9, epochs=1,
                        batch_size=2, save_every=1, device="cpu", amp="none", cpu_threads=1)
            train(TrainConfig(**base, max_batches=1), root/"part")
            train(TrainConfig(**base), root/"resumed", root/"part/last.pt")
            folder = root/"csv"; folder.mkdir()
            record = collect(root/"resumed", folder)
            self.assertEqual(record["observations"], 2)
            self.assertEqual(record["amp_skips"], 0)
            self.assertEqual(set(record["spec"]["inactive"]), {"descriptor", "geometry"})
            self.assertEqual(len(record["val_epochs"]), 1)
            with (folder/"steps.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual([int(row["samples"]) for row in csv_rows], [2, 1])
            epoch = record["train_epochs"][1]
            for key in epoch["loss"]:
                expected = (float(csv_rows[0]["raw_"+key])*2+float(csv_rows[1]["raw_"+key]))/3
                self.assertAlmostEqual(epoch["loss"][key], expected)
            try:
                import matplotlib
            except ImportError:
                self.skipTest("optional matplotlib unavailable; CSV checks passed")
            receipt = render({"B": root/"resumed"}, root/"figures", 2)
            self.assertEqual(receipt["runs"]["B"]["observations"], 2)
            self.assertEqual(len(list((root/"figures").glob("*.png"))), 9)
            self.assertEqual(len(list((root/"figures").glob("*.pdf"))), 9)
            with self.assertRaises(FileExistsError):
                render({"B": root/"resumed"}, root/"figures")


if __name__ == "__main__":
    unittest.main()
