"""Cross-module contracts and frozen-backend regression, CPU only."""
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from PIL import Image
from .types import CropSpec, PromptSpec, PromptPlan, ProposalRecord
from .io import save_case, load_case, read_json, write_json, rgb_hash
from .backend import assign_instances


def fixture():
    rgb = np.zeros((67, 95, 3), np.uint8)
    rgb[20:27, 8:88] = [130, 30, 10]
    c = CropSpec("c", (0, 0, 95, 67))
    p = PromptSpec("p", "c", ((30., 23.),), (), (5., 15., 90., 35.))
    logits = np.full(rgb.shape[:2], -4., np.float32)
    logits[20:27, 8:88] = 4.
    plan = PromptPlan([c], [p], np.zeros(rgb.shape[:2], np.float32))
    record = ProposalRecord(p, c, logits.copy(), logits.copy(), .8)
    return rgb, plan, [record]


class IntegrationTests(unittest.TestCase):
    def test_coarse_prediction_and_cache_export_skip_identity_and_additional_sam(self):
        from .cli import infer, export_coarse_cache
        rgb, plan, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(rgb).save(root/"rgb.png")
            args = SimpleNamespace(image=root/"rgb.png", output=root/"prediction", selection="iterative",
                                   checkpoint=root/"fake.pt", sam_repo=root, device="cpu", max_crops=1,
                                   max_prompts=1, refiner=None, coarse_only=True)
            with patch("sam_once.sam_adapter.NativeSamAdapter") as adapter, patch("sam_once.planner.plan_prompts", return_value=plan), patch("sam_once.backend.assign_instances", side_effect=AssertionError("Identity stage forbidden")), patch("torch.cuda.is_available", return_value=False):
                adapter.return_value.predict.return_value = (records, {"injected_test_model": True})
                infer(args)
                adapter.return_value.predict.assert_called_once()
            with patch("sam_once.sam_adapter.NativeSamAdapter", side_effect=AssertionError("No new SAM")), patch("sam_once.backend.assign_instances", side_effect=AssertionError("No identity stage")):
                export_coarse_cache(SimpleNamespace(cache=root/"prediction/sam_cache", output=root/"replay"))
            for folder in ("prediction", "replay"):
                self.assertFalse((root/folder/"instance_ids.png").exists())
                actual = np.asarray(Image.open(root/folder/"coarse_foreground.png")) > 0
                np.testing.assert_array_equal(actual, records[0].initial_logits > 0)
            self.assertEqual(read_json(root/"replay/inference.json")["sam_calls_this_command"], 0)

    def test_coarse_refiner_conflict_rejected_before_sam(self):
        from .cli import infer
        with patch("sam_once.sam_adapter.NativeSamAdapter", side_effect=AssertionError("Must not load")):
            with self.assertRaisesRegex(ValueError, "omit --refiner"):
                infer(SimpleNamespace(coarse_only=True, refiner=Path("unused.pt")))

    def test_cached_refinement_failure_has_zero_sam_receipt(self):
        from .cli import refine_cache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(cache=root / "missing_cache", refiner=root / "missing.pt",
                                   output=root / "failure", device="cpu")
            with self.assertRaises(FileNotFoundError):
                refine_cache(args)
            audit = read_json(root / "failure/inference.json")
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(audit["sam_calls_this_command"], 0)
            self.assertIn("FileNotFoundError", audit["error"])

    def test_cli_pipeline_exports_native_case_and_failure_audit(self):
        from .cli import infer
        from .sam_adapter import SamInferenceError
        rgb, plan, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(rgb).save(root / "rgb.png")
            args = SimpleNamespace(image=root / "rgb.png", output=root / "prediction", selection="iterative",
                                   checkpoint=root / "fake.pt", sam_repo=root, device="cpu", max_crops=1, max_prompts=1,
                                   refiner=None)
            class FakeAdapter:
                def __init__(self, **kw): pass
                def predict(self, *_):
                    return records, {"injected_test_model": True, "calls": {"image_encoder": 1, "prompt_encoder": 1, "mask_decoder": 1}}
            with patch("sam_once.planner.plan_prompts", return_value=plan), patch("sam_once.sam_adapter.NativeSamAdapter", FakeAdapter), patch("torch.cuda.is_available", return_value=False), patch("mbest_backend.predict", return_value=(np.zeros(rgb.shape[:2], np.uint16), {"status": "empty"})):
                infer(args)
            cached_rgb, _, _, audit = load_case(root / "prediction/sam_cache")
            np.testing.assert_array_equal(cached_rgb, rgb)
            self.assertTrue(audit["sam"]["injected_test_model"])
            for name in ("instance_ids.png", "visible_foreground.png", "unknown_visible.png", "instances_overlay.png"):
                self.assertTrue((root / "prediction" / name).is_file())
            class FailedAdapter(FakeAdapter):
                def predict(self, *_):
                    raise SamInferenceError("witness", {"calls": {"image_encoder": 1}, "retries": 0})
            args.output = root / "failure"
            with patch("sam_once.planner.plan_prompts", return_value=plan), patch("sam_once.sam_adapter.NativeSamAdapter", FailedAdapter):
                with self.assertRaises(SamInferenceError):
                    infer(args)
            failure = read_json(root / "failure/inference.json")
            self.assertEqual(failure["sam_failure"]["retries"], 0)
            self.assertEqual(failure["status"], "failed")

    def test_cache_roundtrip_and_mutation_detection(self):
        rgb, plan, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "case"
            save_case(out, rgb, plan, records, {"test": True})
            rr, pp, pro, audit = load_case(out)
            np.testing.assert_array_equal(rr, rgb)
            np.testing.assert_array_equal(pro[0].initial_logits, records[0].initial_logits)
            self.assertEqual(pp.metadata(), plan.metadata())
            with self.assertRaises(FileExistsError):
                save_case(out, rgb, plan, records, {})
            np.save(out / "ridge.npy", np.ones(rgb.shape[:2], np.float32))
            with self.assertRaises(ValueError):
                load_case(out)

    def test_cache_rejects_coordinate_rebinding_and_missing_hash(self):
        rgb, plan, records = fixture()
        records[0].crop = CropSpec("c", (1, 0, 96, 67))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_case(Path(tmp) / "bad", rgb, plan, records, {})
            records[0].crop = plan.crops[0]
            out = Path(tmp) / "good"
            save_case(out, rgb, plan, records, {})
            meta = read_json(out / "case.json")
            del meta["files_sha256"]["ridge.npy"]
            write_json(out / "case.json", meta)
            with self.assertRaises(ValueError):
                load_case(out)

    def test_frozen_backend_identical_evidence(self):
        from cable_evidence_fusion import Proposal, predict_methods
        from cable_identity_selector import select_identity
        from refine_observed_ids import fill_observed_ids
        rgb, plan, records = fixture()
        c = plan.crops[0]
        p2 = PromptSpec("different_sort_order", "c", ((40., 40.),))
        a = np.full(rgb.shape[:2], -4., np.float32)
        a[8:60, 37:44] = 4.
        records.append(ProposalRecord(p2, c, a.copy(), a.copy(), .8))
        visible = np.logical_or.reduce([r.current_logits > 0 for r in records])
        base = np.zeros(visible.shape, np.uint16)
        base[20:27, 8:35] = 1
        ids, vis, unknown, audit = assign_instances(rgb, records, 1, lambda *_: (base.copy(), {"status": "ok"}))
        props = [Proposal(r.crop.xyxy, r.current_logits > 0, "hq_point", r.score,
                 hashlib.sha256(str(r.crop.xyxy).encode() + np.packbits(r.current_logits > 0).tobytes()).hexdigest()) for r in records]
        filled, _ = fill_observed_ids(base, visible)
        ev = {m: arr for m,arr,_ in predict_methods(rgb, visible, filled, props, ["fill_geodesic", "proposal_graph"])}
        expected, _ = select_identity(rgb, visible, filled, ev["fill_geodesic"], ev["proposal_graph"], 1, "coverage_color")
        np.testing.assert_array_equal(ids, expected)
        self.assertFalse(np.any((ids > 0) & ~vis))
        np.testing.assert_array_equal(unknown, visible & (ids == 0))

    def test_empty_and_failed_geometry_included(self):
        rgb, plan, records = fixture()
        def failed(*_):
            raise IndexError("witness")
        ids, visible, unknown, audit = assign_instances(rgb, records, 1, failed)
        self.assertEqual(audit["status"], "geometry_failed")
        self.assertGreater(ids.sum(), 0)
        ids, visible, unknown, audit = assign_instances(rgb, [], 1, lambda *_: (np.zeros(rgb.shape[:2], np.uint16), {"status": "empty"}))
        self.assertEqual(audit["status"], "empty")
        self.assertFalse(visible.any())

    def test_evaluation_keeps_missing_predictions(self):
        from .evaluate import evaluate
        rgb, plan, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            Image.fromarray(rgb).save(base / "rgb.png")
            Image.fromarray((records[0].current_logits > 0).astype(np.uint8)*255).save(base / "gt.png")
            rows = [{"case_id": ident, "image_path": "rgb.png", "gt_masks": ["gt.png"], "dataset": "testdomain", "split": "check"} for ident in ("ok", "missing")]
            write_json(base / "manifest.json", {"records": rows})
            pred = base / "predictions/ok"
            pred.mkdir(parents=True)
            Image.fromarray((records[0].current_logits > 0).astype(np.uint16)).save(pred / "instance_ids.png")
            Image.open(base / "gt.png").save(pred / "visible_foreground.png")
            Image.fromarray(np.zeros(rgb.shape[:2], np.uint8)).save(pred / "unknown_visible.png")
            write_json(pred / "inference.json", {"status": "ok", "rgb_sha256": rgb_hash(rgb)})
            report = evaluate(base / "manifest.json", base / "predictions", base / "report")
            summary = report["summaries"]["testdomain"]
            self.assertEqual(summary["n"], 2)
            self.assertEqual(summary["failures"], 1)
            self.assertEqual(summary["instance_iou"], .5)
            self.assertAlmostEqual(summary["pq"], 2/3)

    def test_evaluation_native_npy_and_duplicate_rejection(self):
        from .evaluate import evaluate, load_gt_mask
        rgb, _, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(rgb).save(root / "rgb.png")
            gt = records[0].current_logits > 0
            np.save(root / "gt.npy", gt, allow_pickle=False)
            np.testing.assert_array_equal(load_gt_mask(root / "gt.npy", rgb.shape[:2]), gt)
            row = {"case_id": "npy", "image_path": "rgb.png", "gt_masks": ["gt.npy"],
                   "dataset": "testdomain", "split": "check", "rgb_sha256": rgb_hash(rgb)}
            write_json(root / "manifest.json", {"records": [row]})
            report = evaluate(root / "manifest.json", root / "missing", root / "report")
            self.assertEqual(report["summaries"]["testdomain"]["fn"], 1)
            write_json(root / "manifest.json", {"records": [row, row]})
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                evaluate(root / "manifest.json", root / "missing", root / "duplicate_report")


if __name__ == "__main__":
    unittest.main()
