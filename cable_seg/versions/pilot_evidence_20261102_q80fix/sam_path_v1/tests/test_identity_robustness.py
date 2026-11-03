"""Behavioral tests for optional offline identity/stress diagnostics."""
import io
import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from sam_path_v1.data import NativeInstanceDataset
from sam_path_v1.evaluate import evaluate_model
from sam_path_v1.external_data import restricted_npy, rgb_only_diagnostics
from sam_path_v1.identity import identity_events, summarize_identity, visible_identity, difficulty_metadata, difficulty_strata, attention_components
from sam_path_v1.model import ModelConfig, SamPathModel
from sam_path_v1.robustness import apply_occlusion, build_occlusion_plan, fixed_indices
from sam_path_v1.tests.test_engine import fixture_manifest


class IdentityTests(unittest.TestCase):
    def test_fixed_difficulty_bins_and_unavailable_failure_kept(self):
        gt = np.zeros((2, 32, 32), bool)
        gt[0, 2:6, 2:6] = True; gt[0, 11:15, 2:6] = True
        gt[1, 24:28, 24:28] = True
        rgb = np.zeros((32, 32, 3), np.float32)
        rgb[gt.any(0), 0] = 1
        metadata = difficulty_metadata(rgb, gt)
        self.assertEqual(metadata["gt_visible_components"], 3)
        self.assertEqual(metadata["bins"]["visible_fragmentation"], "1-2")
        self.assertEqual(metadata["bins"]["appearance_proxy"], "similar")
        self.assertEqual(metadata["bins"]["proximity_proxy"], "none")
        base = dict(instance_iou=.5, union_iou=.5, tp=1, fp=0, fn=1, matched_iou=.5,
                    empty_prediction=False, failed=False, gt_available=True)
        rows = [{**base, "difficulty": metadata}, {**base, "difficulty": None, "failed": True}]
        strata = difficulty_strata(rows)
        for groups in strata.values():
            self.assertEqual(sum(g["images"] for g in groups), 2)
            self.assertEqual(sum(g["failures"] for g in groups), 1)
            self.assertIn("unavailable", [g["bin"] for g in groups])

    def test_attention_component_mass_has_expected_offline_assignment(self):
        attention = torch.zeros(1, 2, 2, 4, 4)
        attention[:, :, 0, 0, 0] = 1; attention[:, :, 1, 3, 3] = 1
        positions = torch.tensor([[[[.125, .125], [.875, .875]], [[.125, .125], [.875, .875]]]])
        gt = np.zeros((2, 16, 16), bool)
        gt[0, :4, :4] = True; gt[1, 12:, 12:] = True
        outputs = dict(aux=dict(path_attention=attention, path_positions=positions))
        detail, arrays = attention_components(outputs, gt, prediction=dict(masks=gt, queries=[0, 1]))
        self.assertEqual(detail["dominant_gt_instance_by_step"], [[0, 1], [0, 1]])
        self.assertEqual(detail["dominant_instance_changes"], [1, 1])
        np.testing.assert_allclose(detail["gt_visible_mass_by_step"], 1)
        self.assertEqual(arrays["attention"].dtype, np.float16)
        self.assertEqual(detail["matched_prediction_gt"][0]["attention_mass_by_step"], [1., 0.])

    def test_fixed_sample_artifacts_include_baseline_without_attention(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = NativeInstanceDataset(fixture_manifest(root), "validation", 16)
            selections = []
            for variant in ("A", "C"):
                model = SamPathModel(ModelConfig(backend="toy", variant=variant, canvas_size=16,
                    num_queries=2, tracker_steps=2, encoder_amp="none"))
                out = root/variant
                report = evaluate_model(model, ds, identity_diagnostics=True,
                    diagnostic_output=out, attention_samples=1)
                self.assertEqual(report["diagnostic_failures"], 0)
                self.assertEqual(report["diagnostic_samples"]["saved_images"], 1)
                selections.append(report["diagnostic_samples"]["selected_ids"])
                row = next(r for r in report["per_image"] if r.get("diagnostic_artifacts"))
                artifacts = row["diagnostic_artifacts"]
                self.assertTrue((out/artifacts["directory"]/artifacts["panel"]).is_file())
                for key, digest in artifacts["sha256"].items():
                    self.assertEqual(hashlib.sha256((out/artifacts["directory"]/artifacts[key]).read_bytes()).hexdigest(), digest)
                self.assertEqual("attention" in artifacts, variant == "C")
            self.assertEqual(selections[0], selections[1])

    def masks(self):
        gt = np.zeros((2, 64, 64), bool)
        gt[0, 10:14, 5:59] = True
        gt[1, 30:34, 5:59] = True
        return gt

    def test_merge_split_and_whole_visible(self):
        gt = self.masks()
        perfect = visible_identity(gt[::-1], gt)
        self.assertEqual(perfect["whole_visible_recovered"], 2)
        self.assertEqual(perfect["sensitivity"]["0.1"]["merged_predictions"], 0)
        merged = visible_identity(gt.any(0)[None], gt)
        self.assertEqual(merged["whole_visible_recovered"], 0)
        self.assertEqual(merged["sensitivity"]["0.1"]["merged_predictions"], 1)
        split = np.stack((gt[0], gt[0], gt[1]))
        split[0, :, 32:] = False
        split[1, :, :32] = False
        scored = visible_identity(split, gt)
        self.assertEqual(scored["sensitivity"]["0.1"]["fragmented_gt"], 1)
        self.assertEqual(scored["whole_visible_recovered"], 1)

    def test_uncovered_positive_and_negative_are_failures(self):
        gt = self.masks()
        regions = [{"pixels_xy": [[10, 11], [11, 11]]}, {"pixels_xy": [[50, 11], [51, 11]]}]
        events = [dict(id="p", same_instance=True, regions=regions),
                  dict(id="n", same_instance=False, regions=[regions[0], {"pixels_xy": [[10, 31], [11, 31]]}])]
        good = identity_events(gt, events)
        self.assertEqual(good["balanced_accuracy"], 1.)
        empty = identity_events(np.zeros((0, 64, 64), bool), events)
        self.assertEqual(empty["balanced_accuracy"], 0.)
        self.assertEqual(empty["covered_events"], 0)
        merged = identity_events(gt.any(0)[None], events)
        self.assertEqual(merged["balanced_accuracy"], .5)
        self.assertEqual(identity_events(gt, events, failed=True)["balanced_accuracy"], 0.)

    def test_masked_regions_and_overlap_rejected(self):
        gt = self.masks()
        with self.assertRaises(ValueError):
            visible_identity(np.stack((gt[0], gt[0])), gt)
        with self.assertRaises(ValueError):
            identity_events(gt, [dict(id="x", same_instance=True, regions=[{"pixels_xy": [[100, 1]]}]*2)])
        summary = summarize_identity([dict(identity=visible_identity(gt, gt), identity_events=None)])
        self.assertTrue(summary["complete"])

    def test_diagnostics_do_not_change_main_metrics_or_call_count(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            ds = NativeInstanceDataset(fixture_manifest(Path(tmp)), "validation", 16)
            model = SamPathModel(ModelConfig(backend="toy", variant="C", canvas_size=16,
                                            num_queries=2, tracker_steps=2, encoder_amp="none"))
            plain = evaluate_model(model, ds)
            diagnostic = evaluate_model(model, ds, identity_diagnostics=True, attention_diagnostics=True)
            for key in ("instance_iou_macro", "union_iou_macro", "pq_global", "tp", "fp", "fn"):
                self.assertEqual(plain[key], diagnostic[key])
            self.assertEqual(diagnostic["diagnostic_failures"], 0)
            self.assertTrue(diagnostic["identity_diagnostics"]["complete"])
            self.assertTrue(all(r["calls"] == {"encoder": 1, "decoder": 1} for r in diagnostic["per_image"]))


class OcclusionTests(unittest.TestCase):
    def test_stable_subset_independent_of_manifest_order(self):
        rows = [{"id": str(i)} for i in range(100)]
        a = [rows[i]["id"] for i in fixed_indices(rows, 30)]
        reverse = list(reversed(rows))
        b = [reverse[i]["id"] for i in fixed_indices(reverse, 30)]
        self.assertEqual(a, b)

    def test_all_instance_masks_occluded_original_unchanged(self):
        masks = np.zeros((2, 160, 160), bool)
        masks[0, 70:74, 10:150] = True
        masks[1, 80:84, 10:150] = True
        plan = build_occlusion_plan(masks, "fixed", [160, 160])
        self.assertTrue(plan["applicable"])
        self.assertEqual(plan, build_occlusion_plan(masks, "fixed", [160, 160]))
        sample = dict(image=torch.ones(3, 160, 160), masks=torch.from_numpy(masks).float(),
                      valid=torch.ones(1, 160, 160, dtype=torch.bool), id="fixed",
                      meta=dict(original_size=[160, 160]))
        stress = apply_occlusion(sample, plan, 8)
        cover = stress["image"][0] != sample["image"][0]
        self.assertTrue(cover.any())
        self.assertFalse(stress["masks"][:, cover].any())
        self.assertTrue(torch.equal(sample["masks"], torch.from_numpy(masks).float()))
        self.assertTrue(torch.equal(stress["valid"], sample["valid"]))
        self.assertEqual(identity_events(stress["masks"].numpy(), plan["events"])["positive"]["accuracy"], 1.)


class SafeDataTests(unittest.TestCase):
    def test_numeric_numpy_dictionary_is_admitted(self):
        buffer = io.BytesIO()
        np.save(buffer, dict(img=np.zeros((8, 8, 3), np.uint8), pixels={0: [(1, 2)], 1: [(3, 4)]}))
        decoded = restricted_npy(buffer.getvalue())
        self.assertEqual(decoded["img"].shape, (8, 8, 3))

    def test_arbitrary_global_is_rejected(self):
        class Malicious:
            def __reduce__(self):
                return eval, ("1+1",)
        buffer = io.BytesIO()
        np.save(buffer, dict(payload=Malicious()))
        with self.assertRaisesRegex(pickle.UnpicklingError, "Forbidden"):
            restricted_npy(buffer.getvalue())

    def test_external_rgb_only_relocated_fixture_fixed_visuals(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"relocated_audit"
            root.mkdir()
            (root/"rgb").mkdir(); (root/"traces").mkdir()
            rgb = np.zeros((12, 15, 3), np.uint8)
            rgb[3:5, 2:12] = 200
            records = []
            for i in range(2):
                Image.fromarray(rgb).save(root/"rgb"/f"{i}.png")
                np.savez_compressed(root/"traces"/f"{i}.npz", pixels_yx=np.array([[3, 3], [3, 8]]))
                records.append(dict(id=str(i), status="rgb_and_numeric_trace_verified",
                    rgb_path=f"rgb/{i}.png", trace_path=f"traces/{i}.npz",
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
            (root/"audit.json").write_text(json.dumps(dict(commit="software_fixture", records=records)), encoding="utf-8")
            model = SamPathModel(ModelConfig(backend="toy", variant="C", canvas_size=16,
                                            num_queries=2, tracker_steps=2, encoder_amp="none"))
            report = rgb_only_diagnostics(model, root, Path(tmp)/"predictions", sample_visuals=1)
            self.assertEqual(report["images"], 2)
            self.assertEqual(report["failures"], 0)
            self.assertEqual(report["saved_prediction_count"], 1)
            self.assertEqual(len(report["exact_rgb_groups"]), 1)
            self.assertEqual(len(list((Path(tmp)/"predictions"/"predictions").glob("*.npz"))), 1)
            self.assertNotIn("instance_iou", report)


if __name__ == "__main__":
    unittest.main()
