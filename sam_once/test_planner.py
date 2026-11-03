"""CPU checks for native planning and conservative geometric hypotheses.

Run: python tools/padlos_runtime_runner.py sam_once/test_planner.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from sam_once.planner import (PlannerConfig, _fragment_candidates, _safe_negatives,
                              _select_candidates, native_crops, plan_prompts)


def scan_fixture(rgb, config):
    ridge = np.zeros(rgb.shape[:2], np.float32)
    bright = rgb[:, :, 0] > 150
    ridge[bright] = 1
    width = np.full(ridge.shape, 2., np.float32)
    sources = (ridge > 0).astype(np.int16)
    return ridge, width, sources, {"source_names": ["fixture:ridge"], "failures": []}


class PlannerTests(unittest.TestCase):
    def test_full_native_odd_dimension_coverage(self):
        config = PlannerConfig(canvas_size=64, overlap=17)
        crops = native_crops(149, 117, config)
        coverage = np.zeros((117, 149), bool)
        for crop in crops:
            x0, y0, x1, y1 = crop.xyxy
            coverage[y0:y1, x0:x1] = True
            self.assertLessEqual(crop.width, 64)
            self.assertLessEqual(crop.height, 64)
        self.assertTrue(coverage.all())
        self.assertEqual(crops, native_crops(149, 117, config))
        self.assertEqual(native_crops(17, 9, config)[0].xyxy, (0, 0, 17, 9))

    def test_real_geomprompt_thin_line_determinism(self):
        rgb = np.full((81, 113, 3), 25, np.uint8)
        rgb[39:42, 12:101] = 220
        config = PlannerConfig(channels=("gray",), scales=(1., 4., 9., 25.))
        first = plan_prompts(rgb, config)
        second = plan_prompts(rgb, config)
        self.assertGreater(len(first.prompts), 0)
        self.assertEqual(first.metadata(), second.metadata())
        np.testing.assert_array_equal(first.ridge, second.ridge)
        self.assertEqual(first.ridge.dtype, np.float32)
        self.assertGreater(float(first.ridge.max()), 0)
        self.assertLessEqual(float(first.ridge.max()), 1)
        for p in first.prompts:
            self.assertLessEqual(len(p.positive), 3)
            self.assertLessEqual(len(p.negative), 2)
            self.assertLessEqual(p.metadata["sparse_tokens"], 7)
            self.assertFalse(p.metadata["ground_truth_used"])

    def test_parallel_are_independent_hypotheses_and_safe_negatives(self):
        rgb = np.full((81, 113, 3), 20, np.uint8)
        rgb[25, 10:104] = 230
        rgb[45, 10:104] = 230
        with patch("sam_once.planner._scan_tile", side_effect=scan_fixture):
            plan = plan_prompts(rgb)
        self.assertEqual(len(plan.prompts), 2)
        rows = []
        for p in plan.prompts:
            rows.append({y for _, y in p.positive})
            self.assertEqual(len(rows[-1]), 1)
            for x, y in p.negative:
                self.assertEqual(int(rgb[int(y), int(x), 0]), 20)
                self.assertNotIn(y, (25., 45.))
        self.assertNotEqual(rows[0], rows[1])

    def test_crossing_splits_and_abstains_from_negatives(self):
        rgb = np.full((91, 91, 3), 20, np.uint8)
        rgb[45, 10:81] = 230
        rgb[10:81, 45] = 230
        with patch("sam_once.planner._scan_tile", side_effect=scan_fixture):
            plan = plan_prompts(rgb)
        self.assertEqual(len(plan.prompts), 4)
        for p in plan.prompts:
            self.assertTrue(p.metadata["hard_zone"])
            self.assertEqual(len(p.positive), 1)
            self.assertEqual(p.negative, ())
            self.assertNotEqual(p.positive[0], (45., 45.))

    def test_empty_image_and_zero_budget_still_scan_all_crops(self):
        rgb = np.zeros((79, 87, 3), np.uint8)
        config = PlannerConfig(canvas_size=40, overlap=8, max_prompts_per_crop=0)
        with patch("sam_once.planner._scan_tile", side_effect=scan_fixture) as scan:
            plan = plan_prompts(rgb, config)
        self.assertEqual(scan.call_count, len(plan.crops))
        self.assertEqual(plan.audit["empty_crop_count"], len(plan.crops))
        self.assertEqual(plan.audit["status"], "empty_plan")
        self.assertEqual(plan.prompts, [])
        self.assertFalse(plan.ridge.any())

    def test_cap_coordinates_and_odd_edge_crop(self):
        rgb = np.full((57, 71, 3), 20, np.uint8)
        rgb[10:50:8, 4:67] = 220
        config = PlannerConfig(canvas_size=40, overlap=12, max_prompts_per_crop=2)
        with patch("sam_once.planner._scan_tile", side_effect=scan_fixture):
            plan = plan_prompts(rgb, config)
        plan.validate(rgb, 40)
        for c in plan.crops:
            self.assertLessEqual(sum(p.crop_id == c.crop_id for p in plan.prompts), 2)
        self.assertTrue(any(p.positive for p in plan.prompts))

    def test_false_ridge_without_rgb_evidence_is_not_a_positive(self):
        rgb = np.full((65, 81, 3), 80, np.uint8)
        ridge = np.zeros(rgb.shape[:2], np.float32)
        ridge[32, 10:71] = 1
        candidates, audit = _fragment_candidates(rgb, ridge, np.ones_like(ridge) * 2,
                                                 (ridge > 0).astype(np.int16), ["false"], PlannerConfig())
        self.assertEqual(candidates, [])
        self.assertGreater(audit["low_contrast"], 0)

    def test_negative_abstention_on_texture_and_similar_color(self):
        ridge = np.zeros((65, 81), np.float32)
        ridge[32, 10:71] = 1
        candidate = {"hard_zone": False, "width": 2., "center_color": np.array([80.] * 3),
                     "box": (0., 0., 81., 65.)}
        rgb = np.full((65, 81, 3), 80, np.uint8)
        negative, _ = _safe_negatives(rgb, ridge, candidate, PlannerConfig())
        self.assertEqual(negative, ())
        checker = (np.indices(ridge.shape).sum(axis=0) % 2 * 230).astype(np.uint8)
        negative, _ = _safe_negatives(np.repeat(checker[:, :, None], 3, axis=2), ridge,
                                     candidate, PlannerConfig())
        self.assertEqual(negative, ())

    def test_candidate_ordering_and_iterative_diversity(self):
        def item(y, score, x=10):
            return {"points": ((float(x), float(y)),), "coords": np.array([[y, z] for z in range(10, 30)]),
                    "score": score, "angle": 0., "length": 20}
        pool = [item(10, 10.), item(14, 9.), item(60, 8.)]
        config = PlannerConfig(max_prompts_per_crop=2)
        iterative = _select_candidates(pool, config)
        reverse = _select_candidates(list(reversed(pool)), config)
        self.assertEqual([c["points"] for c in iterative], [c["points"] for c in reverse])
        self.assertEqual([c["points"][0][1] for c in iterative], [10., 60.])
        fixed = _select_candidates(pool, replace(config, selection="fixed"))
        self.assertEqual([c["points"][0][1] for c in fixed], [10., 14.])

    def test_failure_audit_and_input_validation(self):
        rgb = np.zeros((7, 9, 3), np.uint8)
        plan = plan_prompts(rgb)
        self.assertEqual(plan.audit["detector_failure_count"], 1)
        self.assertEqual(plan.audit["scanned_crop_count"], 1)
        for bad in (np.zeros((9, 9), np.uint8), np.zeros((9, 9, 3), np.float32)):
            with self.assertRaises(ValueError):
                plan_prompts(bad)
        with self.assertRaises(ValueError):
            plan_prompts(np.zeros((0, 3, 3), np.uint8))
        with self.assertRaises(ValueError):
            PlannerConfig(overlap=1008).validate()


if __name__ == "__main__":
    unittest.main()
