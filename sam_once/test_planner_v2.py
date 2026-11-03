"""CPU tests of observed path geometry and RGB-only prompt pool contracts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sam_once.planner_v2 import (V2Config, _neighbor_pool, arc_samples,
    deduplicate_paths, extract_branch, path_graph, plan_prompts, select_paths, split_paths)


def scan_fixture(rgb, config):
    response = (rgb.max(axis=2) > 150).astype(np.float32)
    candidates, audit = extract_branch(rgb, response, np.full(response.shape, 2., np.float32),
                                       "fixture:ridge", config)
    return response, candidates, {"branches": [audit], "failures": [], "warnings": []}


def straight_candidate(y=30, x0=10, x1=91, color=(220, 220, 220), source="fixture", shape=(100, 120)):
    rgb = np.full((*shape, 3), 20, np.uint8)
    rgb[y, x0:x1] = color
    response = np.zeros(shape, np.float32)
    response[y, x0:x1] = 1
    candidates, _ = extract_branch(rgb, response, np.full(shape, 2., np.float32), source, V2Config())
    return rgb, response, candidates[0]


class PlannerV2Tests(unittest.TestCase):
    def test_staircase_removes_corner_shortcuts_without_cutting_path(self):
        mask = np.zeros((50, 50), bool)
        for n in range(10, 35):
            mask[n, n] = mask[n, n+1] = True
        graph, removed = path_graph(mask)
        self.assertGreater(removed, 0)
        self.assertLessEqual(max(map(len, graph.values())), 2)
        paths, audit = split_paths(mask)
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0][0]), int(mask.sum()))
        self.assertEqual(audit["true_junction_pixels"], 0)

    def test_plus_and_x_crossings_are_split_not_joined(self):
        for diagonal in (False, True):
            mask = np.zeros((91, 91), np.uint8)
            if diagonal:
                cv2.line(mask, (10, 10), (80, 80), 1, 1)
                cv2.line(mask, (10, 80), (80, 10), 1, 1)
            else:
                mask[45, 10:81] = 1
                mask[10:81, 45] = 1
            paths, audit = split_paths(mask.astype(bool))
            self.assertEqual(len(paths), 4)
            self.assertGreater(audit["true_junction_pixels"], 0)
            for coords, hard, closed in paths:
                self.assertTrue(hard)
                self.assertFalse(closed)
                self.assertGreater(np.min(np.linalg.norm(coords-[45, 45], axis=1)), 3)

    def test_u_bend_has_three_arc_points_on_one_observed_path(self):
        rgb = np.full((101, 101, 3), 20, np.uint8)
        cv2.polylines(rgb, [np.array([[25, 15], [25, 80], [75, 80], [75, 15]])], False, (220, 220, 220), 1)
        with patch("sam_once.planner_v2._scan_tile", side_effect=scan_fixture):
            plan = plan_prompts(rgb, V2Config(negative_mode="neighbor", neg_count=2))
        self.assertEqual(len(plan.prompts), 1)
        prompt = plan.prompts[0]
        self.assertEqual(len(prompt.positive), 3)
        self.assertTrue(all(rgb[int(y), int(x), 0] == 220 for x, y in prompt.positive))
        self.assertEqual(prompt.positive[1][1], 80.)
        self.assertEqual({prompt.positive[0][0], prompt.positive[-1][0]}, {25., 75.})
        self.assertFalse(prompt.negative)
        self.assertFalse(prompt.metadata["hard_zone"])

    def test_parallel_paths_are_independent_and_same_color_is_never_neighbor_negative(self):
        rgb = np.full((90, 110, 3), 20, np.uint8)
        rgb[30, 10:101] = 220
        rgb[48, 10:101] = 220
        with patch("sam_once.planner_v2._scan_tile", side_effect=scan_fixture):
            plan = plan_prompts(rgb, V2Config(negative_mode="neighbor", neg_count=2))
        self.assertEqual(len(plan.prompts), 2)
        self.assertEqual({p.positive[0][1] for p in plan.prompts}, {30., 48.})
        self.assertTrue(all(len({y for _, y in p.positive}) == 1 for p in plan.prompts))
        self.assertTrue(all(not p.negative for p in plan.prompts))

    def test_neighbor_pool_requires_color_parallelism_and_measured_gap(self):
        rgb, ridge, one = straight_candidate(y=30, color=(220, 30, 30), source="one")
        other_rgb, other_ridge, other = straight_candidate(y=48, color=(30, 220, 30), source="two")
        rgb[48] = other_rgb[48]
        ridge = np.maximum(ridge, other_ridge)
        points, evidence = _neighbor_pool(rgb, ridge, one, [one, other])
        self.assertEqual(len(points), 2)
        self.assertTrue(all(y == 48 for x, y in points))
        self.assertTrue(all(e["gap_ridge_response_max"] == 0. for e in evidence))
        blocked = ridge.copy()
        blocked[37:42] = 1
        self.assertEqual(_neighbor_pool(rgb, blocked, one, [other])[0], [])
        hard = {**one, "hard_zone": True}
        self.assertEqual(_neighbor_pool(rgb, ridge, hard, [other])[0], [])
        # A visible U connection vetoes negatives even if its arms differ in color.
        connected = ridge.copy()
        connected[30:49, 90] = 1
        self.assertEqual(_neighbor_pool(rgb, connected, one, [other])[0], [])
        textured = rgb.copy()
        textured[48, 10:91] = np.tile(np.array([[0, 255, 0], [255, 0, 255]], np.uint8), (41, 1))[:81]
        self.assertEqual(_neighbor_pool(textured, ridge, one, [other])[0], [])

    def test_false_ridge_without_rgb_contrast_is_discarded(self):
        rgb = np.full((65, 81, 3), 80, np.uint8)
        response = np.zeros(rgb.shape[:2], np.float32)
        response[32, 10:71] = 1
        candidates, audit = extract_branch(rgb, response, response*2, "false", V2Config())
        self.assertEqual(candidates, [])
        self.assertEqual(audit["low_contrast_discarded"], 1)

    def test_crossing_has_multiple_safe_positives_but_no_negative_pool(self):
        rgb = np.full((91, 91, 3), 20, np.uint8)
        rgb[45, 10:81] = 230
        rgb[10:81, 45] = 230
        with patch("sam_once.planner_v2._scan_tile", side_effect=scan_fixture):
            plan = plan_prompts(rgb, V2Config(negative_mode="background", neg_count=2))
        self.assertEqual(len(plan.prompts), 4)
        for prompt in plan.prompts:
            self.assertEqual(len(prompt.positive), 3)
            self.assertTrue(prompt.metadata["hard_zone"])
            self.assertFalse(prompt.negative)
            self.assertTrue(all(np.linalg.norm(np.array([x, y])-[45, 45]) > 3 for x, y in prompt.positive))

    def test_dedup_never_combines_points_or_parallel_paths(self):
        _, _, first = straight_candidate(source="first")
        _, _, rediscovery = straight_candidate(source="another_channel")
        _, _, parallel = straight_candidate(y=35, source="parallel")
        unique, removed = deduplicate_paths([first, rediscovery, parallel])
        reverse, _ = deduplicate_paths([parallel, rediscovery, first])
        self.assertEqual(removed, 1)
        self.assertEqual([c["path_key"] for c in unique], [c["path_key"] for c in reverse])
        self.assertEqual(len(unique), 2)
        self.assertTrue(all(len({y for x, y in c["positive_pool"]}) == 1 for c in unique))

    def test_selection_is_order_independent_and_covers_new_regions(self):
        _, _, first = straight_candidate(y=10, x0=10, x1=35, shape=(100, 100), source="first")
        _, _, second = straight_candidate(y=14, x0=10, x1=35, shape=(100, 100), source="second")
        _, _, distant = straight_candidate(y=75, x0=65, x1=90, shape=(100, 100), source="distant")
        for item, score in zip((first, second, distant), (10., 9., 8.)):
            item["score"] = score
        config = V2Config(max_prompts_per_crop=2)
        chosen = select_paths([first, second, distant], (100, 100), config)
        reverse = select_paths([distant, second, first], (100, 100), config)
        fixed = select_paths([first, second, distant], (100, 100), replace(config, selection="fixed"))
        self.assertEqual([c["path_key"] for c in chosen], [c["path_key"] for c in reverse])
        self.assertEqual([c["sources"] for c in chosen], [["first"], ["distant"]])
        self.assertEqual([c["sources"] for c in fixed], [["first"], ["second"]])

    def test_pools_and_group_selection_independent_of_point_box_ablation(self):
        rgb = np.full((90, 110, 3), 20, np.uint8)
        rgb[30, 10:101] = (220, 30, 30)
        rgb[48, 10:101] = (30, 220, 30)
        with patch("sam_once.planner_v2._scan_tile", side_effect=scan_fixture):
            base = plan_prompts(rgb)
            changed = plan_prompts(rgb, V2Config(pos_count=1, neg_count=2, negative_mode="neighbor", box_mode="expanded"))
        self.assertEqual([p.prompt_id for p in base.prompts], [p.prompt_id for p in changed.prompts])
        for one, two in zip(base.prompts, changed.prompts):
            self.assertEqual(one.metadata["path_key"], two.metadata["path_key"])
            for key in ("positive_pool", "positive_pool_arc_fractions", "negative_pools", "box_local", "box_expanded"):
                self.assertEqual(one.metadata[key], two.metadata[key])
            self.assertEqual(len(one.positive), 3)
            self.assertEqual(len(two.positive), 1)
            self.assertFalse(one.negative)
            self.assertIsNone(one.box)
            self.assertIsNotNone(two.box)
            self.assertLessEqual(two.metadata["sparse_tokens"], 7)

    def test_full_odd_tile_coverage_budget_and_empty_scans(self):
        rgb = np.full((117, 149, 3), 20, np.uint8)
        rgb[10:111:10, 3:145] = 220
        config = V2Config(canvas_size=64, overlap=17, max_prompts_per_crop=2)
        with patch("sam_once.planner_v2._scan_tile", side_effect=scan_fixture) as scanner:
            plan = plan_prompts(rgb, config)
        coverage = np.zeros(rgb.shape[:2], bool)
        for crop in plan.crops:
            x0, y0, x1, y1 = crop.xyxy
            coverage[y0:y1, x0:x1] = True
            self.assertLessEqual(sum(p.crop_id == crop.crop_id for p in plan.prompts), 2)
        self.assertTrue(coverage.all())
        self.assertEqual(scanner.call_count, len(plan.crops))
        with patch("sam_once.planner_v2._scan_tile", side_effect=scan_fixture) as scanner:
            empty = plan_prompts(np.zeros_like(rgb), replace(config, max_prompts_per_crop=0))
        self.assertEqual(scanner.call_count, len(empty.crops))
        self.assertEqual(empty.audit["status"], "empty_plan")
        self.assertEqual(empty.audit["empty_crop_count"], len(empty.crops))

    def test_real_detector_determinism_and_honest_numerical_limit_audit(self):
        rgb = np.full((81, 113, 3), 25, np.uint8)
        cv2.line(rgb, (12, 15), (101, 65), (220, 220, 220), 3)
        config = V2Config(channels=("gray",), scales=(1., 4., 9., 25.))
        first, second = plan_prompts(rgb, config), plan_prompts(rgb, config)
        self.assertGreater(len(first.prompts), 0)
        self.assertEqual(first.metadata(), second.metadata())
        np.testing.assert_array_equal(first.ridge, second.ridge)
        self.assertEqual(first.ridge.dtype, np.float32)
        horizontal = np.full((81, 113, 3), 25, np.uint8)
        horizontal[39:42, 12:101] = 220
        limitation = plan_prompts(horizontal, config)
        self.assertTrue(limitation.audit["crop_audits"][0]["scan"]["warnings"])
        self.assertFalse(limitation.audit["ground_truth_used"])
        self.assertEqual(limitation.audit["sam_calls"], 0)

    def test_configuration_and_input_validation(self):
        for changes in ({"pos_count": 0}, {"neg_count": 3}, {"box_mode": "global"},
                        {"negative_mode": "semantic"}, {"overlap": 1008}):
            with self.assertRaises(ValueError):
                V2Config(**changes).validate()
        with self.assertRaises(ValueError):
            plan_prompts(np.zeros((0, 9, 3), np.uint8))
        tiny = plan_prompts(np.zeros((7, 9, 3), np.uint8))
        self.assertEqual(tiny.audit["detector_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
