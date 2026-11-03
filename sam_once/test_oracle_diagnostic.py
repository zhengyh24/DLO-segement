"""CPU-only checks for isolated, explicitly labelled GT-prompt diagnostics."""
import ast
import copy
from pathlib import Path
import unittest
import numpy as np

from .oracle_diagnostic import oracle_plan


def sample(h=128, w=192):
    rgb = np.full((h, w, 3), 90, np.uint8)
    a, b = np.zeros((h, w), bool), np.zeros((h, w), bool)
    a[24:33, 12:w-12] = True
    b[h-33:h-24, 12:w-12] = True
    return rgb, [a, b]


class OracleDiagnosticTests(unittest.TestCase):
    def test_deterministic_separate_gt_groups_and_flags(self):
        rgb, masks = sample()
        original = copy.deepcopy(masks)
        plan = oracle_plan(rgb, masks)
        self.assertEqual(plan.metadata(), oracle_plan(rgb, masks).metadata())
        self.assertEqual(len(plan.prompts), 2)
        self.assertTrue(plan.audit["ground_truth_used_for_inference"])
        self.assertFalse(plan.audit["deployment_allowed"])
        self.assertEqual(plan.audit["uncovered_gt_indices"], [])
        self.assertEqual(float(plan.ridge.sum()), 0)
        for p in plan.prompts:
            self.assertTrue(p.metadata["diagnostic_only"])
            self.assertTrue(p.metadata["ground_truth_used"])
            self.assertFalse(p.metadata["deployment_allowed"])
            self.assertEqual(len(p.positive), 3)
            self.assertGreater(max(x for x, _ in p.positive)-min(x for x, _ in p.positive), 70)
            for x, y in p.positive:
                self.assertTrue(masks[p.metadata["gt_index"]][int(y), int(x)])
        for before, after in zip(original, masks):
            np.testing.assert_array_equal(before, after)

    def test_crossing_overlap_and_canvas_border_excluded(self):
        rgb, masks = sample()
        masks[1][:] = False
        masks[1][8:120, 92:101] = True
        plan = oracle_plan(rgb, masks)
        overlap = masks[0] & masks[1]
        for p in plan.prompts:
            for x, y in p.positive:
                self.assertFalse(overlap[int(y), int(x)])
                self.assertGreaterEqual(min(x, y), 4)
                other = masks[1-p.metadata["gt_index"]]
                self.assertFalse(other[max(0, int(y)-3):int(y)+4, max(0, int(x)-3):int(x)+4].any())

    def test_negatives_and_box_modes_valid(self):
        rgb, masks = sample()
        for mode in ("background", "other_gt", "mixed"):
            for box in ("none", "expanded", "local"):
                plan = oracle_plan(rgb, masks, 3, 2, mode, box)
                self.assertEqual(plan.audit["actual_negative_points"], 4)
                for p in plan.prompts:
                    target = masks[p.metadata["gt_index"]]
                    for x, y in p.negative:
                        self.assertFalse(target[int(y), int(x)])
                        if mode == "background":
                            self.assertFalse(any(m[int(y), int(x)] for m in masks))
                        elif mode == "other_gt":
                            self.assertTrue(masks[1-p.metadata["gt_index"]][int(y), int(x)])
                    self.assertEqual(p.box is None, box == "none")

    def test_capacity_reports_omissions_without_easy_replacement(self):
        rgb = np.zeros((256, 256, 3), np.uint8)
        masks = []
        for index in range(8):
            m = np.zeros((256, 256), bool)
            y = 10+index*28
            m[y:y+7, 12:244] = True
            masks.append(m)
        # First reserved instance is made fully ambiguous; GT #6/#7 still cannot backfill it.
        masks[1] = masks[0].copy()
        plan = oracle_plan(rgb, masks)
        self.assertEqual([p.metadata["gt_index"] for p in plan.prompts], [2, 3, 4, 5])
        self.assertEqual(plan.audit["uncovered_gt_indices"], [0, 1, 6, 7])
        omissions = plan.audit["crops"][0]["omitted_gt"]
        self.assertEqual(sum(r["reason"] == "six_group_capacity" for r in omissions), 2)

    def test_native_tiling_and_crop_local_coordinates(self):
        rgb = np.zeros((1100, 1300, 3), np.uint8)
        mask = np.zeros((1100, 1300), bool)
        mask[550:559, 10:1290] = True
        plan = oracle_plan(rgb, [mask], positive_count=1)
        self.assertEqual(len(plan.crops), 4)
        crops = {c.crop_id: c for c in plan.crops}
        covered = np.zeros(mask.shape, bool)
        for c in plan.crops:
            x0, y0, x1, y1 = c.xyxy
            self.assertLessEqual(max(c.width, c.height), 1008)
            covered[y0:y1, x0:x1] = True
        self.assertTrue(covered.all())
        for p in plan.prompts:
            c = crops[p.crop_id]
            for x, y in p.positive:
                self.assertTrue(mask[int(y)+c.xyxy[1], int(x)+c.xyxy[0]])

    def test_empty_and_invalid_gt(self):
        rgb, masks = sample()
        plan = oracle_plan(rgb, [])
        self.assertEqual(plan.prompts, [])
        self.assertEqual(plan.audit["actual_groups"], 0)
        with self.assertRaises(ValueError):
            oracle_plan(rgb, [np.zeros((20, 20))])
        with self.assertRaises(ValueError):
            oracle_plan(rgb, masks, positive_count=4)

    def test_oracle_not_imported_or_called_by_deployment(self):
        package = Path(__file__).parent
        oracle_ast = ast.parse((package / "oracle_diagnostic.py").read_text(encoding="utf-8"))
        imported = [n.module or "" for n in ast.walk(oracle_ast) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(any(name in mod for name in ("planner", "sam_adapter", "train", "evaluate")) for mod in imported))
        for name in ("cli.py", "planner.py"):
            tree = ast.parse((package / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("oracle_diagnostic", node.module or "")
                if isinstance(node, ast.Import):
                    self.assertFalse(any("oracle_diagnostic" in n.name for n in node.names))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, "oracle_plan")


if __name__ == "__main__":
    unittest.main()
