"""CPU-only prompt/cache regression tests; no checkpoint or model imports."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from cable_prompt_geometry import (SquareTransform, normalize_points, component_evidence,
    select_prompted_components, width_normalized_limits, endpoint_seeds, sparse_uncovered_seeds)
from stage_a_geomprompt_sam31 import (Candidate, Crop, predict_points, proposal_quality,
    run_crop_sam, run_crop_text)


class FakePredictor:
    def __init__(self, mask):
        self.mask = mask
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "fake"}
        if request["type"] == "add_prompt":
            return {"outputs": {"out_obj_ids": [1], "out_binary_masks": [self.mask]}}
        return {}


class PromptGeometryTests(unittest.TestCase):
    def test_square_padding_roundtrip_and_coordinate_mapping(self):
        image = Image.new("RGB", (128, 64), "white")
        transform = SquareTransform(128, 64)
        self.assertEqual(transform.pad(image).size, (128, 128))
        self.assertEqual(transform.normalize([[64, 32]]), [[.5, .25]])
        raw = np.zeros((128, 128), bool)
        raw[10:20, 50:70] = True
        self.assertTrue(np.array_equal(transform.restore_mask(raw), raw[:64]))
        self.assertEqual(normalize_points([[64, 32]], 128, 64), [[.5, .5]])

    def test_invalid_coordinates_are_not_clamped(self):
        for points in ([[-1, 4]], [[32, 4]], [[float("nan"), 2]]):
            with self.assertRaises(ValueError):
                normalize_points(points, 32, 16)
        with self.assertRaises(ValueError):
            normalize_points([[2, 2]], 32, 16, (16, 16))

    def test_multicomponent_requires_explicit_support(self):
        mask = np.zeros((128, 256), bool)
        mask[30:34, 10:110] = True
        mask[30:34, 150:250] = True
        positives = [[40, 32], [200, 32]]
        legacy = proposal_quality(mask, positives[0])[1]["component_mask"]
        accepted, report = proposal_quality(mask, positives[0], positive_points=positives,
                                             component_mode="prompted")
        self.assertTrue(accepted)
        self.assertEqual(int(legacy.sum()), 400)
        self.assertEqual(int(report["component_mask"].sum()), 800)
        self.assertFalse(report["component_mask"][32, 130])  # no hidden bridge
        one, _ = select_prompted_components(mask, positives, [[180, 32]])
        self.assertEqual(int(one.sum()), 400)

    def test_unsupported_raw_components_remain_auditable(self):
        mask = np.zeros((30, 30), bool)
        mask[2:5, 2:5] = True
        mask[20:24, 20:24] = True
        labels, rows = component_evidence(mask, [[3, 3]])
        self.assertEqual(len(rows), 2)
        self.assertTrue(np.array_equal(labels > 0, mask))
        self.assertEqual([r["positive_count"] for r in rows], [1, 0])

    def test_width_units_scale_area_and_length(self):
        low, high = width_normalized_limits(6), width_normalized_limits(12)
        self.assertEqual(high["point_min_area"], 4*low["point_min_area"])
        self.assertEqual(high["min_long_side"], 2*low["min_long_side"])
        self.assertAlmostEqual(high["negative_offset"], 2*low["negative_offset"])
        with self.assertRaises(ValueError):
            width_normalized_limits(0)

    def test_predict_multiple_points_uses_padded_canvas(self):
        crop = Crop("x", 0, 0, 128, 64)
        fake = FakePredictor(np.zeros((128, 128), bool))
        mask, points, labels = predict_points(fake, "fake", crop, [[64, 32], [32, 16]],
                                               [[4, 8]], model_size=(128, 128))
        self.assertEqual(mask.shape, (64, 128))
        self.assertEqual(labels, [1, 1, 0])
        self.assertEqual(fake.requests[-1]["points"], [[.5, .25], [.25, .125], [.03125, .0625]])
        with self.assertRaises(ValueError):
            predict_points(fake, "fake", crop, [[4, 4]], [[4, 4]])
        with self.assertRaises(ValueError):
            predict_points(fake, "fake", crop, [[i, 4] for i in range(4)])

    def test_point_raw_cache_includes_rejected_proposal(self):
        crop = Crop("unit", 0, 0, 128, 64)
        raw = np.zeros((128, 128), bool)
        raw[10:13, 10:13] = True
        raw[30:34, 60:64] = True
        candidate = Candidate(11, 11, "unit", "test", 1, 12, 0, 0, 0, 0, 0)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            records, union, _ = run_crop_sam(FakePredictor(raw), Image.new("RGB", (128, 64)),
                crop, [candidate], base/"tmp", False, proposal_dir=base/"selected",
                raw_proposal_dir=base/"raw", preserve_aspect=True)
            self.assertFalse(records[0]["accepted"])
            self.assertIsNone(records[0]["mask_path"])
            self.assertFalse(union.any())
            evidence = json.loads(Path(records[0]["raw_evidence_path"]).read_text(encoding="utf-8"))
            self.assertFalse(evidence["accepted"])
            self.assertEqual(len(evidence["components"]), 2)
            with np.load(records[0]["component_labels_path"]) as archive:
                self.assertTrue(np.array_equal(archive["labels"] > 0, raw[:64]))

    def test_text_raw_cache_preserves_rejected_components(self):
        raw = np.zeros((64, 128), bool)
        raw[5:7, 5:7] = True
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            records, union = run_crop_text(FakePredictor(raw), Image.new("RGB", (128, 64)),
                Crop("text", 0, 0, 128, 64), base/"tmp", "cable", raw_proposal_dir=base/"raw")
            self.assertFalse(union.any())
            self.assertEqual(records[0]["raw_component_count"], 1)
            self.assertTrue(Path(records[0]["raw_mask_path"]).exists())

    def test_positive_groups_preserve_supported_pieces(self):
        raw = np.zeros((128, 256), bool)
        raw[30:34, 10:110] = True
        raw[30:34, 150:250] = True
        candidate = Candidate(40, 32, "unit", "test", 1, 12, 0, 0, 0, 0, 0)
        with tempfile.TemporaryDirectory() as folder:
            records, union, _ = run_crop_sam(FakePredictor(raw), Image.new("RGB", (256, 128)),
                Crop("unit", 0, 0, 256, 128), [candidate], Path(folder), False,
                positive_groups=[[[200, 32]]], component_mode="prompted")
            self.assertTrue(records[0]["accepted"])
            self.assertTrue(np.array_equal(union, raw))

    def test_endpoint_seeds_stay_visible_and_grid_is_not_positive(self):
        mask = np.zeros((64, 128), bool)
        mask[30:35, 10:110] = True
        endpoints = endpoint_seeds(mask)
        self.assertEqual(len(endpoints), 2)
        self.assertTrue(all(mask[int(r["y"]), int(r["x"])] for r in endpoints))
        grid = sparse_uncovered_seeds(mask, spacing=16, max_seeds=7)
        self.assertEqual(len(grid), 7)
        self.assertTrue(all(not mask[int(r["y"]), int(r["x"])] for r in grid))
        self.assertTrue(all(r["requires_image_verification"] for r in grid+endpoints))
        self.assertEqual(grid, sparse_uncovered_seeds(mask, spacing=16, max_seeds=7))


if __name__ == "__main__":
    unittest.main()
