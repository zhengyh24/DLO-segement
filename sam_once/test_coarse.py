"""CPU-only semantic checks for raw foreground evidence export."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace
import hashlib
import tempfile
import unittest

import numpy as np
from PIL import Image

from sam_once.coarse import UNOBSERVED_LOGIT, export_coarse
from sam_once.types import CropSpec, PromptPlan, PromptSpec, ProposalRecord


def case(crop_boxes, arrays, shape=(8, 11)):
    rgb = np.full((*shape, 3), 60, np.uint8)
    crops = [CropSpec(f"crop_{i}", box) for i, box in enumerate(crop_boxes)]
    prompts = [PromptSpec(f"prompt_{i}", crop.crop_id, ((0., 0.),)) for i, crop in enumerate(crops)]
    plan = PromptPlan(crops, prompts, np.zeros(shape, np.float32))
    records = [ProposalRecord(prompt, crop, array.copy(), array.copy(), .8)
               for prompt, crop, array in zip(prompts, crops, arrays)]
    return rgb, plan, records


class CoarseTests(unittest.TestCase):
    def test_native_crop_coordinates_and_unobserved_sentinel(self):
        logits = np.array([[3., 0., -2048.], [-2., 4., -1.]], np.float32)
        rgb, plan, records = case([(4, 3, 7, 5)], [logits])
        with tempfile.TemporaryDirectory() as directory:
            audit = {}
            stage = export_coarse(directory, rgb, records, plan, audit)
            self.assertIs(stage, audit["coarse_stage"])
            with np.load(Path(directory) / "coarse_evidence.npz", allow_pickle=False) as arrays:
                maximum, coverage = arrays["max_logits"], arrays["observed_coverage"]
                self.assertEqual(coverage.dtype, np.bool_)
                self.assertEqual(maximum.dtype, np.float32)
                np.testing.assert_array_equal(maximum[3:5, 4:7], logits)
                expected = np.zeros((8, 11), bool)
                expected[3:5, 4:7] = True
                np.testing.assert_array_equal(coverage, expected)
                self.assertTrue(np.all(maximum[~coverage] == UNOBSERVED_LOGIT))
                self.assertEqual(maximum[3, 6], -2048.)
            mask = np.asarray(Image.open(Path(directory) / "coarse_foreground.png"))
            self.assertEqual(set(map(tuple, np.argwhere(mask > 0))), {(3, 4), (4, 5)})
            self.assertEqual(stage["observed_pixels"], 6)
            self.assertEqual(stage["foreground_pixels"], 2)
            self.assertTrue(stage["no_instance_assignment"])
            for name, expected_hash in stage["files_sha256"].items():
                self.assertEqual(hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest(), expected_hash)
            self.assertFalse((Path(directory) / "instance_ids.png").exists())

    def test_overlap_uses_max_preserving_foreground_union(self):
        first = np.array([[2., -2., 0.], [-3., 8., -1.], [-7., -9., -4.]], np.float32)
        second = np.array([[-8., 7., -1.], [4., -10., 1.], [-4., 0., -8.]], np.float32)
        rgb, plan, records = case([(1, 1, 4, 4), (2, 2, 5, 5)], [first, second])
        with tempfile.TemporaryDirectory() as directory:
            export_coarse(directory, rgb, records, plan, {})
            with np.load(Path(directory) / "coarse_evidence.npz") as arrays:
                maximum = arrays["max_logits"]
                np.testing.assert_array_equal(maximum[2:4, 2:4], np.maximum(first[1:, 1:], second[:2, :2]))
            expected_union = np.zeros((8, 11), bool)
            expected_union[1:4, 1:4] |= first > 0
            expected_union[2:5, 2:5] |= second > 0
            mask = np.asarray(Image.open(Path(directory) / "coarse_foreground.png")) > 0
            np.testing.assert_array_equal(mask, expected_union)
            np.testing.assert_array_equal(records[0].initial_logits, first)
            np.testing.assert_array_equal(records[1].current_logits, second)

    def test_single_positive_pixel_is_not_deleted_and_zero_is_not_foreground(self):
        logits = np.full((8, 11), -3., np.float32)
        logits[4, 5] = np.nextafter(np.float32(0), np.float32(1))
        logits[3, 4] = 0.
        rgb, plan, records = case([(0, 0, 11, 8)], [logits])
        with tempfile.TemporaryDirectory() as directory:
            stage = export_coarse(directory, rgb, records, plan, {})
            self.assertEqual(stage["foreground_pixels"], 1)
            self.assertEqual(np.count_nonzero(np.asarray(Image.open(Path(directory) / "coarse_foreground.png"))), 1)
            overlay = np.asarray(Image.open(Path(directory) / "coarse_overlay.png"))
            np.testing.assert_array_equal(overlay[3, 4], rgb[3, 4])

    def test_empty_and_missing_proposals_are_audited_as_unobserved(self):
        rgb, plan, _ = case([(0, 0, 11, 8)], [])
        with tempfile.TemporaryDirectory() as directory:
            stage = export_coarse(directory, rgb, [], plan, {})
            self.assertEqual(stage["status"], "empty")
            self.assertEqual(stage["proposal_count"], 0)
            self.assertEqual(stage["missing_prompt_count"], 1)
            self.assertEqual(stage["observed_pixels"], 0)
            with np.load(Path(directory) / "coarse_evidence.npz") as arrays:
                self.assertFalse(arrays["observed_coverage"].any())
                self.assertTrue(np.all(arrays["max_logits"] == UNOBSERVED_LOGIT))
            self.assertEqual(stage["foreground_pixels"], 0)

    def test_refined_logits_or_provenance_are_rejected_before_writing(self):
        for mode in ("changed", "marked", "visits", "nested"):
            rgb, plan, records = case([(1, 1, 4, 3)], [np.ones((2, 3), np.float32)])
            if mode == "changed":
                records[0].current_logits[0, 0] += 1
            elif mode == "marked":
                records[0].metadata["refined"] = True
            elif mode == "visits":
                records[0].metadata["refinement_visits"] = 1
            else:
                records[0].metadata["source"] = {"refined": True}
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    export_coarse(directory, rgb, records, plan, {})
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_mapping_duplicates_and_modified_prompt_are_rejected(self):
        for mode in ("crop", "prompt", "duplicate"):
            rgb, plan, records = case([(1, 1, 4, 3)], [np.ones((2, 3), np.float32)])
            if mode == "crop":
                records[0].crop = CropSpec("crop_0", (2, 2, 5, 4))
            elif mode == "prompt":
                records[0].prompt = replace(records[0].prompt, positive=((1., 1.),))
            else:
                records.append(records[0])
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    export_coarse(directory, rgb, records, plan, {})
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_observed_negative_foreground_empty_still_has_valid_evidence(self):
        rgb, plan, records = case([(1, 1, 4, 3)], [np.full((2, 3), -2., np.float32)])
        with tempfile.TemporaryDirectory() as directory:
            stage = export_coarse(directory, rgb, records, plan, {})
            self.assertEqual(stage["status"], "ok")
            self.assertEqual(stage["foreground_pixels"], 0)
            self.assertEqual(stage["observed_pixels"], 6)

    def test_existing_outputs_are_not_overwritten(self):
        rgb, plan, records = case([(1, 1, 4, 3)], [np.ones((2, 3), np.float32)])
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "coarse_evidence.npz"
            existing.write_bytes(b"existing evidence")
            with self.assertRaises(FileExistsError):
                export_coarse(directory, rgb, records, plan, {})
            self.assertEqual(existing.read_bytes(), b"existing evidence")
            self.assertFalse((Path(directory) / "coarse_foreground.png").exists())


if __name__ == "__main__":
    unittest.main()
