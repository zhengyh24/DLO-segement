"""Loss invariance, empty-set and gradient contracts for sam_path_v1."""
from __future__ import annotations

import unittest

import torch

from sam_path_v1.losses import SetCriterion, _resize_masks, _visible_geometry
from sam_path_v1.model import ModelConfig, SamPathModel


class LossContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def output(logits):
        if logits.ndim == 3:
            logits = logits[None]
        b, q = logits.shape[:2]
        return {"mask_logits": logits,
                "object_logits": torch.zeros(b, q, requires_grad=True),
                "quality_logits": torch.zeros(b, q, requires_grad=True),
                "aux": {}}

    @staticmethod
    def target_masks(size=8):
        masks = torch.zeros(2, size, size)
        masks[0, 1:3, 1:size-1] = 1
        masks[1, size-3:size-1, 1:size-1] = 1
        return masks

    def test_instance_label_permutation_does_not_change_loss(self):
        masks = self.target_masks()
        logits = torch.stack((masks[1]*10-5, masks[0]*10-5, torch.full((8, 8), -5.)))
        output = self.output(logits)
        valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        criterion = SetCriterion()
        first = criterion(output, [{"id": "x", "masks": masks}], valid)
        second = criterion(output, [{"id": "x", "masks": masks.flip(0)}], valid)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=1e-6)

    def test_matching_finds_instances_independent_of_query_order(self):
        masks = self.target_masks()
        output = self.output(torch.stack((masks[1]*20-10, masks[0]*20-10, torch.full((8, 8), -10.))))
        rows, cols = SetCriterion().match(output, [{"masks": masks}], torch.ones(1, 1, 8, 8))[0]
        self.assertEqual(dict(zip(rows.tolist(), cols.tolist())), {0: 1, 1: 0})

    def test_empty_image_has_finite_negative_supervision(self):
        logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
        output = self.output(logits)
        losses = SetCriterion()(output, [{"id": "empty", "masks": torch.zeros(0, 8, 8)}],
                                torch.ones(1, 1, 8, 8))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertGreater(float(losses["total"]), 0)
        losses["total"].backward()
        self.assertTrue(torch.all(logits.grad > 0), "Unmatched queries need pressure toward empty masks")
        self.assertTrue(torch.all(output["object_logits"].grad > 0))

    def test_native_invalid_pixels_receive_no_direct_mask_gradient(self):
        logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
        valid = torch.ones(1, 1, 8, 8)
        valid[..., 6:, :] = 0
        valid[..., :, 6:] = 0
        masks = self.target_masks()
        criterion = SetCriterion(path_weight=0, descriptor_weight=0, geometry_weight=0, quality_weight=0)
        criterion(self.output(logits), [{"masks": masks}], valid)["total"].backward()
        excluded = ~valid.bool().expand_as(logits)
        self.assertEqual(int(torch.count_nonzero(logits.grad[excluded])), 0)
        self.assertGreater(float(logits.grad[~excluded].abs().sum()), 0)

    def test_invalid_gt_does_not_affect_matching_even_in_partial_coarse_cell(self):
        # Last low-res cell contains one valid native column and three invalid
        # columns. GT outside validity must not change which query is selected.
        logits = torch.tensor([[[[-10., -1.], [-10., -1.]],
                                [[-10., 3.], [-10., 3.]]]])
        output = self.output(logits)
        valid = torch.ones(1, 1, 8, 8)
        valid[..., 5:] = 0
        base = torch.zeros(1, 8, 8)
        base[:, :, 4] = 1
        changed = base.clone()
        changed[:, :, 5:] = 1
        criterion = SetCriterion(mask_bce=1., mask_dice=0.)
        clean = criterion.match(output, [{"masks": base}], valid)[0]
        noisy = criterion.match(output, [{"masks": changed}], valid)[0]
        torch.testing.assert_close(clean[0], noisy[0], rtol=0, atol=0)
        torch.testing.assert_close(clean[1], noisy[1], rtol=0, atol=0)

    def test_more_instances_than_queries_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "never truncate"):
            SetCriterion()(self.output(torch.zeros(1, 1, 8, 8)),
                           [{"id": "overflow", "masks": self.target_masks()}],
                           torch.ones(1, 1, 8, 8))

    def test_no_valid_pixels_and_non_native_gt_are_rejected(self):
        output = self.output(torch.zeros(1, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, "no valid pixels"):
            SetCriterion()(output, [{"masks": self.target_masks()}], torch.zeros(1, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "native pixels"):
            SetCriterion()(output, [{"masks": torch.zeros(1, 7, 8)}], torch.ones(1, 1, 8, 8))

    def test_thin_mask_occupancy_survives_area_reduction(self):
        mask = torch.zeros(1, 16, 16)
        mask[:, :, 7] = 1
        reduced = _resize_masks(mask, (4, 4))
        self.assertGreater(float(reduced.sum()), 0)
        torch.testing.assert_close(reduced.mean(), mask.mean(), rtol=0, atol=0)

    def test_tangent_is_unsigned_and_matches_horizontal_vertical_geometry(self):
        horizontal = torch.zeros(1, 15, 15)
        horizontal[:, 7, 2:13] = 1
        vertical = horizontal.transpose(1, 2)
        _, tangent_h, reliability_h = _visible_geometry(horizontal)
        _, tangent_v, reliability_v = _visible_geometry(vertical)
        self.assertGreater(float(reliability_h[7, 7]), .99)
        self.assertGreater(float(reliability_v[7, 7]), .99)
        torch.testing.assert_close(tangent_h[:, 7, 7], torch.tensor([1., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(tangent_v[:, 7, 7], torch.tensor([-1., 0.]), atol=1e-6, rtol=0)

    def test_d_full_auxiliary_criterion_trains_tracker_and_descriptor(self):
        model = SamPathModel(ModelConfig(variant="D", backend="toy", canvas_size=32,
                                        num_queries=3, tracker_steps=3, init_seed=9))
        outputs = model(torch.rand(1, 3, 32, 32))
        criterion = SetCriterion(mask_bce=0, mask_dice=0, object_weight=0, quality_weight=0,
                                 path_weight=1, descriptor_weight=1, geometry_weight=1)
        losses = criterion(outputs, [{"masks": self.target_masks(32)}], torch.ones(1, 1, 32, 32))
        losses["total"].backward()
        for component in (model.tracker, model.descriptor):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in component.parameters()))
        self.assertGreater(float(losses["path"]), 0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_negative_empty_query_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            SetCriterion(empty_mask_weight=-.1)

    def test_nonfinite_matching_cost_is_rejected(self):
        output = self.output(torch.full((1, 3, 8, 8), float("nan")))
        with self.assertRaisesRegex(FloatingPointError, "matching cost"):
            SetCriterion()(output, [{"masks": self.target_masks()}], torch.ones(1, 1, 8, 8))


if __name__ == "__main__":
    unittest.main()

