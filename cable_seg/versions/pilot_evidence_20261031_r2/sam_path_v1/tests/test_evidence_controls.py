"""Mechanism ablations: forward parity, supervision and per-loss gradient isolation."""
from pathlib import Path
import importlib.util
import importlib
import sys
import unittest

import torch

from sam_path_v1.model import (ModelConfig, SamPathModel, GeometrySemanticDescriptor,
    PlainCNNDescriptor, ParallelAttentionTracker, SoftPathTracker, VARIANT_SPECS,
    variant_loss_metadata)
from sam_path_v1.losses import SetCriterion
from sam_path_v1.train import backward_losses


def has_grad(module):
    return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())


class EvidenceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.image = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(72))
        self.valid = torch.ones(1, 1, 32, 32)
        masks = torch.zeros(2, 32, 32)
        masks[0, 6:10, 2:28] = 1
        masks[1, 21:25, 3:30] = 1
        self.targets = [{"masks": masks, "id": "unit"}]

    def model(self, variant):
        return SamPathModel(ModelConfig(variant=variant, backend="toy", canvas_size=32,
            num_queries=3, tracker_steps=3, init_seed=19))

    def criterion(self, only=None):
        criterion = SetCriterion()
        if only:
            criterion.weights = {name: float(name == only) for name in criterion.weights}
        return criterion

    def run_backward(self, variant, only):
        model = self.model(variant)
        out = model(self.image, self.valid)
        criterion = self.criterion(only)
        losses = criterion(out, self.targets, self.valid)
        backward_losses(losses, out, criterion, torch.amp.GradScaler("cuda", enabled=False),
                        variant_loss_metadata(variant)["gradient_policy"])
        self.assertEqual(out["calls"], {"encoder": 1, "decoder": 1})
        return model

    def test_all_nine_automatic_arms_have_one_forward_and_finite_auxiliary_losses(self):
        for variant in VARIANT_SPECS:
            if variant == "S":
                continue
            with self.subTest(variant=variant):
                model = self.model(variant)
                out = model(self.image, self.valid)
                criterion = SetCriterion()
                losses = criterion(out, self.targets, self.valid)
                self.assertTrue(all(bool(torch.isfinite(x)) for x in losses.values()))
                self.assertEqual(out["calls"], {"encoder": 1, "decoder": 1})
                backward_losses(losses, out, criterion, torch.amp.GradScaler("cuda", enabled=False),
                                variant_loss_metadata(variant)["gradient_policy"])
                self.assertTrue(has_grad(model.base.decoder))
                self.assertTrue(all(p.grad is None for p in model.base.encoder.parameters()))

    def test_c_d_and_mask_only_have_identical_forward_and_initial_state(self):
        reference = self.model("C")
        expected = reference(self.image, self.valid)
        for variant in ("D", "MASK_ONLY", "NO_GEOMETRY", "NO_CONTRAST"):
            other = self.model(variant)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
            actual = other(self.image, self.valid)
            for key in ("mask_logits", "object_logits", "quality_logits"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_mask_only_blocks_mask_gradients_but_keeps_direct_condition_decoder(self):
        for loss in ("mask_bce", "mask_dice"):
            coupled = self.run_backward("C", loss)
            cut = self.run_backward("MASK_ONLY", loss)
            self.assertTrue(has_grad(coupled.tracker))
            self.assertTrue(has_grad(coupled.descriptor))
            self.assertFalse(has_grad(cut.tracker))
            self.assertFalse(has_grad(cut.descriptor))
            self.assertTrue(has_grad(cut.condition))
            self.assertTrue(has_grad(cut.base.decoder))

    def test_mask_only_preserves_quality_and_sam_presence_unlike_full_detach(self):
        for loss in ("quality", "object"):
            with self.subTest(loss=loss):
                coupled = self.run_backward("C", loss)
                cut = self.run_backward("MASK_ONLY", loss)
                detached = self.run_backward("D", loss)
                self.assertTrue(has_grad(cut.tracker))
                self.assertTrue(has_grad(cut.descriptor))
                self.assertFalse(has_grad(detached.tracker))
                self.assertFalse(has_grad(detached.descriptor))
                for (name, p), (_, other) in zip(coupled.named_parameters(), cut.named_parameters()):
                    if p.grad is not None and other.grad is not None:
                        torch.testing.assert_close(p.grad, other.grad, atol=1e-7, rtol=1e-6, msg=name)

    def test_mask_only_keeps_path_descriptor_geometry_auxiliaries(self):
        for loss, expected in (("path", "tracker"), ("descriptor", "descriptor"), ("geometry", "descriptor")):
            self.assertTrue(has_grad(getattr(self.run_backward("MASK_ONLY", loss), expected)))

    def test_auxiliary_removal_is_exact_and_descriptor_decomposition_is_not_double_counted(self):
        results = {}
        for variant in ("C", "NO_GEOMETRY", "NO_CONTRAST"):
            criterion = SetCriterion()
            losses = criterion(self.model(variant)(self.image, self.valid), self.targets, self.valid)
            results[variant] = (losses, criterion.last_components)
            torch.testing.assert_close(losses["descriptor"], sum(criterion.last_components.values()), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(losses["total"], sum(criterion.weights[k]*losses[k] for k in criterion.weights))
        c, no_geo, no_contrast = (results[v] for v in ("C", "NO_GEOMETRY", "NO_CONTRAST"))
        self.assertEqual(float(no_geo[0]["geometry"]), 0.)
        self.assertGreater(float(c[0]["geometry"]), 0.)
        self.assertEqual(float(no_contrast[1]["descriptor_contrast"]), 0.)
        torch.testing.assert_close(no_contrast[0]["descriptor"], c[1]["descriptor_compact"])
        torch.testing.assert_close(c[0]["total"]-no_geo[0]["total"], .1*c[0]["geometry"])
        torch.testing.assert_close(c[0]["total"]-no_contrast[0]["total"], .1*c[1]["descriptor_contrast"])

    def test_parameter_controls_match_active_parameter_budget(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        self.assertEqual(count(GeometrySemanticDescriptor()), 401155)
        self.assertEqual(count(PlainCNNDescriptor()), 400771)
        self.assertEqual(count(SoftPathTracker(8)), 989703)
        self.assertEqual(count(ParallelAttentionTracker(8)), 988800)

    def test_parallel_steps_are_independent_and_padding_has_zero_attention(self):
        module = ParallelAttentionTracker(3).eval()
        descriptor = torch.rand(1, 256, 4, 5)
        queries = torch.rand(1, 2, 256)
        valid = torch.ones(1, 4, 5, dtype=torch.bool)
        valid[..., -1] = False
        before = module(descriptor, queries, valid)
        with torch.no_grad():
            module.step_embeddings.weight[0].add_(torch.linspace(-1, 1, 256))
        after = module(descriptor, queries, valid)
        torch.testing.assert_close(before[0][:, :, 1:], after[0][:, :, 1:], atol=0, rtol=0)
        self.assertFalse(torch.equal(before[0][:, :, 0], after[0][:, :, 0]))
        self.assertEqual(int(torch.count_nonzero(after[2]["path_attention"][..., -1])), 0)

    def test_native_detector_quality_is_inactive(self):
        out = self.model("A")(self.image, self.valid)
        out["aux"] = {"native_detection": True}
        losses = SetCriterion()(out, self.targets, self.valid)
        self.assertEqual(float(losses["quality"]), 0.)
        self.assertGreater(float(losses["object"]), 0.)

    def test_original_abcd_math_is_bit_exact_against_frozen_source(self):
        legacy = Path(__file__).resolve().parents[3] / "full_wireseg_20261029" / "sam_path_v1"
        if not legacy.is_dir():
            self.skipTest("Frozen original package is not included in a portable deployment")
        name = "_frozen_original_sam_path_controls"
        spec = importlib.util.spec_from_file_location(name, legacy / "__init__.py",
                                                     submodule_search_locations=[str(legacy)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        old_model = importlib.import_module(name + ".model")
        old_losses = importlib.import_module(name + ".losses")
        for variant in "ABCD":
            with self.subTest(variant=variant):
                original = old_model.SamPathModel(old_model.ModelConfig(variant=variant, backend="toy",
                    canvas_size=32, num_queries=3, tracker_steps=3, init_seed=19))
                current = self.model(variant)
                original_out, current_out = original(self.image, self.valid), current(self.image, self.valid)
                for key in ("mask_logits", "object_logits", "quality_logits"):
                    torch.testing.assert_close(original_out[key], current_out[key], atol=0, rtol=0)
                original_loss = old_losses.SetCriterion()(original_out, self.targets, self.valid)
                current_loss = SetCriterion()(current_out, self.targets, self.valid)
                for key in original_loss:
                    torch.testing.assert_close(original_loss[key], current_loss[key], atol=0, rtol=0)
                original_loss["total"].backward()
                current_loss["total"].backward()
                for (key, parameter), (_, old_parameter) in zip(current.named_parameters(), original.named_parameters()):
                    self.assertEqual(parameter.grad is None, old_parameter.grad is None, key)
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, old_parameter.grad, atol=0, rtol=0, msg=key)


if __name__ == "__main__":
    unittest.main()
