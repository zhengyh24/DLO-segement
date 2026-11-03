"""Independent CPU contracts for the new automatic SAM-path package.

The toy backend proves tensor, gradient and single-pass contracts, not SAM
accuracy or capacity. No historical package or result is imported.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from sam_path_v1.backbone import Sam31Backbone
from sam_path_v1.model import ModelConfig, SamPathModel


def has_gradient(module):
    return module is not None and any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
    )


def make_model(variant="C", **kwargs):
    return SamPathModel(ModelConfig(
        variant=variant, backend="toy", canvas_size=32,
        num_queries=3, tracker_steps=3, init_seed=71, **kwargs
    ))


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        generator = torch.Generator().manual_seed(123)
        self.images = torch.rand(2, 3, 32, 32, generator=generator)
        self.valid = torch.ones(2, 1, 32, 32, dtype=torch.bool)
        self.valid[0, :, 25:, :] = False
        self.valid[0, :, :, 27:] = False

    def test_all_variants_have_single_encoder_and_single_decode(self):
        for variant in "ABCD":
            with self.subTest(variant=variant):
                model = make_model(variant).eval()
                with patch.object(model.base, "encode", wraps=model.base.encode) as encode:
                    with patch.object(model.base, "decode", wraps=model.base.decode) as decode:
                        result = model(self.images, self.valid)
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(decode.call_count, 1)
                self.assertEqual(result["calls"], {"encoder": 1, "decoder": 1})
                self.assertEqual(result["mask_logits"].shape[:2], (2, 3))
                self.assertEqual(result["object_logits"].shape, (2, 3))
                self.assertEqual(result["quality_logits"].shape, (2, 3))
                self.assertTrue(torch.isfinite(result["mask_logits"]).all())
                self.assertTrue(all(not p.requires_grad for p in model.base.encoder.parameters()))

    def test_c_and_d_same_initial_weights_and_forward_values(self):
        c, d = make_model("C").eval(), make_model("D").eval()
        self.assertEqual(set(c.state_dict()), set(d.state_dict()))
        for key, value in c.state_dict().items():
            torch.testing.assert_close(value, d.state_dict()[key], rtol=0, atol=0)
        result_c, result_d = c(self.images, self.valid), d(self.images, self.valid)
        for key in ("mask_logits", "object_logits", "quality_logits"):
            torch.testing.assert_close(result_c[key], result_d[key], rtol=0, atol=0)
        for key in ("path_attention", "path_positions", "path_coverage", "descriptor"):
            torch.testing.assert_close(result_c["aux"][key], result_d["aux"][key], rtol=0, atol=0)

    def test_c_mask_loss_trains_tracker_descriptor_condition_and_decoder(self):
        model = make_model("C")
        model(self.images, self.valid)["mask_logits"].square().mean().backward()
        self.assertTrue(has_gradient(model.tracker), "C tracker is detached from mask loss")
        self.assertTrue(has_gradient(model.descriptor), "C descriptor is detached from mask loss")
        self.assertTrue(has_gradient(model.condition))
        self.assertTrue(has_gradient(model.base.decoder))
        self.assertTrue(all(p.grad is None for p in model.base.encoder.parameters()))

    def test_d_mask_loss_stops_tracker_and_descriptor_only(self):
        model = make_model("D")
        result = model(self.images, self.valid)
        result["mask_logits"].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.tracker.parameters()))
        self.assertTrue(all(p.grad is None for p in model.descriptor.parameters()))
        self.assertTrue(has_gradient(model.condition), "D must preserve the automatic condition branch")
        self.assertTrue(has_gradient(model.base.decoder), "D must still train the segmentation head")
        self.assertTrue(all(p.grad is None for p in model.base.encoder.parameters()))

    def test_d_auxiliary_route_remains_differentiable(self):
        model = make_model("D")
        result = model(self.images, self.valid)
        # A differentiable route objective, independent of final segmentation.
        result["aux"]["path_attention"].square().sum().backward()
        self.assertTrue(has_gradient(model.tracker))
        self.assertTrue(has_gradient(model.descriptor))
        self.assertTrue(all(p.grad is None for p in model.base.decoder.parameters()))

    def test_shared_initial_components_match_across_ablation_variants(self):
        reference = make_model("C").state_dict()
        for variant in "ABD":
            model = make_model(variant)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, reference[key], rtol=0, atol=0)

    def test_model_construction_does_not_change_cpu_random_stream(self):
        torch.manual_seed(987)
        before = torch.random.get_rng_state().clone()
        make_model("C")
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)

    def test_batch_items_are_independent(self):
        model = make_model("C").eval()
        together = model(self.images, self.valid)
        for index in range(2):
            alone = model(self.images[index:index+1], self.valid[index:index+1])
            for key in ("mask_logits", "object_logits", "quality_logits"):
                torch.testing.assert_close(
                    together[key][index:index+1], alone[key], rtol=2e-5, atol=3e-6
                )

    def test_native_shape_rejected_instead_of_implicitly_resized(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, "natively padded"):
            model(torch.rand(1, 3, 30, 32))
        with self.assertRaisesRegex(ValueError, "valid"):
            model(self.images, self.valid[:, 0])
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            model(self.images, torch.zeros_like(self.valid))
        self.assertEqual(len(model.base.encoder._forward_pre_hooks), 0)

    def test_ground_truth_is_not_a_forward_argument(self):
        with self.assertRaises(TypeError):
            make_model()(self.images, self.valid, masks=torch.ones(2, 3, 32, 32))

    def test_path_attention_excludes_invalid_spatial_cells(self):
        model = make_model("C").eval()
        result = model(self.images, self.valid)
        attention = result["aux"]["path_attention"]
        valid_low = torch.nn.functional.adaptive_max_pool2d(
            self.valid.float(), attention.shape[-2:]
        )[:, 0].bool()
        excluded = ~valid_low[:, None, None].expand_as(attention)
        self.assertEqual(int(torch.count_nonzero(attention[excluded])), 0)
        torch.testing.assert_close(attention.sum((-2, -1)), torch.ones_like(attention.sum((-2, -1))))

    def test_checkpoint_delta_strict_round_trip_and_rejection(self):
        original, restored = make_model("C").eval(), make_model("C").eval()
        with torch.no_grad():
            next(original.condition.parameters()).add_(.125)
        state = original.checkpoint_state_dict()
        restored.load_trainable_state_dict(state)
        torch.testing.assert_close(
            original(self.images, self.valid)["mask_logits"],
            restored(self.images, self.valid)["mask_logits"], rtol=0, atol=0
        )
        broken = dict(state)
        broken.pop(next(iter(broken)))
        with self.assertRaisesRegex(ValueError, "delta mismatch"):
            restored.load_trainable_state_dict(broken)
        broken = dict(state)
        broken["unrecognized_tensor"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "delta mismatch"):
            restored.load_trainable_state_dict(broken)
        broken = dict(state)
        key = next(key for key, value in broken.items() if value.ndim > 0)
        broken[key] = broken[key][0:0]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            restored.load_trainable_state_dict(broken)


class StrictPretrainedAdmission(unittest.TestCase):
    """Exercise the real loader on tiny retained modules; no large checkpoint."""

    @staticmethod
    def tiny_branch():
        branch = Sam31Backbone.__new__(Sam31Backbone)
        nn.Module.__init__(branch)
        branch.trunk = nn.Linear(3, 4)
        branch.neck = nn.ModuleList([nn.Linear(4, 4)])
        branch.prompt_encoder = nn.Linear(4, 4)
        branch.decoder = nn.Linear(4, 4)
        branch.no_mem = nn.Parameter(torch.zeros(1, 1, 256), requires_grad=False)
        return branch

    @staticmethod
    def source_state(branch):
        prefixes = {
            "trunk.": "sam3_model.backbone.vision_backbone.trunk.",
            "neck.": "sam3_model.backbone.vision_backbone.interactive_convs.",
            "prompt_encoder.": "sam2_predictor.model.interactive_sam_prompt_encoder.",
            "decoder.": "sam2_predictor.model.interactive_sam_mask_decoder.",
        }
        source = {}
        for name, tensor in branch.state_dict().items():
            remote = "sam2_predictor.model.interactivity_no_mem_embed" if name == "no_mem" else None
            for local, prefix in prefixes.items():
                if name.startswith(local):
                    remote = prefix + name[len(local):]
                    break
            assert remote is not None
            source[remote] = torch.full_like(tensor, .375)
        return source

    def load(self, branch, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model": source}, path)
            return branch._load_checkpoint(path)

    def test_exact_retained_tensors_load_from_original_prefixes(self):
        branch = self.tiny_branch()
        receipt = self.load(branch, self.source_state(branch))
        self.assertTrue(receipt["strict_retained_state"])
        for tensor in branch.state_dict().values():
            torch.testing.assert_close(tensor, torch.full_like(tensor, .375), rtol=0, atol=0)

    def test_missing_tensor_is_not_silently_random(self):
        branch = self.tiny_branch()
        source = self.source_state(branch)
        source.pop(next(iter(source)))
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load(branch, source)

    def test_wrong_shape_is_rejected(self):
        branch = self.tiny_branch()
        source = self.source_state(branch)
        key = next(iter(source))
        source[key] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            self.load(branch, source)

    def test_normalization_collision_is_rejected(self):
        branch = self.tiny_branch()
        source = self.source_state(branch)
        source["tracker.model.interactivity_no_mem_embed"] = source["sam2_predictor.model.interactivity_no_mem_embed"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(branch, source)

    def test_frozen_native_components_remain_eval_in_training_mode(self):
        branch = self.tiny_branch()
        branch.train()
        self.assertFalse(branch.trunk.training)
        self.assertFalse(branch.neck.training)
        self.assertFalse(branch.prompt_encoder.training)
        self.assertTrue(branch.decoder.training)

    @staticmethod
    def branch_with_geometry_buffer():
        class GeometryBufferDouble(nn.Module):
            # Test the audit protocol without importing the external SAM repo.
            # The actual native formula is checked again by the real loader.
            def __init__(self):
                super().__init__()
                self.input_size = (3, 3)
                self.rope_pt_size = (2, 2)
                self.unstable = False
                self._setup_rope_freqs()

            def _setup_rope_freqs(self):
                phase = torch.arange(18, dtype=torch.float32).reshape(9, 2) / 3
                if self.unstable:
                    phase = phase + .1
                self.register_buffer("freqs_cis", torch.polar(torch.ones_like(phase), phase))

        branch = StrictPretrainedAdmission.tiny_branch()
        branch.trunk = nn.Module()
        block = nn.Module()
        block.attn = GeometryBufferDouble()
        branch.trunk.blocks = nn.ModuleList([block])
        return branch

    def test_rope_checkpoint_buffer_is_ignored_and_new_geometry_is_audited(self):
        branch = self.branch_with_geometry_buffer()
        before = branch.trunk.blocks[0].attn.freqs_cis.clone()
        source = self.source_state(branch)
        key = "sam3_model.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis"
        source[key] = torch.zeros(4, 2, dtype=torch.complex64)  # a different canvas
        receipt = self.load(branch, source)
        torch.testing.assert_close(branch.trunk.blocks[0].attn.freqs_cis, before, rtol=0, atol=0)
        audit = receipt["rope_buffer_audit"][0]
        self.assertEqual(audit["shape"], [9, 2])
        self.assertEqual(audit["dtype"], "torch.complex64")
        self.assertEqual(audit["sha256"], hashlib.sha256(before.numpy().tobytes()).hexdigest())
        self.assertEqual(audit["input_size"], [3, 3])
        self.assertTrue(audit["native_recompute_exact"])
        self.assertTrue(audit["checkpoint_buffer_ignored"])
        self.assertEqual(audit["checkpoint_shape"], [4, 2])

    def test_missing_derived_rope_is_explicit_in_receipt(self):
        branch = self.branch_with_geometry_buffer()
        source = self.source_state(branch)
        source.pop("sam3_model.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis")
        audit = self.load(branch, source)["rope_buffer_audit"][0]
        self.assertFalse(audit["checkpoint_tensor_present"])
        self.assertIsNone(audit["checkpoint_shape"])

    def test_nonreproducible_geometry_buffer_is_rejected(self):
        branch = self.branch_with_geometry_buffer()
        source = self.source_state(branch)
        branch.trunk.blocks[0].attn.unstable = True
        with self.assertRaisesRegex(RuntimeError, "RoPE recomputation mismatch"):
            self.load(branch, source)


if __name__ == "__main__":
    unittest.main()

