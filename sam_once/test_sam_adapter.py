"""CPU contract tests: no checkpoint loading or GPU computation."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn

from sam_once.sam_adapter import NativeSamAdapter, SamInferenceError
from sam_once.types import CropSpec, PromptSpec, PromptPlan


class FakeTrunk(nn.Module):
    def __init__(self, oom=False):
        super().__init__()
        self.oom = oom

    def forward(self, image):
        if self.oom:
            raise torch.cuda.OutOfMemoryError("injected OOM")
        return image.mean(1, keepdim=True)


class FakeBackbone(nn.Module):
    def __init__(self, oom=False):
        super().__init__()
        self.vision_backbone = nn.Module()
        self.vision_backbone.trunk = FakeTrunk(oom)
        self.language_backbone = nn.Identity()

    def forward_image(self, images, **kwargs):
        if kwargs != dict(need_sam3_out=False, need_interactive_out=True, need_propagation_out=False):
            raise AssertionError("Wrong branch selection")
        out = self.vision_backbone.trunk(images)
        return {"interactive": {"backbone_fpn": [SimpleNamespace(tensors=out) for _ in range(3)]}}


class FakePromptEncoder(nn.Module):
    def forward(self, points, boxes, masks):
        coords, labels = points
        self.last_labels = labels.clone()
        # Deliberately give -1 learned-token padding nonzero weight.
        token = coords.sum((1, 2)) * 0.001 + (labels.float()+2).sum(1) * 0.01
        return token[:, None, None], token[:, None, None, None]

    def get_dense_pe(self):
        return torch.zeros(1, 1, 8, 8)


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s0, self.conv_s1 = nn.Identity(), nn.Identity()

    def forward(self, *, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, repeat_image, high_res_features):
        assert not repeat_image and not multimask_output
        assert len(image_embeddings) == len(sparse_prompt_embeddings)
        mask = image_embeddings + dense_prompt_embeddings + high_res_features[0] * .1
        n = len(mask)
        return mask, torch.full((n, 1), .9), torch.zeros(n, 1, 1), torch.ones(n, 1)


class FakeModel(nn.Module):
    def __init__(self, oom=False):
        super().__init__()
        self.detector = nn.Module()
        self.detector.backbone = FakeBackbone(oom)
        self.tracker = nn.Module()
        self.tracker.interactive_sam_prompt_encoder = FakePromptEncoder()
        self.tracker.interactive_sam_mask_decoder = FakeDecoder()
        self.tracker.sam_mask_decoder = nn.Identity()
        self.tracker.directly_add_no_mem_embed = True
        self.tracker.interactivity_no_mem_embed = nn.Parameter(torch.ones(1, 1, 1)*.2)
        self.tracker.pred_obj_scores = True
        self.tracker.object_score_logit_threshold = 0.


def make_adapter(model=None, **kwargs):
    return NativeSamAdapter("does-not-exist", "does-not-exist", "cpu", 3, 5,
                            model=model if model is not None else FakeModel(), canvas_size=8, **kwargs)


def sample():
    rgb = np.arange(9*11*3, dtype=np.uint8).reshape(9, 11, 3)
    crops = [CropSpec("a", (0, 0, 8, 8)), CropSpec("b", (6, 5, 11, 9))]
    prompts = [PromptSpec("p", "a", ((2., 3.),)),
               PromptSpec("q", "b", ((1., 1.), (2., 2.)), ((3., 0.),), (0., 0., 4., 3.))]
    return rgb, PromptPlan(crops, prompts, np.zeros(rgb.shape[:2], np.float32))


class AdapterTests(unittest.TestCase):
    def test_only_confirmed_native_derived_rope_buffer_may_be_missing(self):
        class Attention(nn.Module):
            __module__ = "sam3.model.vitdet"
            def __init__(self):
                super().__init__()
                self.use_rope, self.use_rope_real, self.use_ve_rope = True, False, False
                self.weight = nn.Parameter(torch.zeros(1))
                self._setup_rope_freqs()
            def _setup_rope_freqs(self):
                self.register_buffer("freqs_cis", torch.ones(2, 2, dtype=torch.complex64))
        model = nn.Module()
        owner = model
        for name in ("detector", "backbone", "vision_backbone", "trunk"):
            setattr(owner, name, nn.Module())
            owner = getattr(owner, name)
        block = nn.Module()
        block.attn = Attention()
        owner.blocks = nn.ModuleList([block])
        key = "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis"
        state = {name: value.clone() for name, value in model.state_dict().items() if name != key}
        receipt = NativeSamAdapter._load_exact_cpu_state(model, state)
        self.assertEqual(receipt["native_initialized_derived_buffers"], [key])
        self.assertEqual(receipt["checkpoint_state_tensors"], 1)
        self.assertEqual(receipt["omitted_checkpoint_tensors"], 0)
        with self.assertRaisesRegex(RuntimeError, "missing checkpoint keys"):
            NativeSamAdapter._load_exact_cpu_state(model, {})
        block.attn.freqs_cis.mul_(2)
        with self.assertRaisesRegex(RuntimeError, "not reproducible"):
            NativeSamAdapter._load_exact_cpu_state(model, state)
        block.attn.register_buffer("learned_position", torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, "missing checkpoint keys"):
            NativeSamAdapter._load_exact_cpu_state(model, state)
        del block.attn.learned_position
        del block.attn.freqs_cis
        block.attn.freqs_cis = nn.Parameter(torch.ones(2, 2, dtype=torch.complex64))
        with self.assertRaisesRegex(RuntimeError, "missing checkpoint keys"):
            NativeSamAdapter._load_exact_cpu_state(model, state)

    def test_interactive_residency_places_only_selected_native_modules(self):
        adapter = make_adapter(interactive_only_residency=True)
        visual = adapter.model.detector.backbone.vision_backbone
        visual.interactive_convs = nn.Linear(1, 1)
        visual.convs, visual.propagation_convs = nn.Linear(1, 1), nn.Linear(1, 1)
        visual.position_encoding = nn.Module()
        visual.position_encoding.cache = {(2, 2): torch.ones(1, 2, 2)}
        adapter.tracker.interactive_obj_ptr_proj = nn.Linear(1, 1)
        adapter.tracker.no_obj_ptr_linear = nn.Linear(1, 1)
        selected = (visual.trunk, visual.interactive_convs,
                    adapter.tracker.interactive_sam_prompt_encoder,
                    adapter.tracker.interactive_sam_mask_decoder,
                    adapter.tracker.interactive_obj_ptr_proj, adapter.tracker.no_obj_ptr_linear)
        untouched = (visual.convs, visual.propagation_convs,
                     adapter.model.detector.backbone.language_backbone,
                     adapter.tracker.sam_mask_decoder)
        spies = [mock.patch.object(module, "to", wraps=module.to) for module in selected+untouched]
        started = [spy.start() for spy in spies]
        try:
            receipt = adapter._place_interactive_components(adapter.model)
            self.assertEqual([spy.call_count for spy in started], [1]*6+[0]*4)
        finally:
            for spy in spies:
                spy.stop()
        self.assertGreater(receipt["used_parameter_bytes"], 0)
        self.assertEqual(receipt["gpu_parameter_bytes"], 0)
        self.assertEqual(len(receipt["position_caches"]), 1)
        rgb, plan = sample()
        _, audit = adapter.predict(rgb, plan)
        self.assertEqual(audit["residency_policy"], "experimental_interactive_only")
        self.assertEqual(audit["calls"]["image_encoder"], 1)
        self.assertEqual(audit["calls"]["mask_decoder"], 1)

    def test_strict_cpu_state_checks_missing_shape_and_values(self):
        model = nn.Linear(2, 1)
        state = {"weight": torch.ones(1, 2, dtype=torch.float16),
                 "bias": torch.ones(1, dtype=torch.float16), "omitted_text": torch.zeros(1)}
        receipt = NativeSamAdapter._load_exact_cpu_state(model, state)
        self.assertTrue(receipt["strict_keys_shapes_values"])
        self.assertEqual(receipt["omitted_checkpoint_tensors"], 1)
        self.assertTrue(torch.equal(model.weight, torch.ones(1, 2)))
        with self.assertRaisesRegex(RuntimeError, "missing checkpoint keys"):
            NativeSamAdapter._load_exact_cpu_state(model, {"bias": state["bias"]})
        with self.assertRaises(RuntimeError):
            NativeSamAdapter._load_exact_cpu_state(model, {**state, "weight": torch.zeros(1, 3)})
        with mock.patch.object(model, "load_state_dict"):
            with self.assertRaisesRegex(RuntimeError, "value mismatch"):
                NativeSamAdapter._load_exact_cpu_state(model, {**state, "weight": torch.zeros(1, 2)})

    def test_interactive_builder_uses_cpu_native_modules_and_full_checkpoint(self):
        native = FakeModel().tracker
        native.backbone = FakeBackbone()
        native.backbone.vision_backbone.interactive_convs = nn.Linear(1, 1)
        # Match the native full-predictor checkpoint namespace without constructing it.
        state = {"tracker.model."+key: value for key, value in native.state_dict().items()
                 if not key.startswith("backbone.")}
        state.update({"detector.backbone.vision_backbone."+key: value
                      for key, value in native.backbone.vision_backbone.state_dict().items()})
        def vl_wrapper(*, visual, text, scalp):
            self.assertIsNone(text)
            self.assertEqual(scalp, 0)
            result = FakeBackbone()
            result.vision_backbone, result.language_backbone = visual, text
            return result
        builder = mock.Mock(return_value=native)
        imports = {"sam3.model_builder": SimpleNamespace(build_sam3_multiplex_video_model=builder),
                   "sam3.model.vl_combiner": SimpleNamespace(SAM3VLBackboneTri=vl_wrapper)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"native.pt"
            torch.save({"model": state}, path)
            adapter = NativeSamAdapter(path, "unused", "cpu", 1, 1,
                                       interactive_only_residency=True)
            with mock.patch.dict(sys.modules, imports):
                holder = adapter._build_interactive_only()
        builder.assert_called_once_with(checkpoint_path=None, load_from_HF=False, device="cpu",
                                        use_fa3=False, use_rope_real=False, compile=False)
        self.assertIs(holder.tracker.model, native)
        self.assertIsNone(native.backbone)
        self.assertTrue(adapter._weight_receipt["strict_keys_shapes_values"])
        self.assertEqual(adapter._weight_receipt["omitted_checkpoint_tensors"], 0)

    def test_true_counts_and_native_shapes(self):
        rgb, plan = sample()
        adapter = make_adapter()
        records, audit = adapter.predict(rgb, plan)
        self.assertEqual([r.initial_logits.shape for r in records], [(8, 8), (4, 5)])
        self.assertEqual(audit["calls"], dict(backbone_forward_image=1, image_encoder=1,
            prompt_encoder=1, mask_decoder=1, detector=0, propagation_decoder=0, text_encoder=0))
        self.assertEqual(audit["retries"], 0)
        self.assertFalse(any(p.requires_grad for p in adapter.model.parameters()))
        self.assertFalse(np.shares_memory(records[0].initial_logits, records[0].current_logits))
        self.assertEqual(adapter.tracker.interactive_sam_prompt_encoder.last_labels[1].tolist(), [1,1,0,2,3,-1,-1,-1])

    def test_permutation_and_different_length_isolation(self):
        rgb, plan = sample()
        adapter = make_adapter()
        original, _ = adapter.predict(rgb, plan)
        permuted, _ = adapter.predict(rgb, PromptPlan(plan.crops[::-1], plan.prompts[::-1], plan.ridge))
        isolated, _ = adapter.predict(rgb, PromptPlan(plan.crops[:1], plan.prompts[:1], plan.ridge))
        np.testing.assert_array_equal(original[0].initial_logits, permuted[1].initial_logits)
        np.testing.assert_array_equal(original[1].initial_logits, permuted[0].initial_logits)
        np.testing.assert_array_equal(original[0].initial_logits, isolated[0].initial_logits)

    def test_empty_does_not_load_native_model(self):
        rgb, plan = sample()
        adapter = NativeSamAdapter("missing", "missing", "cpu", 3, 5)
        records, audit = adapter.predict(rgb, PromptPlan(plan.crops, [], plan.ridge))
        self.assertEqual(records, [])
        self.assertIsNone(adapter.model)
        self.assertFalse(any(audit["calls"].values()))

    def test_over_budget_rejected_before_loading(self):
        rgb, plan = sample()
        adapter = NativeSamAdapter("missing", "missing", "cpu", 1, 1)
        with self.assertRaises(SamInferenceError):
            adapter.predict(rgb, plan)
        self.assertFalse(any(adapter.last_audit["calls"].values()))
        self.assertIsNone(adapter.model)

    def test_oom_is_not_retried_and_hooks_removed(self):
        rgb, plan = sample()
        adapter = make_adapter(FakeModel(oom=True))
        with self.assertRaises(SamInferenceError):
            adapter.predict(rgb, plan)
        self.assertEqual(adapter.last_audit["status"], "oom_no_retry")
        self.assertEqual(adapter.last_audit["calls"]["image_encoder"], 1)
        self.assertEqual(adapter.last_audit["calls"]["prompt_encoder"], 0)
        self.assertEqual(adapter.last_audit["retries"], 0)
        self.assertFalse(adapter.model.detector.backbone.vision_backbone.trunk._forward_pre_hooks)

    def test_conflicting_points_and_too_many_tokens(self):
        rgb, plan = sample()
        adapter = make_adapter(prompt_slots=3)
        with self.assertRaises(SamInferenceError):
            adapter.predict(rgb, plan)
        self.assertFalse(any(adapter.last_audit["calls"].values()))
        prompt = PromptSpec("x", "a", ((1., 1.),), ((1., 1.),))
        with self.assertRaises(SamInferenceError):
            adapter.predict(rgb, PromptPlan(plan.crops, [prompt], plan.ridge))

    def test_reject_resized_crop(self):
        rgb, plan = sample()
        crop = CropSpec("large", (0, 0, 11, 9))
        prompt = PromptSpec("x", "large", ((1., 1.),))
        with self.assertRaises(SamInferenceError):
            make_adapter().predict(rgb, PromptPlan([crop], [prompt], plan.ridge))

    def test_normalization_and_padding(self):
        rgb = np.zeros((2, 3, 3), np.uint8)
        crop = CropSpec("c", (0, 0, 3, 2))
        batch = make_adapter().image_batch(rgb, [crop])
        self.assertTrue(batch.is_contiguous())
        self.assertEqual(float(batch[0, 0, 0, 0]), -1.)
        expected = (torch.tensor(127/255, dtype=torch.float16)-.5)/.5
        self.assertEqual(float(batch[0, 0, 7, 7]), float(expected))

    def test_hidden_second_encoder_call_is_rejected(self):
        rgb, plan = sample()
        model = FakeModel()
        backbone = model.detector.backbone
        original = backbone.forward_image
        def repeated(images, **kwargs):
            original(images, **kwargs)
            return original(images, **kwargs)
        backbone.forward_image = repeated
        adapter = make_adapter(model)
        with self.assertRaises(SamInferenceError):
            adapter.predict(rgb, plan)
        self.assertEqual(adapter.last_audit["calls"]["image_encoder"], 2)
        self.assertIs(backbone.forward_image, repeated)

    def test_forbidden_propagation_decoder_is_rejected(self):
        rgb, plan = sample()
        adapter = make_adapter()
        original = adapter._decode
        def forbidden(*args):
            adapter.tracker.sam_mask_decoder(torch.zeros(1))
            return original(*args)
        adapter._decode = forbidden
        with self.assertRaises(SamInferenceError):
            adapter.predict(rgb, plan)
        self.assertEqual(adapter.last_audit["calls"]["propagation_decoder"], 1)
        self.assertEqual(adapter.last_audit["calls"]["mask_decoder"], 0)


def native_cpu_state_check(repo, checkpoint):
    """Optional real checkpoint audit: CUDA must be hidden before this process.

    Only the unregistered position-cache precompute is disabled for this CPU
    contract test. All native parameters and persistent state are constructed.
    No model forward is called, and production builder behavior is untouched.
    """
    import json
    if torch.cuda.is_available():
        raise RuntimeError("Hide CUDA with CUDA_VISIBLE_DEVICES=-1 for this CPU-only audit")
    sys.path.insert(0, str(Path(repo).resolve()))
    import sam3.model_builder as builder
    original_position_class = builder.PositionEmbeddingSine
    adapter = NativeSamAdapter(checkpoint, repo, "cpu", 1, 1, interactive_only_residency=True)
    def cpu_position_cache(*args, **kwargs):
        kwargs["precompute_resolution"] = None
        return original_position_class(*args, **kwargs)
    with mock.patch.object(builder, "PositionEmbeddingSine", side_effect=cpu_position_cache):
        holder = adapter._build_interactive_only()
    assert all(p.device.type == "cpu" for p in holder.parameters())
    assert not torch.cuda.is_initialized(), "CPU contract audit initialized CUDA unexpectedly"
    print(json.dumps({"status": "passed_cpu_native_state_contract", "sam_forward_calls": 0,
                      "cuda_initialized": False, "weight_coverage": adapter._weight_receipt,
                      "residency": adapter._residency_receipt}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--native-cpu-state-check":
        native_cpu_state_check(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
