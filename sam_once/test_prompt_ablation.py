"""CPU tests for frozen prompt conditions and independent run diagnostics."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sam_once.prompt_ablation import (derive_plan, frozen_pool_fingerprint, prompt_input_fingerprint,
                                      summarize_plan, summarize_run)
from sam_once.types import CropSpec, PromptPlan, PromptSpec, ProposalRecord


def fixture():
    crop = CropSpec("crop", (0, 0, 10, 8))
    prompts = []
    for index, pool in enumerate(([(2., 2.), (4., 2.), (6., 2.)], [(3., 5.)])):
        metadata = {"path_key": f"path{index}", "candidate_id": index, "fragment_id": index,
                    "positive_pool": pool, "positive_pool_arc_fractions": [.2, .5, .8] if index == 0 else [.5],
                    "negative_pools": {"background": {"points": [(8., 1.), (8., 6.)], "evidence": [{}, {}]},
                                       "neighbor": {"points": [(5., 4.)] if index == 0 else [], "evidence": [{}]}},
                    "box_local": [1., 1., 8., 7.], "box_expanded": [0., 0., 10., 8.],
                    "instance_identity_verified": False}
        prompts.append(PromptSpec(f"p{index}", crop.crop_id, tuple(pool), metadata=metadata))
    return PromptPlan([crop], prompts, np.zeros((8, 10), np.float32), {"method": "v2", "sam_calls": 0})


def audit(plan):
    entered = int(bool(plan.prompts))
    return {"status": "ok" if entered else "empty_prompts", "retries": 0, "microbatches": entered,
            "requested_crops": len(plan.crops), "requested_prompts": len(plan.prompts),
            "calls": dict(backbone_forward_image=entered, image_encoder=entered, prompt_encoder=entered,
                          mask_decoder=entered, detector=0, propagation_decoder=0, text_encoder=0)}


def records(plan):
    result = []
    for i, prompt in enumerate(plan.prompts):
        logits = np.full((8, 10), -1., np.float32)
        logits[2, 2:7] = 1.
        result.append(ProposalRecord(prompt, plan.crops[0], logits, logits.copy(), -.2 if i == 0 else .7))
    return result


class AblationTests(unittest.TestCase):
    def test_identical_actual_inputs_detected_before_sam_despite_policy_names(self):
        base = fixture()
        one = derive_plan(base, 2, 1, "neighbor", "none")
        two = derive_plan(base, 2, 2, "neighbor", "none")
        self.assertEqual(prompt_input_fingerprint(one), prompt_input_fingerprint(two))
        changed = derive_plan(base, 2, 1, "background", "none")
        self.assertNotEqual(prompt_input_fingerprint(one), prompt_input_fingerprint(changed))

    def test_point_schedules_preserve_selected_groups_and_base_plan(self):
        base = fixture()
        before = deepcopy(base.metadata())
        fingerprint = frozen_pool_fingerprint(base)
        for positive, negative in ((1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (2, 2)):
            derived = derive_plan(base, positive, negative, "neighbor", "none")
            self.assertEqual([p.prompt_id for p in derived.prompts], ["p0", "p1"])
            self.assertEqual([p.metadata["candidate_id"] for p in derived.prompts], [0, 1])
            self.assertEqual(frozen_pool_fingerprint(derived), fingerprint)
            self.assertEqual(len(derived.prompts[1].positive), 1)
            self.assertEqual(len(derived.prompts[1].negative), 0)
            self.assertFalse(np.shares_memory(derived.ridge, base.ridge))
        self.assertEqual(base.metadata(), before)

    def test_midpoint_endpoints_and_insufficient_samples(self):
        self.assertEqual(derive_plan(fixture(), 1, 0).prompts[0].positive, ((4., 2.),))
        two = derive_plan(fixture(), 2, 2)
        self.assertEqual(two.prompts[0].positive, ((2., 2.), (6., 2.)))
        summary = summarize_plan(two)
        self.assertEqual(summary["positive_count_distribution"], {"1": 1, "2": 1})
        self.assertEqual(summary["negative_count_distribution"], {"0": 1, "1": 1})
        self.assertEqual(summary["positive_count_shortfall_groups"], 1)
        self.assertEqual(summary["negative_count_shortfall_groups"], 2)

    def test_box_modes_and_negative_modes_use_only_frozen_values(self):
        base = fixture()
        for mode, expected in (("none", None), ("local", (1., 1., 8., 7.)),
                               ("expanded", (0., 0., 10., 8.))):
            derived = derive_plan(base, 2, 1, "background", mode)
            self.assertEqual(derived.prompts[0].box, expected)
            self.assertEqual(derived.prompts[0].negative, ((8., 1.),))
        del base.prompts[0].metadata["negative_pools"]["neighbor"]
        with self.assertRaises(ValueError):
            derive_plan(base, 2, 1, "neighbor")

    def test_refuses_legacy_metadata_and_invalid_or_duplicate_points(self):
        base = fixture()
        del base.prompts[0].metadata["positive_pool"]
        with self.assertRaises(ValueError):
            derive_plan(base, 2, 0)
        base = fixture()
        base.prompts[0].metadata["negative_pools"]["neighbor"]["points"] = [(2., 2.)]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            derive_plan(base, 2, 1)
        base = fixture()
        base.prompts[0].metadata["positive_pool"] = [(2., 2.)]*3
        with self.assertRaisesRegex(ValueError, "Invalid frozen positive"):
            derive_plan(base, 3, 0)

    def test_token_budget_is_unchanged_and_enforced(self):
        with self.assertRaisesRegex(ValueError, "token budget"):
            derive_plan(fixture(), 3, 2, "background", "local", prompt_slots=6)
        derived = derive_plan(fixture(), 3, 2, "background", "local")
        self.assertEqual(derived.prompts[0].metadata["sparse_tokens"], 7)
        self.assertEqual(summarize_plan(derived)["native_extra_padding_tokens_per_group"], 1)

    def test_raw_native_coverage_negative_scores_and_group_counts(self):
        plan = derive_plan(fixture(), 2, 1)
        result = summarize_run(plan, records(plan), audit(plan))
        self.assertTrue(result["validation"]["passed"])
        self.assertEqual(result["score"]["negative_count"], 1)
        self.assertEqual(result["per_group"][0]["backend_clipped_score_component"], 0.)
        self.assertEqual(result["native_mask_coverage"]["foreground_pixels"], 5)
        self.assertEqual(result["native_mask_coverage"]["overlap_pixels"], 5)
        self.assertEqual(result["native_mask_coverage"]["native_size"], [10, 8])

    def test_repeated_calls_missing_records_and_refinement_fail_validation(self):
        plan = derive_plan(fixture(), 2, 0)
        bad = audit(plan)
        bad["calls"]["mask_decoder"] = 2
        self.assertFalse(summarize_run(plan, records(plan), bad)["validation"]["passed"])
        self.assertFalse(summarize_run(plan, records(plan)[:1], audit(plan))["validation"]["passed"])
        result_records = records(plan)
        result_records[0].current_logits[0, 0] = 1.
        self.assertFalse(summarize_run(plan, result_records, audit(plan))["validation"]["passed"])

    def test_diagnostic_helpers_do_not_change_inputs(self):
        plan = derive_plan(fixture(), 2, 0)
        supplied_audit = audit(plan)
        original_audit, original_plan = deepcopy(supplied_audit), deepcopy(plan.metadata())
        result = summarize_run(plan, records(plan), supplied_audit)
        self.assertEqual(supplied_audit, original_audit)
        self.assertEqual(plan.metadata(), original_plan)
        self.assertFalse(result["sam_feedback_used"])

    def test_empty_plan_has_zero_native_calls(self):
        plan = fixture()
        plan.prompts = []
        summary = summarize_run(plan, [], audit(plan))
        self.assertTrue(summary["validation"]["passed"])
        self.assertEqual(summary["native_mask_coverage"]["foreground_pixels"], 0)


def native_prompt_contract(repo):
    """CPU-only check of original PromptEncoder math, without a model forward."""
    import json
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=-1 for this CPU-only check")
    sys.path.insert(0, str(Path(repo).resolve()))
    from sam3.sam.prompt_encoder import PromptEncoder
    encoder = PromptEncoder(embed_dim=32, image_embedding_size=(2, 2),
                            input_image_size=(32, 32), mask_in_chans=16).eval()
    boxes = torch.tensor([[2., 3., 20., 21.]])
    corners = encoder._embed_points(boxes.reshape(1, 2, 2), torch.tensor([[2, 3]]), pad=False)
    assert torch.equal(corners, encoder._embed_boxes(boxes))
    coords = torch.zeros(2, 8, 2)
    labels = torch.full((2, 8), -1, dtype=torch.int32)
    labels[0, :1], labels[1, :3] = 1, 1
    batched, _ = encoder((coords, labels), boxes=None, masks=None)
    single, _ = encoder((coords[:1], labels[:1]), boxes=None, masks=None)
    assert batched.shape == (2, 9, 32)
    assert torch.equal(batched[:1], single)
    assert torch.equal(batched[0, 1], encoder.not_a_point_embed.weight[0])
    assert not torch.cuda.is_initialized()
    print(json.dumps({"box_corner_embedding_exact": True, "fixed_padding_embedding_batch_isolated": True,
                      "sparse_slots_including_native_extra": 9, "minus_one_is_learned_token": True,
                      "cuda_initialized": False, "scope": "CPU prompt encoder only; no SAM image/mask forward"}))


def v2_freeze_contract(image):
    """Plan the real RGB once; derive every condition before any SAM execution."""
    import json
    from PIL import Image
    from sam_once.planner_v2 import plan_prompts, V2Config
    rgb = np.asarray(Image.open(image).convert("RGB"))
    base = plan_prompts(rgb, V2Config())
    specs = [(p, n, "neighbor" if n else "none", "none")
             for p, n in ((1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (2, 2))]
    specs += [(2, 1, "background", box) for box in ("none", "local", "expanded")]
    variants = [derive_plan(base, *spec) for spec in specs]
    assert len({frozen_pool_fingerprint(plan) for plan in variants}) == 1
    print(json.dumps({"status": "passed_real_v2_frozen_plan_contract", "sam_calls": 0,
        "group_count": len(base.prompts), "variant_count": len(variants),
        "unique_actual_inputs": len({prompt_input_fingerprint(plan) for plan in variants}),
        "variants": [{"condition": spec, "positive_distribution": summarize_plan(plan)["positive_count_distribution"],
                      "negative_distribution": summarize_plan(plan)["negative_count_distribution"]}
                     for spec, plan in zip(specs, variants)]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--native-prompt-contract":
        native_prompt_contract(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--v2-freeze-contract":
        v2_freeze_contract(sys.argv[2])
    else:
        unittest.main(verbosity=2)
