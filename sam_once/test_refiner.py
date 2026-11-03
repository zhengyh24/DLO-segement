"""CPU-only invariants, independent from native SAM feasibility gates."""
from __future__ import annotations

import json
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from PIL import Image

from .types import CropSpec, PromptSpec, ProposalRecord, PromptPlan
from .refiner import (ResidualRefiner, RefinerConfig, ROI, refine_proposals, extract_roi,
                      schedule_rois, proposal_features, load_refiner)
from .train import (match_targets, valid_bce_dice, validate_gate_receipt,
                    load_training_manifest, _checkpoint, validate_native_cache, LazyCaseReader,
                    rollout_case, main as train_main, _fixture as training_fixture)


def fixture(size=256):
    rgb = np.full((size, size, 3), 80, np.uint8)
    ridge = np.zeros((size, size), np.float32)
    ridge[:, size//2] = 1
    crop = CropSpec("c0", (0, 0, size, size))
    prompt = PromptSpec("p0", "c0", ((size/2, size/2),), ((2., 2.),), (1., 1., size-1., size-1.))
    logits = np.full((size, size), -2.0, np.float32)
    logits[:, size//2-3:size//2+3] = 2
    p = ProposalRecord(prompt, crop, logits.copy(), logits.copy(), 0.9, {"original": "kept"})
    return rgb, ridge, [p]


class AddOne(nn.Module):
    def forward(self, x):
        return x[:, 4:5] * 8 + 1


def native_audit():
    sources = {"adapter": "b"*64}
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"cache_kind": "actual_single_batch_sam", "ground_truth_used_for_inference": False,
            "sam": {"status": "ok", "injected_test_model": False, "checkpoint_sha256": "a"*64,
                    "source_fingerprint": fingerprint, "source_files_sha256": sources,
                    "device": "cuda:0", "weight_coverage": {"used_parameters_found": 1, "missing": []},
                    "sam_version": "3.1_multiplex_interactive",
                    "calls": {"backbone_forward_image": 1, "image_encoder": 1, "prompt_encoder": 1,
                              "mask_decoder": 1, "detector": 0, "propagation_decoder": 0, "text_encoder": 0},
                    "microbatches": 1, "retries": 0, "input_resize": False, "canvas_size": 1008,
                    "requested_crops": 1, "requested_prompts": 1, "prompt_slots": 8,
                    "dtype": "torch.bfloat16", "encoder_dtype": "torch.bfloat16",
                    "decoder_dtype": "torch.float32", "precision_policy": "bf16_backbone_fp32_interactive_head",
                    "residency_policy": "native_full_predictor"}}


def stored_case(root, name, split, rgb_offset=0, audit=None, empty=False):
    from .io import save_case
    rgb, ridge, proposals = fixture()
    rgb = rgb + np.uint8(rgb_offset)
    source = root / (name + ".png")
    Image.fromarray(rgb).save(source)
    plan = PromptPlan([proposals[0].crop], [] if empty else [proposals[0].prompt], ridge)
    if empty:
        proposals = []
    save_case(root / name, rgb, plan, proposals, audit or native_audit())
    return {"case_id": name, "split": split, "cluster_id": name,
            "image_path": source.name, "cache_dir": name, "gt_masks": []}


class RefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(2, torch.get_num_threads()))

    def test_initial_identity_and_no_source_mutation(self):
        rgb, ridge, proposals = fixture()
        old = proposals[0].current_logits.copy()
        model = ResidualRefiner()
        out, audit = refine_proposals(rgb, proposals, ridge, model,
                                      RefinerConfig(rounds=1, max_visits=1, max_pixels=256**2))
        np.testing.assert_array_equal(out[0].current_logits, old)
        np.testing.assert_array_equal(proposals[0].current_logits, old)
        self.assertIsNot(out[0].initial_logits, proposals[0].initial_logits)
        self.assertEqual(audit["sam_calls"], 0)
        self.assertEqual(out[0].prompt.prompt_id, "p0")
        self.assertEqual(out[0].metadata, {"original": "kept", "refined": True, "refinement_visits": 1})
        self.assertEqual(proposals[0].metadata, {"original": "kept"})

    def test_untouched_roi_exact_and_original_retained(self):
        rgb, ridge, proposals = fixture(512)
        out, audit = refine_proposals(rgb, proposals, ridge, AddOne(),
                                      RefinerConfig(rounds=1, max_visits=1, max_pixels=256**2))
        changed = np.zeros((512, 512), bool)
        x0, y0, x1, y1 = audit["visits"][0]["xyxy"]
        changed[y0:y1, x0:x1] = True
        np.testing.assert_array_equal(out[0].current_logits[~changed], proposals[0].current_logits[~changed])
        np.testing.assert_allclose(out[0].current_logits[changed], proposals[0].current_logits[changed]+1)
        np.testing.assert_array_equal(out[0].initial_logits, proposals[0].initial_logits)

    def test_budget_and_equal_actual_pixels(self):
        rgb, ridge, proposals = fixture(768)
        results = []
        for mode in ("fixedsingle", "iterative"):
            _, audit = refine_proposals(rgb, proposals, ridge, AddOne(),
                                         RefinerConfig(max_visits=6, max_pixels=6*256**2, mode=mode))
            self.assertEqual(audit["total_visits"], 6)
            results.append(audit["total_model_pixels"])
        self.assertEqual(*results)
        _, audit = refine_proposals(rgb, proposals, ridge, AddOne(),
                                     RefinerConfig(max_visits=100, max_pixels=256**2-1))
        self.assertEqual(audit["total_visits"], 0)

    def test_native_small_crop_padding_and_512(self):
        rgb, ridge, proposals = fixture(137)
        features, valid = extract_roi(rgb, proposals, ridge, ROI(0, (0, 0, 137, 137), 1), 512)
        self.assertEqual(features.shape, (11, 512, 512))
        self.assertEqual(int(valid.sum()), 137*137)
        self.assertEqual(float(features[:, 137:, :].sum()), 0)
        np.testing.assert_allclose(features[:3, :137, :137], 80/255)

    def test_proposal_competition_never_subtracts(self):
        rgb, ridge, proposals = fixture()
        other = ProposalRecord(PromptSpec("p1", "c0", ((90., 90.),)), proposals[0].crop,
                               np.full((256, 256), 10, np.float32), np.full((256, 256), 10, np.float32), 1)
        features = proposal_features(rgb, proposals + [other], ridge, 0)
        np.testing.assert_array_equal(features[4]*8, proposals[0].current_logits)
        self.assertGreater(float(features[6].mean()), 0.99)

    def test_gt_anchor_positive_empty_ambiguous(self):
        rgb, ridge, proposals = fixture()
        first = np.zeros((256, 256), bool)
        first[:, 120:140] = True
        targets, counts = match_targets(proposals, [first], (256, 256))
        np.testing.assert_array_equal(targets[0], first)
        self.assertEqual(counts["positive"], 1)
        targets, counts = match_targets(proposals, [], (256, 256))
        self.assertEqual(targets[0].sum(), 0)
        targets, counts = match_targets(proposals, [first, first.copy()], (256, 256))
        self.assertIsNone(targets[0])
        self.assertEqual(counts["ambiguous"], 1)

    def test_scheduler_unchanged_by_label_matching(self):
        rgb, ridge, proposals = fixture(512)
        before = schedule_rois(rgb, proposals, ridge, RefinerConfig(), 3)
        match_targets(proposals, [np.ones((512, 512), bool)], (512, 512))
        self.assertEqual(before, schedule_rois(rgb, proposals, ridge, RefinerConfig(), 3))

    def test_padding_gradient_zero(self):
        logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
        target, valid = torch.ones_like(logits), torch.zeros_like(logits)
        valid[:, :, :7, :9] = 1
        valid_bce_dice(logits, target, valid).backward()
        self.assertEqual(float(logits.grad[:, :, 7:, :].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, :, :, 9:].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, :, :7, :9].abs().sum()), 0)

    def test_formal_gates_required(self):
        with self.assertRaises(ValueError):
            validate_gate_receipt(None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            path.write_text(json.dumps({"native_parity_pass": True, "capacity_24gb_pass": False}))
            with self.assertRaises(ValueError):
                validate_gate_receipt(path)
            path.write_text(json.dumps({"native_parity_pass": True, "capacity_24gb_pass": True}))
            with self.assertRaisesRegex(ValueError, "hash-bound"):
                validate_gate_receipt(path)

    def test_reject_duplicate_rgb_and_cluster_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [stored_case(root, "a", "train"), stored_case(root, "b", "validation")]
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": "sam_once_training_v1", "records": records}))
            with self.assertRaisesRegex(ValueError, "RGB leakage"):
                load_training_manifest(path)
            records[1]["cluster_id"] = "a"
            path.write_text(json.dumps({"schema": "sam_once_training_v1", "records": records}))
            with self.assertRaisesRegex(ValueError, "cluster leakage"):
                load_training_manifest(path)

    def test_source_rgb_must_match_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [stored_case(root, "a", "train"), stored_case(root, "b", "validation", 1)]
            Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(root / "b.png")
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": "sam_once_training_v1", "records": records}))
            with self.assertRaisesRegex(ValueError, "Source RGB does not match"):
                load_training_manifest(path)

    def test_mock_repeated_and_refined_caches_rejected(self):
        rgb, ridge, proposals = fixture()
        plan = PromptPlan([proposals[0].crop], [proposals[0].prompt], ridge)
        for mutation in (lambda a: a["sam"].update(injected_test_model=True),
                         lambda a: a["sam"]["calls"].update(mask_decoder=2),
                         lambda a: a.update(cache_kind="refined"),
                         lambda a: a["sam"].pop("residency_policy"),
                         lambda a: a["sam"].update(checkpoint_sha256=None)):
            audit = native_audit()
            mutation(audit)
            with self.assertRaises(ValueError):
                validate_native_cache(audit, plan, proposals)
        proposals[0].current_logits[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "untouched"):
            validate_native_cache(native_audit(), plan, proposals)

    def test_empty_cache_retained_without_claiming_sam_success(self):
        audit = native_audit()
        audit["sam"].update(status="empty_prompts", checkpoint_sha256=None, weight_coverage=None,
                            calls={key: 0 for key in audit["sam"]["calls"]}, microbatches=0, requested_prompts=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [stored_case(root, "empty", "train", audit=audit, empty=True),
                       stored_case(root, "valid", "validation", 1)]
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": "sam_once_training_v1", "records": records}))
            cases, scan = load_training_manifest(path)
            self.assertEqual(len(cases), 2)
            self.assertEqual(scan["empty_prompt_zero_call_cases"], 1)
            self.assertEqual(scan["sam_executed_cases"], 1)
            self.assertEqual(scan["empty_prompt_case_ids_by_split"]["train"], ["empty"])
            case = LazyCaseReader().load(cases[0])
            model = AddOne()
            with patch.object(model, "forward", side_effect=AssertionError("Empty must not run a model")):
                result = rollout_case(case, model, RefinerConfig())
            self.assertIsNone(result["loss"])
            self.assertEqual(result["updates"], 0)
            self.assertEqual(result["sam_calls"], 0)
            self.assertTrue(result["empty_proposals"])
            plan = PromptPlan([CropSpec("c0", (0, 0, 256, 256))], [], case["ridge"])
            false_success = copy.deepcopy(audit)
            false_success["sam"]["calls"]["image_encoder"] = 1
            with self.assertRaises(ValueError):
                validate_native_cache(false_success, plan, [])
            plan.crops = [CropSpec("c0", (0, 0, 257, 256))]
            with self.assertRaisesRegex(ValueError, "crop coordinates"):
                validate_native_cache(audit, plan, [])

    def test_all_empty_training_fails_after_one_cycle_with_receipt(self):
        def empty_fixture(split):
            case = training_fixture(split)
            case["proposals"] = []
            return case
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "empty_run"
            with patch("sam_once.train._fixture", side_effect=empty_fixture):
                with self.assertRaisesRegex(RuntimeError, "Bounded training stopped"):
                    train_main(["--smoke", "--output", str(out)])
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["attempts"], 1)
            self.assertEqual(failure["actual_successful_updates"], 0)
            ledger = json.loads((out / "ledger.json").read_text())
            self.assertTrue(ledger[0]["empty_proposals"])

    def test_lazy_reader_bounded_and_frozen_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [stored_case(root, "a", "train"), stored_case(root, "b", "train", 1),
                       stored_case(root, "c", "validation", 2)]
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": "sam_once_training_v1", "records": records}))
            cases, audit = load_training_manifest(path)
            self.assertTrue(all("rgb" not in c and "proposals" not in c for c in cases))
            reader = LazyCaseReader()
            for case in cases:
                self.assertEqual(reader.load(case)["rgb"].shape, (256, 256, 3))
                self.assertLessEqual(len(reader._cache), 2)
            self.assertNotIn("a", reader._cache)
            Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(root / "c.png")
            with self.assertRaisesRegex(ValueError, "changed after isolation scan"):
                reader.load(cases[2])

    def test_gate_receipts_bind_source_and_weight(self):
        from .io import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sam = native_audit()["sam"]
            parity = {"mode": "parity", "status": "passed", "offline_verification_only": True,
                      "batch_audit": sam, "permutation_isolation": [{"passed": True}],
                      "independent_native_reference": [{"passed": True}]}
            capacity = {"mode": "capacity", "status": "passed", "offline_verification_only": True,
                        "batch_audit": copy.deepcopy(sam), "capacity_24gb_pass": True,
                        "capacity": {"passed": True, "not_a_24gb_capacity_claim": False,
                                     "verified_24gb": True, "device_total_memory_bytes": 24*1024**3,
                                     "tested_K": 1, "tested_Q": 1}}
            (root / "parity.json").write_text(json.dumps(parity))
            (root / "capacity.json").write_text(json.dumps(capacity))
            gate = {"native_parity_pass": True, "capacity_24gb_pass": True,
                    "native_parity_receipt": {"path": "parity.json", "sha256": sha256(root / "parity.json")},
                    "capacity_24gb_receipt": {"path": "capacity.json", "sha256": sha256(root / "capacity.json")}}
            (root / "gate.json").write_text(json.dumps(gate))
            result = validate_gate_receipt(root / "gate.json")
            self.assertEqual(result["binding"]["checkpoint_sha256"], "a"*64)
            self.assertEqual(result["binding"]["residency_policy"], "native_full_predictor")
            rgb, ridge, proposals = fixture()
            plan = PromptPlan([proposals[0].crop], [proposals[0].prompt], ridge)
            validate_native_cache(native_audit(), plan, proposals, result)
            other_residency = native_audit()
            other_residency["sam"]["residency_policy"] = "experimental_interactive_only"
            with self.assertRaisesRegex(ValueError, "residency_policy differs"):
                validate_native_cache(other_residency, plan, proposals, result)
            for value in ("experimental_interactive_only", None):
                capacity["batch_audit"]["residency_policy"] = value
                (root / "capacity.json").write_text(json.dumps(capacity))
                gate["capacity_24gb_receipt"]["sha256"] = sha256(root / "capacity.json")
                (root / "gate.json").write_text(json.dumps(gate))
                with self.assertRaisesRegex(ValueError, "different checkpoints"):
                    validate_gate_receipt(root / "gate.json")
            capacity["batch_audit"]["residency_policy"] = "native_full_predictor"
            capacity["batch_audit"]["source_fingerprint"] = {"adapter": "c"*64}
            (root / "capacity.json").write_text(json.dumps(capacity))
            gate["capacity_24gb_receipt"]["sha256"] = sha256(root / "capacity.json")
            (root / "gate.json").write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, "different checkpoints"):
                validate_gate_receipt(root / "gate.json")

    def test_checkpoint_safe_exact_reload(self):
        cuda_was_initialized = torch.cuda.is_initialized()
        model = ResidualRefiner()
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            checkpoint = _checkpoint(model, optimizer, {"smoke_only": True}, 0, 1, [])
            torch.save(checkpoint, path)
            loaded, audit = load_refiner(path)
            self.assertTrue(audit["smoke_only"])
            self.assertIsNone(audit["training_manifest_sha256"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            checkpoint["protocol"]["isolation"] = {"manifest_sha256": "d"*64}
            torch.save(checkpoint, path)
            _, bound_audit = load_refiner(path)
            self.assertEqual(bound_audit["training_manifest_sha256"], "d"*64)
        if not cuda_was_initialized:
            self.assertFalse(torch.cuda.is_initialized(), "CPU checkpointing must not initialize a GPU")


if __name__ == "__main__":
    unittest.main()
