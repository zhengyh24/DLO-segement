"""CPU-only execution checks, not a quality or SAM performance experiment."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from sam_path_v1.checkpoint import (atomic_torch_save, build_from_checkpoint, engine_source_sha256,
                                   read_checkpoint, seed_everything, verify_engine_source)
from sam_path_v1.data import NativeInstanceDataset
from sam_path_v1.evaluate import evaluate_model, main as evaluate_main
from sam_path_v1.inference import decode_instances, padding_valid
from sam_path_v1.metrics import match_scores, score_image, summarize
from sam_path_v1.predict import load_rgb, main as predict_main
from sam_path_v1.train import TrainConfig, train, main as train_main


def fixture_manifest(root):
    root = Path(root)
    records = []
    for index in range(5):
        rgb = np.zeros((12, 13, 3), np.uint8)
        rgb[..., 0] = index*23+10
        rgb[3:6, 2:11, 1:] = 180
        masks = np.zeros((1, 12, 13), bool)
        masks[0, 3:6, 2:11] = True
        image_name, mask_name = f"rgb{index}.png", f"mask{index}.npz"
        Image.fromarray(rgb).save(root/image_name)
        np.savez_compressed(root/mask_name, masks=masks)
        records.append(dict(id=str(index), file_name=image_name, image=image_name, masks=mask_name,
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), shape=[12, 13],
                            num_instances=1, split="train" if index < 3 else "validation",
                            group_id=f"scene-{index}", domain="synthetic_smoke"))
    path = root / "manifest.json"
    path.write_text(json.dumps(dict(grouping_kind="verified_scene", independent_scene_split=True,
                                   data_root=".", records=records)), encoding="utf-8")
    return path


class MetricsTests(unittest.TestCase):
    def test_strict_threshold_and_maximum_cardinality(self):
        total, tp, matched = match_scores(np.array([[.99, .51], [.51, .49]]))
        self.assertAlmostEqual(total, 1.48)
        self.assertEqual(tp, 2)
        self.assertAlmostEqual(matched, 1.02)
        self.assertEqual(match_scores(np.array([[.5]]))[1], 0)

    def test_missing_gt_false_positive_and_failure(self):
        gt = np.zeros((2, 3, 4), bool)
        gt[0, 0, :2] = True; gt[1, 2, :2] = True
        row = score_image(gt[:1], gt)
        self.assertAlmostEqual(row["instance_iou"], .5)
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 0, 1))
        self.assertAlmostEqual(summarize([row])["pq_global"], 2/3)
        empty = np.zeros((0, 3, 4), bool)
        self.assertEqual(score_image(empty, empty)["instance_iou"], 1)
        failed = score_image(empty, empty, failed=True)
        self.assertEqual(failed["instance_iou"], 0)
        self.assertEqual(summarize([failed])["pq_global"], 0)
        self.assertEqual(score_image(gt, empty)["instance_iou"], 0)

    def test_padding_and_overlap_are_ignored(self):
        gt = np.zeros((1, 3, 3), bool); gt[0, 0, 0] = True
        prediction = gt.copy(); prediction[0, 2, 2] = True
        valid = np.ones((3, 3), bool); valid[2, 2] = False
        self.assertEqual(score_image(prediction, gt, valid)["instance_iou"], 1)


class InferenceTests(unittest.TestCase):
    def test_query_ownership_and_native_padding(self):
        valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)
        valid[..., -1] = False
        outputs = {"mask_logits": torch.full((1, 2, 2, 2), 3.),
                   "object_logits": torch.tensor([[2., 1.]])}
        result = decode_instances(outputs, (4, 4), valid)[0]
        self.assertEqual(len(result["masks"]), 1)
        self.assertEqual(result["queries"], [0])
        self.assertFalse(result["union"][:, -1].any())
        self.assertFalse(result["unknown"].any())
        images = torch.zeros((1, 3, 4, 4))
        mask = padding_valid(images, [{"meta": {"original_size": [3, 2]},
                                        "masks": torch.ones((1, 4, 4))}])
        self.assertEqual(mask.sum().item(), 6)

    def test_large_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"large.png"
            Image.new("RGB", (17, 16)).save(path)
            with self.assertRaisesRegex(ValueError, "exceeds canvas"):
                load_rgb(path, 16)


class ProvenanceTests(unittest.TestCase):
    def test_engine_fingerprint_requires_every_frozen_file(self):
        hashes = engine_source_sha256()
        verify_engine_source({"provenance": {"engine_source_sha256": hashes}})
        with self.assertRaisesRegex(ValueError, "no engine_source_sha256"):
            verify_engine_source({})
        with self.assertRaisesRegex(ValueError, "changed=model.py"):
            verify_engine_source({"provenance": {"engine_source_sha256": {**hashes, "model.py": "0"*64}}})
        without_predict = {key: value for key, value in hashes.items() if key != "predict.py"}
        with self.assertRaisesRegex(ValueError, "missing=predict.py"):
            verify_engine_source({"provenance": {"engine_source_sha256": without_predict}})

    def test_bad_source_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"unverified.pt"
            atomic_torch_save(dict(format_version=1, provenance={}), path)
            with patch("sam_path_v1.model.SamPathModel") as constructor:
                with self.assertRaisesRegex(ValueError, "no engine_source_sha256"):
                    build_from_checkpoint(path)
                constructor.assert_not_called()

    def test_incompatible_cublas_setting_fails_clearly(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":bad:"}):
            with self.assertRaisesRegex(ValueError, "fresh Python process"):
                seed_everything(42)


class ResumeTests(unittest.TestCase):
    def test_bit_exact_mid_epoch_resume_and_rgb_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture_manifest(root)
            base = dict(manifest=str(manifest), model=dict(backend="toy", variant="C", canvas_size=16,
                        num_queries=2, tracker_steps=2, encoder_amp="none"), seed=9, epochs=2,
                        batch_size=2, save_every=1, device="cpu", amp="none", cpu_threads=1)
            train(TrainConfig(**base), root/"full")
            (root/"config.json").write_text(json.dumps(dict(**base, max_batches=1)), encoding="utf-8")
            train_main(["--config", str(root/"config.json"), "--output", str(root/"part")])
            result = train(TrainConfig(**base), root/"resumed", root/"part/last.pt")
            whole, resumed = read_checkpoint(root/"full/last.pt"), read_checkpoint(root/"resumed/last.pt")
            for key in whole["model_state"]:
                self.assertTrue(torch.equal(whole["model_state"][key], resumed["model_state"][key]), key)
            for index, state in whole["optimizer"]["state"].items():
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][index][key]), key)
            for key in ("epoch", "batch_cursor", "samples", "opportunities", "successful_updates", "amp_skips"):
                self.assertEqual(whole["progress"][key], resumed["progress"][key], key)
            self.assertEqual(result["progress"]["successful_updates"], 4)
            self.assertEqual(result["progress"]["samples"], 6)
            self.assertTrue(torch.equal(whole["rng"]["torch"], resumed["rng"]["torch"]))
            with self.assertRaises(FileExistsError):
                train(TrainConfig(**base), root/"full")
            changed = {**base, "lr": .001}
            with self.assertRaisesRegex(ValueError, "changes training protocol"):
                train(TrainConfig(**changed), root/"bad_resume", root/"part/last.pt")
            model, _ = build_from_checkpoint(root/"resumed/last.pt")
            # Migration inference validates source, but does not demand the
            # training GPU or runtime. Training resume still rejects this.
            relocated = read_checkpoint(root/"resumed/last.pt")
            relocated["provenance"]["runtime"] = {"device_name": "different deployment hardware"}
            atomic_torch_save(relocated, root/"relocated.pt")
            moved_model, _ = build_from_checkpoint(root/"relocated.pt", device="cpu")
            self.assertFalse(moved_model.training)
            with self.assertRaisesRegex(ValueError, "Exact resume runtime changed"):
                train(TrainConfig(**base), root/"incompatible_runtime", root/"relocated.pt")
            validation = NativeInstanceDataset(manifest, "validation", 16)
            report = evaluate_model(model, validation)
            self.assertEqual(report["images"], 2)
            self.assertEqual(report["failures"], 0)
            self.assertTrue(all(row["calls"] == {"encoder": 1, "decoder": 1} for row in report["per_image"]))
            predict_main(["--checkpoint", str(root/"resumed/last.pt"), "--image", str(root/"rgb0.png"),
                          "--output", str(root/"prediction")])
            with Image.open(root/"prediction/instances.png") as image:
                self.assertEqual(image.size, (13, 12))
            self.assertTrue((root/"prediction/unknown.png").is_file())
            self.assertEqual(evaluate_main(["--checkpoint", str(root/"resumed/last.pt"),
                 "--manifest", str(manifest), "--output", str(root/"evaluation")]), 0)
            # A missing GT file keeps a zero-valued image in macro scoring and
            # explicitly invalidates aggregate PQ instead of inventing its FN.
            (root/"mask3.npz").unlink()
            broken = evaluate_model(model, validation)
            self.assertEqual(broken["images"], 2)
            self.assertEqual(broken["failures"], 1)
            self.assertIsNone(broken["pq_global"])
            self.assertFalse(broken["complete"])


if __name__ == "__main__":
    unittest.main()
