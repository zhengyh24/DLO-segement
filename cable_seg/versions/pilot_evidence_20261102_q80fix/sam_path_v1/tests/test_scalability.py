"""Native coordinate coverage, common query identities, and training-only donors."""
import hashlib
import contextlib
import io
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from sam_path_v1.data import NativeInstanceDataset
from sam_path_v1.model import ModelConfig, SamPathModel
from sam_path_v1.mosaic import NativeMosaicDataset, MosaicDensityError
from sam_path_v1.scalability import tile_origins, factor_tile, predict_native
from sam_path_v1.scalability import main as native_main
from sam_path_v1.train import TrainConfig, train
from sam_path_v1.checkpoint import read_checkpoint


class NativeExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, variant="C", chunk=0):
        return SamPathModel(ModelConfig(backend="toy", variant=variant, canvas_size=16,
            num_queries=3, tracker_steps=2, init_seed=12, decoder_query_chunk=chunk))

    def test_native_tiles_cover_every_pixel_and_align_feature_coordinates(self):
        for length in (1, 16, 17, 24, 25, 48, 57):
            coverage = np.zeros(length, bool)
            for offset in tile_origins(length, 16, 8):
                self.assertEqual(offset % 4, 0)
                coverage[offset:offset+16] = True
            self.assertTrue(coverage.all())

    def test_factor_slice_has_correct_native_origin_and_zero_padded_tail(self):
        spatial = torch.arange(7*9).reshape(1, 1, 7, 9).float()
        factor = {"channels": torch.tensor([[[2., 3.]]]), "spatial": spatial}
        result = factor_tile(factor, 0, 1, 4, 6, 4)
        self.assertEqual(tuple(result.shape), (1, 1, 2, 4, 4))
        torch.testing.assert_close(result[0, 0, 0, :3, :3], 2*spatial[0, 0, 4:, 6:])
        self.assertEqual(float(result[..., -1, :].abs().sum()), 0.)
        self.assertEqual(float(result[..., :, -1].abs().sum()), 0.)

    def test_factorized_condition_and_tracker_equal_materialized_values(self):
        model = self.model()
        low = torch.randn(1, 256, 4, 5)
        valid = torch.ones(1, 4, 5, dtype=torch.bool)
        queries, dense, objects = model.condition(low, valid)
        fq, factor, fo = model.condition.forward_factors(low, valid)
        torch.testing.assert_close(queries, fq, atol=0, rtol=0)
        torch.testing.assert_close(objects, fo, atol=0, rtol=0)
        torch.testing.assert_close(dense, factor["channels"][..., None, None]*factor["spatial"][:, :, None], atol=0, rtol=0)
        tokens, dense, _ = model.tracker(low, queries, valid)
        ft, factor, _ = model.tracker(low, queries, valid, factorized=True)
        torch.testing.assert_close(tokens, ft, atol=0, rtol=0)
        torch.testing.assert_close(dense, factor["channels"][..., None, None]*factor["spatial"][:, :, None], atol=0, rtol=0)

    def test_chunked_decode_matches_single_decode_values_gradients_and_calls(self):
        whole, chunked = self.model(), self.model(chunk=2)
        image = torch.rand(1, 3, 16, 16)
        a, b = whole(image), chunked(image)
        self.assertEqual(a["calls"], {"encoder": 1, "decoder": 1})
        self.assertEqual(b["calls"], {"encoder": 1, "decoder": 2})
        torch.testing.assert_close(a["mask_logits"], b["mask_logits"], atol=2e-6, rtol=2e-5)
        a["mask_logits"].square().mean().backward()
        b["mask_logits"].square().mean().backward()
        for (_, p), (_, other) in zip(whole.named_parameters(), chunked.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, other.grad, atol=1e-6, rtol=3e-5)

    def test_native_prediction_uses_one_global_query_set_without_gt(self):
        model = self.model("A").eval()
        with torch.no_grad():
            model.condition.object_head.weight.zero_(); model.condition.object_head.bias.fill_(10.)
        rgb = np.zeros((25, 29, 3), np.uint8)
        seen_tokens = []
        def fake_decode(features, tokens, dense):
            seen_tokens.append(tokens.detach().clone())
            b, q = tokens.shape[:2]
            # Global query zero (first chunk) always wins and stays the same ID
            # across all tiles, despite different local image content.
            score = 5. if len(seen_tokens) <= 9 else -5.
            return (torch.full((b, q, 4, 4), score), torch.zeros(b, q), torch.zeros(b, q))
        with patch.object(model.base, "decode", side_effect=fake_decode):
            result = predict_native(model, rgb, device="cpu", tile_size=16, stride=8,
                                    patch_size=4, query_chunk=2)
        self.assertEqual(result["calls"], {"encoder": 9, "decoder": 18})
        self.assertEqual(result["instances"].shape, (25, 29))
        self.assertEqual(np.unique(result["instances"]).tolist(), [1])
        for token in seen_tokens[:9]:
            torch.testing.assert_close(token, seen_tokens[0], atol=0, rtol=0)
        self.assertNotIn("masks", inspect.signature(predict_native).parameters)


def mosaic_fixture(root):
    rows = []
    for index in range(7):
        rgb = np.full((16, 16, 3), index*20, np.uint8)
        masks = np.zeros((2, 16, 16), bool)
        masks[0, :, 2:5] = True; masks[1, :, 10:13] = True
        Image.fromarray(rgb).save(root / f"{index}.png")
        np.savez_compressed(root / f"{index}.npz", masks=masks)
        rows.append(dict(id=str(index), file_name=f"{index}.png", image=f"{index}.png",
            masks=f"{index}.npz", shape=[16, 16], num_instances=2,
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), group_id=str(index),
            split="train" if index < 6 else "validation"))
    path = root / "manifest.json"
    path.write_text(json.dumps(dict(grouping_kind="verified_scene", independent_scene_split=True,
                                   records=rows, data_root=".")), encoding="utf-8")
    return path


class MosaicTests(unittest.TestCase):
    def config(self, **kwargs):
        return dict(enabled=True, probability=1., tiles=2, crop_size=8, min_instances=4,
                    max_instances=8, max_attempts=2, crop_candidates=2, **kwargs)

    def test_only_train_donors_native_pixels_unique_identities_and_deterministic_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = mosaic_fixture(Path(tmp))
            base = NativeInstanceDataset(path, "train", 16, seed=3)
            data = NativeMosaicDataset(base, self.config(), 8)
            first, repeated = data[0], data[0]
            self.assertEqual(first["meta"]["mosaic"], repeated["meta"]["mosaic"])
            self.assertTrue(torch.equal(first["image"], repeated["image"]))
            receipt = first["meta"]["mosaic"]
            self.assertEqual(len({r["id"] for r in receipt["recipe"]}), 4)
            self.assertTrue(all(r["split"] == "train" and r["id"] != "6" for r in receipt["recipe"]))
            self.assertTrue(4 <= len(first["masks"]) <= 8)
            # Every donor has constant integer RGB; crop/translate cannot create
            # interpolated colors or map one local instance across donor tiles.
            for tile, recipe in enumerate(receipt["recipe"]):
                y, x = tile//2*8, tile%2*8
                self.assertTrue(torch.allclose(first["image"][:, y:y+8, x:x+8], torch.tensor(int(recipe["id"])*20/255.)))
            validation = NativeInstanceDataset(path, "validation", 16)
            with self.assertRaisesRegex(ValueError, "train split"):
                NativeMosaicDataset(validation, self.config(), 8)

    def test_density_failure_is_reported_without_low_density_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = mosaic_fixture(Path(tmp))
            config = self.config(); config.update(min_instances=20, max_instances=24, max_attempts=1)
            data = NativeMosaicDataset(NativeInstanceDataset(path, "train", 16), config, 24)
            with self.assertRaisesRegex(MosaicDensityError, "no low-density fallback"):
                data[0]

    def test_training_records_mosaic_coverage_and_query_positive_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = mosaic_fixture(root)
            config = TrainConfig(manifest=str(manifest), model=dict(backend="toy", variant="A",
                canvas_size=16, num_queries=8, tracker_steps=2, decoder_query_chunk=2),
                mosaic=self.config(), seed=3, max_batches=1, device="cpu", cpu_threads=1)
            result = train(config, root/"run")
            progress = result["progress"]
            self.assertEqual(progress["mosaic_samples"], 1)
            self.assertTrue(4 <= sum(progress["query_positive_matches"]) <= 8)
            checkpoint = read_checkpoint(root/"run/last.pt")
            self.assertEqual(progress["query_positive_matches"], checkpoint["progress"]["query_positive_matches"])
            events = [json.loads(line) for line in (root/"run/logs.jsonl").read_text().splitlines()]
            update = next(row for row in events if row["event"] == "update")
            self.assertEqual(update["calls"], {"encoder": 1, "decoder": 4})
            self.assertTrue(update["augmentation_receipts"])

    def test_native_cli_identity_failure_denominator_and_empty_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = mosaic_fixture(root)
            checkpoint = root/"fake.pt"; checkpoint.write_bytes(b"mock-only")
            payload = {"train_config": {"object_threshold": .5, "mask_threshold": .5}}
            for mode in ("empty_prediction", "failed_inference", "empty_gt"):
                if mode == "empty_gt":
                    np.savez_compressed(root/"6.npz", masks=np.zeros((0, 16, 16), bool))
                    obj = json.loads(manifest.read_text()); obj["records"][-1]["num_instances"] = 0
                    manifest.write_text(json.dumps(obj))
                prediction = dict(masks=np.zeros((0, 16, 16), bool), instances=np.zeros((16, 16), np.int32),
                                  calls={"encoder": 1, "decoder": 20}, seconds=.1)
                effect = RuntimeError("deliberate native failure") if mode == "failed_inference" else None
                with patch("sam_path_v1.scalability.build_from_checkpoint", return_value=(object(), payload)):
                    with patch("sam_path_v1.scalability.predict_native", return_value=prediction, side_effect=effect):
                        with contextlib.redirect_stdout(io.StringIO()):
                            code = native_main(["--checkpoint", str(checkpoint), "--manifest", str(manifest),
                                "--output", str(root/mode), "--device", "cpu", "--split", "validation"])
                report = json.loads((root/mode/"metrics.json").read_text())
                self.assertTrue(report["identity_diagnostics"]["complete"])
                self.assertEqual(report["identity_diagnostics"]["n_gt"], 0 if mode == "empty_gt" else 2)
                self.assertEqual(code, 2 if mode == "failed_inference" else 0)
                if mode == "failed_inference":
                    self.assertEqual(report["fn"], 2)
                    self.assertEqual(report["identity_diagnostics"]["whole_visible_recovery_rate"], 0.)


if __name__ == "__main__":
    unittest.main()
