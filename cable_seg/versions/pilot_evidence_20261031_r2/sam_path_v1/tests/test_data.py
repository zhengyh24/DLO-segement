"""CPU data-contract tests; tiny generated fixtures, never the 2300-image set."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from sam_path_v1.data import (NativeInstanceDataset, collate_fn, decoded_mask_hash,
                              load_record, read_manifest, safe_relative_path, sha256_file)
from sam_path_v1.prepare import decode_rles, export_parquet, validate_export


def uncompressed(mask):
    counts, run, value = [], 0, False
    for pixel in mask.reshape(-1, order="F"):
        if bool(pixel) == value:
            run += 1
        else:
            counts.append(run)
            run, value = 1, not value
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "rgb").mkdir()
        (self.root / "masks").mkdir()
        self.rows = []
        for index, split in enumerate(("train", "validation")):
            rgb = (np.arange(5 * 7 * 3).reshape(5, 7, 3) + index).astype(np.uint8)
            masks = np.zeros((3, 5, 7), dtype=bool)
            masks[0, 1, 1:4] = True
            masks[1, 1:4, 3] = True
            image_rel, mask_rel = f"rgb/{index}.png", f"masks/{index}.npz"
            Image.fromarray(rgb).save(self.root / image_rel)
            np.savez_compressed(self.root / mask_rel, masks=masks)
            self.rows.append({"id": str(index), "file_name": f"family/{index}.png", "image": image_rel,
                              "masks": mask_rel, "split": split, "group_id": f"group-{index}",
                              "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "shape": [5, 7],
                              "decoded_masks_sha256": decoded_mask_hash(masks), "domain": "fixture"})
        self.obj = {"grouping_kind": "condition_holdout_pilot", "independent_scene_split": False,
                    "records": self.rows, "counts": {"train": 1, "validation": 1}}
        self.manifest = self.root / "manifest.json"
        self.write()

    def write(self, obj=None):
        self.manifest.write_text(json.dumps(self.obj if obj is None else obj), encoding="utf-8")

    def dataset(self, **kwargs):
        return NativeInstanceDataset(self.manifest, allow_condition=True, canvas_size=10, **kwargs)

    def test_native_pixels_gray_padding_overlap_and_variable_instances(self):
        sample = self.dataset()[0]
        rgb, original = load_record(self.root, self.rows[0])
        torch.testing.assert_close(sample["image"][:, :5, :7], torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255)
        self.assertTrue((sample["image"][:, 5:, :] == 0.5).all())
        self.assertTrue((sample["image"][:, :, 7:] == 0.5).all())
        self.assertEqual(sample["masks"].shape, (2, 10, 10))
        self.assertFalse(sample["valid"][0, 1, 3])
        self.assertFalse(sample["valid"][:, 5:].any())
        self.assertFalse(sample["masks"][:, 1, 3].any())
        self.assertEqual(int(sample["valid"].sum()), 34)
        self.assertEqual(sample["meta"]["source_instance_indices"], [0, 1])
        self.assertEqual(sample["meta"]["original_size"], [5, 7])
        batch = collate_fn([sample, self.dataset(split="val")[0]])
        self.assertEqual(batch["images"].shape, (2, 3, 10, 10))
        self.assertEqual(batch["valid"].dtype, torch.bool)
        self.assertEqual(len(batch["targets"]), 2)

    def test_condition_optin_canvas_and_counts(self):
        with self.assertRaisesRegex(ValueError, "allow_condition"):
            NativeInstanceDataset(self.manifest)
        with self.assertRaisesRegex(ValueError, "exceeds canvas"):
            NativeInstanceDataset(self.manifest, canvas_size=6, allow_condition=True)
        obj = deepcopy(self.obj)
        obj["counts"]["train"] = 2
        self.write(obj)
        with self.assertRaisesRegex(ValueError, "counts"):
            self.dataset()

    def test_cross_split_groups_hashes_and_duplicates(self):
        for key in ("group_id", "rgb_sha256", "id", "file_name", "image", "masks"):
            with self.subTest(key=key):
                obj = deepcopy(self.obj)
                obj["records"][1][key] = obj["records"][0][key]
                self.write(obj)
                with self.assertRaises(ValueError):
                    self.dataset()

    def test_portable_path_escape_rejection(self):
        for value in ("../x", "..\\x", "C:\\private\\x", "/tmp/x", "\\\\server\\share", "rgb/a:stream"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative_path(value)

    def test_relocation_and_epoch_determinism(self):
        obj = deepcopy(self.obj)
        obj["data_root"] = "missing-machine-root"
        self.write(obj)
        data = self.dataset(data_root=self.root)
        before = data[0]
        data.epoch = 4
        data.seed = 82
        torch.testing.assert_close(before["image"], data[0]["image"])
        torch.testing.assert_close(before["masks"], data[0]["masks"])

    def test_actual_rgb_hash_and_exif_failure(self):
        path = self.root / self.rows[0]["image"]
        with Image.open(path) as image:
            rgb = np.array(image)
        rgb[0, 0, 0] ^= 1
        Image.fromarray(rgb).save(path)
        with self.assertRaisesRegex(ValueError, "RGB hash"):
            self.dataset()[0]
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(rgb).save(path, exif=exif)
        with self.assertRaisesRegex(ValueError, "EXIF"):
            self.dataset()[0]

    def test_masks_tamper_and_nonbinary_rejected(self):
        path = self.root / self.rows[0]["masks"]
        masks = np.zeros((1, 5, 7), dtype=np.uint8)
        masks[0, 0, 0] = 255
        np.savez_compressed(path, masks=masks)
        with self.assertRaisesRegex(ValueError, "binary"):
            self.dataset()[0]
        np.savez_compressed(path, masks=masks > 0)
        with self.assertRaisesRegex(ValueError, "mask hash"):
            self.dataset()[0]

    def test_no_instance_cap_and_zero_instances(self):
        masks = np.eye(35, dtype=bool).reshape(35, 5, 7)
        row = self.rows[0]
        np.savez_compressed(self.root / row["masks"], masks=masks)
        row["decoded_masks_sha256"] = decoded_mask_hash(masks)
        self.write()
        self.assertEqual(len(self.dataset()[0]["masks"]), 35)
        masks = np.zeros((0, 5, 7), dtype=bool)
        np.savez_compressed(self.root / row["masks"], masks=masks)
        row["decoded_masks_sha256"] = decoded_mask_hash(masks)
        self.write()
        sample = self.dataset()[0]
        self.assertEqual(sample["masks"].shape, (0, 10, 10))
        self.assertEqual(int(sample["valid"].sum()), 35)

    def test_validate_full_membership_without_copy(self):
        receipt = self.root / "receipt.json"
        result = validate_export(self.manifest, self.manifest, allow_condition=True, require_pilot=False,
                                 canvas_size=10, receipt=receipt)
        self.assertEqual(result["records_checked"], 2)
        self.assertEqual(result["overlap_pixels"], 2)
        self.assertEqual(result["max_visible_instances"], 2)
        self.assertEqual(result["source_max"], 3)
        self.assertEqual(result["max_instances"], 24)
        with self.assertRaises(FileExistsError):
            validate_export(self.manifest, allow_condition=True, require_pilot=False, receipt=receipt)
        with self.assertRaisesRegex(ValueError, "2000/300"):
            validate_export(self.manifest, allow_condition=True)

    def test_capacity_counts_visible_after_overlap_and_never_truncates(self):
        row = self.rows[0]
        masks = np.zeros((30, 5, 7), dtype=bool)
        masks[:28, 0, 0] = True  # all these instances disappear after overlap ignore
        masks[28, 1, 1] = True
        masks[29, 2, 2] = True
        np.savez_compressed(self.root / row["masks"], masks=masks)
        row["decoded_masks_sha256"] = decoded_mask_hash(masks)
        self.write()
        result = validate_export(self.manifest, allow_condition=True, require_pilot=False, max_instances=2)
        self.assertEqual(result["source_max"], 30)
        self.assertEqual(result["max_visible_instances"], 2)
        masks = np.eye(35, dtype=bool).reshape(35, 5, 7)
        np.savez_compressed(self.root / row["masks"], masks=masks)
        row["decoded_masks_sha256"] = decoded_mask_hash(masks)
        self.write()
        receipt = self.root / "over_capacity.json"
        with self.assertRaisesRegex(ValueError, "visible=35.*max_instances=24.*no truncation"):
            validate_export(self.manifest, allow_condition=True, require_pilot=False, receipt=receipt)
        self.assertFalse(receipt.exists())
        self.assertEqual(len(self.dataset()[0]["masks"]), 35)
        with self.assertRaisesRegex(ValueError, "positive query capacity"):
            validate_export(self.manifest, allow_condition=True, require_pilot=False, max_instances=0)

    def test_export_capacity_failure_has_no_completion_marker(self):
        output = self.root / "capacity_failure"
        with patch("sam_path_v1.prepare.iter_parquet_rows", return_value=iter(self.raw_rows())), \
                self.assertRaisesRegex(ValueError, "query capacity"):
            export_parquet(self.manifest, self.root, output, allow_condition=True,
                           require_pilot=False, max_instances=1)
        self.assertFalse((output / "completion.json").exists())
        failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["complete"])

    def test_packaged_frozen_receipt_is_membership_only(self):
        path = Path(__file__).resolve().parents[1] / "protocols/pilot2300_manifest.json"
        obj = read_manifest(path, allow_condition=True)
        self.assertEqual(obj["counts"], {"train": 2000, "validation": 300})
        self.assertEqual(len(obj["records"]), 2300)
        self.assertEqual(obj["data_root"], ".")
        self.assertTrue(obj["membership_only"])
        self.assertFalse(obj["contains_decoded_data"])

    def raw_rows(self):
        result = []
        for index, row in enumerate(self.rows):
            rgb, masks = load_record(self.root, row)
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            result.append((self.root / "fixture.parquet", index, {"file_name": row["file_name"],
                           "image": {"bytes": buffer.getvalue(), "path": None},
                           "masks_rle": json.dumps([uncompressed(mask) for mask in masks])}))
        return result

    def test_rle_roundtrip_and_dimension_rejection(self):
        _, masks = load_record(self.root, self.rows[0])
        np.testing.assert_array_equal(decode_rles([uncompressed(mask) for mask in masks], [5, 7]), masks)
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_rles([uncompressed(masks[0])], [7, 5])

    def test_compressed_rle_roundtrip(self):
        try:
            from pycocotools import mask as coco
        except ImportError:
            self.skipTest("pycocotools not installed; uncompressed RLE checks still run")
        _, masks = load_record(self.root, self.rows[0])
        rles = []
        for mask in masks:
            item = coco.encode(np.asfortranarray(mask.astype(np.uint8)))
            item["counts"] = item["counts"].decode("ascii")
            rles.append(item)
        np.testing.assert_array_equal(decode_rles(json.dumps(rles), [5, 7]), masks)

    def test_export_stream_order_preservation_and_completion(self):
        output = self.root / "export"
        with patch("sam_path_v1.prepare.iter_parquet_rows", return_value=iter(self.raw_rows()[::-1])):
            path = export_parquet(self.manifest, self.root, output, allow_condition=True,
                                  require_pilot=False, canvas_size=10)
        obj = read_manifest(path, allow_condition=True)
        self.assertEqual([r["file_name"] for r in obj["records"]], [r["file_name"] for r in self.rows])
        self.assertTrue((output / "completion.json").is_file())
        for old, new in zip(self.rows, obj["records"]):
            np.testing.assert_array_equal(load_record(self.root, old)[1], load_record(output, new)[1])
        with self.assertRaises(FileExistsError):
            export_parquet(self.manifest, self.root, output, allow_condition=True, require_pilot=False)

    def test_missing_and_duplicate_selected_rows_fail_without_completion(self):
        raw = self.raw_rows()
        for suffix, rows in (("missing", raw[:1]), ("duplicate", raw + raw[:1])):
            with self.subTest(suffix=suffix):
                output = self.root / suffix
                with patch("sam_path_v1.prepare.iter_parquet_rows", return_value=iter(rows)), self.assertRaises(ValueError):
                    export_parquet(self.manifest, self.root, output, allow_condition=True, require_pilot=False)
                self.assertFalse((output / "completion.json").exists())
                self.assertTrue((output / "failure.json").is_file())

    def test_real_parquet_stream_tiny_fixture(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed; mock-stream export contract tests still run")
        raw = [r[2] for r in self.raw_rows()[::-1]]
        shard = self.root / "source.parquet"
        pq.write_table(pa.Table.from_pylist(raw), shard)
        path = export_parquet(self.manifest, shard, self.root / "arrow_export", allow_condition=True,
                              require_pilot=False, batch_size=1)
        self.assertEqual(len(read_manifest(path, True)["records"]), 2)


if __name__ == "__main__":
    unittest.main()
