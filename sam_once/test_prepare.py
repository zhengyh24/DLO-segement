"""Tiny temporary fixtures for manifest preparation; no models or real exports."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from sam_once.prepare import prepare, _source_isolation


def write_source(root, name, kind, cases):
    directory = Path(root) / name
    directory.mkdir()
    rows = []
    for ident, split, brightness, group in cases:
        rgb = np.full((11, 17, 3), brightness, np.uint8)
        Image.fromarray(rgb).save(directory / f"{ident}.png")
        masks = np.zeros((1, 11, 17), bool)
        masks[0, 5, 2:15] = True
        np.savez_compressed(directory / f"{ident}.npz", masks=masks)
        rows.append({"id": ident, "split": split, "group_id": group,
                     "image": f"{ident}.png", "masks": f"{ident}.npz", "dataset": "fixture",
                     "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
    path = directory / "manifest.json"
    path.write_text(json.dumps({"grouping_kind": kind, "records": rows,
                                "independent_scene_split": False}), encoding="utf-8")
    return path


class PrepareTests(unittest.TestCase):
    def test_same_grouping_kind_sources_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            a = write_source(directory, "a", "historical_rgb_cluster_holdout", [("a", "train", 10, "g")])
            b = write_source(directory, "b", "historical_rgb_cluster_holdout", [("b", "validation", 200, "g")])
            out = Path(directory) / "out"
            with self.assertRaisesRegex(ValueError, "At most one source"):
                prepare([a, b], Path(directory) / "cache", out)
            self.assertFalse(out.exists())

    def test_cross_source_near_duplicate_cross_split_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a = write_source(directory, "a", "historical_rgb_cluster_holdout", [("a", "train", 10, "g")])
            b = write_source(directory, "b", "condition_holdout_pilot", [("b", "validation", 18, "h")])
            # Different native pixel hashes, but dHash=0 and RGB MAD=8.
            out = Path(directory) / "out"
            with self.assertRaisesRegex(ValueError, "near-duplicate component crosses splits"):
                prepare([a, b], Path(directory) / "cache", out)
            self.assertFalse(out.exists())

    def test_same_split_transitive_components_preserve_group_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = [write_source(directory, name, kind, [(name, "train", value, name)])
                       for name, kind, value in (("a", "historical_rgb_cluster_holdout", 10),
                                                  ("b", "condition_holdout_pilot", 20),
                                                  ("c", "verified_scene", 30))]
            result = prepare(sources, Path(directory) / "cache", Path(directory) / "out")
            self.assertEqual(len({r["cluster_id"] for r in result["records"]}), 1)
            self.assertEqual(len({r["source_cluster_id"] for r in result["records"]}), 3)
            self.assertEqual({r["split"] for r in result["records"]}, {"train"})
            matches = result["cross_source_isolation"]["cross_source_pair_matches"]
            self.assertEqual(len(matches), 2)
            self.assertTrue(all(not m["exact_decoded_rgb"] for m in matches))
            self.assertFalse(result["independent_scene_split"])

    def test_existing_single_source_contract_and_test_to_check(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(directory, "a", "historical_rgb_cluster_holdout",
                                  [("a", "train", 10, "g1"), ("b", "validation", 100, "g2"),
                                   ("c", "test", 200, "g3")])
            out = Path(directory) / "out"
            result = prepare([source], Path(directory) / "cache", out)
            self.assertEqual([r["split"] for r in result["records"]], ["train", "validation", "check"])
            self.assertEqual(result["cache_status"], "paths_only_predictions_not_generated")
            for row in result["records"]:
                with Image.open(out / row["gt_masks"][0]) as im:
                    self.assertEqual(im.size, (17, 11))
                self.assertEqual(row["cluster_id"], row["source_cluster_id"])
            queue = json.loads((out / "inference_inputs.json").read_text())
            self.assertTrue(all("gt_masks" not in r and "split" not in r for r in queue["records"]))

    def test_source_rgb_mutation_and_exact_rgb_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(directory, "a", "historical_rgb_cluster_holdout",
                                  [("a", "train", 10, "g1"), ("b", "validation", 10, "g2")])
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                _source_isolation([source])
            Image.fromarray(np.full((11, 17, 3), 22, np.uint8)).save(source.parent / "a.png")
            with self.assertRaisesRegex(ValueError, "Source RGB changed"):
                _source_isolation([source])

    def test_cross_source_nonmatches_do_not_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            a = write_source(directory, "a", "historical_rgb_cluster_holdout", [("a", "train", 10, "g")])
            b = write_source(directory, "b", "condition_holdout_pilot", [("b", "validation", 200, "h")])
            _, overrides, audit = _source_isolation([a, b])
            self.assertEqual(overrides, {})
            self.assertEqual(audit["cross_source_pair_matches"], [])


if __name__ == "__main__":
    unittest.main()
