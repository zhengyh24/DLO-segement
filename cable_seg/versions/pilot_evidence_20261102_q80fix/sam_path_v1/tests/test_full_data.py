"""Frozen full-data split tests, including pilot exclusions and provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sam_path_v1.full_data import build_protocol, read_sums, real_compatibility
from sam_path_v1.data import read_manifest


class FullDataTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.pilot_rows = [], []
        for condition in ("family/2_cable", "family/4_cable"):
            for i in range(40):
                name = f"{condition}/{i:06d}.png"
                split = "train" if i < 3 else "validation" if i < 5 else "unassigned"
                row = dict(id=name, file_name=name, image="rgb/"+name,
                           masks="masks/"+name.replace(".png", ".npz"),
                           shape=[16, 16], rgb_sha256=hashlib.sha256(name.encode()).hexdigest(),
                           split=split, group_id="condition:"+condition, num_instances=2)
                self.rows.append(row)
                if split != "unassigned":
                    self.pilot_rows.append({**row, "id": f"pilot-{len(self.pilot_rows)}"})
        self.source = dict(grouping_kind="condition_holdout_pilot_with_unassigned_extension",
                           independent_scene_split=False, records=self.rows,
                           counts={"train": 6, "validation": 4, "unassigned": 70})
        self.pilot = dict(grouping_kind="condition_holdout_pilot", independent_scene_split=False,
                          records=self.pilot_rows)
        self.kw = dict(validation_fraction="0.1", test_fraction="0.1", expected_conditions=None)

    def build(self, **kwargs):
        return build_protocol(self.source, self.pilot, **{**self.kw, **kwargs})

    def test_all_members_unique_exact_counts_pilot_preserved_test_unseen(self):
        result = self.build()
        self.assertEqual(result["counts"], {"train": 64, "validation": 8, "test": 8})
        self.assertEqual(len(result["records"]), 80)
        pilot = {r["file_name"]: r["split"] for r in self.pilot_rows}
        for row in result["records"]:
            self.assertEqual(row["group_id"], "rgb_sha256:" + row["rgb_sha256"])
            self.assertEqual(row["source_condition_group"], "condition:"+"/".join(row["file_name"].split("/")[:-1]))
            if row["file_name"] in pilot:
                self.assertEqual(row["split"], pilot[row["file_name"]])
            if row["split"] == "test":
                self.assertEqual(row["source_split"], "unassigned")
                self.assertFalse(row["pilot_member"])
        self.assertFalse(result["independent_scene_split"])
        self.assertFalse(result["independent_condition_split"])

    def test_order_independent_seed_sensitive(self):
        original = self.build()
        self.source["records"] = list(reversed(self.rows))
        self.assertEqual(original, self.build())
        self.assertNotEqual(original["records"], self.build(seed=43)["records"])

    def test_rgb_duplicates_rejected(self):
        self.rows[10]["rgb_sha256"] = self.rows[0]["rgb_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build()

    def test_pilot_hash_or_split_change_rejected(self):
        self.pilot_rows[0]["rgb_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "membership/hash/geometry"):
            self.build()

    def test_unaccounted_historical_assigned_member_rejected(self):
        self.pilot_rows.pop()
        with self.assertRaisesRegex(ValueError, "entire frozen pilot"):
            self.build()

    def test_pinned_quota_and_noninteger_fraction_rejected(self):
        with self.assertRaisesRegex(ValueError, "smaller than pinned"):
            self.build(validation_fraction="0.025")
        with self.assertRaisesRegex(ValueError, "integer counts"):
            self.build(validation_fraction="0.03")

    def test_full_source_inventory_enforced(self):
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            build_protocol(self.source, self.pilot)

    def test_sums_bound_and_missing_file_rejected(self):
        sums = {r[key]: hashlib.sha256(r[key].encode()).hexdigest()
                for r in self.rows for key in ("image", "masks")}
        result = self.build(sums=sums)
        for row in result["records"]:
            self.assertEqual(row["masks_sha256"], sums[row["masks"]])
        sums.pop(self.rows[0]["masks"])
        with self.assertRaisesRegex(ValueError, "missing from SHA256SUMS"):
            self.build(sums=sums)

    def test_manifest_read_explicit_optin_and_honest_grouping(self):
        obj = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            def write(value):
                path.write_text(json.dumps(value), encoding="utf-8")
            write(obj)
            with self.assertRaisesRegex(ValueError, "allow_condition"):
                read_manifest(path)
            self.assertEqual(read_manifest(path, True), obj)
            for changes in ({"independent_condition_split": True}, {"independent_scene_split": True},
                            {"split_unit": "scene"}):
                write({**obj, **changes})
                with self.assertRaises(ValueError):
                    read_manifest(path, True)
            damaged = deepcopy(obj)
            damaged["records"][0]["group_id"] = damaged["records"][0]["source_condition_group"]
            write(damaged)
            with self.assertRaisesRegex(ValueError, "exact decoded RGB hash"):
                read_manifest(path, True)
            damaged = deepcopy(obj)
            damaged["records"][0].pop("source_condition_group")
            write(damaged)
            with self.assertRaisesRegex(ValueError, "source_condition_group"):
                read_manifest(path, True)

    def test_sums_traversal_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SHA256SUMS"
            path.write_text("a"*64 + "  ../secret\n")
            with self.assertRaises(ValueError):
                read_sums(path)
            path.write_text(("a"*64 + "  rgb/a.png\n")*2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_sums(path)

    def test_real_incompatibilities_reported_without_selection(self):
        result = real_compatibility({"records": [dict(shape=[1024, 1024], num_instances=3),
                                                   dict(shape=[2000, 2800], num_instances=67)]})
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["exceeds_canvas"], 1)
        self.assertEqual(result["exceeds_query_capacity"], 1)


if __name__ == "__main__":
    unittest.main()
