"""Small CPU-only tests of validation isolation and selection policy."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import unittest

from sam_once.select_checkpoint import choose_checkpoint, validation_records, checkpoint_manifest_matches


def domain(iou=.5, pq=.4, n=3):
    return {"instance_iou": iou, "pq": pq, "n": n}


def candidate(sha, summaries, **kwargs):
    return {"checkpoint_sha256": sha * 64, "summaries": summaries,
            "smoke_only": False, "status": "ok", **kwargs}


def manifest_row(case, split, digest, **kwargs):
    return {"case_id": case, "split": split, "cluster_id": case + "_cluster",
            "rgb_sha256": digest * 64, **kwargs}


class SelectionTests(unittest.TestCase):
    def test_formal_checkpoint_bound_to_exact_training_manifest(self):
        audit = {"smoke_only": False, "training_manifest_sha256": "a" * 64}
        self.assertTrue(checkpoint_manifest_matches(audit, "a" * 64))
        self.assertFalse(checkpoint_manifest_matches(audit, "b" * 64))
        self.assertFalse(checkpoint_manifest_matches({"smoke_only": False}, "a" * 64))
        self.assertTrue(checkpoint_manifest_matches({"smoke_only": True}, "a" * 64))
        rejected = candidate("a", {"C": domain(.9, .9)}, status="training_manifest_mismatch")
        self.assertIsNone(choose_checkpoint({"C": domain()}, [rejected])["selected"])

    def test_each_domain_and_both_metrics_must_not_regress(self):
        base = {"C": domain(), "W": domain()}
        pool = [candidate("a", {"C": domain(.7, .8), "W": domain(.49, .9)}),
                candidate("b", {"C": domain(.8, .7), "W": domain(.8, .39)})]
        result = choose_checkpoint(base, pool)
        self.assertIsNone(result["selected"])
        self.assertTrue(all(not r["eligible"] for r in result["candidates"]))

    def test_equal_domain_pq_not_image_weighting(self):
        base = {"large": domain(n=100), "small": domain(n=1)}
        a = candidate("a", {"large": domain(.6, .41, 100), "small": domain(.6, .9, 1)})
        b = candidate("b", {"large": domain(.8, .6, 100), "small": domain(.8, .6, 1)})
        self.assertEqual(choose_checkpoint(base, [b, a])["selected"]["checkpoint_sha256"], "a" * 64)

    def test_tie_breaks_iou_then_sha_and_order_independence(self):
        base = {"one": domain()}
        a = candidate("a", {"one": domain(.7, .6)})
        b = candidate("b", {"one": domain(.7, .6)})
        c = candidate("c", {"one": domain(.6, .6)})
        for pool in ([b, a, c], [c, a, b], [c, b, a]):
            self.assertEqual(choose_checkpoint(base, pool)["selected"]["checkpoint_sha256"], "a" * 64)

    def test_smoke_or_failed_checkpoints_are_never_selected(self):
        base = {"one": domain()}
        best = {"one": domain(.99, .99)}
        pool = [candidate("a", best, smoke_only=True), candidate("b", best, status="failed")]
        self.assertIsNone(choose_checkpoint(base, pool)["selected"])

    def test_runtime_failures_cannot_win_a_zero_score_tie(self):
        base = {"one": domain(0., 0.)}
        row = candidate("a", copy.deepcopy(base))
        self.assertIsNone(choose_checkpoint(base, [{**row, "case_failures": [{"case_id": "x"}]}])["selected"])
        self.assertIsNone(choose_checkpoint(base, [row], baseline_failures=[{"case_id": "x"}])["selected"])

    def test_domains_counts_and_nonfinite_metric_checks(self):
        base = {"one": domain()}
        pool = [candidate("a", {}), candidate("b", {"one": domain(n=2)}),
                candidate("c", {"one": domain(float("nan"), .8)})]
        self.assertIsNone(choose_checkpoint(base, pool)["selected"])
        with self.assertRaises(ValueError):
            choose_checkpoint({}, pool)
        with self.assertRaises(ValueError):
            choose_checkpoint({"one": domain(pq=float("inf"))}, pool)

    def test_equality_is_nonregression_and_inputs_unchanged(self):
        base = {"one": domain()}
        pool = [candidate("a", copy.deepcopy(base))]
        before = copy.deepcopy((base, pool))
        result = choose_checkpoint(base, pool)
        self.assertTrue(result["selected"]["eligible"])
        self.assertEqual((base, pool), before)
        self.assertFalse(result["automatic_promotion"])

    def test_only_validation_paths_resolved_and_labels_not_read(self):
        # train/check intentionally have no paths: strict selection must not
        # inspect their RGB, masks or caches to decide a validation checkpoint.
        obj = {"schema": "sam_once_training_v1", "records": [
            manifest_row("train", "train", "a"), manifest_row("check", "check", "b"),
            manifest_row("val", "validation", "c", dataset="C", image_path="image.png",
                         cache_dir="cache", gt_masks=["labels/0.png"])]}
        rows = validation_records(obj, Path.cwd() / "frozen" / "manifest.json")
        self.assertEqual([r["case_id"] for r in rows], ["val"])
        self.assertTrue(Path(rows[0]["image_path"]).is_absolute())
        self.assertTrue(Path(rows[0]["gt_masks"][0]).is_absolute())
        self.assertEqual(rows[0]["split"], "validation")
        self.assertEqual(obj["records"][2]["gt_masks"], ["labels/0.png"])

    def test_no_validation_does_not_fallback_to_train_or_check(self):
        for split in ("train", "check"):
            with self.assertRaises(ValueError):
                validation_records({"schema": "sam_once_training_v1", "records": [
                    manifest_row("x", split, "a")]}, "manifest.json")

    def test_cross_split_cluster_or_rgb_leakage_rejected(self):
        val = manifest_row("val", "validation", "b", dataset="C", image_path="v.png", cache_dir="cache", gt_masks=[])
        for key in ("cluster_id", "rgb_sha256"):
            train = manifest_row("train", "train", "a")
            train[key] = val[key]
            with self.assertRaises(ValueError):
                validation_records({"schema": "sam_once_training_v1", "records": [train, val]}, "manifest.json")

    def test_case_path_alias_and_unknown_split_rejected(self):
        val = manifest_row("val", "validation", "b", dataset="C", image_path="v.png", cache_dir="cache", gt_masks=[])
        for bad in ("../escape", "a/b", "..", "drive:c", "trailing."):
            with self.assertRaises(ValueError):
                validation_records({"schema": "sam_once_training_v1", "records": [{**val, "case_id": bad}]}, "manifest.json")
        with self.assertRaises(ValueError):
            validation_records({"schema": "sam_once_training_v1", "records": [val, {**val, "case_id": "VAL"}]}, "manifest.json")
        with self.assertRaises(ValueError):
            validation_records({"schema": "sam_once_training_v1", "records": [{**val, "split": "test"}]}, "manifest.json")


if __name__ == "__main__":
    unittest.main()
