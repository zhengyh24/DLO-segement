"""GT provenance boundary guards for deployment caches; CPU only."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .io import save_case, load_case, read_json, write_json
from .types import CropSpec, PromptSpec, PromptPlan, ProposalRecord


def fixture():
    rgb = np.zeros((25, 37, 3), np.uint8)
    crop = CropSpec("c", (0, 0, 37, 25))
    prompt = PromptSpec("p", "c", ((15., 12.),))
    logits = np.zeros((25, 37), np.float32)
    plan = PromptPlan([crop], [prompt], np.zeros((25, 37), np.float32))
    proposals = [ProposalRecord(prompt, crop, logits.copy(), logits.copy(), 0.8)]
    return rgb, plan, proposals


class CacheGTGuardTests(unittest.TestCase):
    def test_save_rejects_all_prohibited_flags_before_creating_output(self):
        flags = (("ground_truth_used", True), ("ground_truth_used_for_inference", True),
                 ("diagnostic_only", True), ("deployment_allowed", False))
        with tempfile.TemporaryDirectory() as tmp:
            for location in ("plan", "prompt", "proposal", "audit"):
                for index, (key, value) in enumerate(flags):
                    with self.subTest(location=location, flag=key):
                        rgb, plan, proposals = fixture()
                        audit = {"ground_truth_used_for_inference": False}
                        target = {"plan": plan.audit, "prompt": plan.prompts[0].metadata,
                                  "proposal": proposals[0].metadata, "audit": audit}[location]
                        target["nested_source"] = [{key: value}]
                        output = Path(tmp) / f"{location}_{index}"
                        with self.assertRaisesRegex(ValueError, "GT-assisted/diagnostic"):
                            save_case(output, rgb, plan, proposals, audit)
                        self.assertFalse(output.exists())

    def test_load_rejects_nested_legacy_contamination_even_with_false_outer_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            for location in ("plan", "prompt", "proposal", "audit"):
                with self.subTest(location=location):
                    rgb, plan, proposals = fixture()
                    out = Path(tmp) / location
                    save_case(out, rgb, plan, proposals, {})
                    meta = read_json(out / "case.json")
                    target = {"plan": meta["plan"]["audit"],
                              "prompt": meta["plan"]["prompts"][0]["metadata"],
                              "proposal": meta["proposals"][0]["metadata"], "audit": meta["audit"]}[location]
                    target["parent"] = [{"ground_truth_used_for_inference": True}]
                    self.assertFalse(meta["ground_truth_used_for_inference"])
                    write_json(out / "case.json", meta)
                    # Reject metadata before touching potentially stale payload files.
                    (out / "rgb.png").unlink()
                    with self.assertRaisesRegex(ValueError, "GT-assisted/diagnostic"):
                        load_case(out)

    def test_explicit_gt_free_metadata_roundtrips_unchanged(self):
        rgb, plan, proposals = fixture()
        allowed = {"ground_truth_used": False, "ground_truth_used_for_inference": False,
                   "diagnostic_only": False, "deployment_allowed": True}
        plan.audit.update(allowed)
        plan.prompts[0].metadata.update(allowed)
        proposals[0].metadata.update(allowed)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "free"
            save_case(out, rgb, plan, proposals, {"source": [allowed]})
            cached_rgb, cached_plan, cached_proposals, cached_audit = load_case(out)
            np.testing.assert_array_equal(cached_rgb, rgb)
            self.assertEqual(cached_plan.metadata(), plan.metadata())
            self.assertEqual(cached_proposals[0].metadata, allowed)
            self.assertEqual(cached_audit, {"source": [allowed]})

    def test_oracle_plan_cannot_be_written_as_production_cache(self):
        from .oracle_diagnostic import oracle_plan
        rgb, _, _ = fixture()
        mask = np.zeros(rgb.shape[:2], bool)
        mask[10:17, 5:32] = True
        plan = oracle_plan(rgb, [mask])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "oracle_disguised_as_free"
            with self.assertRaisesRegex(ValueError, "GT-assisted/diagnostic"):
                save_case(out, rgb, plan, [], {"ground_truth_used_for_inference": False})
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
