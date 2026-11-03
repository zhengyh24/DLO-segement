"""Regression tests for bounded density repair, including unchanged valid data."""
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sam_path_v1.data import NativeInstanceDataset
from sam_path_v1.mosaic import NativeMosaicDataset, bounded_candidate_selection
from sam_path_v1.tests.test_scalability import mosaic_fixture


class MosaicRepairTests(unittest.TestCase):
    def test_overdense_maxima_have_a_feasible_choice_without_deleting_instances(self):
        counts = [[5, 3] for _ in range(16)]
        choices = bounded_candidate_selection(counts, 25, 64)
        self.assertEqual(len(choices), 16)
        self.assertEqual(sum(row[i] for row, i in zip(counts, choices)), 64)
        self.assertEqual(choices, bounded_candidate_selection(counts, 25, 64))

    def test_solver_matches_exhaustive_search_including_infeasible_and_zero_crops(self):
        rng=np.random.default_rng(18)
        for _ in range(100):
            counts=rng.integers(0, 8, size=(4,3)).tolist()
            totals=[sum(counts[j][i] for j,i in enumerate(c)) for c in itertools.product(range(3),repeat=4)]
            feasible=[n for n in totals if 8 <= n <= 14]
            chosen=bounded_candidate_selection(counts,8,14)
            if feasible:
                self.assertEqual(sum(row[i] for row,i in zip(counts,chosen)),max(feasible))
            else: self.assertIsNone(chosen)
        self.assertIsNone(bounded_candidate_selection([[5]]*16,25,64))
        self.assertIsNone(bounded_candidate_selection([[0]]*16,25,64))

    def test_original_successful_samples_and_recipes_are_bit_exact(self):
        # The historical module is a test-only immutable reference, not imported
        # by training or inference. Server deployment ships this exact fixture.
        reference=Path(__file__).parent/"fixtures"/"mosaic_r2_reference.py"
        spec=importlib.util.spec_from_file_location("sam_path_v1.mosaic_r2_reference",reference)
        old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as tmp:
            path=mosaic_fixture(Path(tmp))
            a=NativeInstanceDataset(path,"train",16,seed=3)
            b=NativeInstanceDataset(path,"train",16,seed=3)
            cfg=dict(enabled=True,probability=.5,tiles=2,crop_size=8,
                     min_instances=4,max_instances=8,max_attempts=2,crop_candidates=2)
            new=NativeMosaicDataset(a,cfg,8); legacy=old.NativeMosaicDataset(b,cfg,8)
            for epoch in range(3):
                new.epoch=legacy.epoch=epoch
                for index in range(6):
                    expected,actual=legacy[index],new[index]
                    self.assertEqual(expected["meta"],actual["meta"])
                    for key in ("image","masks","valid"):
                        torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
                    recipe=new.inspect_recipe(index)
                    if recipe["kind"]=="mosaic":
                        self.assertEqual({k:v for k,v in recipe.items() if k!="kind"},actual["meta"]["mosaic"])


if __name__ == "__main__":
    unittest.main()
