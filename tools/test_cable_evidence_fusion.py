"""Checks physical crossing behavior, native clipping and metric equivalence."""
import unittest
import cv2
import numpy as np
from cable_evidence_fusion import best_pairs, complete_visible, trace_segments
from run_evidence_fusion_experiment import instance_scores
from evaluate_full_comparison import instances, masks_from_ids


class EvidenceTests(unittest.TestCase):
    def test_crossing_pairs_opposite_arms(self):
        tips = [dict(edge=i+1, direction=np.array(v,float), width=5.,
                     color=np.array([100,120,130],float), signature={})
                for i,v in enumerate([(1,0),(0,1),(-1,0),(0,-1)])]
        self.assertEqual({frozenset(p) for p in best_pairs(tips)}, {frozenset((0,2)),frozenset((1,3))})

    def test_foreground_completion_never_bridges_background(self):
        visible = np.zeros((60,80),bool)
        visible[15:20,5:45] = True
        visible[35:40,50:70] = True
        markers=np.zeros(visible.shape,np.uint16);markers[16,8]=1
        ids=complete_visible(markers,visible)
        self.assertTrue(np.all(ids[15:20,5:45] == 1))
        self.assertFalse(ids[35:40,50:70].any())
        self.assertFalse(ids[~visible].any())

    def test_tracer_does_not_merge_x_crossing(self):
        visible=np.zeros((101,101),np.uint8)
        cv2.line(visible,(10,10),(90,90),1,5)
        cv2.line(visible,(90,10),(10,90),1,5)
        rgb=np.full((101,101,3),140,np.uint8)
        ids,audit=trace_segments(rgb,visible>0,use_evidence=False)
        self.assertEqual(ids[20,20],ids[80,80])
        self.assertEqual(ids[20,80],ids[80,20])
        self.assertNotEqual(ids[20,20],ids[20,80])
        self.assertFalse(ids[visible==0].any())

    def test_fast_metrics_match_existing_evaluator(self):
        rng=np.random.default_rng(15)
        ids=rng.integers(0,7,(31,37),dtype=np.uint16)
        gt=[rng.random((31,37))>.75 for _ in range(4)]
        # Overlapping GT masks are retained as independent instances.
        exact=instances(masks_from_ids(ids),gt)
        fast=instance_scores(ids,gt)
        for k,v in fast.items():
            self.assertAlmostEqual(v,exact[k])


if __name__=='__main__':
    unittest.main()
