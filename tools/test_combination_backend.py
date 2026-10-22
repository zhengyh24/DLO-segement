import unittest
import cv2
import numpy as np
from cable_combination_backend import stitch_ids,choose_continuations,global_path_cover
from cable_evidence_fusion import Proposal


class CombinationTests(unittest.TestCase):
    def scene(self):
        rgb=np.full((80,120,3),125,np.uint8);ids=np.zeros((80,120),np.uint16)
        ids[37:42,10:51]=1;ids[37:42,57:100]=2
        return rgb,ids>0,ids

    def test_continuation_links_identity_without_painting_gap(self):
        rgb,visible,ids=self.scene()
        result,audit=stitch_ids(rgb,visible,ids,[],use_evidence=False)
        self.assertEqual(result[39,20],result[39,80])
        self.assertFalse(result[39,51:57].any())
        self.assertEqual(audit['foreground_changed_pixels'],0)

    def test_independent_masks_can_reject_false_join(self):
        rgb,visible,ids=self.scene();proposals=[]
        for box in [(0,0,120,80),(0,1,120,79)]:
            x0,y0,x1,y1=box
            for ident in [1,2]:
                p=Proposal(box,(ids[y0:y1,x0:x1]==ident),'hq_point',.9,str((box,ident)),quality=.9,width=5.)
                proposals.append(p)
        result,audit=stitch_ids(rgb,visible,ids,proposals,use_evidence=True)
        self.assertNotEqual(result[39,20],result[39,80])

    def test_global_matching_avoids_greedy_endpoint_conflict(self):
        edges=[dict(i=0,j=1,benefit=1.05,ids=[1,2]),dict(i=0,j=2,benefit=.80,ids=[1,3]),dict(i=1,j=3,benefit=.80,ids=[2,4])]
        chosen,audit=choose_continuations(edges,4,True)
        self.assertEqual({(e['i'],e['j']) for e in chosen},{(0,2),(1,3)})

    def test_parallel_wires_remain_separate(self):
        rgb,visible,ids=self.scene();ids[:]=0
        ids[20:25,10:100]=1;ids[45:50,10:100]=2
        result,audit=stitch_ids(rgb,ids>0,ids,[],use_evidence=False)
        self.assertNotEqual(result[22,50],result[47,50])

    def test_cover_preserves_identical_source_partition(self):
        rgb,visible,ids=self.scene()
        result,audit=global_path_cover(rgb,visible,ids,ids,[],ids)
        self.assertTrue(np.array_equal(result>0,visible))
        self.assertTrue(np.array_equal(result==result[39,20],ids==1))
        self.assertTrue(np.array_equal(result==result[39,80],ids==2))


if __name__=='__main__':unittest.main()
