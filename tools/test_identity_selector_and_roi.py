import unittest
import numpy as np
from cable_identity_selector import select_identity
from run_accuracy_roi_probe_v2 import jobs


class SelectionTests(unittest.TestCase):
    def test_exact_query_budget_match_for_small_and_large_images(self):
        for h,w in [(359,640),(1512,2016),(2196,3904)]:
            mask=np.zeros((h,w),bool);mask[h//2:h//2+5,10:w-10]=True
            ids=mask.astype(np.uint16)
            policies,_=jobs(mask,ids,ids)
            self.assertEqual(len(policies['uniform']),len(policies['disagreement']))
            self.assertLessEqual(len(policies['uniform']),4)

    def test_complete_geometry_is_retained_on_single_crop(self):
        visible=np.ones((20,40),bool)
        old=np.ones(visible.shape,np.uint16)
        proposal=np.full(visible.shape,2,np.uint16)
        rgb=np.full((20,40,3),80,np.uint8)
        result,audit=select_identity(rgb,visible,old,old,proposal,1,'coverage_color')
        self.assertTrue(np.array_equal(result,old))
        self.assertEqual(audit['selected_backend'],'fill_geodesic')


if __name__=='__main__':unittest.main()
