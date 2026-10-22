import unittest
import numpy as np
from cable_dual_continuation import dual_stitch


class DualTests(unittest.TestCase):
    def scene(self):
        ids=np.zeros((80,120),np.uint16);ids[37:42,10:51]=1;ids[37:42,57:100]=2
        return np.full((80,120,3),120,np.uint8),ids>0,ids

    def test_no_long_gap_merge_without_independent_support(self):
        rgb,visible,ids=self.scene()
        result,audit=dual_stitch(rgb,visible,ids,[],np.zeros_like(ids))
        self.assertNotEqual(result[39,20],result[39,80])

    def test_geometric_agreement_can_link_without_hidden_painting(self):
        rgb,visible,ids=self.scene()
        result,audit=dual_stitch(rgb,visible,ids,[],visible.astype(np.uint16),True)
        self.assertEqual(result[39,20],result[39,80])
        self.assertFalse(result[~visible].any())
        self.assertEqual(audit['foreground_changed_pixels'],0)

    def test_visible_seam_links_only_under_seam_policy(self):
        rgb,visible,ids=self.scene();ids[:]=0;ids[37:42,10:55]=1;ids[37:42,55:100]=2
        result,audit=dual_stitch(rgb,ids>0,ids,[],np.zeros_like(ids),False)
        self.assertEqual(result[39,20],result[39,80])
        strict,audit=dual_stitch(rgb,ids>0,ids,[],np.zeros_like(ids),True)
        self.assertNotEqual(strict[39,20],strict[39,80])


if __name__=='__main__':unittest.main()
