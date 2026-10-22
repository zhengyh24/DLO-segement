import unittest
import numpy as np
from refine_observed_ids import fill_observed_ids

class ObservedIDsTest(unittest.TestCase):
    def test_no_hidden_bridge_or_unobserved_component(self):
        ids=np.zeros((40,80),np.uint16); ids[10:14,5:25]=1
        mask=np.zeros_like(ids,bool);mask[9:15,5:30]=True;mask[9:15,50:70]=True
        out,audit=fill_observed_ids(ids,mask)
        self.assertGreater(audit['filled_visible_pixels'],0)
        self.assertFalse(out[:,35:].any())
        self.assertFalse(np.any((out>0)&~mask))
        np.testing.assert_array_equal(out[ids>0],ids[ids>0])

if __name__=='__main__':unittest.main()
