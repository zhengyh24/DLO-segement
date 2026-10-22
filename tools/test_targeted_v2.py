"""Focused regression checks for competition semantics and fixed sample budgets."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from train_curvi_targeted_v2 import rank_loss, EndpointSampler


class TargetedTrainingTests(unittest.TestCase):
    def test_multiple_positives_unknown_and_gradient(self):
        ends=[(('image',1,0),('image',k,0)) for k in range(2,6)]
        x=torch.tensor([0.,0.,1.,9.],requires_grad=True)
        loss=rank_loss(x,[1,1,0,-1],ends)
        self.assertAlmostEqual(loss.item(),1.5)
        loss.backward()
        self.assertLess(x.grad[0].item(),0)
        self.assertLess(x.grad[1].item(),0)
        self.assertGreater(x.grad[2].item(),0)
        self.assertEqual(x.grad[3].item(),0)
        self.assertEqual(rank_loss(x,[1,1,1,-1],ends).item(),0)
        self.assertEqual(rank_loss(x,[0,0,0,-1],ends).item(),0)

    def test_different_images_never_compete(self):
        x=torch.tensor([0.,1.],requires_grad=True)
        ends=[(('image1',1,0),('image1',2,0)),(('image2',1,0),('image2',2,0))]
        self.assertEqual(rank_loss(x,[1,0],ends).item(),0)

    def test_sampler_fixed_budget_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            candidates=[dict(key=str(i),a=1,b=i+2,a_end=0,b_end=0,gap=30,
                a_tip=dict(direction=[1,0]),b_tip=dict(direction=[-1,0])) for i in range(3)]
            (path/'graph.json').write_text(json.dumps(dict(candidates=candidates)))
            records=[dict(graph_item='graph.json',candidate_index=i,candidate=str(i),file_name='image',label=y)
                for i,y in enumerate([1,1,0])]
            for hard in (False,True):
                a=EndpointSampler(records,path/'manifest.json',np.random.default_rng(42))
                b=EndpointSampler(records,path/'manifest.json',np.random.default_rng(42))
                for _ in range(20):
                    sampled=a.sample(8,hard)
                    self.assertEqual(sampled,b.sample(8,hard))
                    self.assertEqual(len(sampled),8)
                    self.assertTrue(all(0<=i<3 for i in sampled))


if __name__=='__main__':unittest.main()
