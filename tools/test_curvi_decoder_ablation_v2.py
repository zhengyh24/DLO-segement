"""Deterministic decoder parity and independent-switch regression checks."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curvi_identity.config import Config
from curvi_identity.graph import Graph, decode as frozen_decode
from curvi_decoder_ablation_v2 import decode


def fixture(edges, n=4):
    # Each atomic fragment has three native pixels, separated by real background.
    partition = np.zeros((5, n * 4), np.uint16)
    for i in range(n):
        partition[2, i*4:i*4+3] = i+1
    candidates = [dict(key=f'{a}:{ae}-{b}:{be}', a=a, b=b,
                       a_end=ae, b_end=be, gap=4.) for a,ae,b,be in edges]
    visible = partition > 0
    return Graph(visible, visible.copy(), partition.copy(), partition,
                 {}, candidates, {})


def threshold_reference(graph, scores, threshold):
    # Frozen threshold_ids algorithm, no pipeline/model import needed.
    parent={int(k):int(k) for k in np.unique(graph.partition) if k}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for c,s in zip(graph.candidates,scores):
        if min(s)>=threshold:parent[find(c['a'])]=find(c['b'])
    lookup=np.zeros(int(graph.partition.max())+1,np.uint16);ids={}
    for x in parent:
        root=find(x)
        if root not in ids:ids[root]=len(ids)+1
        lookup[x]=ids[root]
    return lookup[graph.partition]


class DecoderTests(unittest.TestCase):
    def test_full_defaults_random_parity(self):
        rng = np.random.default_rng(20261020)
        graph=fixture([(a,ae,b,be) for a in range(1,5) for b in range(a+1,5)
                       for ae in (0,1) for be in (0,1)])
        original=graph.partition.copy()
        for _ in range(30):
            scores=rng.random((len(graph.candidates),2))
            cfg=replace(Config(),threshold=.45,margin=.04,conflict_threshold=.2)
            actual,audit=decode(graph,scores,cfg)
            expected,old=frozen_decode(graph,scores,cfg)
            np.testing.assert_array_equal(actual,expected)
            for key in ('selected','transitive_vetoed','ids_after','solver'):
                self.assertEqual(audit[key],old[key])
            np.testing.assert_array_equal(actual>0,graph.visible)
            np.testing.assert_array_equal(graph.partition,original)

    def test_threshold_only_parity(self):
        rng=np.random.default_rng(42)
        graph=fixture([(1,0,2,0),(1,0,3,0),(2,1,4,0),(3,0,4,0)])
        for threshold in (.2,.5,.8,1.):
            cfg=replace(Config(),threshold=threshold,conflict_threshold=.1)
            for _ in range(10):
                scores=rng.random((4,2))
                out,_=decode(graph,scores,cfg,use_margin=False,use_matching=False,use_conflict=False)
                np.testing.assert_array_equal(out,threshold_reference(graph,scores,threshold))

    def test_multiple_legal_competitors_margin_really_disabled(self):
        graph=fixture([(1,0,2,0),(1,0,3,0)])
        # Supervision could mark both candidates as same identity. The decoder
        # never receives those labels; disabling margin must retain BOTH.
        scores=np.array([[.99,.99],[.90,.90]])
        cfg=replace(Config(),margin=0.)
        _,enabled=decode(graph,scores,cfg,use_matching=False,use_conflict=False)
        out,disabled=decode(graph,scores,cfg,use_margin=False,use_matching=False,use_conflict=False)
        self.assertEqual(len(enabled['selected']),1)
        self.assertEqual(len(disabled['selected']),2)
        self.assertEqual(len(np.unique(out[out>0])),2)
        _,matched=decode(graph,scores,cfg,use_margin=False,use_matching=True,use_conflict=False)
        self.assertEqual(len(matched['selected']),1)
        self.assertEqual(matched['rejected']['1:0-3:0'],'endpoint_matching')
        np.testing.assert_array_equal(out>0,graph.visible)

    def test_conflict_is_independent(self):
        graph=fixture([(1,0,2,0),(2,1,3,0),(1,1,3,1)])
        scores=[[.99,.99],[.98,.98],[.01,.01]]
        _,on=decode(graph,scores,use_margin=False,use_matching=False)
        out,off=decode(graph,scores,use_margin=False,use_matching=False,use_conflict=False)
        self.assertEqual(len(on['selected']),1)
        self.assertEqual(len(off['selected']),2)
        self.assertEqual(on['rejected']['2:1-3:0'],'transitive_conflict')
        np.testing.assert_array_equal(out>0,graph.visible)

    def test_empty_and_no_candidates(self):
        graph=fixture([])
        out,audit=decode(graph,[])
        np.testing.assert_array_equal(out,graph.partition)
        self.assertEqual(audit['stages'],[])
        graph.partition[:]=0;graph.visible[:]=False
        out,_=decode(graph,[])
        self.assertEqual(int(out.sum()),0)

    def test_invalid_scores(self):
        graph=fixture([(1,0,2,0)])
        for scores in ([[float('nan'),.9]],[[1.1,.9]],[]):
            with self.assertRaises(ValueError):decode(graph,scores)
        with self.assertRaises(ValueError):decode(graph,[[.9,.9]],use_margin=0)


if __name__=='__main__':
    unittest.main(verbosity=2)
