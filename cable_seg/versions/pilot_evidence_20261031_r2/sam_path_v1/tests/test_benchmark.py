from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

import numpy as np
from PIL import Image
import torch

from sam_path_v1.benchmark import (run_benchmark, capacity_cost_scan, finalize_benchmark,
    benchmark_diagnostic_problems, benchmark_forward, CAPACITY_QUERIES)


class RGBModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__(); self.config=SimpleNamespace(canvas_size=16); self.fail=fail
    def forward(self,rgb,valid):
        if self.fail:raise RuntimeError('intentional benchmark failure')
        return dict(mask_logits=rgb[:,:1]*0,object_logits=torch.ones(1,1),calls={'encoder':1,'decoder':1})
    def parameter_counts(self):return dict(total=0,trainable=0,new_modules=0)


class BenchmarkTests(unittest.TestCase):
    def test_only_rgb_required_and_all_repetitions_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.fromarray(np.full((9,13,3),42,np.uint8)).save(Path(folder)/'a.png')
            rows=[dict(id='a',image='a.png',masks='ground_truth_does_not_exist.npz')]
            result=run_benchmark(RGBModel(),rows,folder,device='cpu',warmup=1,repetitions=2)
            self.assertEqual(result['observations'],2)
            self.assertEqual(result['failures'],0)
            self.assertEqual(result['per_observation'][0]['original_size'],[9,13])
            self.assertEqual(result['per_observation'][0]['encoded_views'],1)
            self.assertNotIn('modules_seconds',result['per_observation'][0])
            self.assertEqual(len(result['module_profiles']),1)
            self.assertTrue(result['complete'])
            self.assertEqual(result['diagnostic_failures'],0)
            self.assertTrue(result['runtime']['outer_autocast_disabled'])
            self.assertFalse(result['runtime']['internal_bf16_enabled'])

    def test_model_forward_does_not_inherit_training_autocast(self):
        class CheckAutocast(RGBModel):
            def forward(self,rgb,valid):
                self.autocast_enabled=torch.is_autocast_enabled('cpu')
                return super().forward(rgb,valid)
        model=CheckAutocast()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            benchmark_forward(model,torch.zeros(1,3,16,16),torch.ones(1,1,16,16,dtype=torch.bool))
        self.assertFalse(model.autocast_enabled)

    def test_profile_failure_retains_valid_main_timing_but_is_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.fromarray(np.zeros((8,8,3),np.uint8)).save(Path(folder)/'a.png')
            with patch('sam_path_v1.benchmark.profile_modules', side_effect=RuntimeError('profile failed')):
                result=run_benchmark(RGBModel(),[dict(id='a',image='a.png')],folder,device='cpu',warmup=1,repetitions=1)
            self.assertTrue(result['timing_complete'])
            self.assertIsNotNone(result['summaries']['end_to_end_seconds'])
            self.assertEqual(result['failures'],0)
            self.assertEqual(result['diagnostic_failures'],1)
            self.assertFalse(result['complete'])

    def test_capacity_oom_recorded_and_parameters_restored(self):
        class CapacityModel(RGBModel):
            def __init__(self):
                super().__init__(); self.condition=torch.nn.Module()
                self.condition.queries=torch.nn.Parameter(torch.zeros(24,4))
            def forward(self,rgb,valid):
                if len(self.condition.queries)==48: raise torch.OutOfMemoryError('fixture out of memory')
                return super().forward(rgb,valid)
        with tempfile.TemporaryDirectory() as folder:
            Image.fromarray(np.zeros((8,8,3),np.uint8)).save(Path(folder)/'a.png')
            model=CapacityModel(); original=model.condition.queries
            result=capacity_cost_scan(model,dict(id='a',image='a.png'),folder,'cpu')
            self.assertIs(model.condition.queries,original)
            self.assertEqual([r['queries'] for r in result['results']],list(CAPACITY_QUERIES))
            self.assertEqual(result['results'][3]['failure_kind'],'oom')
            self.assertEqual(result['diagnostic_failures'],1)
            self.assertFalse(result['complete'])
            self.assertEqual(result['results'][4]['forward_seconds']['n'],10)

    def test_required_capacity_points_and_module_evidence_are_independent_of_flags(self):
        report=dict(images=1,observations=1,repetitions=1,failures=0,
            per_observation=[dict(id='a',repeat=0,error=None)],
            module_profiles=[dict(id='a',modules_seconds={key:.01 for key in ('encoder','decoder','condition','descriptor','tracker')})],
            query_capacity_cost=dict(available=True,results=[dict(queries=q,trained_at_this_capacity=False,accuracy_evaluated=False,
                forward_seconds=dict(n=10,mean=.01,median=.01,p95=.01,maximum=.01)) for q in CAPACITY_QUERIES]))
        self.assertTrue(finalize_benchmark(deepcopy(report),'C')['complete'])
        for change in ('missing','duplicate','nonfinite','missing_module'):
            broken=deepcopy(report)
            if change=='missing': broken['query_capacity_cost']['results'].pop()
            elif change=='duplicate': broken['query_capacity_cost']['results'][-1]['queries']=48
            elif change=='nonfinite': broken['query_capacity_cost']['results'][0]['forward_seconds']['mean']=float('nan')
            else: del broken['module_profiles'][0]['modules_seconds']['tracker']
            broken['complete']=True; broken['diagnostic_failures']=0
            self.assertTrue(benchmark_diagnostic_problems(broken,'C'),change)
            self.assertFalse(finalize_benchmark(broken,'C')['complete'],change)
        native=deepcopy(report); del native['query_capacity_cost']
        native['module_profiles'][0]['modules_seconds']={'encoder':.01,'decoder':.01}
        self.assertTrue(finalize_benchmark(native,'S')['complete'])
    def test_no_implicit_resize(self):
        from sam_path_v1.benchmark import rgb_batch
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'large.png';Image.fromarray(np.zeros((17,16,3),np.uint8)).save(path)
            with self.assertRaises(ValueError):rgb_batch(path,16,'cpu')


if __name__=='__main__':unittest.main()
