import copy
import hashlib
import unittest

from sam_path_v1.evidence_protocol import build_matrix, additional_evaluations, validate_membership
from sam_path_v1.train import TrainConfig


class ProtocolTests(unittest.TestCase):
    def test_finite_matrix_counts_fresh_seeds_and_shared_defaults(self):
        template = dict(manifest='pilot.json', model=dict(variant='C',backend='sam31'),data_root='data',
                        allow_condition=True,device='cuda',seed=42,epochs=5)
        unchanged = copy.deepcopy(template)
        jobs = build_matrix(template, 'study', sam2_repo='sam2',sam2_checkpoint='sam2.pt')
        self.assertEqual(template, unchanged)
        self.assertEqual(len(jobs),42)
        self.assertEqual(len({j['id'] for j in jobs}),42)
        self.assertEqual(sum(j['stage']=='main' for j in jobs),30)
        self.assertEqual(sum(j['stage']=='sam21' for j in jobs),9)
        self.assertEqual(sum(j['stage']=='native80' for j in jobs),3)
        for job in jobs:
            config=TrainConfig(**job['config'])
            self.assertEqual(config.epochs,5)
            self.assertIsNone(config.max_batches)
            self.assertEqual(config.model['init_seed'],job['seed'])
            self.assertEqual(config.manifest,'pilot.json')
        extra=additional_evaluations(jobs,'runs',real_manifest='w.json',real_root='wire',c135_root='c135',handloom_root='handloom')
        self.assertEqual(len(extra),30)
        self.assertEqual(sum(x['expected_images']==110 for x in extra),18)
        self.assertTrue(all(x['inputs'] and x['report'] for x in extra))

    def test_membership_rejects_overlap_duplicate_and_changed_counts(self):
        def row(i,split):return dict(id=str(i),rgb_sha256=hashlib.sha256(str(i).encode()).hexdigest(),split=split)
        pilot=dict(grouping_kind='condition_holdout_pilot',records=[row(i,'train' if i<2000 else 'validation') for i in range(2300)])
        full=dict(records=[row(i,'test') for i in range(2300,4100)])
        self.assertFalse(validate_membership(pilot,full)['test_is_new_blind'])
        bad=copy.deepcopy(full);bad['records'][0]['rgb_sha256']=pilot['records'][0]['rgb_sha256']
        with self.assertRaises(ValueError):validate_membership(pilot,bad)
        bad=copy.deepcopy(full);bad['records'][1]['id']=bad['records'][0]['id']
        with self.assertRaises(ValueError):validate_membership(pilot,bad)
        bad=copy.deepcopy(full);bad['records'].pop()
        with self.assertRaises(ValueError):validate_membership(pilot,bad)


if __name__=='__main__':unittest.main()
