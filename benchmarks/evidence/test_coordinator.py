import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import coordinator as c


class CoordinatorTests(unittest.TestCase):
    def test_real_frozen_roster_routes_exact_work(self):
        workspace=Path(__file__).resolve().parents[2]
        old=Path('/original/runs_r2'); new=Path('/separate/evaluations')
        protocol=c.read(workspace/'results/sam_tracker/protocol_fixture.json')
        original=copy.deepcopy(protocol); protocol=c.relocate(protocol,old,new)
        c.set_engine(workspace/'cable_seg/versions'/c.R2)
        from sam_path_v1.evidence_run import EvidenceQueue
        expected=EvidenceQueue(protocol,new).expected_evaluations
        plan=c.commands(protocol,expected,new)
        self.assertEqual(len(plan),157)
        self.assertEqual(c.Counter(row['engine'] for row in plan),{c.R2:139,c.RECOVERY:18})
        self.assertEqual([row['id'] for row in expected],[row['id'] for row in EvidenceQueue(original,old).expected_evaluations])
        for row in plan:
            argv=row['arguments'];destination=Path(argv[argv.index('--output')+1]);checkpoint=Path(argv[argv.index('--checkpoint')+1]);
            self.assertEqual(destination,Path(row['report']).parent)
            self.assertTrue(destination.is_relative_to(new));self.assertTrue(checkpoint.is_relative_to(new/'train'))
            self.assertEqual(checkpoint.name,(row.get('tag') or 'best')+'.pt')
            if row['kind']=='synthetic':self.assertIn('--identity-diagnostics',argv)
        for old_item,new_item in zip(original['additional_evaluations'],protocol['additional_evaluations']):
            self.assertEqual(old_item['inputs'],new_item['inputs'])

    def test_extra_commands_refuse_checkpoint_escape(self):
        old=Path('/old');new=Path('/new')
        item=dict(module='scalability',job='native80_C_seed42',tag='best',report='/old/real/dir/metrics.json',
            arguments=['--checkpoint','/another/checkpoint.pt','--output','/old/real/dir'])
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            c.relocate({'additional_evaluations':[item]},old,new)

    def test_resume_adopts_finished_child_and_refuses_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);task={'id':'test_A','report':str(root/'test/a/best/metrics.json')}
            result=Path(task['report']);c.write(result,{'complete':True})
            history=root/'attempts/test_A/0001';c.write(history/'exit.json',dict(code=0,signature='x'))
            coordinator=object.__new__(c.Coordinator);coordinator.root=root;coordinator.args=SimpleNamespace(retry_failed=False)
            with patch.object(coordinator,'signature',return_value='x'),patch.object(coordinator,'register',return_value='adopted'):
                self.assertEqual(coordinator.run_task(task,allow_execute=False),'adopted')
                c.write(history/'receipt.json',dict(signature='x',complete=True,files=c.inventory(result.parent)))
                c.write(result,{'complete':False})
                with self.assertRaisesRegex(ValueError,'artifacts changed'):coordinator.run_task(task,allow_execute=False)

    def test_finalize_cannot_launch_unfinished_gpu_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);task={'id':'test_A','report':str(root/'test/a/best/metrics.json')}
            c.write(root/'attempts/test_A/0001/exit.json',dict(code=1,signature='x'))
            coordinator=object.__new__(c.Coordinator);coordinator.root=root;coordinator.args=SimpleNamespace(retry_failed=False)
            with patch.object(coordinator,'signature',return_value='x'),patch('subprocess.run') as run:
                with self.assertRaisesRegex(ValueError,'Finalize requires'):coordinator.run_task(task,allow_execute=False)
                run.assert_not_called()

    def test_failed_retry_preserves_previous_output_and_never_uses_old_training_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); dest=root/'test/a/best'
            task={'id':'test_A','engine':c.R2,'module':'evaluate','arguments':[], 'report':str(dest/'metrics.json')}
            c.write(task['report'],{'complete':False}); history=root/'attempts/test_A/0001'
            c.write(history/'exit.json',dict(code=2,signature='x'))
            c.write(history/'receipt.json',dict(signature='x',files=c.inventory(dest),complete=False))
            coordinator=object.__new__(c.Coordinator);coordinator.root=root;coordinator.args=SimpleNamespace(retry_failed=True)
            coordinator.engine_files={c.R2:{}};coordinator.sources={c.R2:root/'engine'};coordinator.lock_fd=4;coordinator.gpu_lock_fd=7
            coordinator.queue=SimpleNamespace(task_receipts=[])
            def child(argv,**kwargs):
                spec=c.read(argv[-1]);self.assertEqual(spec['lock_fds'],[4,7]);self.assertEqual(kwargs['pass_fds'],(4,7))
                c.write(spec['exit_receipt'],dict(code=0,signature='x'));c.write(task['report'],{'complete':True})
            with patch.object(coordinator,'signature',return_value='x'),patch.object(coordinator,'register',return_value='done'),patch('subprocess.run',side_effect=child):
                self.assertEqual(coordinator.run_task(task),'done')
            self.assertFalse(c.read(history/'preserved_output/metrics.json')['complete'])
            self.assertTrue(c.read(task['report'])['complete'])


if __name__=='__main__':unittest.main()
