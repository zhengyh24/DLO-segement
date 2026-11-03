"""Local software checks. All experiment numbers in these fixtures are synthetic."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

import portable_bundle as package


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else package.json_bytes(value))
    return path


def link(target, folder):
    folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        folder.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != 'nt':
            raise
        # Junctions exercise the same Linux symlink layout on non-elevated Windows.
        subprocess.run(['powershell','-NoProfile','-Command',
                        'New-Item -ItemType Junction -Path $env:WIRE_TEST_LINK -Target $env:WIRE_TEST_TARGET | Out-Null'],
                       env={**os.environ,'WIRE_TEST_LINK':str(folder),'WIRE_TEST_TARGET':str(target)},check=True)


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.merged = root/'merged'
        self.output = self.merged/'final_results'
        for name, value in [('completion.json',{'status':'fixture_partial'}),('expected_evaluations.json',[]),('task_receipts.json',{})]:
            write(self.merged/name, value)
        write(self.output/'analysis.json', {'complete':False})
        write(self.output/'RESULTS_ZH.html', b'<a href="../curves/A/epoch_overview.png">loss</a>')
        write(self.output/'per_event'/'test.csv',b'image_id,positive\nfixture,1\n')
        write(self.merged/'test'/'diagnostic_samples'/'predictions.npz',os.urandom(257))
        manifest = write(root/'manifest.json', {'records':[]})
        self.spec = dict(schema=1, merged_runs=str(self.merged), report=str(self.output), versions={}, inputs=[], evidence={})
        self.log = b'line\n'*(package.CHUNK//2)
        for label, job in [('original','A'),('recovery','C')]:
            folder = root/label
            runs = folder/'runs'
            write(runs/'completion.json',{'status':'failed' if label=='original' else 'trained'})
            write(runs/'train'/job/'logs.jsonl', self.log)
            write(runs/'train'/job/'last.pt',b'weights-excluded')
            write(runs/'curves'/job/'epoch_overview.png',b'plot-fixture')
            for part in ('train','curves'):
                link(runs/part/job,self.merged/part/job)
            write(folder/'engine'/'model.py',b'# software fixture\n')
            config = write(folder/'config.json',{'manifest':str(manifest)})
            protocol = write(folder/'protocol.json',dict(jobs=[dict(id=job,config_path=str(config),config={'manifest':str(manifest)})],test_manifest=str(manifest)))
            self.spec['versions'][label]=dict(source=str(folder/'engine'),runs=str(runs),protocol=str(protocol),jobs=[job])
        self.spec['evidence']['first_failure']=str(write(root/'early_fail'/'fail.log',b'fixture failed'))

    def test_linked_artifacts_failures_large_file_retry_and_verify(self):
        receipt = package.build(self.spec,self.output)
        self.assertEqual(receipt,package.verify(self.output))
        manifest=package.read_json(self.output/package.MANIFEST)
        self.assertEqual(manifest['files']['runs/train/A/logs.jsonl']['sha256'],hashlib.sha256(self.log).hexdigest())
        self.assertEqual(manifest['identical_parent_files_by_archive_member']['history/original/train/A/logs.jsonl'],'runs/train/A/logs.jsonl')
        for member in ('runs/curves/A/epoch_overview.png','runs/final_results/per_event/test.csv','runs/test/diagnostic_samples/predictions.npz',
                       'versions/original/source/model.py','versions/recovery/source/model.py','evidence/first_failure/fail.log','history/original/completion.json'):
            self.assertIn(member,manifest['files'])
        self.assertFalse(any(name.endswith('.pt') for name in manifest['files']))
        self.assertEqual(package.build(self.spec,self.output),receipt)
        # Interrupted after committing the zip but before either sidecar existed.
        (self.output/package.RECEIPT).unlink()
        (self.output/package.MANIFEST).unlink()
        self.assertEqual(package.build(self.spec,self.output),receipt)
        (self.output/package.MANIFEST).write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'manifest differs'):
            package.verify(self.output)

    def test_external_directory_link_is_rejected(self):
        outside=self.merged.parent/'unrelated'
        write(outside/'evidence.json',{})
        link(outside,self.merged/'unapproved_link')
        with self.assertRaisesRegex(ValueError,'outside declared'):
            package.build(self.spec,self.output)
        self.assertFalse((self.output/package.RECEIPT).exists())

    def test_complete_claim_requires_parent_training_receipts(self):
        write(self.output/'analysis.json',{'complete':True})
        with self.assertRaisesRegex(ValueError,'unverified parent artifacts'):
            package.build(self.spec,self.output)
        self.assertFalse((self.output/package.ARCHIVE).exists())


if __name__=='__main__':
    unittest.main()
