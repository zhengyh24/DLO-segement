"""Extract a trusted task-generated code archive and run it outside the project."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        with tarfile.open(a.archive) as tf:
            members=tf.getmembers()
            for item in members:
                path=Path(item.name)
                if not item.isfile() or path.is_absolute() or '..' in path.parts or path.parts[0]!='sacm_frontend':
                    raise ValueError('Unexpected archive member')
                if path.suffix not in ('.py','.md','.txt','.json') and path.name!='LICENSE':
                    raise ValueError('Non-source archive member')
            tf.extractall(root,filter='data')
        run=subprocess.run([sys.executable,str(root/'sacm_frontend/check.py')],cwd=root,
            capture_output=True,text=True)
        if run.returncode:
            print(run.stdout);print(run.stderr);raise RuntimeError('Isolated package checks failed')
        checks=json.loads((root/'sacm_frontend/checks/software_smoke.json').read_text())
        report=dict(archive=str(a.archive),files=len(members),isolated_directory=True,
            exit_code=run.returncode,checks=checks,credentials_included=False,
            note='Source allowlist only. This does not validate a Linux GPU installation.')
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
