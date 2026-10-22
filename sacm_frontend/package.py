"""Allowlist source bundle: no checkpoints, datasets, logs or credentials."""
import argparse
from pathlib import Path
import tarfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise FileExistsError(a.out)
    files=list(root.glob('*.py'))+[root/'README.md',root/'requirements.txt']
    for folder,patterns in [('configs',['*.json']),('vendor',['*.py','LICENSE']),('implement-stage',['*.md','*.json'])]:
        for pattern in patterns:files.extend((root/folder).glob(pattern))
    with tarfile.open(a.out,'w:gz') as tf:
        for path in sorted(set(files)):tf.add(path,arcname='sacm_frontend/'+path.relative_to(root).as_posix(),recursive=False)
    print(f'PACKAGED {len(set(files))} files; {a.out.stat().st_size} bytes; {a.out}')
if __name__=='__main__':main()
