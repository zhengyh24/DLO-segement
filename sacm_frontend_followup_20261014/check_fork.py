"""Checks a forked continuation against an uninterrupted training fixture."""
import sys,json,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
root=Path('outputs/sacm_followup_20261014/fork_software_check');root.mkdir(exist_ok=False)
rows=[]
for i,split in enumerate(['train','validation']):
 rgb=np.random.default_rng(i).integers(0,256,(35,47,3),dtype=np.uint8);mask=np.zeros((1,35,47),bool);mask[:,10:14,:]=True
 Image.fromarray(rgb).save(root/f'{i}.png');np.savez_compressed(root/f'{i}.npz',masks=mask)
 rows.append(dict(id=str(i),image=f'{i}.png',masks=f'{i}.npz',split=split,group_id=f'fixture{i}',rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
mp=root/'manifest.json';mp.write_text(json.dumps(dict(grouping_kind='verified_scene',independent_scene_split=True,records=rows)))
cfg=json.loads(Path('sacm_frontend/configs/B.json').read_text());cfg.update(backbone='tiny_test',tile=32,stride=24,epochs=1)
cp=root/'one.json';cp.write_text(json.dumps(cfg));cfg2=dict(cfg,epochs=2);cp2=root/'two.json';cp2.write_text(json.dumps(cfg2))
def run(script,config,out,extra=[]):
 subprocess.run([sys.executable,script,'--manifest',str(mp),'--config',str(config),'--out',str(root/out),'--device','cpu','--software-smoke',*extra],check=True)
run('sacm_frontend/train.py',cp,'parent');parent=root/'parent/last.pt';digest=hashlib.sha256(parent.read_bytes()).hexdigest()
run('sacm_frontend_followup_20261014/train.py',cp2,'fork',['--fork-from',str(parent)])
run('sacm_frontend/train.py',cp2,'full')
a=torch.load(root/'fork/last.pt',weights_only=True);b=torch.load(root/'full/last.pt',weights_only=True)
assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
assert hashlib.sha256(parent.read_bytes()).hexdigest()==digest
(root/'check.json').write_text(json.dumps(dict(passed=True,fork_matches_uninterrupted_exactly=True,parent_unchanged=True,accuracy_claim=False)))
print('FORK_CHECK_PASS')
