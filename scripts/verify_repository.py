"""Validate published Python sources and bundled dataset without using the GPU."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def main():
    sources = []
    for folder in ('tools', 'scripts', 'sacm_frontend', 'sacm_frontend_followup_20261014',
                   'sacm_fullframe_20261020', 'curvi_instance_v1', 'curvi_identity', 'curvi_experiments_20261022'):
        for p in (ROOT/folder).rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            sources.append(p)
    root = ROOT/'data/wireseg_demo'
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    groups, rgb_hashes, ids = {}, {}, set()
    counts = {}
    for row in manifest['records']:
        assert row['id'] not in ids, 'duplicate ID'
        ids.add(row['id'])
        for key, hashkey in (('image', 'image_sha256'), ('masks', 'masks_sha256')):
            p = (root/row[key]).resolve()
            assert p.is_relative_to(root.resolve()), 'path escaped data directory'
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row[hashkey], str(p)
        rgb = np.asarray(Image.open(root/row['image']).convert('RGB'))
        with np.load(root/row['masks'], allow_pickle=False) as z:
            masks = z['masks']
        assert list(rgb.shape[:2]) == row['shape']
        assert masks.dtype == bool and masks.shape == (row['num_instances'], *row['shape'])
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256']
        if row.get('decoded_masks_sha256'):
            assert hashlib.sha256(str(tuple(masks.shape)).encode('ascii') + b'|bool|' + np.ascontiguousarray(masks).tobytes()).hexdigest() == row['decoded_masks_sha256']
        for store, key in ((groups,row['group_id']), (rgb_hashes,row['rgb_sha256'])):
            assert key not in store or store[key] == row['split'], 'split overlap'
            store[key] = row['split']
        counts[row['split']] = counts.get(row['split'],0) + 1
    assert counts == manifest['counts'] == {'train':36, 'validation':12}
    print(json.dumps({'status':'passed','python_files':len(sources),'native_samples':len(ids),'counts':counts}))

if __name__ == '__main__':
    main()
