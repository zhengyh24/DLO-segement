"""Bind verified scene, geometry and pixel results and prepare actual render details."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root',type=Path,required=True)
root = parser.parse_args().run_root.resolve()
receipts = root/'receipts'
manifest = root/'data/manifest.jsonl'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


rows = [json.loads(line) for line in manifest.read_text().splitlines()]
assert len(rows) == 12
assert len({row['image_id'] for row in rows}) == 12
common_digests = set()
scenes = []
for tier in ('easy1','medium2','easy3'):
    folder = root/'data'/tier
    expected_blend = sha(folder/'scene.blend')
    tile = {'scene_id':tier}
    for check in ('cabinet_revision','close_data','scene_contacts','support_connections'):
        path = receipts/f'qa_{tier}_{check}.json'
        report = read(path)
        assert report['status'] in ('passed','passed_with_explicit_scope'), (path,report.get('failures'))
        assert report['blend_sha256'] == expected_blend
        tile[check] = {'file':path.name,'sha256':sha(path),'status':report['status']}
        if check == 'cabinet_revision':
            common_digests.add(report['common_cabinet_geometry_sha256'])
            tile['target_meshes_preserved'] = report['target_meshes_value_equal']
            tile['mounts_connected'] = report['cabinet_components_mounted']
            tile['visible_geometry'] = report['views']
        if check == 'close_data':
            assert report['num_views'] == 4
            for view in report['views']:
                for file,digest in view['file_sha256'].items():
                    assert sha(folder/view['view']/file) == digest, file
    scenes.append(tile)
assert len(common_digests) == 1, 'Different background geometry across tiers'
for row in rows:
    for key,digest in row['view_file_sha256'].items():
        assert sha(root/'data'/row['paths'][key]) == digest

summary = {
    'status':'passed_with_explicit_scope','artifact_root':str(root/'data'),
    'manifest_sha256':sha(manifest),'num_scenes':3,'num_views':12,
    'background_geometry_identical_across_difficulties':True,'scenes':scenes,
    'scope': 'All source target meshes/materials, solved axes and calibrated viewpoints retained. Evaluated clamp and cabinet models checked for attachment and target intersection; full pixel hierarchy and sampled first-hit geometry checked for all 12 renders. Simulated material and sensor parameters inherited.',
}
(receipts/'final_qa.json').write_text(json.dumps(summary,indent=2)+'\n')

# Select the front clamp from the part-ID pixels of the hard view.
new_view = root/'data/easy3/view_000'
parts = read(new_view.parent/'part_map.json')
band = next(int(pid) for pid,p in parts.items()
            if p['attached_assembly_id']==3 and 'clamp_band' in p['feature_tags'])
ys,xs = np.where(np.load(new_view/'part_id.npy') == band)
cx,cy = int(np.median(xs)),int(np.median(ys))
length = 240
cx = max(length//2,min(1024-length//2,cx))
cy = max(length//2,min(1024-length//2,cy))
bounds = cx-length//2,cy-length//2,cx+length//2,cy+length//2
panel = Image.open(new_view/'rgb.png').convert('RGB').crop(bounds)

assets = root/'delivery/final/assets'
assets.mkdir(parents=True,exist_ok=True)
panel.save(assets/'clamp_detail.png')
(receipts/'clamp_comparison_source.json').write_text(json.dumps({
    'crop_bounds_xyxy':bounds,'source':'easy3/view_000/rgb.png',
    'source_sha256':sha(new_view/'rgb.png'),
},indent=2)+'\n')
print(json.dumps({'scenes':len(scenes),'frames':len(rows),'manifest_sha256':sha(manifest),
                  'primary_clamp_detail':str(assets/'clamp_detail.png')},ensure_ascii=False))
