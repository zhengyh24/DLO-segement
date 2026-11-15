"""Check exact retained harnesses, aligned views, and evaluated new hardware."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_common import fingerprint
from qa_scene_contacts import mesh, overlap

CLAMP = 'round_cushioned_cable_clamp'
CABINET = 'industrial_cabinet_component'


def read(path):
    return json.loads(path.read_text())


def mesh_set(parts, labels):
    deps = bpy.context.evaluated_depsgraph_get()
    return {obj.name: mesh(obj, deps) for obj in bpy.context.scene.objects
            if obj.type in {'MESH', 'FONT', 'CURVE'} and labels(parts.get(str(obj.pass_index), {}))}


def connected_components(infos, root_name, tolerance=.00015):
    """Find contact paths to the mounting plate through evaluated geometry."""
    names = list(infos)
    adjacent = {name: set() for name in names}
    for i, name in enumerate(names):
        a = infos[name]
        for other in names[i+1:]:
            b = infos[other]
            if np.any(a['hi'] + tolerance < b['lo']) or np.any(b['hi'] + tolerance < a['lo']):
                continue
            hit = a['bvh'].overlap(b['bvh'])
            if not hit:
                # The bounding boxes already intersect to the stated tolerance.
                ab = any(b['bvh'].find_nearest(Vector(v), tolerance)[0] is not None for v in a['v'])
                ba = any(a['bvh'].find_nearest(Vector(v), tolerance)[0] is not None for v in b['v'])
                if not (ab or ba):
                    continue
            adjacent[name].add(other)
            adjacent[other].add(name)
    seen = {root_name}
    queue = [root_name]
    while queue:
        name = queue.pop()
        unseen = adjacent[name] - seen
        seen.update(unseen)
        queue.extend(unseen)
    return seen


def audit(old, current):
    old_parts = read(old / 'part_map.json')
    new_parts = read(current / 'part_map.json')
    retained = set(read(current / 'scene.json')['retained_assembly_ids'])
    before_target_parts = {pid: part for pid, part in old_parts.items()
                           if part['target'] and part['assembly_id'] in retained}
    assert {pid: part for pid, part in new_parts.items() if part['target']} == before_target_parts
    # Check the actual target meshes, their material slots and material nodes.
    bpy.ops.wm.open_mainfile(filepath=str(old / 'scene.blend'))
    before = fingerprint()
    bpy.ops.wm.open_mainfile(filepath=str(current / 'scene.blend'))
    after = fingerprint()
    unchanged = [part['object_name'] for part in before_target_parts.values()]
    for name in unchanged:
        assert before['objects'][name] == after['objects'][name], name
        for material in before['objects'][name]['materials']:
            assert before['materials'][material] == after['materials'][material], material
    with np.load(old / 'centerline_gt.npz') as a, np.load(current / 'centerline_gt.npz') as b:
        wire_ids = {part['wire_id'] for part in before_target_parts.values() if part['wire_id']}
        expected_axes = {key for key in a.files if int(key.split('_')[1]) in
                         (retained if key.startswith('assembly_') else wire_ids)}
        assert set(b.files) == expected_axes
        for key in b.files:
            assert np.array_equal(a[key], b[key]), key
    for key in ('world', 'color_management', 'cycles_invariants', 'render_invariants'):
        assert before[key] == after[key], key
    for name, old_obj in before['objects'].items():
        if old_obj['type'] == 'LIGHT':
            assert old_obj == after['objects'][name], name

    tracked = {p['object_name']: p for p in new_parts.values()}
    targets = mesh_set(new_parts, lambda p: p.get('target'))
    clamps = mesh_set(new_parts, lambda p: CLAMP in p.get('feature_tags', []))
    contacts = []
    for name, fixture in clamps.items():
        info = tracked[name]
        assert not info['target'] and info['assembly_id'] == 0 and info['semantic_name'] == 'clip_tie'
        assert fixture['closed'], (name, 'expected a solid closed clamp component')
        for target, wire in targets.items():
            if not overlap(fixture, wire):
                continue
            count = len(fixture['bvh'].overlap(wire['bvh']))
            if count:
                t = tracked[target]
                assert 'clamp_liner' in info['feature_tags'], (name, target, count)
                assert info['attached_assembly_id'] == t['assembly_id'], (name, target)
                assert t['semantic_name'] in {'cloth_cover', 'braid_cover', 'tape_cover'}, (name, target)
                contacts.append({'liner': name, 'jacket': target, 'triangle_pairs': count})
    roots = 'cream equipment mounting panel'
    cabinet = mesh_set(new_parts, lambda p: p.get('object_name') == roots or CABINET in p.get('feature_tags', []))
    # Apply the structural check to modeled hardware, apart from thin ink/lettering.
    structural = {name: info for name, info in cabinet.items()
                  if 'equipment_marking' not in tracked[name]['feature_tags']}
    connected = connected_components(structural, roots)
    unknown_mounts = sorted(set(structural) - connected)

    frame_records = []
    camera_delta = 0.
    for view in sorted(current.glob('view_*')):
        cam = read(view / 'camera.json')
        prior = read(old / view.name / 'camera.json')
        assert cam['K'] == prior['K']
        assert cam['image_wh'] == prior['image_wh'] == [1024, 1024]
        error = np.max(abs(np.asarray(cam['T_world_from_camera']) - prior['T_world_from_camera']))
        assert error < 1e-6, (view.name, error)
        camera_delta = max(camera_delta, float(error))
        assert abs(cam['actual_nearest_cable_surface_distance_m']-prior['actual_nearest_cable_surface_distance_m']) < 1e-6
        assert cam['visible_assembly_ids'] == sorted(retained)
        visible = read(view / 'part_visibility.json')
        band_pixels = {str(p['attached_assembly_id']): int(visible.get(part_id, 0))
                       for part_id,p in new_parts.items() if 'clamp_band' in p.get('feature_tags', [])}
        assert len(band_pixels) == len(cam['visible_assembly_ids']) and min(band_pixels.values()) >= 60, band_pixels
        component_pixels = sum(int(visible.get(part_id, 0)) for part_id,p in new_parts.items()
                               if CABINET in p.get('feature_tags', []))
        assert component_pixels > 20000, (view.name, component_pixels)
        frame_records.append({'view': view.name, 'band_visible_pixels_by_assembly': band_pixels,
                              'cabinet_component_visible_pixels': component_pixels})
    assert len(frame_records) == 4
    common = {name: after['objects'][name] for name,part in tracked.items()
              if not part['attached_assembly_id']}
    common_digest = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
    return {'status':'passed' if not unknown_mounts else 'failed',
            'source_scene':str(old), 'target_meshes_value_equal':len(unchanged),
            'blend_sha256':hashlib.sha256((current/'scene.blend').read_bytes()).hexdigest(),
            'common_cabinet_geometry_sha256':common_digest,
            'lens_and_pose_max_abs_delta':camera_delta,'views':frame_records,
            'clamp_components_checked':len(clamps), 'liner_jacket_contacts':contacts,
            'cabinet_components_mounted':len(connected)-1,
            'cabinet_components_checked':len(structural)-1,
            'unconnected_structural_parts':unknown_mounts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    result = audit(args.old, args.scene)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('views','liner_jacket_contacts')}),flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)
