"""Frozen RGB-only benchmark runners; no ground-truth access in inference.

PADLOS retains the released PGM/depth/ACM algorithms and defaults. Our coarse
front end reproduces the C135 recipe and extends it to overlapping 1024 crops
for larger images. HQ refines only point proposals. Evaluation is separate.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from model_paths import SAM31_REPO, SAM31_CHECKPOINT, HQ_CHECKPOINT

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('outputs/full_comparison_20261009')
os.environ.setdefault('MPLBACKEND', 'Agg')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    records = []
    for image in sorted(Path('datasets/PADLOS_C1C2C3/data/test_imgs').glob('c[123]_*.png')):
        records.append({'dataset': 'c135', 'name': image.stem, 'image': image.as_posix(),
                        'group': image.stem.split('_')[0], 'image_sha256': sha(image),
                        'truth': f'datasets/PADLOS_C1C2C3/data/test_labels/{image.name}'})
    sys.path.insert(0, str(ROOT / 'third_party/wireseg_deps'))
    import pyarrow.parquet as pq
    wire_root = Path('datasets/WireSeg-36K-real')
    inventory = json.loads((wire_root / 'image_manifest.json').read_text(encoding='utf-8'))['records']
    expected = {(r['shard'], r['row']): r['image_sha256'] for r in inventory}
    for shard in sorted(wire_root.glob('real-*.parquet')):
        # Read RGB and descriptive metadata only, never labels or model scores.
        table = pq.read_table(shard, columns=['image', 'file_name', 'scene_family'])
        for index, r in enumerate(table.to_pylist()):
            raw = r['image']['bytes']
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == expected[(shard.name, index)]
            name = f'{shard.stem}_{index:03d}'
            image = OUT / 'inputs/wire300' / f'{name}.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(raw)
            records.append({'dataset': 'wire300', 'name': name, 'image': image.as_posix(),
                            'group': r['scene_family'], 'image_sha256': digest,
                            'shard': shard.as_posix(), 'row': index, 'file_name': r['file_name']})
    for folder in sorted(Path('datasets/scene_000000').glob('view_*')):
        image = folder / 'rgb.png'
        if image.exists():
            records.append({'dataset': 'synthetic20', 'name': folder.name, 'image': image.as_posix(),
                            'group': 'one_synthetic_scene', 'image_sha256': sha(image),
                            'truth_instances': [x.as_posix() for x in sorted((folder/'instances').glob('*_mask.png'))]})
    for image in sorted(Path('datasets/cabinet').glob('*.png')):
        records.append({'dataset': 'cabinet4', 'name': f'cabinet_{image.stem}', 'image': image.as_posix(),
                        'group': 'unlabelled_cabinet', 'image_sha256': sha(image)})
    counts = {d: sum(r['dataset'] == d for r in records) for d in ['c135', 'wire300', 'synthetic20', 'cabinet4']}
    assert counts == {'c135': 135, 'wire300': 300, 'synthetic20': 20, 'cabinet4': 4}, counts
    dump(OUT/'dataset_manifest.json', {'counts': counts, 'ground_truth_read': False, 'records': records})
    print(counts, flush=True)


def freeze():
    paths = [Path(__file__), Path('tools/stage_a_geomprompt_sam31.py'),
             Path('tools/geomprompt_tuned_detector.py'), Path('tools/cable_feedback_policy.py'),
             Path('tools/cable_prompt_geometry.py'), Path('tools/mbest_backend.py'),
             Path('tools/refine_observed_ids.py'), Path('configs/semantic_champion_20261001.json'),
             Path('configs/identity_champion_20261001.json')]
    paths += sorted(Path('datasets/PADLOS_official').rglob('*.py'))
    payload = {'source_sha256': {p.as_posix(): sha(p) for p in paths},
               'dataset_manifest_sha256': sha(OUT/'dataset_manifest.json'),
               'weights': {p.as_posix(): sha(p) for p in [Path('.models/padlos/pgm.pth'),
                   Path('.models/padlos/depth_anything_v2_vits.pth'), Path('.models/hqsam_b/sam_hq_vit_b.pth'),
                   SAM31_CHECKPOINT]},
               'seed': 20260923, 'new_training': False, 'test_label_tuning': False,
               'padlos_commit': '1b951197f177202df28a16f0cb97a344355c5448',
               'padlos_weight_revision': '79879e7d2a8c56ca39f1170168da344af474ee41',
               'ours': {'tile_size': 1024, 'overlap': 256, 'local_text_size': 256,
                        'width_area_factor': 2.0, 'max_points_per_crop': 6,
                        'hq': 'mask_box_refiner_inspired', 'adaptive_feedback': False},
               'padlos': {'encoder': 'vits', 'input_size': 518, 'threshold': 0.05,
                          'acm': {'w_bend': .4, 'w_color': .3, 'w_width': .1, 'w_len': .1, 'k_paths': 100}},
               'scope': 'C135 historical regression; Wire300 first external challenge; synthetic20 correlated views; cabinet4 qualitative',
               'timeout_seconds_per_cpu_case': 180}
    target = OUT/'frozen_protocol.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != payload:
            raise ValueError('Frozen protocol differs; create a new experiment directory')
    else:
        dump(target, payload)
    print('Protocol frozen before benchmark label evaluation', flush=True)


def cases(datasets):
    frozen = json.loads((OUT/'frozen_protocol.json').read_text(encoding='utf-8'))
    for path, digest in frozen['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Frozen source changed: {path}')
    if sha(OUT/'dataset_manifest.json') != frozen['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    all_cases = json.loads((OUT/'dataset_manifest.json').read_text(encoding='utf-8'))['records']
    for r in all_cases:
        if r['dataset'] in datasets:
            if sha(r['image']) != r['image_sha256']:
                raise ValueError(f'Input changed: {r["name"]}')
            yield r


def folder(r, method):
    return OUT / r['dataset'] / method / r['name']


def coarse(records):
    from stage_a_geomprompt_sam31 import (Crop, load_sam31, regular_crops, run_crop_text,
        run_crop_sam, make_square_crop, select_spread, candidates_in_crop, clean_semantic_mask)
    import geomprompt_tuned_detector as detector
    saved = sys.argv
    sys.argv = ['detector']
    dargs = detector.parse_args()
    sys.argv = saved
    predictor = load_sam31(SAM31_CHECKPOINT, SAM31_REPO)
    for idx, r in enumerate(records, 1):
        dest = folder(r, 'ours_coarse')
        if (dest/'inference.json').exists():
            continue
        started = time.perf_counter()
        image = Image.open(r['image']).convert('RGB')
        rgb = np.asarray(image)
        crops = regular_crops(image.width, image.height, 1024, 256)
        text_union = np.zeros(rgb.shape[:2], bool)
        text_records, candidates, diagnostics = [], [], []
        for cidx, crop in enumerate(crops):
            tr, tm = run_crop_text(predictor, image, crop, dest/'crop_cache', 'cable',
                                  proposal_dir=dest/'proposals', raw_proposal_dir=dest/'raw')
            text_records.extend(tr)
            text_union[crop.y0:crop.y1, crop.x0:crop.x1] |= tm
            found, diag = detector.detect_crop_tuned(rgb, crop, dargs, cidx)
            candidates.extend(found)
            diagnostics.append(diag)
        candidates = detector.global_deduplicate(candidates, dargs.global_min_distance)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        candidates = [detector.recenter_candidate(c, lab) for c in candidates]
        min_span = detector.image_scaled_min_span(dargs.global_min_span, image.width, image.height)
        filtered, topology = detector.global_topology_filter(candidates, dargs.global_link_radius,
            dargs.global_min_group_size, min_span, dargs.global_min_mean_tube_score,
            dargs.global_max_chromatic_longitudinal_std, dargs.global_max_achromatic_longitudinal_std,
            dargs.global_min_chroma)
        if len(crops) == 1:
            filtered = select_spread(filtered, 6, dargs.local_min_distance)
        confirmed, local_records = [], []
        for cidx, c in enumerate(filtered):
            crop = make_square_crop(c.x, c.y, 256, image.width, image.height, f'gate_{cidx:03d}', 'local_cable_text_confirmation')
            _, tm = run_crop_text(predictor, image, crop, dest/'crop_cache', 'cable')
            px = int(np.clip(round(c.x - crop.x0), 0, crop.width-1))
            py = int(np.clip(round(c.y - crop.y0), 0, crop.height-1))
            passed = bool(tm[py, px])
            local_records.append({'point': [c.x, c.y], 'passed': passed})
            if passed:
                confirmed.append(c)
        point_union = np.zeros_like(text_union)
        point_records = []
        for crop in crops:
            current = confirmed if len(crops) == 1 else candidates_in_crop(confirmed, crop, 0, 6)
            if not current:
                continue
            pr, pm, _ = run_crop_sam(predictor, image, crop, current, dest/'crop_cache', True,
                width_area_factor=2.0, proposal_dir=dest/'proposals', raw_proposal_dir=dest/'raw')
            point_records.extend(pr)
            point_union[crop.y0:crop.y1, crop.x0:crop.x1] |= pm
        Image.fromarray(text_union.astype(np.uint8)*255).save(dest/'text_mask.png')
        union = clean_semantic_mask(text_union | clean_semantic_mask(point_union))
        Image.fromarray(union.astype(np.uint8)*255).save(dest/'mask.png')
        dump(dest/'inference.json', {'image_sha256': r['image_sha256'], 'seconds': time.perf_counter()-started,
            'ground_truth_used': False, 'text_proposals': text_records, 'point_proposals': point_records,
            'candidates': [asdict(c) for c in confirmed], 'local_text_verification': local_records,
            'diagnostics': diagnostics, 'topology': topology, 'crop_count': len(crops)})
        print(f'coarse [{idx}/{len(records)}] {r["dataset"]}/{r["name"]}: {time.perf_counter()-started:.1f}s', flush=True)


def hq(records):
    import torch
    from stage_a_geomprompt_sam31 import Candidate, Crop, proposal_quality, clean_semantic_mask
    from cable_feedback_policy import spread_interior_points
    sys.path.insert(0, str(ROOT/'third_party/hqsam_deps'))
    from segment_anything_hq import SamPredictor, sam_model_registry
    model = SamPredictor(sam_model_registry['vit_b'](checkpoint=str(HQ_CHECKPOINT)).to('cuda').eval())
    for idx, r in enumerate(records, 1):
        dest = folder(r, 'ours_hq')
        if (dest/'inference.json').exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        source = folder(r, 'ours_coarse')
        details = json.loads((source/'inference.json').read_text(encoding='utf-8'))
        image = Image.open(r['image']).convert('RGB')
        union = np.asarray(Image.open(source/'text_mask.png')) > 0
        started = time.perf_counter()
        last_box, proposals = None, []
        for i, old in enumerate(details['point_proposals']):
            x0, y0, x1, y1 = old['crop_xyxy']
            crop = Crop(f'fixed_{i:03d}', x0, y0, x1, y1)
            candidate = Candidate(**old['candidate'])
            coarse_mask = np.asarray(Image.open(old['mask_path'])) > 0 if old.get('mask_path') else None
            pts, labels = np.array(old['points_local'], np.float32), np.array(old['labels'])
            positive, negative = pts[labels == 1].tolist(), pts[labels == 0].tolist()
            if last_box != old['crop_xyxy']:
                model.set_image(np.asarray(image.crop((x0, y0, x1, y1))))
                last_box = old['crop_xyxy']
            box, mask_input = None, None
            if coarse_mask is not None and coarse_mask.any():
                anchors = spread_interior_points(coarse_mask, 3)
                positive = [positive[0]] + [q for q in anchors if np.linalg.norm(np.array(q)-positive[0]) > candidate.width*2][:2]
                yy, xx = np.where(coarse_mask)
                margin = max(4, int(candidate.width))
                box = np.array([max(0, xx.min()-margin), max(0, yy.min()-margin),
                    min(crop.width-1, xx.max()+margin), min(crop.height-1, yy.max()+margin)], np.float32)
                mh, mw = model.input_size
                probability = cv2.GaussianBlur(coarse_mask.astype(np.float32), (0, 0), 1.5)*.9+.05
                resized = cv2.resize(probability, (mw, mh), interpolation=cv2.INTER_LINEAR)
                canvas = np.full((1024, 1024), .05, np.float32)
                canvas[:mh, :mw] = resized
                soft = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_LINEAR)
                mask_input = np.log(soft/(1-soft))[None]
            with torch.inference_mode():
                masks, scores, _ = model.predict(point_coords=np.array(positive+negative, np.float32),
                    point_labels=np.array([1]*len(positive)+[0]*len(negative), np.int32),
                    box=box, mask_input=mask_input, multimask_output=False)
            accepted, quality = proposal_quality(masks[0], positive[0], candidate_width=candidate.width,
                width_area_factor=2.0, positive_points=positive, negative_points=negative,
                component_mode='prompted' if len(positive)>1 else 'seed')
            component = quality.pop('component_mask', np.zeros_like(masks[0]))
            selected = component if accepted else np.zeros_like(component)
            union[y0:y1, x0:x1] |= selected
            ppath = dest/'proposals'/f'{i:03d}.png'
            ppath.parent.mkdir(exist_ok=True)
            Image.fromarray(selected.astype(np.uint8)*255).save(ppath)
            proposals.append({'accepted': accepted, 'quality': quality, 'score': float(scores[0]),
                              'mask_path': ppath.as_posix(), 'points': positive+negative})
        union = clean_semantic_mask(union)
        Image.fromarray(union.astype(np.uint8)*255).save(dest/'mask.png')
        dump(dest/'inference.json', {'image_sha256': r['image_sha256'], 'ground_truth_used': False,
                                    'seconds': time.perf_counter()-started, 'proposals': proposals})
        print(f'hq [{idx}/{len(records)}] {r["dataset"]}/{r["name"]}', flush=True)


def padlos_pgm(records):
    import torch
    sys.path.insert(0, str(ROOT/'datasets/PADLOS_official'))
    from PDSegmentor.benchmark_pgm import load_model, build_transform, postprocess
    from PDSegmentor.depthinfer import build_model
    device = torch.device('cuda')
    model = load_model(Path('.models/padlos/pgm.pth'), device, 'vits', 1)
    depth_model = build_model('vits', Path('.models/padlos/depth_anything_v2_vits.pth'), device)
    transform = build_transform(518)
    with torch.inference_mode():
        for idx, r in enumerate(records, 1):
            dest = folder(r, 'padlos_pgm')
            if (dest/'inference.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            rgb = np.asarray(Image.open(r['image']).convert('RGB'))
            torch.cuda.synchronize()
            began = time.perf_counter()
            inp = transform({'image': rgb.astype(np.float32)/255})['image']
            tensor = torch.from_numpy(inp).unsqueeze(0).to(device).float()
            mask = postprocess(model(tensor), rgb.shape[:2], .05)
            # Official DepthAnythingV2.infer_image accepts BGR and uses 518.
            depth = depth_model.infer_image(np.ascontiguousarray(rgb[:,:,::-1]), 518)
            depth_gray = ((depth-depth.min())/max(float(depth.max()-depth.min()), 1e-8)*255).astype(np.uint8)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-began
            Image.fromarray(mask).save(dest/'mask.png')
            Image.fromarray(depth_gray).save(dest/'depth.png')
            dump(dest/'inference.json', {'seconds': seconds, 'image_sha256': r['image_sha256'],
                'ground_truth_used': False, 'threshold': .05, 'input_size': 518})
            print(f'pgm [{idx}/{len(records)}] {r["dataset"]}/{r["name"]}: {seconds:.2f}s', flush=True)


def acm_one(r):
    sys.path.insert(0, str(ROOT/'third_party/padlos_deps'))
    sys.path.insert(0, str(ROOT/'datasets/PADLOS_official'))
    from instance_extractor.acm import LeeCCNumbaNeighborWireSegmenter, initialize_endpoint_nodes, reconstruct_instances
    source, dest = folder(r, 'padlos_pgm'), folder(r, 'padlos_acm')
    dest.mkdir(parents=True, exist_ok=True)
    # Relative ASCII paths preserve OpenCV's Windows path compatibility.
    segmenter = LeeCCNumbaNeighborWireSegmenter(r['image'], (source/'mask.png').as_posix(), (source/'depth.png').as_posix())
    began = time.perf_counter()
    segmenter.preprocess()
    segmenter.build_graph()
    initialize_endpoint_nodes(segmenter)
    _, paths = segmenter.optimize_wires_global(w_bend=.4, w_color=.3, w_width=.1, w_len=.1, k_paths=100)
    rendered = reconstruct_instances(segmenter, paths)
    Image.fromarray(rendered).save(dest/'instances_rgb.png')
    colors, inverse = np.unique(rendered.reshape(-1, 3), axis=0, return_inverse=True)
    lookup = np.zeros(len(colors), np.uint16)
    counter = 0
    for i, color in enumerate(colors):
        if np.any(color):
            counter += 1
            lookup[i] = counter
    Image.fromarray(lookup[inverse].reshape(rendered.shape[:2])).save(dest/'instance_ids.png')
    Image.fromarray(np.any(rendered != 0, axis=2).astype(np.uint8)*255).save(dest/'mask.png')
    dump(dest/'inference.json', {'status': 'ok', 'ground_truth_used': False, 'seconds': time.perf_counter()-began,
        'path_count': len(paths), 'rendered_id_count': counter, 'palette_collision': len(paths)>counter})


def cpu_batch(records, method):
    for idx, r in enumerate(records, 1):
        dest = folder(r, method)
        if (dest/'inference.json').exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        if method == 'ours_mbest':
            command = [sys.executable, 'tools/embedded_runner.py', 'tools/mbest_backend.py',
                '--image', r['image'], '--mask', (folder(r, 'ours_hq')/'mask.png').as_posix(),
                '--output', dest.as_posix(), '--fill-visible-band']
        else:
            command = [sys.executable, 'tools/embedded_runner.py', str(Path(__file__).relative_to(ROOT)),
                       '--stage', 'acm_one', '--case', r['name'], '--datasets', r['dataset']]
        status, error = 'ok', None
        try:
            completed = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace', timeout=180)
            (dest/'process.log').write_text(completed.stdout+'\n'+completed.stderr, encoding='utf-8')
            if completed.returncode != 0:
                status, error = 'process_failed', completed.stderr[-3000:]
        except subprocess.TimeoutExpired as ex:
            status, error = 'timeout', str(ex)
        if not (dest/'inference.json').exists():
            shape = np.asarray(Image.open(r['image'])).shape[:2]
            Image.fromarray(np.zeros(shape, np.uint16)).save(dest/'instance_ids.png')
            dump(dest/'inference.json', {'status': status if status != 'ok' else 'missing_prediction',
                'error': error, 'ground_truth_used': False})
        audit = json.loads((dest/'inference.json').read_text(encoding='utf-8'))
        audit['process_seconds_including_import'] = time.perf_counter()-began
        dump(dest/'inference.json', audit)
        print(f'{method} [{idx}/{len(records)}] {r["dataset"]}/{r["name"]}: {audit.get("status")}', flush=True)


def main():
    # Runtime-only compatibility: the released ACM prints Chinese diagnostics
    # in its existing spline fallback; Windows redirected stdout may be cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace', line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['prepare','freeze','coarse','hq','pgm','acm','mbest','acm_one'])
    parser.add_argument('--datasets', nargs='+', default=['c135','wire300','synthetic20','cabinet4'])
    parser.add_argument('--case')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'freeze':
        freeze()
    else:
        records = list(cases(args.datasets))
        if args.case:
            records = [r for r in records if r['name'] == args.case]
        {'coarse': coarse, 'hq': hq, 'pgm': padlos_pgm,
         'acm': lambda rr: cpu_batch(rr, 'padlos_acm'),
         'mbest': lambda rr: cpu_batch(rr, 'ours_mbest'),
         'acm_one': lambda rr: acm_one(rr[0])}[args.stage](records)


if __name__ == '__main__':
    main()
