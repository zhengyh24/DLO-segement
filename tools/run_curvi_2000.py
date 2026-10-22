"""Resumable local 2000-image experiment; CWD remains the project root.

All expensive artifacts live on D:. RGB frontend runs in a separate process
with an RGB-only manifest. GT enters only offline pair supervision/evaluation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from curvi_identity.config import Config
from curvi_identity.io import read_json, read_rgb, read_mask, rgb_hash

DEFAULT_OUT = ROOT/'run'
DATA = ROOT/'data/WireSeg-36K'
PILOT = ROOT/'outputs/curvi_identity_pilot_20261014'


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def status(out, stage, **kwargs):
    value = dict(stage=stage, updated=stamp(), pid=os.getpid(), **kwargs)
    save(out/'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def selected_rows(groups, columns):
    from curvi_identity.data import parquet_module
    pq = parquet_module(); by_shard = defaultdict(list)
    for r in groups['records']: by_shard[r['shard']].append(r)
    for shard, items in sorted(by_shard.items()):
        p = pq.ParquetFile(DATA/'data'/shard); offset = 0
        for rg in range(p.num_row_groups):
            count = p.metadata.row_group(rg).num_rows
            current = [r for r in items if offset <= r['row'] < offset+count]
            if current:
                table = p.read_row_group(rg, columns=columns)
                for r in current:
                    row = table.slice(r['row']-offset, 1).to_pylist()[0]
                    if row['file_name'] != r['file_name']: raise ValueError('Dataset index changed')
                    yield r, row
                del table
            offset += count


def setup(out):
    from curvi_identity.data import group_map, decoded_rgb
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'groups.json').exists():
        inventory = read_json(ROOT/'outputs/wireseg_download_check_20261013/image_inventory.json')['records']
        excluded = {r['file_name'] for r in read_json(PILOT/'groups.json')['records']}
        pools = defaultdict(list)
        for r in inventory:
            if r['file_name'] not in excluded: pools[f"{r['scene_family']}/{r['n_cable']}_cable"].append(r)
        val = {'data_center/8_cable', 'flying_wire/4_cable', 'wire_on_plane/4_cable'}
        train_conditions = sorted(set(pools)-val)
        assert len(train_conditions) == 6 and len(pools) == 9
        quotas = {c: 333+(i < 2) for i, c in enumerate(train_conditions)}
        quotas.update({c: 100 for c in val})
        rng = np.random.default_rng(20261014); records = []
        for c, pool in sorted(pools.items()):
            pool = sorted(pool, key=lambda r: r['file_name'])
            for index in sorted(rng.choice(len(pool), quotas[c], replace=False)):
                row = {k: pool[index][k] for k in ('file_name','shard','row','scene_family','n_cable')}
                records.append(dict(row, split='validation' if c in val else 'train', group_id='condition:'+c))
        save(out/'groups.json', dict(grouping_kind='condition_holdout_pilot', independent_scene_split=False,
             grouping_basis='Whole family/count conditions held out; simulation/run/view IDs unavailable. No independent-scene claim.',
             selection_basis='Seeded filename-only stratified sample; exclude all 27 previously inspected pilot images. No GT, QA or prediction selection.',
             seed=20261014, quotas=quotas, records=records))
    groups, _ = group_map(out/'groups.json')
    assert Counter(r['split'] for r in groups['records']) == {'train':2000, 'validation':300}
    # The pre-existing pilot protocol already cross-checked the original frozen baseline.
    frozen = read_json(PILOT/'frontend_protocol.json')
    for name, digest in frozen['source_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Frozen source changed: '+name)
    for name, digest in frozen['weights_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Frozen weight changed: '+name)
    if not (out/'frontend_protocol.json').exists(): save(out/'frontend_protocol.json', frozen)
    cfg = Config(training_enabled=True)
    if (out/'config.json').exists():
        if read_json(out/'config.json') != cfg.to_dict(): raise ValueError('Run config changed')
    else: save(out/'config.json', cfg.to_dict())
    if not (out/'protocol.json').exists():
        save(out/'protocol.json', dict(created=stamp(), train_images=2000, validation_images=300,
            grouping_kind=groups['grouping_kind'], independent_scene_split=False,
            validation_role='Checkpoint selection by minimum bidirectional BCE; no blind test claim.',
            training=cfg.to_dict(), training_scope='PairNet-v1 only; SAM/HQ frozen',
            resolution='Native 1024 RGB; unresized 128/256 crops',
            inference_gt=False, real300_used=False, prior_pilot_images_excluded=True,
            pair_storage='per-image NPZs, all purity-qualified candidates; zero-pair images retained',
            baselines=['train-positive-prior pair classifier','no-link graph partition'],
            decoder_ablation=['full_default','bidirectional_threshold_only'],
            thresholds_fixed_before_results=dict(threshold=.8, margin=.1, conflict_threshold=.2),
            epochs=10, seed=20261013, cpu_threads=4, num_workers=0,
            resume='Epoch-boundary checkpoint restores optimizer and random states; interrupted epoch is replayed.',
            limitations=['Condition holdout not verified scene independence','Single seed','No claim against coverage_color without a same-input run'],
            sources='Existing curvi_identity implementation: R21-inspired continuation, R03/R05/R06/R19-inspired fragment association; not a Curvi-Tracker reproduction.'))
    # Snapshot and lock all code used by the new orchestration. No old source is edited.
    if not (out/'source_hashes.json').exists():
        sources = list((ROOT/'curvi_identity').glob('*.py')) + [ROOT/'tools'/n for n in
            ('run_curvi_2000.py','train_curvi_2000.py','predict_curvi_2000.py','evaluate_curvi_2000.py','run_curvi_pilot_frontend.py')]
        hashes = {}
        for source in sources:
            relative = source.relative_to(ROOT); target = out/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            hashes[relative.as_posix()] = sha(source)
        save(out/'source_hashes.json', hashes)
    if not (out/'inference_inputs.json').exists():
        dest = out/'rgb'; dest.mkdir(exist_ok=True); records=[]; hashes={}
        for r, row in selected_rows(groups, ['file_name','image']):
            rgb = decoded_rgb(row['image'])
            if rgb.shape != (1024,1024,3): raise ValueError('Unexpected native resolution')
            digest = rgb_hash(rgb)
            if digest in hashes: raise ValueError('Exact selected RGB duplicate: '+r['file_name'])
            hashes[digest] = r['file_name']
            name = f"{r['scene_family']}_{r['n_cable']}_{Path(r['file_name']).stem}"
            path = dest/(name+'.png')
            if path.exists():
                if rgb_hash(read_rgb(path)) != digest: raise ValueError('Export mismatch')
            else: Image.fromarray(rgb).save(path)
            records.append(dict(dataset='scale2300', name=name, image=path.as_posix(), image_sha256=sha(path),
                                rgb_sha256=digest, file_name=r['file_name']))
            if len(records)%100 == 0: status(out, 'export_rgb', completed=len(records), total=2300)
        assert len(records)==2300
        save(out/'inference_inputs.json', dict(records=records, ground_truth_read=False, native_rgb=True))
        save(out/'rgb_dedup.json', dict(images=2300, unique_pixel_hashes=len(hashes), duplicates=0,
                                      scope='Exact selected RGB only; not scene-independence certification'))
    if not (out/'geometry_duplicate_check.json').exists():
        hashes={}; duplicates=[]
        for r, row in selected_rows(groups, ['file_name','masks_rle']):
            key = hashlib.sha256(json.dumps(sorted(json.dumps(x,sort_keys=True) for x in json.loads(row['masks_rle']))).encode()).hexdigest()
            if key in hashes: duplicates.append(dict(first=hashes[key], second=r, cross_split=hashes[key]['split']!=r['split']))
            hashes[key]=r
        save(out/'geometry_duplicate_check.json', dict(images=2300, duplicates=duplicates,
             scope='Exact per-instance RLE equality only; no near-duplicate or scene-independence certification'))
    if any(r['cross_split'] for r in read_json(out/'geometry_duplicate_check.json')['duplicates']):
        raise ValueError('Cross-split identical geometry; stop for protocol review')
    status(out, 'setup_complete', train_images=2000, validation_images=300)


def finalize(out):
    rows=[]
    for r in read_json(out/'inference_inputs.json')['records']:
        for stage in ('ours_coarse','ours_hq'):
            directory = out/'frontend/scale2300'/stage/r['name']
            info = read_json(directory/'inference.json')
            if info['image_sha256'] != r['image_sha256'] or info['ground_truth_used'] is not False:
                raise ValueError('Frontend provenance mismatch')
            if read_mask(directory/'mask.png').shape != (1024,1024): raise ValueError('Non-native mask')
        rows.append(dict(file_name=r['file_name'], foreground=(directory/'mask.png').as_posix(),
                         rgb_sha256=r['rgb_sha256'], producer='Frozen SAM3.1/GeoMPrompt + HQ; frontend_protocol.json', gt_used=False))
    save(out/'predictions.json', dict(records=rows, independent_scene_split=False))


def pairs(out):
    from curvi_identity.data import group_map, arc_labels, decode_masks
    from curvi_identity.graph import build_graph
    from curvi_identity.features import extract_pair
    from curvi_identity.train import validate_manifest
    cfg=Config.read(out/'config.json'); groups,_=group_map(out/'groups.json')
    inputs={r['file_name']:r for r in read_json(out/'inference_inputs.json')['records']}
    predictions=read_json(out/'predictions.json'); pred={r['file_name']:r for r in predictions['records']}
    prepared=out/'prepared'; prepared.mkdir(exist_ok=True); records=[]; images=[]
    for r, row in selected_rows(groups,['file_name','masks_rle']):
        inp=inputs[r['file_name']]; directory=prepared/'by_image'/inp['name']; done=directory/'done.json'
        if done.exists():
            obj=read_json(done)
            if obj['image']['rgb_sha256'] != inp['rgb_sha256']: raise ValueError('Pair resume hash mismatch')
            if any(not (prepared/p['path']).exists() for p in obj['records']): raise ValueError('Missing resumed pair')
        else:
            if shutil.disk_usage(out).free < 8*1024**3: raise RuntimeError('D: free space below 8 GiB; stop without deletion')
            directory.mkdir(parents=True,exist_ok=True); rgb=read_rgb(inp['image']); cache=pred[r['file_name']]
            if rgb_hash(rgb)!=cache['rgb_sha256']: raise ValueError('RGB changed before supervision')
            fg=read_mask(cache['foreground'])>0; graph=build_graph(fg,cfg)
            assigned,stats=arc_labels(graph,decode_masks(row['masks_rle'],fg.shape),cfg.purity)
            current=[]
            for c in graph.candidates:
                if c['a'] not in assigned or c['b'] not in assigned: continue
                label=int(assigned[c['a']]==assigned[c['b']]); s,l,g=extract_pair(rgb,graph,c,cfg)
                path=directory/f'{len(current):04d}.npz'
                with path.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f,small=s,large=l,geometry=g,label=np.float32(label))
                os.replace(path.with_suffix('.tmp'),path)
                current.append(dict(path=path.relative_to(prepared).as_posix(), file_name=r['file_name'],
                                    group_id=r['group_id'],split=r['split'],rgb_sha256=cache['rgb_sha256'],label=label,candidate=c['key']))
            audit=dict(file_name=r['file_name'],shard=str(DATA/'data'/r['shard']),row=r['row'],split=r['split'],
                       group_id=r['group_id'],rgb_sha256=cache['rgb_sha256'],scene_family=r['scene_family'],
                       prediction_producer=cache['producer'],graph=graph.audit,supervision=stats,pairs=len(current),
                       positive=sum(p['label'] for p in current),ignored_pairs=len(graph.candidates)-len(current))
            obj=dict(image=audit,records=current);save(done,obj)
        records.extend(obj['records']);images.append(obj['image'])
        if len(images)%10==0: status(out,'prepare_pairs',images=len(images),total=2300,pairs=len(records))
    assert len(images)==2300
    manifest=dict(schema=1,kind='predicted_native_pair_supervision',config=cfg.to_dict(),groups=groups,
                  prediction_manifest=predictions,images=images,records=records,training_performed=False,real_subset_used=False)
    validate_manifest(manifest,cfg); save(prepared/'manifest.json',manifest)
    summary={}
    for split in ('train','validation'):
        rr=[r for r in records if r['split']==split]; ii=[r for r in images if r['split']==split]
        summary[split]=dict(images=len(ii),pairs=len(rr),positive=sum(r['label'] for r in rr),
                           negative=sum(1-r['label'] for r in rr),images_without_pairs=[r['file_name'] for r in ii if r['pairs']==0])
    save(out/'preparation_summary.json',summary)
    status(out,'pairs_complete',splits=summary)


def child(out, script, args, label):
    logs=out/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'{label}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    command=[sys.executable,'-X','utf8','-u','tools/padlos_runtime_runner.py',script,*map(str,args)]
    with path.open('w',encoding='utf-8') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status(out,label,child_pid=p.pid,log=path.as_posix())
        code=p.wait()
    if code: raise RuntimeError(f'{label} failed with exit {code}; see {path}')


def run(out):
    setup(out)
    for relative,digest in read_json(out/'source_hashes.json').items():
        if sha(ROOT/relative)!=digest: raise ValueError('Source changed after freeze: '+relative)
    for stage in ('coarse','hq'):
        if not (out/f'{stage}_status.json').exists():
            child(out,'tools/run_curvi_pilot_frontend.py',[stage,'--output',out],stage)
    finalize(out)
    if not (out/'prepared/manifest.json').exists(): pairs(out)
    if not (out/'fit/completion.json').exists(): child(out,'tools/train_curvi_2000.py',['--output',out],'training')
    if not (out/'validation_predictions/completion.json').exists(): child(out,'tools/predict_curvi_2000.py',['--output',out],'identity_inference')
    if not (out/'evaluation.json').exists(): child(out,'tools/evaluate_curvi_2000.py',['--output',out],'evaluation')
    status(out,'complete',report=(out/'results.md').as_posix())


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace');sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['setup','pairs','run']);p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    a=p.parse_args();out=a.output.resolve();os.chdir(ROOT)
    try:
        out.mkdir(parents=True,exist_ok=True)
        # Windows releases this byte-range lock even if the supervisor crashes.
        # It prevents two launchers from writing the same run concurrently.
        import msvcrt
        with (out/'run.lock').open('a+b') as lock:
            if lock.tell()==0:lock.write(b'0');lock.flush()
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            {'setup':setup,'pairs':pairs,'run':run}[a.stage](out)
    except Exception:
        out.mkdir(parents=True,exist_ok=True);status(out,'failed',error=traceback.format_exc());raise
