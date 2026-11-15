"""Export and verify a portable training archive from the approved v7 frames."""
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path, required=True)
root = parser.parse_args().run_root.resolve()
source = root / 'data'
delivery = root / 'delivery'
name = 'panel_harness_rgbd_v7'
dataset = delivery / name
archive = delivery / (name + '.zip')
source_manifest = source / 'manifest.jsonl'
qa = json.loads((root / 'receipts/final_qa.json').read_text())
tiers = {'easy1': ('easy', 1), 'medium2': ('medium', 2), 'easy3': ('hard', 3)}
depth_keys = {
    'depth_input': 'input',
    'depth_z_gt': 'z_gt',
    'depth_range_gt': 'range_gt',
    'depth_valid': 'valid',
}
mask_keys = ('assembly', 'branch', 'wire', 'junction', 'semantic', 'part_id',
             'annotation_valid', 'branch_annotation_valid')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


source_hash = sha(source_manifest.read_bytes())
assert qa['status'] == 'passed_with_explicit_scope'
assert qa['manifest_sha256'] == source_hash
dataset.mkdir()
rows = []
scene_files = {}
original_rows = [json.loads(line) for line in source_manifest.read_text().splitlines()]
for original in original_rows:
    scene = original['scene_id']
    tier, count = tiers[scene]
    view = Path(original['paths']['rgb']).parent.name
    stem = f'{tier}_{view}'
    paths = {}
    original_files = {}
    for key, digest in original['view_file_sha256'].items():
        value = (source / original['paths'][key]).read_bytes()
        assert sha(value) == digest, (stem, key)
        original_files[key] = value

    def put(key, relative, data):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths[key] = relative

    def array(key):
        return np.load(io.BytesIO(original_files[key]), allow_pickle=False)

    put('rgb', f'rgb/{stem}.png', original_files['rgb'])
    put('camera', f'camera/{stem}.json', original_files['camera'])
    for key, folder in depth_keys.items():
        put(key, f'depth/{folder}/{stem}.npy', original_files[key])

    for key in (*mask_keys, 'foreground'):
        if key == 'foreground':
            pixels = (array('assembly') > 0).astype(np.uint8) * 255
        elif key.endswith('_valid'):
            pixels = array(key).astype(np.uint8) * 255
        else:
            pixels = array(key).astype(np.uint16)
        paths[key] = f'mask/{key}/{stem}.png'
        save_png(dataset / paths[key], pixels)

    rgb = np.asarray(Image.open(io.BytesIO(original_files['rgb'])).convert('RGB'))
    paths['rgbd'] = f'rgbd/{stem}.npz'
    rgbd = dataset / paths['rgbd']
    rgbd.parent.mkdir(exist_ok=True)
    np.savez_compressed(rgbd, rgb=rgb, depth=array('depth_input'), depth_valid=array('depth_valid'))

    if scene not in scene_files:
        relative = Path('scene') / tier
        (dataset / relative).mkdir(parents=True)
        keys = ('scene.json', 'centerline_gt.npz', 'topology.json', 'part_map.json', 'feature_anchors.json', 'cabinet_revision.json', 'support_routes.json')
        scene_files[scene] = {}
        for key in keys:
            path = relative / key
            shutil.copyfile(source / scene / key, dataset / path)
            scene_files[scene][key] = path.as_posix()

    rows.append({
        'sample_id': stem, 'scene_id': scene, 'difficulty': tier, 'view_id': view,
        'group_id': original['group_id'], 'split': original['split'],
        'assembly_count': count, 'visible_assembly_ids': original['visible_assembly_ids'],
        'width': rgb.shape[1], 'height': rgb.shape[0], 'source_image_id': original['image_id'],
        'paths': paths, 'scene_files': scene_files[scene],
        'sha256': {key: sha((dataset / path).read_bytes()) for key, path in paths.items()},
    })

assert Counter(row['difficulty'] for row in rows) == {'easy': 4, 'medium': 4, 'hard': 4}
assert len({row['sample_id'] for row in rows}) == len(rows)
(dataset / 'manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
save_json(dataset / 'dataset_card.json', {
    'name': name, 'frames': len(rows), 'resolution': [1024, 1024],
    'assemblies_by_difficulty': {'easy': 1, 'medium': 2, 'hard': 3},
    'depth_units': 'm', 'rgbd_depth': 'simulated axial Z',
    'original_manifest_sha256': source_hash,
    'source_qa_sha256': sha((root / 'receipts/final_qa.json').read_bytes()),
    'split': 'quality_pilot_only', 'scene_families': 1,
    'revision': 'Circular cushioned P clamps and modeled control-cabinet background. Exact retained cable meshes and original camera poses.',
})
(dataset / 'README.md').write_text("""# 控制柜近景线束 RGB-D 数据集（v7）

保留原 1、2、3 束的线束几何与相机机位。卡箍使用圆形金属带、缓冲衬垫和带安装孔的紧固耳。控制柜导轨、端子、线槽和模块为三维部件，和夹具一样参加 RGB、深度和部件标签渲染。卡箍是辅助部件，保留在语义及部件掩码中。

| 档位 | 线束数量 | 视角数 | 帧编号前缀 |
| --- | ---: | ---: | --- |
| 简单 | 1 | 4 | easy |
| 中等 | 2 | 4 | medium |
| 困难 | 3 | 4 | hard |

12 帧均为 1024×1024。以 `manifest.jsonl` 为入口，每行一帧，包含相对路径、原始帧编号、场景组和逐文件 SHA-256。三档使用同一场景家族和同一组相机机位。划分字段为 `quality_pilot_only`。

| 目录 | 内容 |
| --- | --- |
| `rgb/` | 原始 RGB PNG |
| `rgbd/` | NPZ，`rgb` 为 H×W×3 的 uint8 RGB，`depth` 为 H×W 的 float32 模拟轴向深度（米），`depth_valid` 为有效掩码 |
| `depth/` | 原始 NPY，`input` 为模拟轴向深度，`z_gt` 为相机轴 Z 真值，`range_gt` 为射线距离真值，`valid` 为模拟深度有效掩码。深度单位均为米 |
| `mask/` | `foreground` 为整束前景二值图（0/255）。`assembly`、`branch`、`wire`、`junction`、`semantic`、`part_id` 为原始数值 ID 的 16 位 PNG。两种标注有效性图使用 0/255 |
| `camera/` | 原始内参 K、世界与相机之间的双向变换。相机坐标采用 OpenCV 约定，像素中心为索引 + 0.5 |
| `scene/` | 各档的中心线、拓扑、部件 ID 映射和场景元数据 |

整束掩码覆盖可见包覆、热缩和导线。单线掩码只标可见导线。交汇区使用独立的 `junction` 层，并在 `branch_annotation_valid` 中置为无效。原始实例 ID 跨档保留，因此简单档的整束 ID 为 3，中等档为 2 和 3。读取 16 位掩码时保留整数像素值。场景 JSON 中的云端路径用于来源追踪，读取数据使用清单中的相对路径。

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(".")  # 在解压后的数据集目录运行
sample = json.loads((root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
with np.load(root / sample["paths"]["rgbd"], allow_pickle=False) as frame:
    rgb, depth_m = frame["rgb"], frame["depth"]
    valid = frame["depth_valid"]
assembly_ids = np.asarray(Image.open(root / sample["paths"]["assembly"]))
camera = json.loads((root / sample["paths"]["camera"]).read_text(encoding="utf-8"))
```
""", encoding='utf-8')

# Read the actual archive back. Verify all file bytes plus lossless decoding.
files = sorted(p for p in dataset.rglob('*') if p.is_file())
hashes = {p.relative_to(dataset).as_posix(): sha(p.read_bytes()) for p in files}
(dataset / 'SHA256SUMS').write_text(''.join(f'{digest}  {path}\n' for path, digest in hashes.items()))
files.append(dataset / 'SHA256SUMS')
temporary = archive.with_suffix('.zip.partial')
with zipfile.ZipFile(temporary, 'w') as package:
    for path in files:
        compression = zipfile.ZIP_STORED if path.suffix in {'.png', '.npz'} else zipfile.ZIP_DEFLATED
        package.write(path, Path(name) / path.relative_to(dataset), compress_type=compression, compresslevel=5)

with zipfile.ZipFile(temporary) as package:
    assert package.testzip() is None
    assert len(package.namelist()) == len(files)
    assert set(package.namelist()) == {f'{name}/{path}' for path in (*hashes, 'SHA256SUMS')}
    for path, digest in hashes.items():
        assert sha(package.read(f'{name}/{path}')) == digest, path
    for row, original in zip(rows, original_rows, strict=True):
        read = lambda key: package.read(f'{name}/' + row['paths'][key])
        original_array = lambda key: np.load(source / original['paths'][key], allow_pickle=False)
        assert read('rgb') == (source / original['paths']['rgb']).read_bytes()
        assert read('camera') == (source / original['paths']['camera']).read_bytes()
        for key in depth_keys:
            assert read(key) == (source / original['paths'][key]).read_bytes()
        for key in (*mask_keys, 'foreground'):
            with Image.open(io.BytesIO(read(key))) as image:
                mask = np.asarray(image)
                assert mask.shape == (1024, 1024)
                if key == 'foreground':
                    expected = (original_array('assembly') > 0) * 255
                elif key.endswith('_valid'):
                    expected = original_array(key).astype(np.uint8) * 255
                else:
                    assert image.mode in {'I', 'I;16'}
                    expected = original_array(key)
                assert np.array_equal(mask, expected), (row['sample_id'], key)
        with np.load(io.BytesIO(read('rgbd')), allow_pickle=False) as frame:
            assert frame['rgb'].shape == (1024, 1024, 3) and frame['rgb'].dtype == np.uint8
            with Image.open(io.BytesIO(read('rgb'))) as image:
                assert np.array_equal(frame['rgb'], np.asarray(image.convert('RGB')))
            assert frame['depth'].dtype == np.float32
            assert np.array_equal(frame['depth'], original_array('depth_input'))
            assert np.array_equal(frame['depth_valid'], original_array('depth_valid'))
assert not archive.exists(), archive
temporary.rename(archive)
receipt = {'status': 'passed', 'archive': str(archive), 'bytes': archive.stat().st_size,
           'sha256': sha(archive.read_bytes()), 'frames': len(rows), 'files': len(files),
           'source_manifest_sha256': source_hash}
save_json(root / 'receipts/dataset_archive.json', receipt)
print(json.dumps(receipt), flush=True)
