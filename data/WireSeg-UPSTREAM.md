---
license: cc-by-4.0
pretty_name: WireSeg-36K
task_categories:
- image-segmentation
- depth-estimation
tags:
- DLO
- segmentation
- cables
- wire
- sam
- real image
- synthetic image
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
- config_name: real
  data_files:
  - split: test
    path: real/*.parquet
---

# WireSeg-36K

A 36,000-image dataset for **cable / wire instance segmentation**: physically simulated
deformable linear objects (DLOs) rendered over diverse backgrounds, with exact per-instance
masks and per-image metric depth. Single category: `cable`. Generated with **DeformX**.

- **Images:** 36,000 (1024x1024 RGB)
- **Instances:** 176,387 cable masks (COCO RLE)
- **Categories:** 1 (`cable`)
- **Modalities:** RGB, instance masks, per-image metric depth

## Scenes

| scene_family | cable counts | images | description |
|---|---|---|---|
| `data_center`   | 4 / 8 / 16 | 6,000  | wires routed on server racks |
| `flying_wire`   | 2 / 4 / 8  | 15,000 | suspended / catenary wires |
| `wire_on_plane` | 2 / 4 / 8  | 15,000 | wires draped on a flat surface |

## Files & layout

| path | what |
|---|---|
| `data/*.parquet` (config `default`) | RGB image + per-instance COCO-RLE masks + metadata, one row per image |
| `real/*.parquet` (config `real`) | **real-image eval subset** (300 imgs) — RGB + COCO-RLE masks, one row per image |
| `annotations.json` | full COCO ground truth (36,000 images / 176,387 annotations, single category `cable`) |
| `depth/*.tar.gz` | per-image metric depth, float32 `[1024,1024]` `.npy`, packed in 100 gzipped tar shards |

> The `image-only` branch holds the RGB+masks release without depth.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("DeformX/WireSeg-36K", split="train")   # rgb + masks + metadata
ex = ds[0]
ex["image"]                              # PIL.Image (RGB)

import json
from pycocotools import mask as M
masks = json.loads(ex["masks_rle"])      # list of {"size":[h,w], "counts": <RLE str>}
m0 = M.decode(masks[0])                   # HxW uint8 instance mask
```

**Depth.** Download and extract the `depth/*.tar.gz` shards; each member is
`<scene_family>/<n>_cable/<id>.npy`. The depth for an image is the `.npy` at the same
relative path as its `file_name` (swap `.png` -> `.npy`):

```python
import numpy as np
d = np.load(f"depth/{ex['file_name'][:-4]}.npy")   # float32 [1024,1024], metric
```

## Data fields (Parquet `default`)

`image` (RGB), `file_name`, `scene_family`, `n_cable`, `masks_rle`, `bboxes`, `areas`,
`num_instances`, `wire_area_frac`, `det_f1_iou_50`, `det_f1_iou_75`, `det_map_50_95`,
`jaccard_j`, `category`. Masks / bboxes / areas are JSON-encoded per-instance lists
(RLE = COCO compressed).

## Real eval subset (config `real`)

A companion set of **300 real cable images** with hand-verified ground truth, for evaluating
sim-to-real transfer of the synthetic `default` set. These were cherry-picked as the
**disagreement slice** between base **SAM 3** (prompt `"cable"`) and a cable-tuned **LoRA** —
i.e. the images the two models most disagreed on, so they concentrate the hard/informative
cases. Masks are COCO-RLE, single category `cable`.

- **Images:** 300 real (variable resolution) — `scene_family` = `real_easy` (224) / `real_hard` (76)
- **Instances:** 2,119 cable masks (COCO RLE)

```python
from datasets import load_dataset
real = load_dataset("DeformX/WireSeg-36K", "real", split="test")
ex = real[0]                                  # ex["image"] -> PIL.Image
```

Fields: `image`, `file_name`, `scene_family`, `masks_rle`, `bboxes`, `areas`, `num_instances`,
`category`, `width`, `height` — the mask/bbox/area columns match the `default` config (same
JSON-encoded per-instance COCO-RLE), so decoding code is shared. The synthetic-only QA columns
(`n_cable`, `wire_area_frac`, `det_*`, `jaccard_j`) are omitted; `width`/`height` are added
because real images are not a fixed size.

## Splits

The `default` config provides a single `train` split; users should split as needed (recommended:
stratify by `scene_family` / `n_cable`). The `real` config is a held-out `test` set.

## Generation

Wires are simulated and rendered with **DeformX** and composited over background imagery;
ground-truth instance masks and metric depth come directly from the renderer (no human
annotation), exported to COCO RLE.

## Limitations & responsible use

- **Synthetic** foreground; expect a domain gap to real imagery.
- Backgrounds are real photographs composited in; they may contain incidental scene content.
  Redistribution is under the dataset license below.

## Citation

```bibtex
@inproceedings{yang2026deformx,
  title     = {DeformX: A Versatile Co-Simulation Framework for
               Deformable Linear Objects},
  author    = {Yang, Yi and Fei, Xiang and Wang, Lehong and Li, Chenhao
               and Dai, Zilin and Kou, Henry and Li, Lu and Choset, Howie},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent
               Robots and Systems (IROS)},
  year      = {2026},
  organization = {IEEE}
}
```

## License

`cc-by-4.0`.
