# 数据

本仓库提供 **48 张 WireSeg-36K 合成图像**及逐实例可见掩码，适合验证输入输出、调试训练和协作开发。图像保持 1024×1024，实例重叠保留在布尔 NPZ 中。

| 划分 | 条件数 | 每条件张数 | 合计 |
|---|---:|---:|---:|
| train | 6 | 6 | 36 |
| validation | 3 | 4 | 12 |

抽样在既有 2000/300 划分内部按固定文件名哈希完成，不改变组别。条件包括 data_center、flying_wire、wire_on_plane 及不同线缆数量。这是 condition holdout，未取得独立仿真场景 ID。

```text
wireseg_demo/
  manifest.json      # 文件名、来源分片/行、划分、尺寸、哈希
  rgb/*.png          # 原生 RGB
  masks/*.npz        # masks: [实例数, 高, 宽]，bool
manifests/
  pilot2300.json     # 完整 2000 train + 300 validation 名单
```

```python
import json
import numpy as np
from PIL import Image
from pathlib import Path

root = Path('data/wireseg_demo')
row = json.loads((root/'manifest.json').read_text(encoding='utf-8'))['records'][0]
rgb = np.array(Image.open(root/row['image']).convert('RGB'))
masks = np.load(root/row['masks'], allow_pickle=False)['masks']
```

需要完整冻结子集时，从 [WireSeg-36K](https://huggingface.co/datasets/DeformX/WireSeg-36K) 获取对应 `data/*.parquet`，执行：

```bash
python scripts/export_wireseg_subset.py --dataset-root <WireSeg-36K目录> --manifest data/manifests/pilot2300.json --output data/wireseg_full
```

来源：DeformX / WireSeg-36K，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。作者及引用见 [原始数据卡](WireSeg-UPSTREAM.md)。本仓库做了子集选择和 RGB/NPZ 格式导出，未修改像素、掩码几何或原划分。
