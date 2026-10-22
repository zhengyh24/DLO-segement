# DLO-segement

从单张 RGB 图像分割线缆，输出逐根实例 ID、可见前景和未知身份区域。

| 分支 | 技术路线 | 入口 |
|---|---|---|
| [`main`](https://github.com/zhengyh24/DLO-segement/tree/main) | SAM3.1 + 几何提示 + HQ-SAM + coverage_color 身份恢复 | `tools/run_accuracy_candidate.py` |
| [`sacm`](https://github.com/zhengyh24/DLO-segement/tree/sacm) | 双层 Adapter 前景学习、拓扑损失及原生整图训练 | `sacm_frontend/train.py` |
| [`curvi-tracker`](https://github.com/zhengyh24/DLO-segement/tree/curvi-tracker) | 片段配对与带实例记忆的分块迭代网络 | `curvi_instance_v1/run.py` |

当前本机网络比较将在完整运行结束后发布到独立的 `curvi-auto-20261022` 分支。

## 主线流程

`RGB → SAM3.1 / 几何提示 → HQ-SAM 前景细化 → 提案与骨架关联 → coverage_color → 实例 ID`

`coverage_color` 是目前保留的历史回归基线：

| 数据 | 实例 IoU | PQ |
|---|---:|---:|
| C135 | 83.05% | 76.71% |
| Wire300 | 57.28% | 37.14% |

[数字来源](results/coverage_color.json)：既有完整实验，属于历史回归。仓库中的小数据集用于安装验证和协作开发。

## 开始使用

Python 3.12，先安装适合 GPU 的 PyTorch，再按 [安装说明](docs/SETUP.md) 准备 SAM3.1、HQ-SAM 权重及 mBEST 扩展。

```bash
python -m pip install -r requirements.txt
python scripts/verify_repository.py
python tools/run_accuracy_candidate.py --image data/wireseg_demo/rgb/000061.png --output outputs/example
```

示例图路径以 `data/wireseg_demo/manifest.json` 为准。输出在 `outputs/example/candidate/`：`instance_ids.png`、`visible_foreground.png`、`unknown_visible.png` 和叠加预览。

## 目录

- `tools/`：推理、身份恢复、评价和算法检查。
- `configs/`：SAM3.1 路线的版本化配置。
- `data/`：48 张 WireSeg 原生样本、实例标注及完整 2000/300 划分清单。
- `third_party/mBEST/`：固定版本的几何后端源码与许可证。
- `docs/`：安装、方法映射、数据来源和导出清单。

数据使用见 [data/README.md](data/README.md)，协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)，第三方来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
