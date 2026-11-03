# DLO-segement

当前分支 **sam+tracker** 发布单张 RGB 线缆实例分割的训练实现与结果。视觉特征由 SAM 编码器共享，自动实例条件网络与可学习路径模块向 mask decoder 提供逐实例引导。主协议每图一次编码、一次批量解码。Q80 原生分块扩展单独计时。

|实验|完成情况|结果入口|
|---|---|---|
|全量 WireSeg B/C|各 162,000 次更新，best/last 均已在固定 1,800 图测试|[逐图结果与 loss](results/sam_tracker/full_wireseg_BC)|
|2,000/300 机制验证|42/42 组训练完成，合计 420,000 次成功更新、0 次 AMP 跳步|[训练汇总](results/sam_tracker/pilot/training/summary.csv) · [每轮验证](results/sam_tracker/pilot/training/epochs.csv)|
|机制验证正式评价|北京时间 2026-11-03 10:22 快照，33/157 项已通过|[任务清单](results/sam_tracker/pilot/evaluation/snapshot.json) · [已完成固定测试](results/sam_tracker/pilot/evaluation/synthetic_summary.csv)|
|PADLOS 同划分对照|固定 1,800 图，保留失败样本|[对照报告数据](results/sam_tracker/padlos_test1800)|

复现实验见 [训练入口和引擎版本](docs/SAM_TRACKER.md)，科学结论、失败记录与文件格式见 [结果索引](results/sam_tracker/README.md)。早期单次提示代码在 [sam_once](sam_once/)，旧 A/B/C/D pilot 在 [sam_path_v1](sam_path_v1/)。

## 从 main 继承的基线

从单张 RGB 图像分割线缆，输出逐根实例 ID、可见前景和未知身份区域。

| 分支 | 技术路线 | 入口 |
|---|---|---|
| [`main`](https://github.com/zhengyh24/DLO-segement/tree/main) | SAM3.1 + 几何提示 + HQ-SAM + coverage_color 身份恢复 | `tools/run_accuracy_candidate.py` |
| [`sacm`](https://github.com/zhengyh24/DLO-segement/tree/sacm) | 双层 Adapter 前景学习、拓扑损失及原生整图训练 | `sacm_frontend/train.py` |
| [`curvi-tracker`](https://github.com/zhengyh24/DLO-segement/tree/curvi-tracker) | 片段配对与带实例记忆的分块迭代网络 | `curvi_instance_v1/run.py` |


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
