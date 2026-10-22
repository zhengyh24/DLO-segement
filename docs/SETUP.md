# 安装与运行

在仓库根目录执行以下命令。使用 Python 3.12 和独立虚拟环境，GPU 版 PyTorch / torchvision 按本机 CUDA 环境安装。

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

## SAM3.1 与权重

```bash
gh repo clone facebookresearch/sam3 external/sam3
git -C external/sam3 checkout 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
python -m pip install -e external/sam3
```

按 [SAM3.1 发布说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) 获取 `sam3.1_multiplex_fp16.pt`，按 [HQ-SAM 说明](https://github.com/SysCV/sam-hq) 获取 `sam_hq_vit_b.pth`，放入 `weights/`。

默认路径及可覆盖变量：

| 内容 | 默认位置 | 环境变量 |
|---|---|---|
| SAM3.1 源码 | `external/sam3` | `SAM31_REPO` |
| SAM3.1 权重 | `weights/sam3.1_multiplex_fp16.pt` | `SAM31_CHECKPOINT` |
| HQ-SAM 权重 | `weights/sam_hq_vit_b.pth` | `HQ_CHECKPOINT` |

## mBEST

需要本机 C/C++ 编译工具链：

```bash
python third_party/mBEST/build.py
python third_party/mBEST/build.py --numba
```

## 推理

```bash
python tools/run_accuracy_candidate.py --image <RGB图像> --output outputs/example
```

处理流程分阶段运行以控制显存。`--cache-root`、`--dataset`、`--case` 可用于复用已有前端缓存；`tools/run_combination_candidate.py` 提供基于该缓存的组合后端实验。

历史整套基准评价入口为 `tools/evaluate_full_comparison.py`，需要完整数据及对应预测目录。日常安装验证先运行 `python scripts/verify_repository.py`，再使用 `data/wireseg_demo/manifest.json` 中的样本。
