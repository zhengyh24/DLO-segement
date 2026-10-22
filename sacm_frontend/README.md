# 双层 Adapter 前景模型

冻结 SAM ViT 编码器，在每层 MLP 并联内部 Adapter，并在块输出添加外部 Adapter。各层特征投影到 32 通道后学习加权融合，再解码线缆前景。

| 配置 | 原生细节分支 | 逐根均衡损失 | soft-clDice |
|---|---|---|---|
| A | 无 | 无 | 无 |
| B | 有 | 无 | 无 |
| C | 无 | 有 | 无 |
| D | 有 | 有 | 无 |
| E | 有 | 有 | 0.1 |

默认 512×512 原生裁块，推理步长 384；边界补零并忽略 padding。基础损失为 BCE + 0.4 Dice，逐根均衡项权重 0.25。验证前景 IoU 选择 `best.pt`，同时保留 `last.pt`、优化器和随机数状态。

训练数据使用相对路径 manifest，图像为 RGB，NPZ 的 `masks` 为逐实例可见布尔掩码。使用仓库中的 condition-holdout 样本时加 `--allow-condition-pilot`。

```bash
python sacm_frontend/train.py --help
python sacm_frontend/predict.py --help
```

从中断周期恢复：在原命令中添加 `--resume <原输出目录>/last.pt`。本实现使用自定义单前景解码器，未实现原论文的全部候选排序模块。

`vendor/` 保留编码器源码与原许可证。原生整图变体见 `../sacm_fullframe_20261020/`。
