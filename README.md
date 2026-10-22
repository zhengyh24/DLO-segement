# DLO-segement · Curvi-Tracker

面向细长线缆的身份学习与分块迭代实例分割。本分支保留两个阶段的实现：

| 模块 | 输入与方法 | 输出 |
|---|---|---|
| `curvi_identity/` | RGB + 预测前景，双尺度 CNN 对原子片段配对，图约束解码 | 逐根实例 ID |
| `curvi_instance_v1/` | RGB 自动种子 + query/anchor 实例记忆 + 局部条件解码 + 出口调度 | 原生实例 ID、可见前景、unknown |

这些是受 Curvi-Tracker 启发的项目实现。直接实例 v1 使用 32/64/128/192 通道 CNN、128 维状态和 GRU，共 647,395 个参数；无需 SAM 预训练权重。

## 使用直接实例 v1

```bash
python -m pip install -r curvi_instance_v1/requirements.txt
python scripts/verify_repository.py
python curvi_instance_v1/run.py check --manifest data/wireseg_demo/manifest.json --output outputs/curvi_check
python curvi_instance_v1/run.py train --manifest data/wireseg_demo/manifest.json --output outputs/curvi_demo --kind smoke --allow-condition-pilot --updates 12 --batch 2 --tile 256 --steps 3 --val-every 6
python curvi_instance_v1/run.py predict --help
```

默认使用 256×256 原生窗口、128 步长。推理由 RGB 自动找种子，逐根传播状态，融合局部掩码并处理冲突。详细结构见 [模块说明](curvi_instance_v1/README.md)。

## 已有验证

v1 的 7 组功能检查通过；2+2 图烟雾训练完成 12 次成功更新。一张人工同色十字图上，300 步训练后两个实例的条件 IoU 均为 1.0，验证了状态表达能力。真实 RGB 烟雾推理仍未检出实例，这些记录不代表实际分割精度。凭据见 `results/curvi_v1/`。

正在进行的四架构比较、调优和多种子复核不包含在本分支，完整结束后另发 `curvi-auto-20261022`。

[SAM3.1 主线](https://github.com/zhengyh24/DLO-segement/tree/main) · [SACM 分支](https://github.com/zhengyh24/DLO-segement/tree/sacm) · [数据](data/README.md) · [提交规范](CONTRIBUTING.md)
