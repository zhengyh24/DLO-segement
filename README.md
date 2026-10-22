# DLO-segement · SACM

训练线缆可见前景，再接入主线实例后端。本分支包含双层 Adapter、原生细节分支、拓扑损失和原生整图训练。

| 目录 | 内容 |
|---|---|
| `sacm_frontend/` | 冻结 SAM 编码器，训练内部/外部 Adapter 与前景解码器；A–E 消融配置 |
| `sacm_frontend_followup_20261014/` | 拓扑损失及后续训练实现 |
| `sacm_fullframe_20261020/` | 原生整图、分块注意力、padding 对齐及裁剪/整图配对实验 |
| `results/sacm_fullframe/` | 已完成实验的配置、验证门槛和固定后端结果 |
| `data/wireseg_demo/` | 36 训练 + 12 验证原生样本，含实例掩码 |

## 使用

Python 3.12，安装 GPU 版 PyTorch，再执行：

```bash
python -m pip install -r sacm_frontend/requirements.txt
python sacm_frontend/check.py
python sacm_frontend/train.py --manifest data/wireseg_demo/manifest.json --config sacm_frontend/configs/D.json --encoder weights/sam_hq_vit_b.pth --out outputs/sacm_demo --allow-condition-pilot
python sacm_frontend/predict.py --checkpoint outputs/sacm_demo/best.pt --image data/wireseg_demo/rgb/000061.png --out outputs/sacm_prediction
```

编码器权重按 [安装说明](docs/SETUP.md) 获取。模型输出前景概率和二值 mask；逐根身份由 `main` 的后端恢复。

## 已有结果

已完成的裁剪/整图配对实验各训练 5 周期、5000 次成功更新。47 张历史图上的 8 组固定后端配置均未超过匹配的 coverage_color，因此主线仍保留原前端。早期消融台账见 `results/sacm_trials_20261015.json`，残差细节融合见 `tools/sacm_residual_fusion_20261015.py`。详细结果见 [实验摘要](results/sacm_fullframe/README.md)。

这是依据 SACM 思路构建的双层 Adapter 改造版本。模型结构和损失说明见 [模块说明](sacm_frontend/README.md)。

[SAM3.1 主线](https://github.com/zhengyh24/DLO-segement/tree/main) · [Curvi 分支](https://github.com/zhengyh24/DLO-segement/tree/curvi-tracker) · [数据](data/README.md) · [提交规范](CONTRIBUTING.md)
