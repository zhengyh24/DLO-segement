# 片段身份配对

从已有预测前景提取骨架原子段，使用 128×128 和 256×256 原生窗口的共享 CNN，加 16 维几何特征，判断两个片段是否属于同一根线缆。网络共 138,817 个可训练参数。

图解码综合双向分数、端点互斥和组间冲突约束，输出逐根 ID。输入前景保持不变；局部片段配对与直接实例 v1 可分别研究。

```bash
python -m pip install -r curvi_identity/requirements.txt
python tools/run_curvi_identity.py self-test --output outputs/pairnet_check
python tools/run_curvi_identity.py prepare --help
python tools/run_curvi_identity.py train --help
python tools/run_curvi_identity.py infer --help
```

准备数据时提供原分组文件和独立预测前景缓存。训练复制 `default_config.json` 到新配置并设置 `training_enabled: true`。`default_config.json` 给出模型、候选筛选和图解码参数；目录默认为 `data/WireSeg-36K`。

端点难例采样、竞争排序损失及固定分数解码消融分别在 `tools/train_curvi_targeted_v2.py`、`train_curvi_targeted_v3.py`、`curvi_decoder_ablation_v2.py`。对应检查为 `test_targeted_v2.py`、`test_targeted_v3.py` 和 `test_curvi_decoder_ablation_v2.py`。历史2000图训练适配器保留于 `run_curvi_2000.py` / `run_gtunion_2000.py`，使用独立生成的图结构及前景缓存。
