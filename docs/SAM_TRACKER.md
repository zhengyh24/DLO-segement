# 运行 SAM + tracker

模型输入为原始 RGB 和 padding 的有效区域。A 使用自动实例条件网络和 SAM mask decoder。B 增加递归 tracker，C 再加入几何和语义描述符。S 使用原生实例头、固定文本 `cable` 和 200 个查询；只训练最后八个视觉块 Q/V 的 rank-8 LoRA。D 与 MASK_ONLY 分别检验全部 decoder 反馈和仅 mask 反馈。CNN、NO_GEOMETRY、NO_CONTRAST、PARALLEL 分别控制描述符参数量、几何监督、对比项和递归计算。

|要重现的结果|对应源码根目录|
|---|---|
|历史 A/B/C/D pilot|项目根目录的 `sam_path_v1`|
|全量 B/C 固定测试|`cable_seg/versions/full_wireseg_20261029`|
|30 组主消融、9 组 SAM2.1|`cable_seg/versions/pilot_evidence_20261031_r2`|
|3 组 Q80|`cable_seg/versions/pilot_evidence_20261102_q80fix`|

另保留初版 `pilot_evidence_20261031`，用于复核原生 SAM 空提示导致准入失败的记录。各版模块都叫 `sam_path_v1`。从选定的源码根运行，checkpoint 会校验对应引擎的文件 SHA-256，混用不同版本会失败。

## 环境与数据

实验使用 Python 3.12、CUDA PyTorch 和 RTX 4090。安装对应引擎目录里的 `sam_path_v1/requirements.txt`，绘图需要 Matplotlib。SAM3.1 multiplex 和 SAM2.1 Hiera-Tiny 使用官方源码和权重，SAM2 另需 Hydra/OmegaConf。各配置的实际基础权重哈希、可训练参数数目在每组 `model_provenance.json` 中。获取方式见 [第三方来源](../THIRD_PARTY.md)。

仓库发布训练/测试成员及 RGB、mask 哈希。准备 WireSeg 的原生 RGB/NPZ，按 manifest 中的相对路径放置，解压对应的 `pilot_manifest.json.gz` 或 `synthetic_36k_manifest.json.gz`。文件路径前缀已便携化，原始 manifest 字节哈希另列在 `split_receipt.json`。同一解码 RGB 不能跨划分。main 中的 48 图 demo 适合检查环境，正式训练使用原来的 2,000 训练图和 300 验证图。

## 训练和测试

以下命令从仓库根目录执行。替换数据、上游仓库与权重路径。配置工具只替换本机资源路径，保留种子、学习率、损失、阈值、步数与容量。

```bash
python scripts/prepare_tracker_config.py --job main_C_seed42 \
  --data /data/pilot --manifest /data/pilot/manifest.json \
  --sam-repo /models/sam3 --checkpoint /models/sam3.1_multiplex_fp16.pt \
  --output local/config_C.json
python cable_seg/versions/pilot_evidence_20261031_r2/sam_path_v1/run.py check
python cable_seg/versions/pilot_evidence_20261031_r2/sam_path_v1/run.py train \
  --config local/config_C.json --output local/C
python cable_seg/versions/pilot_evidence_20261031_r2/sam_path_v1/run.py evaluate \
  --checkpoint local/C/best.pt --manifest /data/full/manifest.json \
  --data-root /data/full --allow-condition --split test --device cuda \
  --identity-diagnostics --output local/C_test
```

每个输出目录都应是新目录。Q80 A/B/C 选择 `native80_{A,B,C}_seed42` 配置及 q80fix 源码。SAM2.1 选择 `sam21_{A,B,C}_seed{42,43,44}`，向配置工具传 SAM2 源码和基础权重。训练及正式评测的大型权重由原实验存储保存；在分支中可以按 `training/checkpoints.csv` 核对文件字节数及 SHA-256。

## 评测与结果复核

冻结引擎包含 `evaluate`、`robustness`、`benchmark`、`scalability`、`external_data`、`evidence_plot`。Linux 上已有原协议、checkpoint 和产物目录时，[接续程序](../benchmarks/evidence)可按版本核验并组合结果。便携入口的 `WIRESEG_EXPERIMENTS` 和 `WIRESEG_VERSIONS` 可覆盖实验存储和源码目录。`preflight` 先检查输入。`run` 等待 Q80 归档，核对 42 组、84 份 checkpoint 后才评分。表中独立结果文件可直接阅读，无需启动训练或接续程序。
