# 训练、评价与服务器执行说明

以下均从项目根目录执行；代码包是 `sam_path_v1`，不依赖旧训练包。正式数据与 SAM 权重路径由配置提供。当前 CPU toy 检查只证明工程运行与恢复一致，不构成 SAM 效果或性能结果。

## 配置与训练

配置是严格 JSON，未知键直接报错。以下示例用于服务器，路径需改成服务器真实路径；`allow_condition=true` 仅表示显式接受 condition holdout pilot，不能把它称为独立场景划分。

```json
{
  "manifest": "/data/wireseg/manifest.json",
  "data_root": "/data/wireseg",
  "allow_condition": true,
  "model": {
    "variant": "C",
    "backend": "sam31",
    "sam_repo": "/opt/sam3",
    "sam_checkpoint": "/models/sam3.1_multiplex.pt",
    "canvas_size": 1036,
    "num_queries": 24,
    "tracker_steps": 8,
    "encoder_amp": "bf16"
  },
  "seed": 42,
  "epochs": 5,
  "batch_size": 1,
  "lr": 0.0001,
  "decoder_lr": 0.00001,
  "weight_decay": 0.01,
  "device": "cuda:0",
  "amp": "none",
  "workers": 0,
  "save_every": 100,
  "validate_every": 1,
  "object_threshold": 0.5,
  "mask_threshold": 0.5
}
```

```bash
python -m sam_path_v1.train --config configs/C.json --output runs/C_seed42
python -m sam_path_v1.train --config configs/C_extended.json --resume runs/C_seed42/last.pt --output runs/C_seed42_resume01
```

所有输出目录必须不存在。恢复写新目录；不会覆盖父实验。`seed` 自动注入 `model.init_seed`；显式冲突报错。A–D 配对须保持相同 seed、数据、query 数、预算与超参数。AdamW 只优化 `requires_grad=True` 参数；原解码器默认学习率 1e-5，新增模块默认 1e-4。`amp=none` 让 head FP32，SAM 编码器可独立使用 BF16。外层 fp16 才启用 GradScaler；BF16/FP32 非有限梯度直接停止。

`max_batches` 是累计优化器更新机会的上限，用于有界 smoke 和续训测试。它包含 AMP 跳步，不能当成成功更新数。`logs.jsonl` 每次记录累计 opportunities、successful_updates、amp_skips、samples、native_pixels、supervised_pixels、耗时及逐项损失。图像解码失败或非有限 loss 会停止训练，不静默跳过。数据集不增强，`workers=0`，每 epoch 使用独立确定性排列，checkpoint 保存下一 batch cursor、optimizer、scaler、Python/NumPy/Torch/CUDA RNG。

`last.pt` 与 `best.pt` 都原子替换；best 仅按 validation 的实例 IoU 宏平均选择，平局用 global PQ。阈值在配置中冻结。只有完成验证才会产生 best；中途预算停止可能只有 last。恢复时若父 best 可核验会保留其副本，否则 `completion.json` 明确说明 best 不可用。

SAM checkpoint 保存可训练 delta、新模块、模型配置与冻结基础权重/官方关键源码 SHA，不每轮重复保存整个 SAM 编码器。toy checkpoint 保存完整模型，包括冻结随机编码器。训练本包源码、manifest、运行时版本及设备信息也审计；恢复时若发生不兼容变更直接拒绝。此精确续训承诺针对相同代码、依赖、硬件类型、精度和 batch 协议；不承诺跨硬件浮点逐位相同。仅从可信来源加载 `.pt`，完整训练 checkpoint 含 Python RNG/优化器结构。

## 自动评价与预测

```bash
python -m sam_path_v1.evaluate --checkpoint runs/C_seed42/best.pt --manifest /data/wireseg/manifest.json --data-root /data/wireseg --split validation --allow-condition --device cuda:0 --output reports/C_best_validation
python -m sam_path_v1.evaluate --checkpoint runs/C_seed42/last.pt --manifest /data/wireseg/manifest.json --data-root /data/wireseg --split test --allow-condition --device cuda:0 --output reports/C_last_test
python -m sam_path_v1.predict --checkpoint runs/C_seed42/best.pt --image /data/upright_rgb.png --device cuda:0 --output predictions/example01
```

迁移后可以传 `--sam-repo` 和 `--sam-checkpoint` 覆盖路径，实际权重 SHA 与关键官方源码 SHA 必须匹配。本包 10 个训练/模型/计分/部署源码也默认逐文件 SHA 校验；缺失、额外文件或内容变化在构建模型前直接报错，并列出文件名。使用训练时冻结的整包与对应 checkpoint，不修改 checkpoint 内的 provenance 绕过检查。迁移推理允许不同 GPU/运行时；精确训练续训仍要求原运行时匹配。评价和预测阈值读取 checkpoint，不提供 test/check 调阈值开关。测试集本身不能选 best，最佳与末轮应分别报告。

服务器开始前先以正式 query/step/canvas 配置运行 `check --real-sam`。确定性算法可能降低速度；某个运行时若不支持对应算子的确定性实现会报错，不能静默切换后继续声称精确续训。单 GPU 作业建议启动进程前固定 `CUDA_VISIBLE_DEVICES`；可见设备数量及 `CUBLAS_WORKSPACE_CONFIG` 属于续训运行时合同。支持的 workspace 配置为 `:4096:8`（默认）或 `:16:8`，其他值会提前给出清晰错误。当前 CPU 测试确认恢复逐位一致，真实 GPU 梯度 smoke 不等于已经验证跨中断 GPU 逐位恢复。

模型只接收 RGB 与由原图尺寸生成的 padding validity；数据集的 GT overlap validity 只参与 loss/metric。每次模型调用强制记录一个 encoder 与一个 batched decoder。原图只右下补到 canvas，超过 canvas 立即报错。不能把大图隐式缩小，也不自动切块。

输出 `instances.png` 是 uint16 可见实例 ID、`union.png` 是可见前景、`unknown.png` 当前全零，另有 `prediction.npz` 与 SHA/尺寸/调用计数 receipt。当前模型没有独立校准的未知身份拒识头，因此全零 unknown 是明确限制；不从低置信度背景猜测未知前景，也不补隐藏段。query 以 object probability 与 mask probability 的乘积竞争每个可见像素；固定阈值后赋予唯一 ID，相等时使用较小 query 索引。quality 是预测 IoU 回归值，不直接充当概率参与过滤。

主指标为每图 GT 数归一化 Hungarian 最大和 IoU，然后对全图宏平均；PQ 用严格 IoU>0.5、优先最大 TP 的一对一匹配，再全域累计。另报 union IoU、域别、空预测、失败图及逐图结果。empty GT 与 empty prediction 同时出现计 1；任何推理失败该图计 0。padding 与重叠标注区域被排除；只出现在忽略区域的预测不计 FP。若 GT 文件不可读，图仍以零分进入宏平均，但 global PQ 置 null、complete=false，避免编造未知 FN。评价进程保留失败报告并以退出码 2 告知调用者。训练不会用不完整 GT 评价选择 best。

## CPU smoke

```bash
python -m unittest sam_path_v1.tests.test_engine -v
```

测试生成 5 张 12×13 合成图，pad 到 16，不读真实数据。C toy 连续训练 4 次 AdamW 更新，对照运行 1 次后保存并续训，逐张量比较模型、AdamW 与 Torch RNG；另跑 train/evaluate/predict CLI 参数解析，检查原生输出尺寸、调用计数、拒覆盖、严格 PQ 阈值、最大 TP 优先、空图/失败处理与大图拒绝。此测试没有衡量分割精度。Windows 项目内嵌 Python 的 `_pth` 不自动包含工作目录，可用：

```powershell
.\.sam31_python312_embed\python.exe -c "import sys,unittest; sys.path.insert(0,'.'); unittest.main(module='sam_path_v1.tests.test_engine',verbosity=2)"
```
