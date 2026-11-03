## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-23
- Verification Status: real inference receipts and native cache evidence checked; per-case status below
- Version Label: fresh_coarse_visual_v1

# 一次 SAM 粗分割：四图实跑

本轮重新执行真实 SAM3.1 推理，只输出粗前景及独立原始提案，不运行修复网络或实例后端。固定 v2 几何规划、每图一块、最多六个独立提示组、每组 1 正点、0 负点、无框；logit > 0。

四张图均为此前反复使用的历史 validation 样本；该配置来自此前探索，不是盲测或独立验证，不证明泛化提升。

## 实验材料

| 材料 | 来源及处理 |
|---|---|
| RGB | 冻结 evaluation_inputs.json 中的四张原图；逐图核对原生 RGB SHA，无裁剪或缩放。 |
| 模型 | 真实预训练 SAM3.1；checkpoint SHA、设备、精度及 residency_policy 逐图记录于 metrics.json。 |
| 提示 | 仅 RGB 几何规划；编号表示彼此独立的提示组，不表示真实线缆实例。 |
| 粗分割 | 原始提案 logit 在原生坐标逐像素取 max，阈值严格 > 0；无删小区域、形态学或实例处理。 |
| 标注 | 既有 NPZ masks，经冻结 SHA 验证后合并为二值前景；仅在全部预测阶段结束后读入。 |
| 图片 | PIL 从真实像素绘制；绿色叠加和 TP/FN/FP 着色仅用于展示，没有生成式编辑。 |
| 失败处理 | 预测失败保留在四图分母内，按空预测计分并标识；标注完整性失败则指标不可评价，不静默剔除。 |
| 追溯 | [PROTOCOL.json](PROTOCOL.json)、[RUN_COMPLETION.json](RUN_COMPLETION.json)、[metrics.json](metrics.json)；清单 SHA `73f00e7a0594204dab999db37db378808e6d906d7294d5f5b9745da680807a20`。 |

## 逐图二值前景指标

| 图像 | 状态 | IoU | 精确率 | 召回率 | 冷启动 | SAM 推理段 |
|---|---|---:|---:|---:|---:|---:|
| c135_c1_2 | ok | 77.9% | 80.5% | 96.0% | 45.00s | 1.96s |
| c135_c1_20 | ok | 56.2% | 77.9% | 66.8% | 63.85s | 1.95s |
| wire300_real-002_090 | ok | 26.8% | 52.8% | 35.3% | 62.89s | 0.70s |
| wire300_real-002_099 | ok | 21.1% | 49.7% | 26.8% | 130.07s | 1.66s |

域均为图像等权算术平均；不是池化像素指标。IoU/精确率/召回率分母为零时按 0，与项目既有二值评价一致。

| 域 | 图数 | IoU | 精确率 | 召回率 | 预测失败 |
|---|---:|---:|---:|---:|---:|
| c135 | 2 | 67.0% | 79.2% | 81.4% | 0 |
| wire300 | 2 | 24.0% | 51.3% | 31.1% | 0 |

## 实际调用审计

| 图像 | backbone | image encoder | prompt encoder | mask decoder | detector / propagation / text | 微批 / 重试 | 核验 |
|---|---:|---:|---:|---:|---|---|---|
| c135_c1_2 | 1 | 1 | 1 | 1 | 0 / 0 / 0 | 1 / 0 | True |
| c135_c1_20 | 1 | 1 | 1 | 1 | 0 / 0 / 0 | 1 / 0 | True |
| wire300_real-002_090 | 1 | 1 | 1 | 1 | 0 / 0 / 0 | 1 / 0 | True |
| wire300_real-002_099 | 1 | 1 | 1 | 1 | 0 / 0 / 0 | 1 / 0 | True |

每图一次批量前向；四张图是四次独立运行。backbone wrapper 和 image encoder 计数描述同一前向路径，不应相加成两次 SAM。

冷启动时间来自 `end_to_end_cold_seconds`，包含模型装载、规划、SAM 和粗分割导出；SAM 推理段来自 `sam.inference_seconds`，不是纯 GPU kernel 时间。进程总耗时另记于 metrics.json。共享 GPU 可能受其他负载影响，且没有匹配的同机同口径基线，因此不报告加速比。

## 可视化与原始掩膜

每张单图提供原图、全部自动提示、粗前景叠加、离线误差四个面板。所有面板像素按 1:1 粘贴；较小图片仅补白。绿色为 TP，红色为 FN，青色为 FP。原始二值掩膜与独立提案缓存仍可直接检查。

![c135 全图结果](c135_summary.png)

![wire300 全图结果](wire300_summary.png)

- c135_c1_2：[四面板图](c135_c1_2.png) · [原始二值掩膜](runs/c135_c1_2/coarse_foreground.png) · [推理审计](runs/c135_c1_2/inference.json) · [独立提案缓存](runs/c135_c1_2/sam_cache/case.json)
- c135_c1_20：[四面板图](c135_c1_20.png) · [原始二值掩膜](runs/c135_c1_20/coarse_foreground.png) · [推理审计](runs/c135_c1_20/inference.json) · [独立提案缓存](runs/c135_c1_20/sam_cache/case.json)
- wire300_real-002_090：[四面板图](wire300_real-002_090.png) · [原始二值掩膜](runs/wire300_real-002_090/coarse_foreground.png) · [推理审计](runs/wire300_real-002_090/inference.json) · [独立提案缓存](runs/wire300_real-002_090/sam_cache/case.json)
- wire300_real-002_099：[四面板图](wire300_real-002_099.png) · [原始二值掩膜](runs/wire300_real-002_099/coarse_foreground.png) · [推理审计](runs/wire300_real-002_099/inference.json) · [独立提案缓存](runs/wire300_real-002_099/sam_cache/case.json)
