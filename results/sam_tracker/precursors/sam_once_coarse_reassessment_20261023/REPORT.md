## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (cached-output reassessment)
- Origin Date: 2026-10-23
- Verification Status: 44 automatic caches checked; all coarse masks equal frozen raw unions; zero new SAM calls
- Version Label: coarse_stage_clarification_v1

# 一次SAM只负责粗分割：目标澄清与缓存复核

**用户约束：SAM3.1全流程只调用一次，产出粗分割；后续算法负责局部分块迭代与最终实例分割。当前仍先不加修复网络。**

前一轮实现包含SAM后的固定身份后端，但我用最终实例PQ给前端提示选优，并把它表述成“一次SAM方案能否达到实例分割”。这种阶段目标绑定过紧。保留旧结果作为完整链路对照，不再把该排序当作粗分割提示的最优结论。

正确分工：RGB与几何提示规划 → 一次真实批量SAM → 独立软提案＋可见粗前景 → 不调用SAM的分块处理与身份关联 → 最终实例。粗前景允许多根线相连、一个提案包含多根或只覆盖局部；后处理可以拆分/合并提案。当前粗导出接口已分离，新的后处理迭代方法尚未在本轮实现或验证。

## 阶段评价

- 粗分割：前景召回/精度/IoU、细线中心线覆盖、局部缺口和误粘连。优先避免整段遗漏，同时控制大块背景进入；全图都判前景并非合格粗分割。
- 最终链路：后处理完成后再评实例IoU/PQ、误并/碎片和端到端耗时。最终实例指标仍是部署标准，但不是SAM粗输出自身必须达到的指标。
- 本轮中心线召回是GT并集骨架与粗前景的精确像素重叠，无膨胀容忍；它只作离线诊断。每根GT的覆盖率也只用于描述遗漏，不做实例匹配。

## 同一批冻结预测的粗前景结果

每域仅2张历史验证图，表中按图平均。这是事后更正评价侧重点，未重新推理或改变掩码阈值，也没有据此宣称泛化最优。旧baseline行使用旧ours_hq可见掩码，包含旧多次调用流程，不能当作单次SAM对照。

| 配置 | C135 前景IoU / 精度 / 召回 | Wire 前景IoU / 精度 / 召回 |
|---|---:|---:|
| v1_control | 25.91% / 80.18% / 27.70% | 9.08% / 72.45% / 9.30% |
| v2_p1n0_neighbor_none | 67.05% / 79.22% / 81.42% | 23.97% / 51.25% / 31.07% |
| v2_p2n0_neighbor_none | 66.25% / 77.16% / 82.65% | 15.34% / 53.23% / 19.44% |
| v2_p3n0_neighbor_none | 65.89% / 76.22% / 83.28% | 13.15% / 51.71% / 16.82% |
| v2_p1n1_neighbor_none | 67.56% / 79.60% / 81.74% | 22.96% / 51.50% / 29.61% |
| v2_p2n1_neighbor_none | 66.49% / 77.38% / 82.73% | 18.24% / 56.83% / 23.56% |
| v2_p3n1_neighbor_none | 65.99% / 76.33% / 83.28% | 13.25% / 50.26% / 16.93% |
| v2_p2n2_neighbor_none | 66.53% / 77.41% / 82.77% | 17.16% / 56.11% / 21.96% |
| v2_p2n1_background_none | 66.49% / 77.41% / 82.65% | 17.75% / 55.93% / 22.74% |
| v2_p2n1_neighbor_local | 54.84% / 79.95% / 63.52% | 11.48% / 65.40% / 12.15% |
| v2_p2n1_neighbor_expanded | 61.66% / 75.12% / 76.74% | 15.44% / 61.59% / 17.00% |
| old_coverage_color | 85.08% / 88.61% / 95.48% | 59.52% / 64.85% / 87.86% |

## 提示语义与后续优化

粗分割提示优先覆盖尚未被询问的可信线缆区域，减少同一区域重复占用预算。默认负点应有真实背景证据；其他线缆不属于全局粗前景的负类。邻线负点只能作为某个局部提案的排除证据，其邻线本身仍需被其他提案覆盖。

不要求一组提示最终对应一根完整线缆，也不要求SAM在交叉处确定最终身份。但也不能假设把多根线的正点塞进一个交互提示组，就一定得到所有线缆并集；多目标提示响应需要另作消融。保留各个软提案是为了给后处理提供证据，不是把SAM提案ID固定成最终实例ID。

后处理可根据RGB和几何证据重新发现候选区域、修边、拆粘连、处理有证据的可见断口并关联身份；它的覆盖范围不必永远受初始粗mask硬限制。不能在没有图像证据时把遮挡段画成前景。新迭代算法的效果与成本仍需独立验证。

旧后端的9次几何错误发生在SAM之后，不改变此处的原始粗前景。它们影响最终实例结果，应与粗分割漏检分开诊断。

## 已分离的入口

```powershell
.sam31_python312_embed/python.exe tools/sam_once.py predict --coarse-only --planner v2 --image IMAGE.png --output outputs/COARSE_NEW --max-crops 1 --max-prompts 6 --interactive-only-residency
.sam31_python312_embed/python.exe tools/sam_once.py export-coarse --cache PATH/sam_cache --output outputs/COARSE_FROM_CACHE_NEW
```

`--coarse-only`跳过身份后端和修复网络，保存粗前景、最大logit证据图、有提案证据的像素覆盖层及独立原始提案。`export-coarse`重用已有初始缓存，新增SAM调用0次。logit融合不是经过校准的概率；没有提案覆盖的像素单独标记，不视为确定背景。现有全链路predict保持兼容。

本目录的coarse_outputs保留44份自动配置的粗导出，每份均与旧visible_foreground逐像素相同。results.json保留全部配置、中心线和每GT覆盖诊断；不会覆盖或重排原实验结果。
