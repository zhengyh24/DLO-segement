# 不使用修复网络：一次批量SAM实验

2026-10-22，在既有 c1_0 原生640×360图上完成一次部署流程：几何提示规划 → SAM3.1一次批量 → 冻结coverage_color。没有加载或训练修复网络，未根据GT调整提示或参数。

**这张图上结果明显低于旧基线，当前提示方案不具备替换资格。**

## 同图离线指标

| 指标 | 历史coverage_color | 单次SAM，无修复网络 |
|---|---:|---:|
| 实例IoU | 91.08% | 21.57% |
| PQ（严格IoU>0.5） | 91.08% | 0.00% |
| 可见前景IoU | 92.04% | 44.21% |
| 可见前景召回率 | 94.77% | 45.91% |
| 预测实例数（GT为2） | 2 | 6 |
| 实例TP / FP / FN | 2 / 0 / 0 | 0 / 6 / 2 |

PQ为0表示没有预测实例与整根GT的IoU超过0.5，并非完全没有检出线缆。新结果共有8003个真实前景像素被检出，9428个真实前景像素漏检。

![原图、GT、旧基线与本次结果](comparison.png)

## 观察与诊断

![实际六组提示；圆点为正点，矩形为局部框](prompts.png)

- 六组提示均落在交叉点左侧：黄色线左上部分3组、深绿色线左下部分3组。交叉点右侧两段没有分配到提示。图像全覆盖扫描不等于可信候选或提示已覆盖全部线缆。
- 六个SAM提案均是局部片段，每个提案与完整GT的最佳IoU仅约4.9%–23.3%；两根GT的前景覆盖率分别38.74%与51.55%。漏检在实例关联前已经存在。
- 后端正常运行，mBEST得到4条路径；coverage_color按冻结规则选择proposal_graph。六个提案形成6组、合并边0，因此最终仍为6个碎片实例。
- 提示集中、局部框范围和提案碎片是直接观察到的问题。尚未做点提示/框尺度等对照，不能单独确定各因素的因果贡献。下一步优先检查几何候选覆盖、提示分配与框尺度，继续验证无修复网络路线；本次不按该图GT调参或追加试跑。

## 一次调用与成本

- K=1裁块，Q=6独立提示；真实骨干1次、提示编码1次、掩码解码1次；文本检测、视频传播均0次；修复网络0次。无拆批、降采样或OOM重试。
- 实测SAM批次2.929秒；SAM构造/加载及推理合计13.889秒；几何规划5.769秒；实例后端2.397秒；完整流程（含加载与导出）23.224秒。
- 本机RTX4060 Laptop 8GB；峰值allocated显存2,654,208,000字节，约2.47GiB，reserved约2.92GiB。存在另一项GPU训练，因此耗时是并行负载下的观察值。旧基线仅有缓存回放时间，**不能计算旧新端到端加速比**。
- 本轮采用显式`--interactive-only-residency`：只迁移原生交互运算所需模块到GPU，保持原checkpoint数值与运算。899个保留checkpoint tensor严格核验，32个原生RoPE派生buffer重算相等；驻留策略、源码和权重指纹均在审计中。
- 第一次启动在RoPE派生buffer校验处失败，SAM前向计数全部为0，记录保留于`predictions/c1_0/inference.json`。修正加载器后写新目录`predictions_after_loader_fix/c1_0`，该次完整推理成功且只有一批SAM前向。没有使用初次分割输出来调整提示，因为初次未产生分割。

## 证据与复现

- [完整数值及逐提案诊断](results.json)
- [完整推理与模块计数审计](predictions_after_loader_fix/c1_0/inference.json)
- [实际提示计划](predictions_after_loader_fix/c1_0/prompt_plan.json)
- [实验前协议](protocol.json)、[源码快照指纹](source_snapshot.json)、[离线评价脚本](evaluate_case.py)
- [软件检查](software_checks.json)：67项CPU测试、编译检查均通过。

从项目根运行（换用新输出目录）：

```powershell
.sam31_python312_embed/python.exe tools/sam_once.py predict --image datasets/PADLOS_C1C2C3/data/test_imgs/c1_0.png --output outputs/sam_once_norefiner_NEW/c1_0 --max-crops 1 --max-prompts 6 --interactive-only-residency
```

这是一个已见历史图的回归实验，不是C135/Wire总体成绩。当前完整原生参考等价性与24GB容量门槛仍待验证，未开展正式训练或替换旧基线，其他GPU任务未被停止或修改。
