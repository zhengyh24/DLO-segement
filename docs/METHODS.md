# 代码导航

| 模块 | 文件 | 作用 |
|---|---|---|
| 自动推理 | `tools/run_accuracy_candidate.py` | 串行执行前端、细化、几何和身份融合 |
| SAM3.1 / 几何提示 | `tools/stage_a_geomprompt_sam31.py`、`tools/geomprompt_tuned_detector.py` | 原生裁块、文本与几何候选、自动提示 |
| 前端协议 | `tools/full_comparison_inference.py` | 固定 SAM/HQ 推理流程 |
| 几何恢复 | `tools/mbest_backend.py`、`tools/refine_observed_ids.py` | 骨架路径及可见区域补全 |
| 身份融合 | `tools/cable_evidence_fusion.py`、`tools/cable_identity_selector.py` | 提案图、像素分区和 coverage_color 选择器 |
| 组合后端 | `tools/cable_combination_backend.py`、`tools/cable_dual_continuation.py` | 路径覆盖与双向延伸候选 |
| 评价 | `tools/evaluate_full_comparison.py` | 原生坐标、EXIF 对齐和实例指标 |

主线保留 coverage_color；组合后端对应此前的局部开发实验。SACM 和 Curvi 的训练实现分别放在各自分支。

发布版只调整外部模型路径和安装入口。原始源码 SHA-256 与发布调整记录见 `export_sources_main.json`。
