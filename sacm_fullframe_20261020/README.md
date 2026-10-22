# 原生整图与裁剪配对

`NativeForeground` 保留非整倍数图像的原始坐标，通过 padding、位置编码适配及分块查询注意力处理完整图像。训练使用相同父权重、样本顺序和优化器更新预算，比较 crop / full 两种输入方式。

```bash
python -m sacm_fullframe_20261020.train --config <本地配置.json> --out outputs/native_full --cycles 5
python -m sacm_fullframe_20261020.predict --help
```

配置模板见 `results/sacm_fullframe/crop_config.json` 和 `full_config.json`。复现实验需配置对应父权重及合成/真实 manifest，并更新文件哈希。`preflight.py` 保留历史精确回放配置；`controller.py` 为两 GPU 的配对控制器，单 GPU 使用训练入口逐个执行。
