# HANDLOOM 公开真实交叉测试集审计

结论：该资源可以支持**给定交叉位置和目标路径后的局部上下穿越分类**，不能直接提供本项目所需的无提示 RGB 实例分割或遮挡两侧身份关联真值。本次只做数据与官方代码审计，不加载模型、不产生预测、不修改冻结源码、协议或队列。

固定来源为 `vainaviv/handloom` 的提交 `4b23ef54ccb85d814965f8ae1c22e91f3e5171ed`，目录 `data/real_data/under_over_crossing_real_test_set`。所用提交和逐文件 Git blob SHA-1 来自已保存的 tracer110 `source_manifest.json`，并非重新选择最新版本。[官方目录](https://github.com/vainaviv/handloom/tree/4b23ef54ccb85d814965f8ae1c22e91f3e5171ed/data/real_data/under_over_crossing_real_test_set)

| 检查项 | 全量实读结果 |
|---|---|
| 下载与来源校验 | 76 / 76 个 NPY；125,656 字节；全部 Git blob SHA-1、大小匹配，并记录 SHA-256 |
| 图像字段 | `crop_img`，全部 20×20×3、uint8，三个通道完全相同，是灰度图块 |
| 路径字段 | `spline_pixels`，73 条有 4 点、2 条有 5 点、1 条有 6 点；二维坐标均在 0–19 |
| 类别字段 | `under_over`：0 为 under，35 条；1 为 over，41 条 |
| 精确图块重复 | 解码像素 SHA-256 得到 76 个唯一图块，无重复组 |
| 原始完整图像与场景 | 无原图 ID、scene ID、裁剪到原图的正式映射；完整原图数量、原图重复和场景独立性均未知 |
| 可见性与身份真值 | 无实例 mask、完整线缆 ID、竞争分支身份、点级可见性、正负关联事件或完整端点标注 |

原 NPY 使用已验证的受限 NumPy 反序列化器读取：限定输入大小和 scalar-dict 包装，只允许明确白名单类型，拒绝任意 pickle global。下载的官方 Python 文件仅作为文本核查，没有执行。17 份官方源码/文档另有 Git blob SHA-1 和 SHA-256 记录。

**提示依赖。** 官方 evaluator 明确选择本目录。dataset 读取样本提供的路径点，将路径画进输入通道 0；推荐配置将中心位置写进通道 1，并用首尾路径点的方向旋转输入。因此二分类标签是相对于所指定路径段而言，不是仅凭图块就定义了“哪个实例”的无提示任务。20×20 裁剪本身已给出局部交叉位置。该子集没有全局起点字段；完整 HANDLOOM pipeline 则先从起始像素追踪，再获取交叉位置和条件路径。公开子集没有逐条注明条件路径来自人工还是 tracer，不将其来源擅自认定为 GT。[官方评估入口](https://github.com/vainaviv/handloom/blob/4b23ef54ccb85d814965f8ae1c22e91f3e5171ed/model_training/analysis.py#L202)、[条件输入构造](https://github.com/vainaviv/handloom/blob/4b23ef54ccb85d814965f8ae1c22e91f3e5171ed/model_training/src/dataset.py#L402)、[方向对齐](https://github.com/vainaviv/handloom/blob/4b23ef54ccb85d814965f8ae1c22e91f3e5171ed/model_training/src/dataset.py#L205)、[完整流程](https://github.com/vainaviv/handloom/blob/4b23ef54ccb85d814965f8ae1c22e91f3e5171ed/handloom_pipeline/tracer_knot_detection.py#L669)

**适用边界。** 未来若单独研究“给定路径的局部 under/over 分类”，可另行制定配对评价协议。当前不能据此计算实例 IoU/PQ、merge/split、whole-visible、完整线缆恢复率、自然遮挡两侧正负关联准确率或自动交叉定位性能；也不能把稀疏路径膨胀成分割 GT。路径可能经过不可见部分，且上下关系不等于不同线缆身份。缺少源图映射时，不将 76 个图块视为 76 张独立场景图；与 tracer110、官方训练集的场景重叠仍未知。

处理决定：作为**审计通过但主指标不准入**的公开补充资源保留，不加入已冻结模型评分队列。现有自然身份事件标注缺口不会因此自动补齐。

本目录证据：`DATA_AUDIT.json` 含全量字段、计数、逐记录 SHA、可用性判定与官方行号；`source_manifest.json` 固定下载来源；`official_source_hashes.json` 固定源码证据；`raw/` 保留原 NPY；`audit_crossing.py` 可离线重新核验；`all76_crops_contact_sheet.png` 展示全部 76 条，放大仅用于查看，未增加图像信息。最后的 `AUDIT_FILES_SHA256.json` 对审计交付物记录 SHA-256（其自身除外）。
