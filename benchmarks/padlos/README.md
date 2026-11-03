# PADLOS 固定测试对照

`run_padlos.py` 是原测试驱动的本地路径适配版，使用发布的测试成员和 B 的正式测试报告检查数据一致性。需单独安装上游依赖，将固定版本 PADLOS 源码放在仓库根的 `datasets/PADLOS_official`，将其权重放在 `.models/padlos`。具体提交、权重 SHA、输入尺度、阈值及失败策略见 [协议](../../results/sam_tracker/padlos_test1800/protocol.json)。

设置 `PADLOS_CACHE` 和新输出目录 `PADLOS_RUN`。缓存目录下的 `data/rgb/<ID>.png` 与 `data/masks/<ID>.npz` 要对应 [测试视图](../../results/sam_tracker/padlos_test1800/test_view.json.gz) 内的同一个 ID 和像素哈希。准备完成后用 `python benchmarks/padlos/run_padlos.py` 运行。对照速度脚本 `benchmark_latency.py --method B` 另需 `TRACKER_B_BEST`、`SAM3_REPO` 和 `SAM3_BASE`。
