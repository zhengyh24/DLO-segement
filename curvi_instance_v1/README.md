# Curvi Instance v1

`RGB → 自动种子 → 实例 query / anchor → 当前实例 mask → GRU 更新 → 出口邻窗 → 全图融合`

- 编码器：深度可分离残差 CNN，32/64/128/192 通道，GroupNorm + SiLU。
- 发现头：从 RGB 特征预测种子热图，滑窗聚合后选峰值。
- 实例状态：128 维 query 动态更新，初始 anchor 固定；用预测 mask 池化特征更新 GRU。
- 条件解码：U-Net 跳接读取 seed 与历史 mask，FiLM 注入 query/anchor。
- 出口头：N、NE、E、SE、S、SW、W、NW 八方向调度相邻窗口。
- 输出：实例 ID、可见总前景和冲突 unknown 层。

训练展开 3 个窗口。损失由 mask BCE/Dice、其他实例负例、种子 BCE/Dice、出口 BCE 构成；历史 teacher forcing 从 0.5 退火到 0。query 跨步反向传播，空间历史反馈 detach。

数据使用原生像素，padding 和重叠身份歧义区域不计入监督。条件验证使用 GT 起点和轨迹，其 `conditional_target_iou` 与自动 RGB 全流程实例指标分别记录。

主要入口：`run.py train`、`predict`、`check`、`prepare`、`learnability`。所有参数可用 `--help` 查看。`prepare` 默认从随仓库提供的样本集中生成 2+2 小副本；完整数据按 `data/manifests/pilot2300.json` 导出。
