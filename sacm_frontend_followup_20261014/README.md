# 后续训练与拓扑损失

在 `sacm_frontend` 上增加独立的损失与训练入口，保留原分支实现便于对照。

- `losses.py`：后续前景/拓扑损失。
- `train.py`：独立配置、权重与训练日志。
- `check.py`、`check_fork.py`：损失和分支兼容性检查。

从仓库根执行 `python -m sacm_frontend_followup_20261014.train --help` 查看参数。
