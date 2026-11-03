"""Write a Chinese interpretation from the actual fixed evaluation results."""
import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
QUESTIONS = (
    ("B", "A", "加入 tracker"),
    ("B", "S", "与原生 LoRA 比较"),
    ("C", "B", "加入描述符"),
    ("C", "CNN", "与等参数 CNN 比较"),
    ("C", "D", "保留全部 decoder 反馈"),
    ("C", "MASK_ONLY", "保留 mask 经路径的反馈"),
    ("C", "NO_GEOMETRY", "加入几何监督"),
    ("C", "NO_CONTRAST", "加入对比监督"),
    ("C", "PARALLEL", "递归与并行 attention 比较"),
)


def paired(index, left, right, metric):
    differences = []
    for seed in SEEDS:
        a, b = index.get((left, seed), {}), index.get((right, seed), {})
        if a.get(metric) is None or b.get(metric) is None:
            return []
        differences.append(100 * (a[metric] - b[metric]))
    return differences


def display(values):
    if not values:
        return "缺数据"
    return " / ".join(f"{v:+.2f}" for v in values) + f"（均值 {statistics.mean(values):+.2f}）"


def direction(values):
    if not values:
        return "缺少完整三种子"
    if min(values) > 0:
        return "三个种子均有提升"
    if max(values) < 0:
        return "三个种子均有下降"
    return "种子间未显示一致增益"


def write_report(directory):
    a = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
    lines = ["# 单视角 WireSeg 机制验证", "",
             f"训练核验 {a['successful_training']}/42。正式评价完整性标记为 {a['complete']}。"
             "表中差值单位为百分点，按 42、43、44 顺序列出并给出均值。", ""]
    for stage, label in (("main", "SAM3.1"), ("sam21", "SAM2.1 Hiera-Tiny")):
        for tag in ("best", "last"):
            rows = {(r["variant"], r["seed"]): r for r in a["rows"]
                    if r["stage"] == stage and r["checkpoint"] == tag and r["complete"] and not r["failures"]}
            lines += [f"## {label} 固定测试 / {tag}", "",
                      "|检验|对照|实例 IoU 配对差|PQ 配对差|实例 IoU 方向|",
                      "|---|---|---|---|---|"]
            for left, right, title in QUESTIONS:
                if stage == "sam21" and right not in ("A", "B"):
                    continue
                iou, pq = (paired(rows, left, right, m) for m in ("instance_iou_macro", "pq_global"))
                lines.append(f"|{title}|{left} − {right}|{display(iou)}|{display(pq)}|{direction(iou)}|")
            lines.append("")
    lines += ["## 可控遮挡两侧的身份关联", "",
              "使用具有两侧可见片段的正对和邻近不同根的负对，分别核算覆盖与关联。"
              "同一底图的各级遮挡保持配对。以下列出 best 模型平衡关联正确率的种子配对差。", "",
              "|遮挡倍线宽|B − A|C − B|C − S|", "|---:|---|---|---|"]
    for severity in (0, 1, 2, 4, 8):
        rows = {(r["variant"], r["seed"]): r for r in a["robustness"] if r["severity"] == severity
                and r["complete"] and not r["failures"] and not r["diagnostic_failures"]}
        cells = [display(paired(rows, l, b, "balanced_event_accuracy")) for l, b in (("B", "A"), ("C", "B"), ("C", "S"))]
        lines.append("|" + str(severity) + "|" + "|".join(cells) + "|")
    lines += ["", "## 实证边界", "", "S 使用固定文本、200 个原生查询和最后八块的 Q/V LoRA。"
              "A 包含自动实例条件网络。两者的查询容量、可训练参数、原生头、目标函数不同。"
              "D 切断多种反馈，MASK_ONLY 单独隔离 mask 路径。",
              "", "自然遮挡缺少完整事件真值。Wire300 和 C135 是历史真实挑战回归。"
              "Q80 的 3 组各用一个种子并采用原生分块和分批解码，处理像素量和调用次数见真实集报告。"
              "HANDLOOM tracer 的 110 条记录对应 87 个唯一 RGB，外部轨迹支持度为描述性指标。",
              "", "第二骨干的 A/B/C 均经过重新适配训练。效率表记录主实验的同步延迟、显存、参数和查询增长的成本。"
              "配对符号描述本轮种子的方向，跨场景置信度仍需独立采集的真实场景和实例标注。"]
    lines += ["", "## Q80 查询使用情况", "",
              "|配置|有过正匹配的查询数 / 80|每查询正匹配数 min / median / max|", "|---|---:|---|"]
    for variant in ("A", "B", "C"):
        path = directory.parent/"train"/f"native80_{variant}_seed42"/"completion.json"
        if path.is_file():
            matches = json.loads(path.read_text())["progress"].get("query_positive_matches", [])
            if len(matches) == 80:
                lines.append(f"|{variant}|{sum(v > 0 for v in matches)}|{min(matches)} / {statistics.median(matches):g} / {max(matches)}|")
    lines.append("")
    lines.append("Q80 的 10,000 个锚点位置包含 4,872 张拼图和 5,128 张普通整图，三组使用同一配方。拼图共使用 77,952 个原生裁块，每次采用 16 个不同训练来源。与 Q24 主实验比较时，查询容量和训练图像曝光量同时变化。")
    index = {(r["variant"], r["seed"]): r for r in a["rows"] if r["stage"] == "main" and
             r["checkpoint"] == "best" and r["complete"] and not r["failures"]}
    lines += ["", "## 下一轮全量确认", ""]
    if a["complete"]:
        sb, sc = paired(index, "S", "B", "instance_iou_macro"), paired(index, "S", "C", "instance_iou_macro")
        if sb and sc and min(sb + sc) > 0:
            lines.append("原生 LoRA 在全部三种子上的实例 IoU 高于 B 和 C，应优先做 S 的全量确认，并作为路径方法的必备对照。")
        else:
            lines.append("保留 S 和 A 作为全量确认的原生适配与自动条件对照。")
        cb = paired(index, "C", "B", "instance_iou_macro")
        lines.append(f"C 相对 B 的实例 IoU {direction(cb)}。"
                     "下一轮对描述符的判断应同时核对遮挡关联与固定测试 PQ，避免仅凭单项最优选择模型。")
    else:
        lines.append("正式评价仍有缺失或失败，先核对任务审计表和失败分母，再决定全量配置。")
    lines += ["", "原始统计、三种子标准差、逐图/逐事件指标、失败案例、完整 loss 图表与计时见同目录 RESULTS_ZH.html。", ""]
    (directory / "CLAIMS_ZH.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    write_report(parser.parse_args().output)
