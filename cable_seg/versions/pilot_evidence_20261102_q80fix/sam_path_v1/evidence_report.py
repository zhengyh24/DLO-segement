"""Complete evidence audit, Chinese report and post-completion archive.

The queue writes its final completion after build_report, then invokes
--package-only. Scientific limitations are distinct from execution failures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np

from .checkpoint import atomic_json, new_directory, sha256
from .benchmark import benchmark_diagnostic_problems
from .metrics import summarize

METRICS = ("instance_iou_macro", "pq_global", "union_iou_macro")
LOSSES = ("mask_bce", "mask_dice", "object", "quality", "path", "descriptor", "geometry", "total")
COMPONENTS = ("descriptor_contrast", "descriptor_compact")
CONTRASTS = (("B", "A"), ("B", "S"), ("C", "B"), ("C", "D"), ("C", "MASK_ONLY"),
             ("C", "CNN"), ("C", "NO_GEOMETRY"), ("C", "NO_CONTRAST"), ("C", "PARALLEL"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) or ["no_observations"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for key, v in row.items()})


def finite_mapping(value, keys):
    return isinstance(value, dict) and all(key in value and isinstance(value[key], (int, float)) and math.isfinite(value[key]) for key in keys)


def report_path(runs, value):
    path = Path(value)
    return path if path.is_absolute() else Path(runs)/path


def audit_training(job, runs, train_count=2000, validation_count=300):
    folder = Path(runs)/"train"/job["id"]
    problems, expected = [], train_count*5
    result = dict(job=job["id"], stage=job["stage"], variant=job["variant"], seed=job["seed"],
                  expected_updates=expected, problems=problems, complete=False)
    try:
        completion = read(folder/"completion.json")
        p = completion["progress"]
        fields = dict(epoch=5, batch_cursor=0, opportunities=expected, successful_updates=expected, amp_skips=0, samples=expected)
        if completion.get("status") != "completed" or any(p.get(k) != v for k, v in fields.items()):
            problems.append("训练预算/成功更新/AMP跳步不符")
        result.update(successful_updates=p.get("successful_updates"), amp_skips=p.get("amp_skips"),
                      training_seconds=p.get("elapsed_seconds"), best_epoch=p.get("best_epoch"))
        provenance = read(folder/"provenance.json")
        result["parameters"] = provenance.get("parameter_counts")
        if provenance.get("resume_from") is not None:
            problems.append("意外从其他实验恢复")
        val_members = None
        for epoch in range(1, 6):
            val = read(folder/f"validation_epoch_{epoch:04d}.json")
            rows = val.get("per_image", [])
            members = {r["id"] for r in rows}
            if val_members is None: val_members = members
            losses, components = val.get("losses", {}), val.get("loss_components", {})
            if val.get("images") != validation_count or len(rows) != validation_count or len(members) != validation_count or members != val_members or val.get("failures") or not val.get("complete"):
                problems.append(f"验证epoch{epoch}成员/失败覆盖不完整")
            if (not losses.get("complete") or losses.get("observed_images") != validation_count or losses.get("failed_images") != 0
                    or not finite_mapping(losses.get("mean"), LOSSES) or not finite_mapping(losses.get("weighted_mean"), LOSSES[:-1])):
                problems.append(f"验证epoch{epoch}原始/加权loss不完整")
            if not components.get("complete") or components.get("observed_images") != validation_count or not finite_mapping(components.get("mean"), COMPONENTS) or not finite_mapping(components.get("weighted_mean"), COMPONENTS):
                problems.append(f"验证epoch{epoch}descriptor分解不完整")
            if any(not finite_mapping(r.get("loss"), LOSSES) or not finite_mapping(r.get("weighted_loss"), LOSSES[:-1])
                   or not finite_mapping(r.get("loss_components"), COMPONENTS) or not finite_mapping(r.get("weighted_loss_components"), COMPONENTS) or r.get("loss_error") for r in rows):
                problems.append(f"验证epoch{epoch}逐图loss覆盖不完整")
        updates, train_epochs, val_epochs = 0, [], []
        with (folder/"logs.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith("\n"): raise ValueError(f"日志未完成行{line_number}")
                event = json.loads(line)
                if event.get("event") == "update":
                    updates += 1
                    if event.get("successful_updates") != updates or event.get("opportunities") != updates or event.get("amp_skipped") or event.get("amp_skips"):
                        raise ValueError(f"第{updates}次更新不连续或跳步")
                    for key, names in (("loss", LOSSES), ("weighted_loss", LOSSES[:-1]), ("loss_components", COMPONENTS), ("weighted_loss_components", COMPONENTS)):
                        if not finite_mapping(event.get(key), names): raise ValueError(f"更新{updates}:{key}缺失/非有限")
                elif event.get("event") == "epoch_train": train_epochs.append(event["epoch"])
                elif event.get("event") == "validation": val_epochs.append(event["epoch"])
        if updates != expected or train_epochs != [1, 2, 3, 4, 5] or val_epochs != [1, 2, 3, 4, 5]:
            problems.append("JSONL更新次数或5轮训练/验证事件不完整")
        result["logged_updates"] = updates
        curves = Path(runs)/"curves"/job["id"]
        receipt = read(curves/"curve_receipt.json")
        curve_run = receipt.get("runs", {}).get(job["id"], {})
        if curve_run.get("observations") != expected or curve_run.get("amp_skips") or curve_run.get("incomplete_validation_epochs"):
            problems.append("绘图凭据的更新/验证覆盖不符")
        step_csv = curves/job["id"]/"steps.csv"
        if not step_csv.is_file() or sha256(step_csv) != curve_run.get("step_csv_sha256"):
            problems.append("loss逐步CSV哈希不符")
        else:
            with step_csv.open(encoding="utf-8-sig") as stream:
                if sum(1 for _ in csv.DictReader(stream)) != expected: problems.append("loss逐步CSV行数不符")
        for stem in [*("loss_"+key for key in LOSSES), "epoch_overview", *("loss_"+key for key in COMPONENTS)]:
            for suffix in ("png", "pdf"):
                if not (curves/(stem+"."+suffix)).is_file(): problems.append(f"缺loss绘图:{stem}.{suffix}")
        for name in ("epochs.csv", "metrics.csv", "descriptor_components_steps.csv", "descriptor_components_epochs.csv"):
            if not (curves/job["id"]/name).is_file(): problems.append("缺loss CSV:"+name)
        component_csv = curves/job["id"]/"descriptor_components_steps.csv"
        if component_csv.is_file():
            with component_csv.open(encoding="utf-8-sig") as stream:
                if sum(1 for _ in csv.DictReader(stream)) != expected: problems.append("descriptor逐步CSV行数不符")
        result["curve_receipt"] = str(curves/"curve_receipt.json")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    result["complete"] = not problems
    return result


def paired_bootstrap(before, after, metric="instance_iou", repetitions=1000, seed=20261031):
    left, right = {str(r["id"]): r for r in before}, {str(r["id"]): r for r in after}
    if len(left) != len(before) or len(right) != len(after) or set(left) != set(right) or not left:
        return dict(complete=False, reason="底图ID缺失/重复/不配对", n_pairs=len(set(left)&set(right)))
    delta = np.array([right[k][metric]-left[k][metric] for k in sorted(left)], dtype=float)
    if not np.isfinite(delta).all(): return dict(complete=False, reason="非有限指标", n_pairs=len(delta))
    rng = np.random.default_rng(seed)
    values = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(repetitions)]
    return dict(complete=True, n_pairs=len(delta), mean_difference_pp=100*float(delta.mean()),
        ci95_difference_pp=(100*np.quantile(values, [.025, .975])).tolist(), improved=int((delta > 0).sum()),
        equal=int((delta == 0).sum()), degraded=int((delta < 0).sum()), bootstrap_seed=seed, repetitions=repetitions,
        unit="paired base image; descriptive image resampling, not independent-scene inference")


def seed_groups(rows):
    result = []
    for stage, variant, tag in sorted({(r["stage"], r["variant"], r["checkpoint"]) for r in rows}):
        subset = [r for r in rows if (r["stage"], r["variant"], r["checkpoint"]) == (stage, variant, tag)]
        expected = [42] if stage == "native80" else [42, 43, 44]
        seeds = sorted(r["seed"] for r in subset)
        row = dict(stage=stage, variant=variant, checkpoint=tag, seeds=seeds, expected_seeds=expected,
                   seed_coverage_complete=seeds == expected, failures=sum(r["failures"] for r in subset))
        for key in METRICS:
            values = [r[key] for r in subset if r.get(key) is not None]
            row[key+"_mean"] = float(np.mean(values)) if len(values) == len(subset) and values else None
            row[key+"_std"] = float(np.std(values, ddof=1)) if len(values) == len(subset) and len(values) > 1 else None
        result.append(row)
    return result


def metric_row(job, tag, report, task_id):
    identity = report.get("identity_diagnostics", {})
    primary = identity.get("sensitivity", {}).get("0.1", {})
    return dict(task_id=task_id, job=job["id"], stage=job["stage"], variant=job["variant"], seed=job["seed"], checkpoint=tag,
        images=report.get("images"), failures=report.get("failures"), complete=report.get("complete"),
        **{key: report.get(key) for key in METRICS}, tp=report.get("tp"), fp=report.get("fp"), fn=report.get("fn"),
        whole_visible_recovery=identity.get("whole_visible_recovery_rate"), identity_complete=identity.get("complete"),
        merge_rate=primary.get("merge_rate"), fragmentation_rate=primary.get("fragmentation_rate"),
        diagnostic_failures=report.get("diagnostic_failures", 0))


def aggregate(protocol, runs):
    runs = Path(runs)
    missing, problems = [], []
    jobs = {job["id"]: job for job in protocol["jobs"]}
    if any(protocol.get("counts", {}).get(key) != value for key, value in (("train", 2000), ("validation", 300), ("test", 1800))):
        problems.append("冻结数据预算必须为2000训练/300验证/1800测试")
    main_variants = ("A", "B", "C", "D", "MASK_ONLY", "CNN", "NO_GEOMETRY", "NO_CONTRAST", "PARALLEL", "S")
    required_jobs = {(stage, variant, seed) for stage, variants, seeds in
        (("main", main_variants, (42, 43, 44)), ("sam21", ("A", "B", "C"), (42, 43, 44)),
         ("native80", ("A", "B", "C"), (42,))) for variant in variants for seed in seeds}
    if {(j["stage"], j["variant"], j["seed"]) for j in jobs.values()} != required_jobs:
        problems.append("42组配置/训练种子组合与固定矩阵不一致")
    training = [audit_training(job, runs, protocol.get("counts", {}).get("train", 2000), protocol.get("counts", {}).get("validation", 300)) for job in protocol["jobs"]]
    for stage, expected_count in (("main", 30), ("sam21", 9), ("native80", 3)):
        if sum(job["stage"] == stage for job in jobs.values()) != expected_count: problems.append(f"{stage}训练数量非{expected_count}")
    if len(jobs) != 42: problems.append("预定训练非42组")
    path = runs/"expected_evaluations.json"
    expected = read(path) if path.is_file() else protocol.get("expected_evaluations", [])
    if not expected: missing.append("expected_evaluations.json")
    if len({r["id"] for r in expected}) != len(expected): problems.append("预期任务ID重复")
    required = {"synthetic": 84, "robustness": 30, "benchmark": 13, "metrics": 12, "external": 18}
    for kind, count in required.items():
        if sum(r.get("kind") == kind for r in expected) != count: problems.append(f"预期{kind}任务不是{count}份")
    if protocol.get("expected_evaluations") is not None and expected != protocol["expected_evaluations"]: problems.append("冻结协议与预期评价不一致")
    path = runs/"task_receipts.json"
    receipts = read(path) if path.is_file() else {}
    if not receipts: missing.append("task_receipts.json")
    tasks = {item["id"]: item for item in receipts.get("tasks", [])}
    if receipts.get("expected") != expected: problems.append("task_receipts期望列表不符")
    task_audit, rows, real, external, benchmarks, costs, stress = [], [], [], [], [], [], []
    reports, stress_reports = {}, {}
    additional = {r["id"]: r for r in protocol.get("additional_evaluations", [])}
    test_members = None
    if Path(protocol.get("test_manifest", "")).is_file():
        test_members = {r["id"] for r in read(protocol["test_manifest"])["records"] if r["split"] == "test"}
    for item in expected:
        item = {**additional.get(item["id"], {}), **item}
        ident, kind = item["id"], item.get("kind", item.get("phase", ""))
        path = report_path(runs, item["report"])
        entry = dict(**item, problems=[])
        receipt = tasks.get(ident)
        if receipt is None: entry["problems"].append("任务凭据缺失")
        elif receipt.get("exit_code") != 0 or not receipt.get("complete") or receipt.get("failures") or receipt.get("diagnostic_failures") or receipt.get("missing"):
            entry["problems"].append("任务失败/退出异常/不完整")
        try:
            report = read(path); reports[ident] = report
            if receipt and receipt.get("sha256") != sha256(path): entry["problems"].append("报告SHA不符")
            job = jobs.get(item.get("job"), dict(id=item.get("job", ident), stage="unknown", variant="unknown", seed=None))
            tag = item.get("tag", "best")
            if kind in ("synthetic", "synthetic_test", "test", "metrics", "real", "native", "native_real", "real_native"):
                per, n = report.get("per_image", []), item.get("expected_images")
                members = {r["id"] for r in per}
                if report.get("images") != n or len(per) != n or len(members) != n: entry["problems"].append("图像分母不完整")
                if report.get("failures") or not report.get("complete") or report.get("diagnostic_failures"):
                    entry["problems"].append("报告含推理/诊断失败或未知GT")
                summary = summarize(per) if per else {}
                for key in (*METRICS, "tp", "fp", "fn", "failures"):
                    if report.get(key) != summary.get(key): entry["problems"].append("逐图复算不符:"+key)
                row = metric_row(job, tag, report, ident)
                if kind in ("synthetic", "synthetic_test", "test"):
                    rows.append(row)
                    if test_members is None or members != test_members: entry["problems"].append("固定test成员未核验或不符")
                    if not row["identity_complete"]: entry["problems"].append("身份诊断分母不完整")
                else:
                    native = [r["prediction"] for r in per if r.get("prediction")]
                    row.update(dataset=item.get("domain", item.get("dataset", item.get("phase", "real"))),
                        mean_encoded_views=float(np.mean([r["encoded_views"] for r in native])) if native else None,
                        maximum_decoder_calls=max([r["calls"]["decoder"] for r in native], default=None),
                        native_peak_allocated_bytes=max([r.get("peak_cuda_allocated_bytes") or 0 for r in native], default=0))
                    real.append(row)
            elif kind in ("robustness", "occlusion", "occlusion_test"):
                base_members = None
                for severity in (0, 1, 2, 4, 8):
                    detail = read(path.parent/f"severity_{severity}.json")
                    per, n = detail.get("per_image", []), item.get("expected_images", 300)
                    members = {r["id"] for r in per}
                    if base_members is None: base_members = members
                    if detail.get("images") != n or len(per) != n or len(members) != n or members != base_members: entry["problems"].append(f"遮挡{severity}底图分母不完整")
                    if detail.get("failures") or detail.get("diagnostic_failures") or not detail.get("identity_diagnostics", {}).get("complete"): entry["problems"].append(f"遮挡{severity}失败或诊断不完整")
                    stress_reports[(job["id"], severity)] = detail
                    row = metric_row(job, tag, detail, ident)
                    ev = detail.get("identity_diagnostics", {}).get("events", {})
                    row.update(severity=severity, applicable_images=detail.get("applicable_images"),
                        positive_events=ev.get("positive", {}).get("events"), positive_accuracy=ev.get("positive", {}).get("accuracy"),
                        negative_events=ev.get("negative", {}).get("events"), negative_accuracy=ev.get("negative", {}).get("accuracy"),
                        event_coverage=ev.get("coverage"), balanced_event_accuracy=ev.get("balanced_accuracy"))
                    stress.append(row)
            elif kind in ("external", "handloom", "external_diagnostic"):
                per = report.get("per_image", [])
                if report.get("images") != item.get("expected_images", 110) or len(per) != item.get("expected_images", 110): entry["problems"].append("HANDLOOM分母不完整")
                if report.get("failures") or any(r.get("status") != "predicted" for r in per): entry["problems"].append("HANDLOOM保留失败记录")
                observed = report.get("descriptive_group_summary", {})
                if observed.get("observed_unique_rgb") != 87 or not observed.get("complete"): entry["problems"].append("HANDLOOM87个RGB组不完整")
                external.append(dict(job=job["id"], variant=job["variant"], seed=job["seed"], checkpoint=tag,
                    images=report.get("images"), failures=report.get("failures"), unique_rgb=observed.get("observed_unique_rgb"),
                    descriptive_trace_support=observed.get("macro_reference_trace_support"), task_id=ident))
            elif kind in ("benchmark", "efficiency"):
                per = report.get("per_observation", [])
                if report.get("images") != item.get("expected_images", 100) or len(per) != report.get("images", 0)*report.get("repetitions", 0): entry["problems"].append("效率观测数不完整")
                if report.get("failures") or any(r.get("error") for r in per): entry["problems"].append("效率测量包含失败")
                diagnostics = benchmark_diagnostic_problems(report, job["variant"])
                if report.get("diagnostic_failures") or not report.get("complete") or diagnostics:
                    entry["problems"].append("成本/模块诊断失败或覆盖不完整:"+"; ".join(diagnostics))
                entry["diagnostic_failures"] = max(report.get("diagnostic_failures", 0), len(diagnostics))
                parameters = report.get("parameters", {})
                timing = report.get("summaries", {}).get("end_to_end_seconds") or {}
                benchmarks.append(dict(job=job["id"], stage=job["stage"], variant=job["variant"], seed=job["seed"], images=report.get("images"), observations=len(per), failures=report.get("failures"),
                    diagnostic_failures=entry["diagnostic_failures"], diagnostic_problems=diagnostics, complete=not entry["problems"], runtime=report.get("runtime"),
                    **{key+"_ms": 1000*timing[key] if timing.get(key) is not None else None for key in ("median", "p95", "mean")},
                    peak_allocated_mib=max([r.get("peak_allocated_bytes") or 0 for r in per], default=0)/1024**2,
                    peak_reserved_mib=max([r.get("peak_reserved_bytes") or 0 for r in per], default=0)/1024**2,
                    new_parameters=parameters.get("new_modules"), trainable_parameters=parameters.get("trainable"), total_parameters=parameters.get("total"), gpu=report.get("gpu"), task_id=ident))
                costs.extend(dict(job=job["id"], stage=job["stage"], **cost) for cost in report.get("query_capacity_cost", {}).get("results", []))
            else: entry["problems"].append("未知任务类型:"+str(kind))
        except Exception as exc:
            entry["problems"].append(f"{type(exc).__name__}: {exc}"); missing.append(ident)
        entry["complete"] = not entry["problems"]
        task_audit.append(entry)
    groups = seed_groups(rows)
    if any(not r["seed_coverage_complete"] for r in groups): problems.append("三种子或native80单种子覆盖不完整")
    if len(rows) != len(jobs)*2: problems.append("合成best/last未覆盖所有训练")
    contrasts = []
    for stage in ("main", "sam21"):
        for left, right in CONTRASTS:
            for key in METRICS:
                differences = []
                for seed in (42, 43, 44):
                    pair = {r["variant"]: r for r in rows if r["stage"] == stage and r["seed"] == seed and r["checkpoint"] == "best" and r["variant"] in (left, right)}
                    if len(pair) == 2 and all(pair[v].get(key) is not None for v in pair): differences.append(dict(seed=seed, difference_pp=100*(pair[left][key]-pair[right][key])))
                if differences:
                    values = [r["difference_pp"] for r in differences]
                    contrasts.append(dict(stage=stage, contrast=left+" minus "+right, metric=key, pairs=differences, mean_difference_pp=float(np.mean(values)), sd_difference_pp=float(np.std(values, ddof=1)) if len(values) > 1 else None, all_three_positive=len(values) == 3 and all(v > 0 for v in values), complete=len(values) == 3))
    paired = []
    for (job, level), report in stress_reports.items():
        if level and (job, 0) in stress_reports:
            for metric in ("instance_iou", "union_iou"):
                paired.append(dict(contrast="stress minus clean", job=job, severity=level, metric=metric, **paired_bootstrap(stress_reports[(job, 0)]["per_image"], report["per_image"], metric)))
    for level in (0, 1, 2, 4, 8):
        for seed in (42, 43, 44):
            left, right = f"main_C_seed{seed}", f"main_B_seed{seed}"
            if (left, level) in stress_reports and (right, level) in stress_reports: paired.append(dict(contrast="C minus B", seed=seed, severity=level, metric="instance_iou", **paired_bootstrap(stress_reports[(right, level)]["per_image"], stress_reports[(left, level)]["per_image"])))
    result = dict(training=training, rows=rows, groups=groups, contrasts=contrasts, robustness=stress, robustness_paired=paired,
        real=real, external=external, benchmarks=benchmarks, query_capacity_cost=costs, task_audit=task_audit, expected_evaluations=expected,
        missing=missing, problems=problems, expected_training=42, successful_training=sum(r["complete"] for r in training), expected_successful_updates=420000,
        complete=not missing and not problems and len(training) == 42 and all(r["complete"] for r in training) and bool(expected) and all(r["complete"] for r in task_audit) and bool(receipts.get("complete")),
        scientific_gaps=["未取得可靠独立真实场景划分及完整可见实例标注", "固定合成test已报告过，非新盲测", "native80仅单种子工程验证", "HANDLOOM仅轨迹描述性诊断", "HANDLOOM另76个20x20交叉图块已审计，其官方分类依赖给定路径/中心/方向，缺实例与场景真值，未纳入无提示身份评分", "未开展多视角重建"],
        statistics="三训练种子均值/样本SD与配对差；遮挡按底图配对bootstrap1000次，seed20261031；无场景独立推断或ICCV录用承诺")
    return result, reports, stress_reports


def make_figures(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(out)/"figures"; folder.mkdir()
    created = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"): fig.savefig(folder/(name+"."+extension), dpi=180)
        plt.close(fig); created.append("figures/"+name+".png")
    for stage in ("main", "sam21"):
        rows = [r for r in result["rows"] if r["stage"] == stage and r["checkpoint"] == "best"]
        variants = sorted({r["variant"] for r in rows})
        if not variants: continue
        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(variants)*1.1), 4))
        for ax, metric, label in zip(axes, METRICS[:2], ("Instance IoU (%)", "PQ (%)")):
            for i, variant in enumerate(variants):
                subset = sorted((r for r in rows if r["variant"] == variant and r.get(metric) is not None), key=lambda r: r["seed"])
                values = np.array([r[metric]*100 for r in subset])
                if len(values):
                    ax.bar(i, values.mean(), color="#b8d5e8", width=.65)
                    ax.scatter(i+np.linspace(-.12, .12, len(values)), values, color="#163c57", zorder=3)
                    if len(values) > 1: ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), color="#163c57", capsize=4)
            ax.set_xticks(range(len(variants)), variants, rotation=40, ha="right"); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
        fig.suptitle(stage+" | best; dots=training seeds; whiskers=sample SD"); save(fig, stage+"_three_seed_accuracy")
    rows = [r for r in result["benchmarks"] if r["median_ms"] is not None]
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key, label, divisor in zip(axes, ("median_ms", "peak_allocated_mib", "new_parameters"), ("End-to-end median (ms)", "Peak allocated (MiB)", "New parameters (M)"), (1, 1, 1e6)):
            ax.bar(range(len(rows)), [r[key]/divisor if r.get(key) is not None else np.nan for r in rows], color="#69a696")
            ax.set_xticks(range(len(rows)), [r["job"] for r in rows], rotation=75, ha="right", fontsize=7); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
        fig.suptitle("Measured inference cost; failures retained in tables"); save(fig, "inference_cost")
    if result["query_capacity_cost"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for job in sorted({r["job"] for r in result["query_capacity_cost"]}):
            rows = [r for r in result["query_capacity_cost"] if r["job"] == job and not r.get("error")
                    and finite_mapping(r.get("forward_seconds"), ("median",))]
            if rows:
                axes[0].plot([r["queries"] for r in rows], [r["forward_seconds"]["median"]*1000 for r in rows], "o-", label=job)
                memory = [r for r in rows if r.get("peak_allocated_bytes") is not None]
                axes[1].plot([r["queries"] for r in memory], [r["peak_allocated_bytes"]/1024**2 for r in memory], "o-", label=job)
        for ax, ylabel in zip(axes, ("Forward median (ms)", "Peak allocated (MiB)")):
            ax.set_xlabel("Query count"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=7)
        fig.suptitle("Cost-only repeated-query microbenchmark; no accuracy claim"); save(fig, "query_capacity_cost")
    if result["robustness"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for variant in ("A", "B", "C", "D", "PARALLEL", "S"):
            rows = [r for r in result["robustness"] if r["variant"] == variant]
            for ax, metric in zip(axes, ("instance_iou_macro", "balanced_event_accuracy")):
                values = [[r[metric] for r in rows if r["severity"] == level and r.get(metric) is not None] for level in (0, 1, 2, 4, 8)]
                means = [np.mean(v) if v else np.nan for v in values]
                if np.isfinite(means).any(): ax.plot((0, 1, 2, 4, 8), 100*np.asarray(means), "o-", label=variant)
        for ax, label in zip(axes, ("Instance IoU (%)", "Balanced event accuracy (%)")):
            ax.set_xlabel("Artificial opaque gap / local width"); ax.set_ylabel(label); ax.grid(alpha=.2)
            if ax.lines: ax.legend()
        fig.suptitle("Controlled artificial occlusion; paired base images"); save(fig, "occlusion_response")
    return created


def table(rows, columns):
    if not rows: return "<p class='missing'>暂无完整观测，不能据此作效果结论。</p>"
    def cell(value):
        return html.escape(f"{value:.4f}" if isinstance(value, float) else "—" if value is None else json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
    return "<div class='scroll'><table><thead><tr>"+"".join("<th>"+html.escape(label)+"</th>" for _, label in columns)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+cell(row.get(key))+"</td>" for key, _ in columns)+"</tr>" for row in rows)+"</tbody></table></div>"


def diagnostic_exports(protocol, result, reports, stress_reports):
    """Post-hoc inspection tables only; never feed checkpoint selection or tuning."""
    jobs = {job["id"]: job for job in protocol["jobs"]}
    tasks = {item["id"]: item for item in result["expected_evaluations"]}
    sources = {}
    path = Path(protocol.get("test_manifest", ""))
    if path.is_file():
        sources = {row["id"]: row.get("image") for row in read(path)["records"]}
    failures, strata = [], []
    all_reports = [(ident, report, tasks.get(ident, {})) for ident, report in reports.items()]
    for (job, level), report in stress_reports.items():
        all_reports.append((job+"_severity_"+str(level), report,
            dict(job=job, tag="best", kind="robustness", severity=level,
                 report=f"robustness/{job}/severity_{level}.json")))
    for ident, report, task in all_reports:
        job = jobs.get(task.get("job"), {})
        metadata = dict(task_id=ident, job=task.get("job"), stage=job.get("stage"),
            variant=job.get("variant"), seed=job.get("seed"), checkpoint=task.get("tag"),
            kind=task.get("kind"), severity=task.get("severity"), source_report=task.get("report"))
        for row in report.get("per_image", report.get("per_observation", [])):
            errors = {key:value for key,value in row.items() if "error" in key and value}
            failed = bool(row.get("failed") or row.get("status") == "failed" or errors)
            empty = bool(row.get("empty_prediction") or row.get("n_pred") == 0 or row.get("predicted_instances") == 0)
            if failed or empty:
                failures.append(dict(**metadata, record_type="image", image_id=row.get("id"),
                    image_reference=row.get("image", row.get("rgb_path", sources.get(row.get("id")))),
                    repeat=row.get("repeat"), failed=failed, empty_prediction=empty, errors=errors,
                    instance_iou=row.get("instance_iou"), n_pred=row.get("n_pred", row.get("predicted_instances")),
                    n_gt=row.get("n_gt"), gt_available=row.get("gt_available"), failure_kind=row.get("failure_kind")))
        for category, records in (("module_profile", report.get("module_profiles", [])),
                ("query_capacity", report.get("query_capacity_cost", {}).get("results", []))):
            for row in records:
                if row.get("error"):
                    failures.append(dict(**metadata, record_type=category, image_id=row.get("id"),
                        queries=row.get("queries"), failed=True, errors=dict(error=row["error"]), failure_kind=row.get("failure_kind")))
        for axis, bins in report.get("difficulty_strata", {}).items():
            for row in bins:
                strata.append(dict(**metadata, axis=axis, **row))
    selected, comparisons = {}, []
    for row in result["rows"]:
        selected[(row["stage"], row["variant"], row["seed"], row["checkpoint"])] = row
    for stage, seed, tag in sorted({(r["stage"], r["seed"], r["checkpoint"]) for r in result["rows"]}):
        for after, before in (("C", "B"), ("B", "A")):
            left, right = selected.get((stage, before, seed, tag)), selected.get((stage, after, seed, tag))
            if left is None or right is None: continue
            left_rows = {r["id"]: r for r in reports[left["task_id"]].get("per_image", [])}
            right_rows = {r["id"]: r for r in reports[right["task_id"]].get("per_image", [])}
            candidates = []
            for ident in left_rows.keys() & right_rows.keys():
                a,b = left_rows[ident],right_rows[ident]
                if not finite_mapping(a, ("instance_iou",)) or not finite_mapping(b, ("instance_iou",)): continue
                delta = 100*(b["instance_iou"]-a["instance_iou"])
                candidates.append(dict(stage=stage, seed=seed, checkpoint=tag, contrast=after+" minus "+before,
                    image_id=ident, image_reference=sources.get(ident), difference_pp=delta, degraded=delta < 0,
                    before_instance_iou=a["instance_iou"], after_instance_iou=b["instance_iou"],
                    before_failed=a.get("failed"), after_failed=b.get("failed"),
                    before_report=tasks[left["task_id"]].get("report"), after_report=tasks[right["task_id"]].get("report"),
                    difficulty=b.get("difficulty"),
                    selection="post-hoc lowest 20 paired IoU deltas, stable image-ID tie break; inspection only, no tuning"))
            for rank,row in enumerate(sorted(candidates,key=lambda r:(r["difference_pp"],str(r["image_id"])))[:20],1):
                comparisons.append(dict(rank=rank, **row))
    return dict(failures=failures, paired_degradations=comparisons, difficulty_strata=strata)


def build_report(protocol, runs, output):
    runs, out = Path(runs), new_directory(output)
    result, reports, stress_reports = aggregate(protocol, runs)
    result.update(diagnostic_exports(protocol, result, reports, stress_reports))
    plots = make_figures(result, out); result["figures"] = plots
    atomic_json(out/"analysis.json", result)
    for key in ("rows", "groups", "contrasts", "robustness", "robustness_paired", "real", "external", "benchmarks", "query_capacity_cost", "training", "task_audit", "failures", "paired_degradations", "difficulty_strata"):
        write_csv(out/(key+".csv"), result[key])
    (out/"per_image").mkdir(); (out/"per_event").mkdir()
    for ident, report in list(reports.items())+[(job+"_severity_"+str(level), r) for (job, level), r in stress_reports.items()]:
        rows = report.get("per_image", report.get("per_observation", []))
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in ident)
        if rows: write_csv(out/"per_image"/(safe+".csv"), rows)
        events = [dict(image_id=row["id"], **event) for row in rows for event in (row.get("identity_events") or {}).get("events", [])]
        if events: write_csv(out/"per_event"/(safe+".csv"), events)
    parts = ["<h1>WireSeg 单视角实验补强报告</h1><p>执行证据完整：<strong>"+str(result["complete"])+"</strong>；训练核验 "+str(result["successful_training"])+"/42。证据完整不等于方法主张成立或达到ICCV录用标准。</p>",
        "<p>每组2,000训练图、5轮。旧全量B/C保留；按冻结验证规则选best，同时报告last。无多视角重建。表中率值为0–1，图中为百分比。</p>",
        "<h2>1. 路径、描述符与联合训练贡献</h2><p>B−A/B−S检验tracker；C−B/C−CNN检验描述符；D阻断decoder整体反馈，MASK_ONLY单独阻断mask梯度。几何、对比监督及无递归模块分别消融；不同total loss不直接比较。C描述符含401,155参数，整体新增3,636,235参数；NO_GEOMETRY保留771参数几何头以匹配结构，但该头不更新。</p>",
        table(result["groups"], [("stage","阶段"),("variant","配置"),("checkpoint","检查点"),("seeds","种子"),("instance_iou_macro_mean","实例IoU均值"),("instance_iou_macro_std","种子SD"),("pq_global_mean","PQ均值"),("pq_global_std","种子SD"),("failures","失败")]),
        table(result["contrasts"], [("stage","阶段"),("contrast","对照"),("metric","指标"),("mean_difference_pp","均值差pp"),("sd_difference_pp","差值SD"),("all_three_positive","三种子均正"),("complete","完整")]),
        "<h2>2. 身份关联与可控遮挡</h2><p>未覆盖正/负对均判错。完整可见恢复要求单预测精度/召回≥0.90，每个≥16像素GT分量覆盖≥0.80；不评价隐藏像素。合并/碎裂按重叠≥max(16像素,10% GT面积)定义，5%/20%敏感性结果保存在原报告。无有效切点图保留不重抽。</p>",
        table(result["rows"], [("job","训练"),("checkpoint","检查点"),("whole_visible_recovery","完整可见恢复"),("merge_rate","合并率"),("fragmentation_rate","碎裂率"),("failures","失败"),("identity_complete","身份分母完整")]),
        table(result["robustness"], [("job","训练"),("severity","遮挡倍线宽"),("images","全部底图"),("applicable_images","有效切点"),("positive_events","正对数"),("positive_accuracy","正对正确率"),("negative_events","负对数"),("negative_accuracy","负对正确率"),("balanced_event_accuracy","平衡正确率"),("failures","失败")]),
        "<p>人工遮挡不等于自然遮挡。配对底图bootstrap固定seed=20261031、1000次；同底图全部扰动配对，不当独立样本。区间为图像集合描述，不假设场景独立。<a href='robustness_paired.csv'>配对差值与95%区间</a>。</p>",
        "<p><a href='difficulty_strata.csv'>全部难度分层</a>：实例数、可见碎片数、3像素膨胀邻近比例、实例平均RGB相似度，阈值随源码冻结。邻近与RGB是固定代理量，不是真实交叉或自然遮挡标注；未知GT归入unavailable，不隐去失败。</p>",
        "<h2>3. 真实回归与外部数据</h2><p>Wire300/C135已被历史开发使用；Wire300为模型分歧筛选困难集。Q80单种子原生裁块验证必须报告真实编码/分批解码次数，不能声称一次整图推理。native80训练mosaic由16张训练来源图原生crop构造，来源图曝光量与主2000样本分支不同；本表单列为容量/工程验证。</p>",
        table(result["real"], [("job","训练"),("dataset","真实集"),("checkpoint","检查点"),("images","数量"),("instance_iou_macro","实例IoU"),("pq_global","PQ"),("failures","失败"),("mean_encoded_views","平均编码块"),("maximum_decoder_calls","最大decoder调用")]),
        "<p>HANDLOOM 110记录/87唯一RGB/20重复组。缺完整可见mask、轨迹可见性与可靠场景ID；仅按RGB组等权汇总描述性轨迹支持，不作实例准确率或独立场景结论。</p>",
        table(result["external"], [("job","训练"),("checkpoint","检查点"),("images","记录数"),("unique_rgb","唯一RGB"),("descriptive_trace_support","描述性轨迹支持"),("failures","失败")]),
        "<h2>4. 实测成本与第二骨干</h2><p>完整延迟含PNG读取、原生padding、H2D、同步前向、排他实例解码/D2H；不含GT读取评分或保存输出。主计时、模块诊断和Q扫描统一禁用外层autocast，适配器内部精度配置保留：CUDA BF16在S覆盖原生检测分支，其余模型只覆盖冻结视觉编码器。参数dtype与实际配置见benchmarks.csv的runtime及原始JSON；不能用训练AMP代替此计时精度说明。一次批量decoder不等于恒定成本。Q微基准重复已训练查询，只测成本，不报告容量扩展精度。SAM2.1重新训练结果不是跨骨干零训练即插即用。</p>",
        table(result["benchmarks"], [("job","训练"),("median_ms","延迟中位ms"),("p95_ms","p95 ms"),("peak_allocated_mib","allocated MiB"),("peak_reserved_mib","reserved MiB"),("new_parameters","新增参数"),("trainable_parameters","可训练参数"),("failures","推理失败"),("diagnostic_failures","诊断失败"),("complete","完整")]),
        "<p>模块诊断缺失、成本点缺失或OOM均保留记录，并令执行证据不完整；已完成的主延迟测量仍可查看。<a href='query_capacity_cost.csv'>全部查询成本点（含失败）</a>。</p>"]
    for image in plots:
        parts.append("<figure><a href='"+image.replace(".png", ".pdf")+"'><img src='"+image+"' alt='"+html.escape(Path(image).stem)+"'></a><figcaption>点击图像下载PDF，数字见对应CSV。</figcaption></figure>")
    parts.append("<h2>5. 全部loss曲线与复现</h2><p>每组10,000成功更新、5轮300图验证；原始/加权步骤CSV、逐轮记录和JSONL均保留。contrast/compact是descriptor分解，不重复计入total。</p><ul>")
    for job in protocol["jobs"]:
        curve = runs/"curves"/job["id"]/"epoch_overview.png"
        if curve.is_file():
            relative = Path(os.path.relpath(curve, out)).as_posix()
            parts.append("<li><a href='"+html.escape(relative, quote=True)+"'>"+html.escape(job["id"])+"：loss总览</a> · <a href='"+html.escape((Path(relative).parent/"curve_receipt.json").as_posix(), quote=True)+"'>绘图凭据</a></li>")
    parts.append("</ul><h2>失败与证据边界</h2><p><a href='failures.csv'>全部失败/错误/空预测</a>；<a href='paired_degradations.csv'>逐种子best/last配对最差20张</a>。C−B与B−A按实例IoU差值升序、图像ID稳定排序，保留原报告与图像索引；属于事后诊断，不能用于调参或重新选择检查点。最差20张仍可能没有负差，degraded字段区分实际退化。</p>"+table([r for r in result["training"] if not r["complete"]], [("job","训练"),("problems","证据问题")])+table([r for r in result["task_audit"] if not r["complete"]], [("id","任务"),("problems","证据问题")]))
    parts.append("<p>全局问题："+html.escape(json.dumps(result["problems"]+result["missing"], ensure_ascii=False))+"</p><ul>"+"".join("<li>"+html.escape(g)+"</li>" for g in result["scientific_gaps"])+"</ul><p>独立真实场景不足是科学证据缺口，不冒称软件失败。负结果、失败和退化配置均保留，不自动晋升方法或承诺录用。</p>")
    if protocol.get("previous_failed_attempt"):
        parts.append("<h2>正式训练前的失败记录</h2><p>第一版在原生SAM3.1 LoRA准入时因空点提示grid_sample反向与严格确定性冲突停止，未开始正式训练。独立r2只短路空提示操作，全部42训练配置内容保持不变，16结构重新准入。原源码、失败日志和检查点保留；归档包含prior_attempts/admission_attempt1_failure.zip。</p><p>"+html.escape(protocol.get("revision_reason", ""))+"</p>")
    document = "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>WireSeg实验补强</title><style>body{max-width:1350px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui,sans-serif;color:#18232b}h2{margin-top:36px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #cbd5df;padding:7px;text-align:left}th{background:#eaf1f6}.scroll{overflow:auto;margin:16px 0}img{max-width:100%;height:auto}figure{margin:24px 0}.missing{color:#755126}a{color:#12537c}</style></head><body>"+"".join(parts)+"</body></html>"
    (out/"RESULTS_ZH.html").write_text(document, encoding="utf-8")
    (out/"RESULTS_ZH.txt").write_text("WireSeg实验补强\n执行证据完整："+str(result["complete"])+"\n训练核验："+str(result["successful_training"])+"/42\n完整表与图：RESULTS_ZH.html\n"+"\n".join(result["scientific_gaps"])+"\n不将实验自动认定为达到ICCV录用标准。\n", encoding="utf-8")
    return result


def package_results(protocol, runs, output):
    runs, out = Path(runs).resolve(), Path(output).resolve()
    if not (runs/"completion.json").is_file() or not (out/"analysis.json").is_file(): raise ValueError("Final completion and analysis must exist before packaging")
    files, suffixes = {}, {".json", ".jsonl", ".csv", ".png", ".pdf", ".log", ".md", ".html", ".txt", ".npz"}
    for path in runs.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes and "smoke" not in path.relative_to(runs).parts and path.name not in ("package_receipt.json", "download_manifest.json"):
            files["runs/"+path.relative_to(runs).as_posix()] = path
    if not out.is_relative_to(runs):
        for path in out.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes and path.name not in ("package_receipt.json", "download_manifest.json"): files["report/"+path.relative_to(out).as_posix()] = path
    source = Path(__file__).resolve().parent
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".json", ".sh", ".md", ".txt") and "__pycache__" not in path.parts: files["source/sam_path_v1/"+path.relative_to(source).as_posix()] = path
    for job in protocol["jobs"]:
        path = Path(job["config_path"])
        if not path.is_file(): raise FileNotFoundError("Frozen config unavailable for archive: "+str(path))
        files["frozen_configs/"+job["id"]+".json"] = path
        path = Path(job["config"]["manifest"])
        if not path.is_file(): raise FileNotFoundError("Training manifest unavailable for archive: "+str(path))
        files["protocol_inputs/"+sha256(path)[:16]+"_"+path.name] = path
    path = Path(protocol["test_manifest"])
    if path.is_file(): files["protocol_inputs/"+sha256(path)[:16]+"_"+path.name] = path
    for name in protocol.get("frozen_inputs", []):
        path=Path(name)
        if not path.is_file(): raise FileNotFoundError("Frozen admission input unavailable: "+name)
        files["protocol_inputs/"+sha256(path)[:16]+"_"+path.name]=path
    if protocol.get("previous_failed_attempt"):
        path=Path(protocol["previous_failed_attempt"]).parent/"admission_attempt1_failure.zip"
        if not path.is_file(): raise FileNotFoundError("Previous failed admission evidence unavailable: "+str(path))
        files["prior_attempts/"+path.name]=path
    for item in protocol.get("additional_evaluations", []):
        for name in item.get("inputs", []):
            path = Path(name)
            if not path.is_file(): raise FileNotFoundError("Frozen external input unavailable for archive: "+name)
            files["protocol_inputs/"+sha256(path)[:16]+"_"+path.name] = path
    archive, receipts = out/"pilot_evidence_results.zip", {}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            receipts[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            bundle.writestr(name, data)
        atomic_json(out/"download_manifest.json", dict(files=receipts, server_originals_preserved=True, includes_model_weights=False,
            note="completed-data snapshot; supervisor may later append packaging notifications"))
        bundle.writestr("download_manifest.json", (out/"download_manifest.json").read_bytes())
    receipt = dict(path=str(archive), sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)+1, includes_model_weights=False, completion_included=True)
    atomic_json(out/"package_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--runs", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args(argv); protocol = read(args.protocol)
    if args.package_only: print(json.dumps(package_results(protocol, args.runs, args.output)))
    else:
        result = build_report(protocol, args.runs, args.output)
        print(json.dumps(dict(complete=result["complete"], rows=len(result["rows"]), missing=result["missing"], problems=result["problems"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
