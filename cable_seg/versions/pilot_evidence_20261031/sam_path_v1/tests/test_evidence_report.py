"""Reporting fixtures are software evidence only, never model accuracy results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from sam_path_v1.checkpoint import atomic_json, sha256
from sam_path_v1.evidence_report import (LOSSES, COMPONENTS, aggregate, audit_training,
    build_report, package_results, paired_bootstrap, seed_groups, diagnostic_exports)
from sam_path_v1.metrics import summarize


def metric_report(value=.5, failure=False):
    rows = [dict(id=str(i), instance_iou=value if not (failure and i == 0) else 0.,
        union_iou=value, tp=1, fp=1, fn=1, matched_iou=value, n_pred=2, n_gt=2,
        empty_prediction=False, failed=bool(failure and i == 0), gt_available=True,
        identity_events=dict(events=[dict(id="event", same_instance=True, eligible=True,
            covered=not failure, correct=not failure)])) for i in range(2)]
    if failure:
        rows[0].update(tp=0, fp=0, fn=2, matched_iou=0., union_iou=0., n_pred=0, empty_prediction=True)
    return dict(**summarize(rows), per_image=rows, diagnostic_failures=0,
        identity_diagnostics=dict(complete=True, whole_visible_recovery_rate=.4,
            sensitivity={"0.1": dict(merge_rate=.1, fragmentation_rate=.2)},
            events=dict(positive=dict(events=2, accuracy=.5), negative=dict(events=2, accuracy=.25),
                        coverage=.75, balanced_accuracy=.375)))


def software_fixture(root):
    """Small all-phase fixture deliberately fails the real 42-run completeness gate."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    jobs, expected, receipts = [], [], []
    manifest = root/"manifest.json"
    atomic_json(manifest, dict(records=[dict(id=str(i), split="test") for i in range(2)]))
    def task(ident, job, kind, report, payload, **extra):
        path = root/report; path.parent.mkdir(parents=True, exist_ok=True); atomic_json(path, payload)
        entry = dict(id=ident, phase=kind, job=job, tag="best", report=report,
                     expected_images=payload.get("images", 2), kind=kind, **extra)
        expected.append(entry)
        receipts.append(dict(**entry, exit_code=0, complete=True, failures=0,
                             diagnostic_failures=0, missing=[], sha256=sha256(path)))
    for variant in ("B", "C"):
        for seed in (42, 43, 44):
            ident = f"main_{variant}_seed{seed}"
            config = root/"configs"/(ident+".json"); config.parent.mkdir(exist_ok=True)
            atomic_json(config, dict(manifest=str(manifest)))
            jobs.append(dict(id=ident, stage="main", variant=variant, seed=seed,
                             config_path=str(config), config=dict(manifest=str(manifest))))
            task("test_"+ident, ident, "synthetic", "test/"+ident+"/best/metrics.json", metric_report(.4+.05*(variant == "C")+.001*(seed-42)))
    for variant in ("B", "C"):
        job = f"main_{variant}_seed42"
        completion = root/"robustness"/job/"completion.json"
        completion.parent.mkdir(parents=True)
        for level in (0, 1, 2, 4, 8):
            report = metric_report(.5-level*.01+.03*(variant == "C"))
            report.update(applicable_images=2, severity_widths=level)
            atomic_json(completion.parent/f"severity_{level}.json", report)
        task("robust_"+job, job, "robustness", str(completion.relative_to(root)), dict(images=2, status="completed"))
    real = metric_report(.3)
    for row in real["per_image"]:
        row["prediction"] = dict(encoded_views=4, calls=dict(encoder=4, decoder=80), peak_cuda_allocated_bytes=1024)
    task("real", jobs[0]["id"], "metrics", "real/wire/metrics.json", real, domain="wire300")
    external = dict(images=110, failures=0, per_image=[dict(id=str(i), status="predicted") for i in range(110)],
        descriptive_group_summary=dict(observed_unique_rgb=87, complete=True, macro_reference_trace_support=.7))
    task("external", jobs[0]["id"], "external", "external/diagnostics.json", external)
    benchmark = dict(images=2, repetitions=1, observations=2, failures=0, diagnostic_failures=0, complete=True,
        per_observation=[dict(id=str(i), repeat=0, peak_allocated_bytes=1024**2, peak_reserved_bytes=2*1024**2, error=None) for i in range(2)],
        module_profiles=[dict(id=str(i), modules_seconds={key:.001 for key in ("encoder", "condition", "descriptor", "tracker", "decoder")}) for i in range(2)],
        parameters=dict(new_modules=200, trainable=300, total=1000), gpu="software fixture",
        summaries=dict(end_to_end_seconds=dict(median=.01, mean=.011, p95=.012)),
        query_capacity_cost=dict(available=True, results=[dict(queries=q, trained_at_this_capacity=False, accuracy_evaluated=False,
            forward_seconds=dict(n=10, mean=q/1000, median=q/1000, p95=q/1000, maximum=q/1000), peak_allocated_bytes=q*1024) for q in (8, 16, 24, 48, 80)]))
    task("benchmark", jobs[3]["id"], "benchmark", "benchmark/benchmark.json", benchmark)
    protocol = dict(jobs=jobs, counts=dict(train=2000, validation=300), test_manifest=str(manifest))
    atomic_json(root/"protocol.json", protocol)
    atomic_json(root/"expected_evaluations.json", expected)
    atomic_json(root/"task_receipts.json", dict(expected=expected, tasks=receipts, complete=True, missing_expected=[]))
    return protocol


class EvidenceReportTests(unittest.TestCase):
    def test_training_complete_then_loss_or_curve_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = dict(id="fixture", stage="main", variant="C", seed=42)
            folder = root/"train"/"fixture"; folder.mkdir(parents=True)
            atomic_json(folder/"completion.json", dict(status="completed", progress=dict(epoch=5,
                batch_cursor=0, opportunities=10, successful_updates=10, amp_skips=0, samples=10)))
            atomic_json(folder/"provenance.json", dict(resume_from=None, parameter_counts={}))
            raw = {key: 0. for key in LOSSES}; weighted = {key: 0. for key in LOSSES[:-1]}
            components = {key: 0. for key in COMPONENTS}
            for epoch in range(1, 6):
                report = metric_report()
                for row in report["per_image"]:
                    row.update(loss=raw, weighted_loss=weighted, loss_components=components, weighted_loss_components=components)
                report.update(losses=dict(complete=True, observed_images=2, failed_images=0, mean=raw, weighted_mean=weighted),
                              loss_components=dict(complete=True, observed_images=2, mean=components, weighted_mean=components))
                atomic_json(folder/f"validation_epoch_{epoch:04d}.json", report)
            events = []
            for epoch in range(1, 6):
                for step in (epoch*2-1, epoch*2):
                    events.append(dict(event="update", opportunities=step, successful_updates=step, amp_skips=0, amp_skipped=False,
                        loss=raw, weighted_loss=weighted, loss_components=components, weighted_loss_components=components))
                events.extend([dict(event="epoch_train", epoch=epoch), dict(event="validation", epoch=epoch)])
            (folder/"logs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in events), encoding="utf-8")
            curves = root/"curves"/"fixture"; (curves/"fixture").mkdir(parents=True)
            for stem in [*("loss_"+key for key in LOSSES), "epoch_overview", *("loss_"+key for key in COMPONENTS)]:
                for extension in ("png", "pdf"): (curves/(stem+"."+extension)).write_bytes(b"software presence fixture")
            for name in ("steps.csv", "descriptor_components_steps.csv"):
                (curves/"fixture"/name).write_text("step\n"+"".join(str(i)+"\n" for i in range(10)), encoding="utf-8")
            for name in ("epochs.csv", "metrics.csv", "descriptor_components_epochs.csv"):
                (curves/"fixture"/name).write_text("fixture\n", encoding="utf-8")
            atomic_json(curves/"curve_receipt.json", dict(runs=dict(fixture=dict(observations=10,
                amp_skips=0, incomplete_validation_epochs=[], step_csv_sha256=sha256(curves/"fixture"/"steps.csv")))))
            self.assertTrue(audit_training(job, root, 2, 2)["complete"])
            last = folder/"validation_epoch_0005.json"
            payload = json.loads(last.read_text()); payload["per_image"][0]["loss_components"] = None
            atomic_json(last, payload)
            self.assertFalse(audit_training(job, root, 2, 2)["complete"])
            payload["per_image"][0]["loss_components"] = components; atomic_json(last, payload)
            (curves/"fixture"/"steps.csv").write_text("step\ncorrupt\n")
            self.assertFalse(audit_training(job, root, 2, 2)["complete"])

    def test_paired_images_reorder_and_failure_kept(self):
        before = [dict(id="a", instance_iou=.7), dict(id="b", instance_iou=.5)]
        after = [dict(id="b", instance_iou=.6), dict(id="a", instance_iou=0.)]
        a = paired_bootstrap(before, after)
        self.assertEqual(a, paired_bootstrap(list(reversed(before)), list(reversed(after))))
        self.assertAlmostEqual(a["mean_difference_pp"], -30.)
        self.assertEqual(a["n_pairs"], 2)
        self.assertFalse(paired_bootstrap(before, after[:1])["complete"])

    def test_missing_seed_is_not_presented_as_complete(self):
        rows = [dict(stage="main", variant="C", checkpoint="best", seed=seed, failures=0,
                     instance_iou_macro=.5, pq_global=.4, union_iou_macro=.6) for seed in (42, 43)]
        self.assertFalse(seed_groups(rows)[0]["seed_coverage_complete"])

    def test_declared_test_failures_and_missing_phases_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); protocol = software_fixture(root)
            expected = json.loads((root/"expected_evaluations.json").read_text())
            first = root/expected[0]["report"]
            atomic_json(first, metric_report(failure=True))
            result, _, _ = aggregate(protocol, root)
            self.assertFalse(result["complete"])
            self.assertEqual(result["rows"][0]["images"], 2)
            self.assertEqual(result["rows"][0]["failures"], 1)
            self.assertTrue(result["task_audit"][0]["problems"])
            self.assertEqual(len(result["robustness"]), 10)
            self.assertEqual(len(result["real"]), 1)
            self.assertEqual(len(result["external"]), 1)
            self.assertEqual(len(result["benchmarks"]), 1)

    def test_benchmark_false_success_cannot_hide_module_or_capacity_failures(self):
        for failure in ("oom", "missing_capacity", "missing_profile"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); protocol=software_fixture(root)
                path=root/"benchmark"/"benchmark.json"; report=json.loads(path.read_text())
                if failure == "oom": report["query_capacity_cost"]["results"][-1].update(error="OutOfMemoryError: fixture", failure_kind="oom")
                elif failure == "missing_capacity": report["query_capacity_cost"]["results"].pop()
                else: report["module_profiles"].pop()
                atomic_json(path,report)
                receipt_path=root/"task_receipts.json"; receipts=json.loads(receipt_path.read_text())
                next(r for r in receipts["tasks"] if r["id"] == "benchmark")["sha256"]=sha256(path)
                atomic_json(receipt_path,receipts)
                result,_,_=aggregate(protocol,root)
                task=next(r for r in result["task_audit"] if r["id"] == "benchmark")
                self.assertFalse(task["complete"])
                self.assertGreater(task["diagnostic_failures"],0)
                self.assertEqual(result["benchmarks"][0]["median_ms"],10.)
                self.assertFalse(result["complete"])

    def test_posthoc_tables_keep_failures_empty_and_fixed_worst_twenty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); protocol=software_fixture(root)
            result,reports,stress=aggregate(protocol,root)
            for variant in ("B","C"):
                reports[f"test_main_{variant}_seed42"]["per_image"]=[dict(id=f"{i:02d}",instance_iou=.5 if variant=="B" else i/100,
                    failed=variant=="C" and i==0,empty_prediction=variant=="C" and i==1,
                    error="fixture failure" if variant=="C" and i==0 else None) for i in range(25)]
            reports["test_main_C_seed42"]["difficulty_strata"]={"instance_count":[dict(bin="1-4",images=25,failures=1)]}
            result=diagnostic_exports(protocol,result,reports,stress)
            failures=[r for r in result["failures"] if r["task_id"]=="test_main_C_seed42"]
            self.assertEqual([r["image_id"] for r in failures],["00","01"])
            pairs=[r for r in result["paired_degradations"] if r["seed"]==42]
            self.assertEqual(len(pairs),20)
            self.assertEqual([r["rank"] for r in pairs],list(range(1,21)))
            self.assertEqual(pairs[0]["image_id"],"00")
            self.assertIn("no tuning",pairs[0]["selection"])
            self.assertEqual(result["difficulty_strata"][0]["images"],25)
            self.assertEqual(result["difficulty_strata"][0]["axis"],"instance_count")

    def test_real_html_figures_and_archive_hashes(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib needed for report rendering")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); protocol = software_fixture(root)
            out = root/"final_results"
            result = build_report(protocol, root, out)
            self.assertFalse(result["complete"])
            document = (out/"RESULTS_ZH.html").read_text(encoding="utf-8")
            self.assertIn("<table>", document)
            self.assertIn("<img src='figures/", document)
            self.assertNotIn("<pre>", document)
            for filename in ("failures.csv", "paired_degradations.csv", "difficulty_strata.csv"):
                self.assertTrue((out/filename).is_file())
                self.assertIn(filename,document)
            for name in result["figures"]:
                self.assertTrue((out/name).read_bytes().startswith(b"\x89PNG"))
                self.assertTrue((out/name.replace(".png", ".pdf")).is_file())
            with self.assertRaises(ValueError): package_results(protocol, root, out)
            atomic_json(root/"completion.json", dict(status="software_fixture_incomplete_evidence"))
            (root/"last.pt").write_bytes(b"never include model weights")
            receipt = package_results(protocol, root, out)
            with zipfile.ZipFile(receipt["path"]) as archive:
                self.assertIn("runs/completion.json", archive.namelist())
                self.assertNotIn("runs/last.pt", archive.namelist())
                self.assertIn("source/sam_path_v1/identity.py", archive.namelist())
                manifest = json.loads(archive.read("download_manifest.json"))
                for name, item in manifest["files"].items():
                    data = archive.read(name)
                    self.assertEqual(len(data), item["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(sha256(receipt["path"]), receipt["sha256"])


if __name__ == "__main__": unittest.main()
