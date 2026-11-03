"""Offline checks of published engine identities, budgets, splits and reported scores."""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/sam_tracker"
PILOT = DATA / "pilot"


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def check_report(report, members):
    per = report["per_image"]
    assert len(per) == report["images"] == 1800
    assert {r["id"] for r in per} == members
    assert sum(r["failed"] for r in per) == report["failures"]
    iou = sum(r["instance_iou"] for r in per) / len(per)
    assert math.isclose(iou, report["instance_iou_macro"], abs_tol=1e-10)
    tp, fp, fn = (sum(r[k] for r in per) for k in ("tp", "fp", "fn"))
    pq = sum(r["matched_iou"] for r in per) / (tp + 0.5*(fp+fn))
    assert math.isclose(pq, report["pq_global"], abs_tol=1e-10)


def main():
    for line in (DATA / "source_files.sha256").read_text().splitlines():
        expected, path = line.split("  ", 1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    subsets = {}
    for stem, splits in (("pilot", ("train", "validation")), ("synthetic_36k", ("test",))):
        manifest = read_json(PILOT/"data"/(stem+"_manifest.json.gz"))
        for split in splits:
            subsets[split] = [r for r in manifest["records"] if r["split"] == split]
    assert [len(v) for v in subsets.values()] == [2000, 300, 1800]
    assert len({r["rgb_sha256"] for subset in subsets.values() for r in subset}) == 4100
    members = {r["id"] for r in subsets["test"]}
    with (PILOT/"training/summary.csv").open(encoding="utf-8") as stream:
        jobs = list(csv.DictReader(stream))
    assert Counter(j["stage"] for j in jobs) == {"main":30, "sam21":9, "native80":3}
    for j in jobs:
        assert (int(j["updates"]), int(j["amp_skips"])) == (10000, 0)
        d = PILOT/"training"/j["job"]
        version = "pilot_evidence_20261102_q80fix" if j["stage"] == "native80" else "pilot_evidence_20261031_r2"
        engine = ROOT/"cable_seg/versions"/version/"sam_path_v1"
        for file, digest in read_json(d/"model_provenance.json")["engine_source_sha256"].items():
            assert hashlib.sha256((engine/file).read_bytes()).hexdigest() == digest
        assert len(read_json(d/"validation_epochs.json.gz")) == 5
        with gzip.open(d/"curves"/j["job"]/"steps.csv.gz", "rt", encoding="utf-8-sig") as f:
            assert sum(1 for _ in csv.DictReader(f)) == 10000
    receipts = read_json(PILOT/"evaluation/task_receipts.json")
    snapshot = read_json(PILOT/"evaluation/snapshot.json")
    assert len(receipts) == snapshot["registered"] and len(receipts) + len(snapshot["pending"]) == 157
    for r in receipts:
        report = read_json(PILOT/"evaluation"/r["id"]/"report.json.gz")
        if r["kind"] == "synthetic":
            check_report(report, members)
    reports = list((DATA/"full_wireseg_BC/test").rglob("metrics.json.gz"))
    assert len(reports) == 4
    for p in reports:
        check_report(read_json(p), members)
    print(f"Verified {len(jobs)} training runs, 420,000 updates, {len(receipts)}/157 registered tasks, and four full-data reports.")


if __name__ == "__main__":
    main()
