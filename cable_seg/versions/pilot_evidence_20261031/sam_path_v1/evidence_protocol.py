"""The finite, user-authorized 2,000-image experiment matrix.

This writes configurations, never starts training or reassigns a data split.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SEEDS = (42, 43, 44)
MAIN_VARIANTS = ("A", "B", "C", "D", "MASK_ONLY", "CNN", "NO_GEOMETRY", "NO_CONTRAST", "PARALLEL", "S")


def build_matrix(template, root, *, sam2_repo, sam2_checkpoint):
    root = Path(root)
    jobs = []
    for stage, seeds, variants in (("main", SEEDS, MAIN_VARIANTS),
                                   ("sam21", SEEDS, ("A", "B", "C")),
                                   ("native80", (42,), ("A", "B", "C"))):
        for seed in seeds:
            for variant in variants:
                config = copy.deepcopy(template)
                config.update(seed=seed, epochs=5, batch_size=1, max_batches=None,
                              validate_every=1, save_every=1000, workers=0,
                              object_threshold=.5, mask_threshold=.5)
                config["model"].update(variant=variant, init_seed=seed)
                if variant == "S":
                    config["model"].update(backend="sam31_native", num_queries=200)
                if stage == "sam21":
                    config["model"].update(backend="sam21", canvas_size=1024,
                                           sam2_repo=str(sam2_repo), sam2_checkpoint=str(sam2_checkpoint))
                if stage == "native80":
                    config["model"].update(num_queries=80, decoder_query_chunk=4)
                    config["mosaic"] = {"enabled": True, "probability": .5, "tiles": 4,
                                        "crop_size": 256, "min_instances": 25, "max_instances": 64}
                ident = f"{stage}_{variant}_seed{seed}"
                jobs.append(dict(id=ident, stage=stage, variant=variant, seed=seed,
                                 config=config, config_path=str(root / "configs" / f"{ident}.json")))
    return jobs


def validate_membership(pilot, full):
    train = [r for r in pilot["records"] if r["split"] == "train"]
    val = [r for r in pilot["records"] if r["split"] == "validation"]
    test = [r for r in full["records"] if r["split"] == "test"]
    if (len(train), len(val), len(test)) != (2000, 300, 1800):
        raise ValueError("Expected the original 2000/300 pilot and fixed full-study 1800 test")
    groups = [{r["rgb_sha256"] for r in rows} for rows in (train, val, test)]
    for name, rows, hashes in zip(("train", "validation", "test"), (train,val,test), groups):
        if len({str(r["id"]) for r in rows}) != len(rows) or len(hashes) != len(rows):
            raise ValueError(f"Duplicate IDs or exact RGBs in {name}")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Pilot train/validation and fixed test RGB hashes overlap")
    return dict(train=2000, validation=300, test=1800,
                membership_sha256={name:hashlib.sha256(json.dumps(sorted((str(r["id"]),r["rgb_sha256"]) for r in rows), separators=(",", ":")).encode()).hexdigest()
                    for name,rows in (("train",train),("validation",val),("test",test))},
                pilot_grouping=pilot["grouping_kind"], independent_scene_split=False,
                test_is_new_blind=False,
                note="Fixed synthetic test previously reported for full-data B/C; exploratory confirmation, not a newly blind test")


def additional_evaluations(jobs, run_root, *, real_manifest, real_root, c135_root, handloom_root):
    root, c135_root, handloom_root = map(Path, (run_root, c135_root, handloom_root))
    items = []
    for job in jobs:
        if job["stage"] == "native80":
            for tag in ("best", "last"):
                for domain, manifest, data, count in (("wire300", Path(real_manifest), Path(real_root), 300),
                    ("c135", c135_root/"manifest.json", c135_root, 135)):
                    ident = "real_"+domain+"_"+job["id"]+"_"+tag
                    dest = root/"real"/domain/job["id"]/tag
                    items.append(dict(id=ident, phase="real_regression", kind="metrics", job=job["id"], tag=tag,
                        domain=domain, module="scalability", report=str(dest/"metrics.json"), expected_images=count,
                        inputs=[str(manifest)], arguments=["--checkpoint",str(root/"train"/job["id"]/(tag+".pt")),
                        "--manifest",str(manifest),"--data-root",str(data),"--output",str(dest),"--device","cuda"]))
        if job["stage"] == "main" and job["variant"] in ("A", "B", "C"):
            for tag in ("best", "last"):
                ident = "handloom_"+job["id"]+"_"+tag
                dest = root/"external"/job["id"]/tag
                items.append(dict(id=ident, phase="external_diagnostics", kind="external", job=job["id"], tag=tag,
                    module="external_data", report=str(dest/"diagnostics.json"), expected_images=110,
                    inputs=[str(handloom_root/"audit.json"),str(handloom_root/"source_manifest.json")],
                    arguments=["--checkpoint",str(root/"train"/job["id"]/(tag+".pt")),"--audit-root",str(handloom_root),
                               "--output",str(dest),"--device","cuda","--sample-visuals","12"]))
    return items


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", required=True)
    p.add_argument("--full-manifest", required=True)
    p.add_argument("--full-data-root", required=True)
    p.add_argument("--sam2-repo", required=True)
    p.add_argument("--sam2-checkpoint", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--real-manifest", required=True); p.add_argument("--real-data-root", required=True)
    p.add_argument("--c135-root", required=True); p.add_argument("--handloom-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    template = json.loads(Path(args.template).read_text(encoding="utf-8-sig"))
    pilot_path, test_path = Path(template["manifest"]), Path(args.full_manifest)
    membership = validate_membership(json.loads(pilot_path.read_text()), json.loads(test_path.read_text()))
    jobs = build_matrix(template, root, sam2_repo=args.sam2_repo, sam2_checkpoint=args.sam2_checkpoint)
    for job in jobs:
        Path(job["config_path"]).write_text(json.dumps(job["config"], indent=2)+"\n", encoding="utf-8")
    protocol = dict(schema=1, name="pilot_evidence_20261031", counts=membership, jobs=jobs,
                    test_manifest=str(test_path.resolve()), test_data_root=str(Path(args.full_data_root).resolve()),
                    pilot_manifest_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
                    test_manifest_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                    selection="validation instance IoU, tie-break global PQ; best and last reported",
                    no_full_training=True, no_automation_notifications=True,
                    main_runs=30, second_backbone_runs=9, capacity_feasibility_runs=3,
                    initialization="Independent original backbone initialization; never full-data or pilot warm-start",
                    native80_scope="single-seed engineering feasibility, not three-seed real generalization evidence")
    protocol["additional_evaluations"] = additional_evaluations(jobs, args.run_root,
        real_manifest=args.real_manifest, real_root=args.real_data_root, c135_root=args.c135_root, handloom_root=args.handloom_root)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"protocol": str(root / "protocol.json"), "runs": len(jobs), "counts": membership}))


if __name__ == "__main__":
    main()
