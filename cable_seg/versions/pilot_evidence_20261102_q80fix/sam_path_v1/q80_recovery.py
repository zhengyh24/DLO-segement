"""Finite recovery for exactly three Q80 trainings; preserve the failed parent."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import zipfile

from .checkpoint import atomic_json, read_checkpoint, sha256, verify_engine_source
from .evidence_run import (EvidenceQueue, read, utc, verify_snapshot,
                           verify_log_membership)
from .full_run import verify_training, check_disk
from .train import TrainConfig


class Q80Recovery(EvidenceQueue):
    def expected_tasks(self):
        # Test scoring belongs to the separate, unchanged 42-model analysis
        # protocol. This recovery is authorized only to finish the three runs.
        return []

    def admit(self):
        jobs=self.protocol["jobs"]
        if [j["id"] for j in jobs] != [f"native80_{v}_seed42" for v in "ABC"]:
            raise ValueError("Recovery must contain exactly native80 A/B/C seed42")
        parent=Path(self.protocol["recovery"]["parent_runs"])
        terminal=read(parent/"completion.json")
        if terminal["status"]!="failed" or len(terminal["training"])!=39:
            raise ValueError("Parent must be the preserved 39-completed, Q80-failed attempt")
        expected={f"{stage}_{v}_seed{s}" for stage,variants in (
            ("main",["A","B","C","D","MASK_ONLY","CNN","NO_GEOMETRY","NO_CONTRAST","PARALLEL","S"]),
            ("sam21",list("ABC"))) for v in variants for s in (42,43,44)}
        if {j["id"] for j in terminal["training"]} != expected:
            raise ValueError("Unexpected completed parent jobs")
        previous_protocol=read(self.protocol["recovery"]["parent_protocol"])
        old_jobs={j["id"]:j for j in previous_protocol["jobs"]}
        for job in jobs:
            if job["config"]!=old_jobs[job["id"]]["config"]:
                raise ValueError("Recovery changed the frozen training configuration")
            if read(job["config_path"])!=read(old_jobs[job["id"]]["config_path"]):
                raise ValueError("Recovery configuration file changed")
        self.parent_snapshot={str(parent/"completion.json"):sha256(parent/"completion.json")}
        # Preserve the weights, training/validation logs, and code by content.
        for job in terminal["training"]:
            dest=parent/"train"/job["id"]
            config=asdict(TrainConfig(**read(dest/"config.json")))
            verify_training(dest, config, {"train":2000,"validation":300})
            for path in dest.iterdir():
                if path.is_file(): self.parent_snapshot[str(path)]=sha256(path)
        for path in (parent/"train"/"native80_A_seed42").iterdir():
            if path.is_file(): self.parent_snapshot[str(path)]=sha256(path)
        old_source=Path(self.protocol["recovery"]["parent_source"])
        for path in old_source.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                self.parent_snapshot[str(path)]=sha256(path)
        atomic_json(self.root/"parent_preservation.json",dict(created_utc=utc(),
            sha256=self.parent_snapshot, completed_training_preserved=39,
            failed_A_successful_updates=3306,
            restart_policy="Q80 A/B/C independently initialize from the original base; no parent continuation"))
        super().admit()
        admission=read(self.root/"admission.json")
        admission.update(main_runs=0,sam21_runs=0,native80_runs=3,
                         scope="Only three Q80 trainings; 39 parent runs preserved")
        atomic_json(self.root/"admission.json",admission)
        self.phase="exhaustive_mosaic_preflight"
        self.command("mosaic_all_10000", "q80_preflight", ["--config", jobs[0]["config_path"],
            "--output",self.root/"mosaic_preflight","--threads",4])
        data_check=read(self.root/"mosaic_preflight"/"completion.json")
        if (data_check.get("passed") is not True or data_check.get("checked")!=10000
                or data_check.get("failures")!=0 or not data_check.get("known_failed_seed_repaired")):
            raise ValueError("All-position data admission failed")
        for name in ("completion.json","recipes.jsonl"):
            path=self.root/"mosaic_preflight"/name
            self.snapshot[str(path.resolve())]=sha256(path)
        admission.update(exhaustive_mosaic_preflight=data_check,
                         exhaustive_mosaic_sha256=sha256(self.root/"mosaic_preflight"/"completion.json"))
        atomic_json(self.root/"admission.json",admission)
        atomic_json(self.root/"freeze_after_data_admission.json",dict(frozen_files=self.snapshot))

    def verify_recipes(self, dest):
        with (self.root/"mosaic_preflight"/"recipes.jsonl").open() as f:
            plans={(x["epoch_index"],x["id"]):x for x in map(json.loads,f)}
        checked=repairs=0
        with (dest/"logs.jsonl").open() as f:
            for row in map(json.loads,f):
                if row["event"]!="update": continue
                plan=plans[(row["epoch_index"],row["ids"][0])]
                expected=[] if plan["kind"]=="native" else [
                    {k:v for k,v in plan.items() if k not in ("epoch_index","index","id","kind")}]
                if row.get("augmentation_receipts")!=expected:
                    raise ValueError("Training recipe differs from exhaustive preflight")
                checked+=1; repairs+=int(bool(expected and expected[0].get("repair")))
        if checked!=10000: raise ValueError("Incomplete recipe verification")
        return dict(checked=checked,repaired_samples=repairs,exact_preflight_correspondence=True)

    def run(self):
        self.admit()
        self.train_all()
        audits=[]
        for job in self.protocol["jobs"]:
            self.job,self.phase=job["id"],"final_training_verification"
            dest=self.root/"train"/self.job
            config=asdict(TrainConfig(**read(job["config_path"])))
            verify_training(dest,config,{"train":2000,"validation":300})
            verify_log_membership(dest,config,self.train_ids,self.validation_ids)
            recipes=self.verify_recipes(dest)
            payload=read_checkpoint(dest/"last.pt"); verify_engine_source(payload)
            steps=sorted({int(v["step"]) for v in payload["optimizer"]["state"].values() if "step" in v})
            if not steps or max(steps)!=10000 or min(steps)<1 or max(steps)>10000:
                raise ValueError("Optimizer state does not confirm the cumulative budget")
            del payload
            for tag in ("best","last"):
                receipt=dest/(tag+"_reload.json")
                self.command("final_reload_"+self.job+"_"+tag,"runtime_admission",[
                    "--checkpoint",dest/(tag+".pt"),"--manifest",config["manifest"],
                    "--data-root",config["data_root"],"--device",config["device"],"--output",receipt])
                if not read(receipt).get("complete"): raise ValueError("Final checkpoint reload failed")
            files=list((self.root/"curves"/self.job).rglob("*"))
            if not all(any(p.suffix==ext for p in files) for ext in (".png",".pdf",".csv")):
                raise ValueError("Loss plots/CSV absent")
            audits.append(dict(id=self.job,successful_updates=10000,validation_epochs=5,
                optimizer_step_values=steps,recipes=recipes,best_sha256=sha256(dest/"best.pt"),
                last_sha256=sha256(dest/"last.pt")))
        verify_snapshot(self.snapshot); verify_snapshot(self.parent_snapshot)
        self.phase,self.job="completed",None
        atomic_json(self.root/"completion.json",dict(status="completed",finished_utc=utc(),
            scope="Three Q80 training runs and loss curves; final test evaluation has not run",
            training=self.finished,audits=audits,parent_39_unchanged=True,
            parent_failure_preserved=True,exhaustive_mosaic_preflight_passed=True,
            initialization="three fresh runs from identical base and seed42; failed parent A retained",
            tests_executed=False))
        self.event(training_verified=True,parent_39_unchanged=True)
        archive=self.root/"q80_training_results.zip"
        hashes={}
        with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for path in sorted(self.root.rglob("*")):
                if not path.is_file() or path.suffix in (".pt",".zip") or "smoke" in path.relative_to(self.root).parts:
                    continue
                rel=path.relative_to(self.root).as_posix()
                hashes[rel]=sha256(path); z.write(path,rel)
            z.writestr("archive_member_sha256.json",json.dumps(hashes,indent=2))
        atomic_json(self.root/"package_receipt.json",dict(archive=str(archive),bytes=archive.stat().st_size,
                    sha256=sha256(archive),members=hashes,weights_retained_on_server=True))


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","output","lock"): p.add_argument("--"+name,required=True)
    args=p.parse_args()
    lock=Path(args.lock).open("a+")
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=Path(args.output); check_disk(root.parent,Path(__file__).parent)
    root.mkdir(exist_ok=False)
    for name in ("stdout","smoke_configs","smoke","train","curves"):
        (root/name).mkdir()
    protocol=read(args.protocol); atomic_json(root/"protocol.json",protocol)
    queue=Q80Recovery(protocol,root,lock.fileno(),5,args.protocol)
    try:
        queue.run()
    except Exception as exc:
        queue.phase="failed"; queue.event(error=f"{type(exc).__name__}: {exc}")
        atomic_json(root/"completion.json",dict(status="failed",error=f"{type(exc).__name__}: {exc}",
            finished_utc=utc(),training=queue.finished,retried=False,scope="Q80 recovery only"))
        raise
    finally:
        lock.close()


if __name__=="__main__":
    main()
