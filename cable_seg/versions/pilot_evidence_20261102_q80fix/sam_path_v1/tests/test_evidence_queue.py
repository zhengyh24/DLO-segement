"""Evidence queue integrity tests; no real models, data or network calls."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sam_path_v1.checkpoint import atomic_json, sha256
from sam_path_v1.evidence_run import (EvidenceQueue, exact_members, expected_calls, verify_snapshot,
                                    verify_test_report, verify_smoke, verify_log_membership)


class EvidenceQueueContracts(unittest.TestCase):
    def test_member_check_rejects_same_count_duplicate_or_wrong_member(self):
        exact_members([{"id":"a"},{"id":"b"}], {"a","b"}, "test")
        for rows in ([{"id":"a"},{"id":"a"}], [{"id":"a"},{"id":"c"}]):
            with self.assertRaisesRegex(ValueError, "IDs"):
                exact_members(rows, {"a","b"}, "test")

    def test_forward_counts_distinguish_chunking_from_native_queries(self):
        self.assertEqual(expected_calls({"model":{"num_queries":80,"decoder_query_chunk":4,"backend":"sam31"}}), {"encoder":1,"decoder":20})
        self.assertEqual(expected_calls({"model":{"num_queries":200,"decoder_query_chunk":4,"backend":"sam31_native"}}), {"encoder":1,"decoder":1})

    def test_freeze_includes_original_protocol_external_inputs_and_official_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); repo=root/"official"; repo.mkdir()
            paths={name:root/name for name in ("protocol.json","original.json","manifest.json","test.json","weights.pt","config.json","annotations.json")}
            for path in paths.values(): path.write_text("{}")
            (repo/"encoder.py").write_text("frozen=True")
            config={"manifest":str(paths["manifest.json"]),"model":{"backend":"sam31","sam_repo":str(repo),"sam_checkpoint":str(paths["weights.pt"])}}
            job={"id":"main_B_seed42","stage":"main","variant":"B","seed":42,"config_path":str(paths["config.json"])}
            protocol={"jobs":[job],"test_manifest":str(paths["test.json"]),"additional_evaluations":[
                {"id":"external","phase":"external","report":"external/report.json","arguments":[],"inputs":[str(paths["annotations.json"])]}]}
            q=EvidenceQueue(protocol,root,protocol_path=paths["original.json"])
            with patch("sam_path_v1.evidence_run.source_snapshot",return_value={}): q.freeze([config])
            for path in (paths["original.json"],paths["annotations.json"],repo/"encoder.py",paths["weights.pt"]):
                self.assertIn(str(path.resolve()),q.snapshot)
            verify_snapshot(q.snapshot)
            (repo/"encoder.py").write_text("frozen=oops")
            with self.assertRaisesRegex(RuntimeError,"Frozen file changed"): verify_snapshot(q.snapshot)

    def test_test_provenance_progress_and_unreadable_gt(self):
        config={"epochs":5,"object_threshold":.5,"mask_threshold":.5}
        report=dict(images=2,per_image=[{"id":"a","n_gt":1},{"id":"b","n_gt":2}],
                    manifest_sha256="m",checkpoint_sha256="c",complete=True,
                    checkpoint_progress={"epoch":5,"successful_updates":10000,"amp_skips":0},
                    object_threshold=.5,mask_threshold=.5)
        self.assertTrue(verify_test_report(report,{"a","b"},"m","c",config,"last"))
        report["per_image"][1]["n_gt"]=None
        self.assertFalse(verify_test_report(report,{"a","b"},"m","c",config,"last"))
        report["checkpoint_sha256"]="changed"
        with self.assertRaisesRegex(ValueError,"provenance"): verify_test_report(report,{"a","b"},"m","c",config,"last")

    def test_allowed_exit_two_and_missing_tasks_never_count_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            protocol={"jobs":[],"additional_evaluations":[
                {"id":"one","phase":"real","report":"one.json","expected_images":2},
                {"id":"two","phase":"external","report":"two.json","expected_images":3}]}
            q=EvidenceQueue(protocol,root)
            atomic_json(root/"one.json",dict(images=2,complete=True,failures=0))
            q.record_task("one",2)
            receipt=q.write_task_receipts()
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["missing_expected"],["two"])
            q.record_task("two",0)
            receipt=q.write_task_receipts()
            self.assertFalse(receipt["complete"])
            self.assertTrue(receipt["tasks"][1]["missing"])

    def test_logs_reject_invented_final_counter_and_duplicate_epoch_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config={"epochs":1,"model":{"backend":"sam31","num_queries":24}}
            rows=[]
            for i,ident in enumerate(("a","b"),1):
                rows.append(dict(event="update",epoch_index=0,successful_updates=i,opportunities=i,
                                 amp_skips=0,amp_skipped=False,calls={"encoder":1,"decoder":1},ids=[ident],loss={"total":.1}))
            rows.extend([dict(event="epoch_train",epoch=1),dict(event="validation",epoch=1)])
            atomic_json(root/"validation_epoch_0001.json",dict(per_image=[{"id":"v"}]))
            path=root/"logs.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            verify_log_membership(root,config,{"a","b"},{"v"})
            rows[1]["ids"]=["a"]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError,"evidence mismatch"):
                verify_log_membership(root,config,{"a","b"},{"v"})

    def test_smoke_must_have_two_finite_actual_update_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/"source.py").write_text("source");(root/"weight.pt").write_text("weight")
            config={"model":{"backend":"sam31","num_queries":24,"sam_checkpoint":str(root/"weight.pt"),"init_seed":42},"seed":42}
            progress={"successful_updates":2,"opportunities":2,"amp_skips":0,"samples":2}
            saved={"train_config":{**config,"max_batches":2,"save_every":1},"progress":progress,
                   "base_receipt":{"checkpoint_sha256":sha256(root/"weight.pt"),"strict_retained_state":True,
                                   "repo":str(root),"source_sha256":{"source.py":sha256(root/"source.py")}}}
            atomic_json(root/"completion.json",dict(status="budget_stopped",progress=progress))
            rows=[dict(event="update",successful_updates=i,opportunities=i,amp_skipped=False,
                       calls={"encoder":1,"decoder":1},loss={"total":.1}) for i in (1,2)]
            (root/"logs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with patch("sam_path_v1.evidence_run.read_checkpoint",return_value=saved), patch("sam_path_v1.evidence_run.verify_engine_source"):
                self.assertIs(verify_smoke(root,config),saved)
                rows[1]["calls"]["encoder"]=2
                (root/"logs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                with self.assertRaisesRegex(ValueError,"configured call count"): verify_smoke(root,config)


if __name__ == "__main__": unittest.main()
