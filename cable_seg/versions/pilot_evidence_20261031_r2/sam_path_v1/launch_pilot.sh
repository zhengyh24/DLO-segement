#!/usr/bin/env bash
# Run from the project root, in the foreground (tmux may own that foreground).
# Usage: bash sam_path_v1/launch_pilot.sh CONFIG_DIR NEW_OUTPUT_ROOT SMOKE_RUN_DIR [REALCHECK_JSON]
# PYTHON_BIN may name one Python executable; it is never evaluated as shell code.
# Requires the server's matching, successful TWO-update C smoke. No skip switch.
set -Eeuo pipefail
umask 027

if (( $# < 3 || $# > 4 )); then
    printf '%s\n' "Usage: $0 CONFIG_DIR NEW_OUTPUT_ROOT SMOKE_RUN_DIR [REALCHECK_JSON]" >&2
    exit 64
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || { printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2; exit 69; }
command -v flock >/dev/null || { printf '%s\n' 'flock is required; refusing an unlocked pilot' >&2; exit 69; }
command -v tee >/dev/null || exit 69
PROJECT_ROOT="$(pwd -P)"
[[ -f "$PROJECT_ROOT/sam_path_v1/run.py" ]] || { printf '%s\n' 'Run from the project root containing sam_path_v1/run.py' >&2; exit 64; }
resolve_path() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}
CONFIG_DIR="$(resolve_path "$1")"
OUTPUT_ROOT="$(resolve_path "$2")"
SMOKE_ROOT="$(resolve_path "$3")"
REALCHECK=""
if (( $# == 4 )); then REALCHECK="$(resolve_path "$4")"; fi
[[ -d "$CONFIG_DIR" ]] || { printf 'Config directory not found: %s\n' "$CONFIG_DIR" >&2; exit 66; }
[[ -d "$SMOKE_ROOT" ]] || { printf 'Smoke run not found: %s\n' "$SMOKE_ROOT" >&2; exit 66; }
[[ ! -e "$OUTPUT_ROOT" ]] || { printf 'Output already exists; refusing overwrite: %s\n' "$OUTPUT_ROOT" >&2; exit 73; }

# One lock for every pilot launched from this project, including different run roots.
# Do not unlink a flock file: replacing its inode would permit a second launcher.
LOCK_FILE="$PROJECT_ROOT/.sam_path_v1.pilot.lock"
exec 9>>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'Another pilot holds %s; exiting without starting a model\n' "$LOCK_FILE" >&2
    exit 75
fi
printf 'pid=%s\nproject=%s\noutput=%s\nstarted_utc=%s\n' "$$" "$PROJECT_ROOT" "$OUTPUT_ROOT" "$(date -u +%FT%TZ)" > "$LOCK_FILE"
mkdir -p -- "$(dirname -- "$OUTPUT_ROOT")"
mkdir -- "$OUTPUT_ROOT"
printf '%s\n' "$$" > "$OUTPUT_ROOT/launcher.pid"
CURRENT_VARIANT=""
PHASE="preflight"

log() {
    printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUTPUT_ROOT/launcher.log"
}

write_status() {
    "$PYTHON_BIN" - "$OUTPUT_ROOT" "$1" "$2" "$CURRENT_VARIANT" "$$" "$CONFIG_DIR" "$SMOKE_ROOT" <<'PY'
import datetime, json, os, pathlib, sys
root, phase, code, arm, pid, configs, smoke = sys.argv[1:]
root = pathlib.Path(root)
completed = [v for v in "ABCD" if (root / v / "pilot_arm_summary.json").is_file()]
child_file = root / (arm + ".pid") if arm else None
child_pid = int(child_file.read_text().strip()) if child_file and child_file.is_file() else None
obj = dict(status=phase, exit_code=int(code), variant=arm or None, launcher_pid=int(pid),
           child_pid=child_pid, completed_variants=completed, config_dir=configs, smoke_run=smoke,
           updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           progress_log=str(root / arm / "logs.jsonl") if arm else None)
temporary = root / ".status.json.tmp"
temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, root / "status.json")
with (root / "launcher_events.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(obj) + "\n")
PY
}

on_exit() {
    code=$?
    trap - EXIT INT TERM
    set +e
    if (( code != 0 )); then PHASE="failed"; fi
    write_status "$PHASE" "$code"
    if (( code != 0 )); then
        "$PYTHON_BIN" - "$OUTPUT_ROOT" "$code" "$CURRENT_VARIANT" <<'PY'
import datetime, json, pathlib, sys
root, code, current = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
summaries = {}
for arm in "ABCD":
    path = root / arm / "pilot_arm_summary.json"
    if path.is_file():
        summaries[arm] = json.loads(path.read_text(encoding="utf-8"))
destination = root / "pilot_completion.json"
if not destination.exists():
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(dict(status="failed", exit_code=code, failed_variant=current or None,
                       completed_variants=list(summaries), variants=summaries,
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), stream, indent=2)
PY
    fi
    log "Pilot exit=$code status=$PHASE variant=${CURRENT_VARIANT:-none}; no other process was terminated"
    exit "$code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_disk() {
    "$PYTHON_BIN" - "$PROJECT_ROOT" "$OUTPUT_ROOT" <<'PY'
import shutil, sys
minimum = 3 * 1024**3
for path in dict.fromkeys(sys.argv[1:]):
    free = shutil.disk_usage(path).free
    print(f"Disk admission: {path}: {free / 1024**3:.3f} GiB free", flush=True)
    if free < minimum:
        raise SystemExit(f"Refusing training: {path} has less than 3 GiB free")
PY
}

validate_config() {
    "$PYTHON_BIN" - "$CONFIG_DIR" "$SMOKE_ROOT" "$REALCHECK" "$OUTPUT_ROOT" <<'PY'
from dataclasses import asdict
import json, math, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
from sam_path_v1.checkpoint import read_checkpoint, seed_everything, sha256, verify_engine_source
from sam_path_v1.data import NativeInstanceDataset
from sam_path_v1.train import TrainConfig, _runtime_receipt

configs_root, smoke_root, realcheck_path, out = map(pathlib.Path, sys.argv[1:])
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
configs = {}
for arm in "ABCD":
    config = asdict(TrainConfig(**read(configs_root / f"{arm}.json")))
    if config["model"]["variant"] != arm or config["model"]["backend"] != "sam31":
        raise ValueError(f"{arm}.json must be its named SAM3.1 variant")
    if not config["device"].startswith("cuda") or config["max_batches"] is not None:
        raise ValueError("Pilot requires CUDA and full configured epochs, without a smoke max_batches limit")
    configs[arm] = config
reference = configs["C"]
def paired(value, smoke=False):
    value = json.loads(json.dumps(value))
    value["model"].pop("variant")
    if smoke:
        value.pop("max_batches", None); value.pop("save_every", None)
    return value
if any(paired(config) != paired(reference) for config in configs.values()):
    raise ValueError("A/B/C/D must differ only in model.variant")
for split in ("train", "validation"):
    NativeInstanceDataset(reference["manifest"], split, reference["model"]["canvas_size"],
                          reference["seed"], reference["allow_condition"], reference["data_root"])
completion = read(smoke_root / "completion.json")
saved = read_checkpoint(smoke_root / "last.pt")
verify_engine_source(saved)
smoke_config = asdict(TrainConfig(**saved["train_config"]))
if smoke_config["model"]["variant"] != "C" or smoke_config["max_batches"] != 2:
    raise ValueError("Admission requires the C configuration's explicit two-update smoke")
if paired(smoke_config, True) != paired(reference, True):
    raise ValueError("Smoke training configuration differs from C (except max_batches/save_every)")
if read(smoke_root / "config.json") != saved["train_config"]:
    raise ValueError("Smoke config.json differs from its actual checkpoint")
progress = saved["progress"]
if completion["status"] not in ("budget_stopped", "completed"):
    raise ValueError("Smoke did not complete its bounded run")
for key, required in (("opportunities", 2), ("successful_updates", 2), ("amp_skips", 0)):
    if progress[key] != required or completion["progress"][key] != required:
        raise ValueError(f"Smoke {key} must equal {required}")
steps = [float(state["step"]) for state in saved["optimizer"]["state"].values() if "step" in state]
if not steps or any(step != 2 for step in steps):
    raise ValueError("Smoke AdamW state does not confirm two successful updates")
updates = [json.loads(line) for line in (smoke_root / "logs.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
updates = [event for event in updates if event.get("event") == "update"]
if len(updates) != 2 or any(event["amp_skipped"] or event["calls"] != {"encoder": 1, "decoder": 1}
                          or not all(math.isfinite(value) for value in event["loss"].values()) for event in updates):
    raise ValueError("Smoke update logs are incomplete, nonfinite, skipped, or violate one-pass inference")
seed_everything(reference["seed"])
if saved["provenance"]["runtime"] != _runtime_receipt(reference["device"]):
    raise ValueError("Run the smoke on this server with the same visible devices/runtime before launching")
if saved["provenance"]["manifest_sha256"] != sha256(reference["manifest"]):
    raise ValueError("Smoke manifest hash differs from pilot data")
base = saved["base_receipt"]
if base["checkpoint_sha256"] != sha256(reference["model"]["sam_checkpoint"]):
    raise ValueError("Frozen SAM checkpoint changed after smoke")
official_files = {"sam3/model/vitdet.py", "sam3/model/necks.py", "sam3/sam/mask_decoder.py",
                  "sam3/sam/prompt_encoder.py", "sam3/sam/transformer.py"}
if set(base["source_sha256"]) != official_files:
    raise ValueError("Incomplete official SAM source receipt")
for name in official_files:
    if sha256(pathlib.Path(reference["model"]["sam_repo"]) / name) != base["source_sha256"][name]:
        raise ValueError(f"Official SAM source changed after smoke: {name}")
real_receipt = None
if sys.argv[3]:
    real_receipt = read(realcheck_path)
    if (real_receipt.get("calls") != {"encoder": 1, "decoder": 1}
            or real_receipt.get("frozen_clean") is not True
            or any(not real_receipt.get("gradient_checks", {}).get(name, {}).get(key, False)
                   for name in ("condition", "tracker", "descriptor") for key in ("finite", "nonzero"))
            or real_receipt.get("output_shape", [0, 0])[1] != reference["model"]["num_queries"]
            or real_receipt.get("base_receipt", {}).get("checkpoint_sha256") != base["checkpoint_sha256"]
            or real_receipt.get("base_receipt", {}).get("canvas_size") != reference["model"]["canvas_size"]):
        raise ValueError("Optional real SAM check failed gradient/query/canvas/base admission")
    if not real_receipt.get("loss") or not all(math.isfinite(value) for value in real_receipt["loss"].values()):
        raise ValueError("Optional real SAM check has nonfinite or missing losses")
admission = dict(passed=True, smoke=str(smoke_root), smoke_checkpoint_sha256=sha256(smoke_root / "last.pt"),
                 smoke_successful_updates=2, configs_sha256={arm: sha256(configs_root / f"{arm}.json") for arm in "ABCD"},
                 source_sha256=saved["provenance"]["engine_source_sha256"],
                 base_receipt=base,
                 manifest_sha256=saved["provenance"]["manifest_sha256"], runtime=saved["provenance"]["runtime"],
                 realcheck=str(realcheck_path) if real_receipt else None)
with (out / "smoke_admission.json").open("x", encoding="utf-8") as stream:
    json.dump(admission, stream, indent=2)
print("Admission passed: paired A/B/C/D configs and matching two-successful-update C smoke", flush=True)
PY
}

write_status "$PHASE" 0
log "Launcher pid=$$; project=$PROJECT_ROOT; output=$OUTPUT_ROOT"
check_disk 2>&1 | tee -a "$OUTPUT_ROOT/launcher.log"
validate_config 2>&1 | tee -a "$OUTPUT_ROOT/launcher.log"

for CURRENT_VARIANT in A B C D; do
    PHASE="running"
    check_disk 2>&1 | tee -a "$OUTPUT_ROOT/launcher.log"
    [[ ! -e "$OUTPUT_ROOT/$CURRENT_VARIANT" ]] || { log "Arm output already exists; refusing overwrite"; exit 73; }
    write_status "$PHASE" 0
    log "Starting $CURRENT_VARIANT in foreground; model PID will be recorded in $CURRENT_VARIANT.pid"
    "$PYTHON_BIN" -u - "$CONFIG_DIR/$CURRENT_VARIANT.json" "$OUTPUT_ROOT/$CURRENT_VARIANT" \
        "$OUTPUT_ROOT/$CURRENT_VARIANT.pid" "$OUTPUT_ROOT/smoke_admission.json" "$CURRENT_VARIANT" <<'PY' 2>&1 | tee "$OUTPUT_ROOT/$CURRENT_VARIANT.stdout.log"
import json, os, pathlib, runpy, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
from sam_path_v1.checkpoint import sha256, verify_engine_source
config, output, pid_path, admission_path, arm = sys.argv[1:]
admission = json.loads(pathlib.Path(admission_path).read_text(encoding="utf-8"))
if sha256(config) != admission["configs_sha256"][arm]:
    raise ValueError("Pilot config changed after admission")
verify_engine_source({"provenance": {"engine_source_sha256": admission["source_sha256"]}})
training_config = json.loads(pathlib.Path(config).read_text(encoding="utf-8"))
if sha256(training_config["manifest"]) != admission["manifest_sha256"]:
    raise ValueError("Pilot manifest changed after admission")
base = admission["base_receipt"]
if sha256(training_config["model"]["sam_checkpoint"]) != base["checkpoint_sha256"]:
    raise ValueError("Base SAM weights changed after admission")
for name, expected in base["source_sha256"].items():
    if sha256(pathlib.Path(training_config["model"]["sam_repo"]) / name) != expected:
        raise ValueError(f"Official SAM source changed after admission: {name}")
with pathlib.Path(pid_path).open("x", encoding="utf-8") as stream:
    stream.write(str(os.getpid()) + "\n")
status_path = pathlib.Path(admission_path).parent / "status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
status["child_pid"] = os.getpid()
temporary = status_path.with_name(".child_status.json.tmp")
temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, status_path)
print(f"Training variant={arm} pid={os.getpid()} config={config} output={output}", flush=True)
sys.argv = ["sam_path_v1.train", "--config", config, "--output", output]
runpy.run_module("sam_path_v1.train", run_name="__main__")
PY
    "$PYTHON_BIN" - "$OUTPUT_ROOT/$CURRENT_VARIANT" "$CONFIG_DIR/$CURRENT_VARIANT.json" <<'PY' 2>&1 | tee -a "$OUTPUT_ROOT/launcher.log"
import json, pathlib, sys
root, config_path = map(pathlib.Path, sys.argv[1:])
def read(path): return json.loads(path.read_text(encoding="utf-8"))
config, completion = read(config_path), read(root / "completion.json")
progress = completion["progress"]
if completion["status"] != "completed" or progress["epoch"] != config["epochs"] or progress["batch_cursor"] != 0:
    raise ValueError("Arm did not complete its configured epochs; subsequent arms are blocked")
if not all((root / name).is_file() for name in ("best.pt", "last.pt")):
    raise ValueError("Arm is missing best or last checkpoint")
validation = read(root / f"validation_epoch_{progress['epoch']:04d}.json")
if not validation["complete"] or validation["failures"]:
    raise ValueError("Final validation is incomplete or contains failures; inspect its retained per-image report")
summary = dict(variant=config["model"]["variant"], completion=completion,
               last_validation={key: value for key, value in validation.items() if key != "per_image"},
               best_checkpoint=str(root / "best.pt"), last_checkpoint=str(root / "last.pt"),
               best_instance_iou=progress["best_instance_iou"], best_pq=progress["best_pq"],
               best_epoch=progress["best_epoch"], selection="training validation only; no repeated evaluation")
with (root / "pilot_arm_summary.json").open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
print(f"Completed {summary['variant']}: {progress['successful_updates']} successful AdamW updates", flush=True)
PY
    PHASE="between_arms"
    write_status "$PHASE" 0
done

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import datetime, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
arms = {arm: json.loads((root / arm / "pilot_arm_summary.json").read_text(encoding="utf-8")) for arm in "ABCD"}
with (root / "pilot_completion.json").open("x", encoding="utf-8") as stream:
    json.dump(dict(status="completed", exit_code=0, order=list("ABCD"), variants=arms,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), stream, indent=2)
PY
CURRENT_VARIANT=""
PHASE="completed"
log "All four variants finished; summary: $OUTPUT_ROOT/pilot_completion.json"
