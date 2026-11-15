"""Render the three v7 cabinet configurations, validate, and export RGB-D."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "pipeline"
SEED = HERE / "seed"
CASES = {"easy1": (3,), "medium2": (2, 3), "easy3": (1, 2, 3)}


def run_stage(command, log):
    print(f"[{log.stem}]", flush=True)
    with log.open("wb") as stream:
        result = subprocess.run([str(arg) for arg in command], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-24:])
        raise SystemExit(f"{log.stem} failed ({result.returncode}). Log: {log}\n{tail}")


def verify_seed():
    provenance = json.loads((SEED / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["seed_sha256"].items():
        if hashlib.sha256((SEED / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Seed checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender", help="Path to Blender 4.2, or blender on PATH")
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU", help="GPU uses NVIDIA CUDA")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True, help="Directory for this new run")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not (blender := shutil.which(args.blender)):
        parser.error("Blender executable not found")
    verify_seed()
    root = args.output.resolve()
    if root.exists():
        parser.error(f"Output directory already exists: {root}")
    logs = root / "logs"
    logs.mkdir(parents=True)
    (root / "receipts").mkdir()

    def blender_stage(stage, script, *options):
        run_stage([blender, "-b", "--factory-startup", "-t", args.threads, "--python-exit-code", 1,
                   "--python", SCRIPTS / script, "--", *options], logs / f"{stage}.log")

    def python_stage(script, *options):
        run_stage([sys.executable, SCRIPTS / script, *options], logs / f"{Path(script).stem}.log")

    physics_input = ("--contract", SEED / "input_contract.json", "--solution", SEED / "solved_rods.npz",
                     "--solution-receipt", SEED / "solve_receipt.json")
    parent_root = root / "parent"
    blender_stage("parent_scene", "build_parent_scene.py", *physics_input, "--output", parent_root)
    parent = parent_root / "easy3"
    cameras = json.loads((SEED / "cameras.json").read_text(encoding="utf-8"))
    for view, camera in cameras.items():
        directory = parent / view
        directory.mkdir()
        (directory / "camera.json").write_text(json.dumps(camera, indent=2) + "\n", encoding="utf-8")

    for scene, retained in CASES.items():
        blender_stage(f"render_{scene}", "render_close_physics.py", *physics_input, "--source-scene",
                      parent, "--keep-assemblies", *retained, "--output", root / "data",
                      "--scene-id", scene, "--size", 1024, "--samples", 256, "--views", 4,
                      "--device", args.device)
        for audit in ("cabinet_revision", "close_data", "scene_contacts", "support_connections"):
            name = f"qa_{scene}_{audit}"
            options = ("--old", parent) if audit == "cabinet_revision" else ()
            blender_stage(name, f"qa_{audit}.py", "--scene", root / "data" / scene,
                          "--out", root / "receipts" / f"{name}.json", *options)

    python_stage("build_sparse_manifest.py", "--run-root", root)
    python_stage("finalize_v7.py", "--run-root", root)
    python_stage("build_cabinet_report.py", "--data", root / "data", "--output", root / "delivery" / "final",
                 "--qa", root / "receipts" / "final_qa.json")
    python_stage("export_dataset_archive.py", "--run-root", root)
    print(f"Dataset: {root / 'delivery' / 'panel_harness_rgbd_v7.zip'}")
    print(f"Viewer:  {root / 'delivery' / 'final' / 'index.html'}")


if __name__ == "__main__":
    main()
