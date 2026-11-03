"""Portable CLI, also works with Windows embedded Python that omits cwd."""
from pathlib import Path
import runpy
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
COMMANDS = ("train", "evaluate", "predict", "prepare", "check", "package", "configure")
if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    raise SystemExit("Usage: python sam_path_v1/run.py {"+"|".join(COMMANDS)+"} [arguments]")
command = sys.argv.pop(1)
runpy.run_module("sam_path_v1." + command, run_name="__main__")
