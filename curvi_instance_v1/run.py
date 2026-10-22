"""Entry point for Windows embedded Python, whose sys.path omits the cwd.

Normal Python also supports `python -m curvi_instance_v1.train` etc.
"""
import runpy
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "train", "predict", "check", "learnability"):
    raise SystemExit("Usage: python curvi_instance_v1/run.py {prepare|train|predict|check|learnability} [arguments]")
command = sys.argv.pop(1)
runpy.run_module("curvi_instance_v1."+command, run_name="__main__")
