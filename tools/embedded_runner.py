"""Launch a project script with the local embeddable SAM3.1 Python runtime."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: embedded_runner.py SCRIPT [ARGS...]")
    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
