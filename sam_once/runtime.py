"""Explicit local dependency bootstrap; no environment or source changes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def bootstrap():
    for path in (ROOT, ROOT / "tools", ROOT / "third_party/padlos_deps"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # Keep this package ahead of tools/sam_once.py in import resolution.
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
