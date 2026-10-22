"""Run from project root through padlos_runtime_runner.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if __name__ == '__main__':
    from curvi_identity.cli import main
    main()
