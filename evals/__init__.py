"""Evals run as `python -m evals.<name>` from the repo root; put `src/` on the path like pytest does."""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
