"""Pytest configuration: make src/ importable from any test.

This file lets `from src.recommender import ...` work both locally
and in CI (GitHub Actions runs from the repo root on Linux).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Allow `from src.recommender import ...` (used by tests)
sys.path.insert(0, str(ROOT))

# Allow `from recommender import ...` (used by src/main.py and src/api.py)
sys.path.insert(0, str(ROOT / "src"))