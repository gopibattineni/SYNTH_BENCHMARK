#!/usr/bin/env python3
"""Run Utility Drop Analysis → Results/Utility_Drop_Analysis/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.utility_drop.pipeline import run_utility_drop_analysis


def main() -> None:
    run_utility_drop_analysis()


if __name__ == "__main__":
    main()
