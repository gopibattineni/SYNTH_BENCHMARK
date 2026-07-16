#!/usr/bin/env python3
"""Run representative-metrics dual trade-off analysis (Cancer & Mushroom).

Primary: Classifier-Independent (average over 10 classifiers × seed means)
Supplementary: Best Classifier (max F1 per generator × dataset)

Usage:
    python run_representative_tradeoff.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.representative_tradeoff import run_representative_tradeoff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    run_representative_tradeoff()


if __name__ == "__main__":
    main()
