#!/usr/bin/env python3
"""Run the Cancer & Mushroom two-dataset case study assessment.

Usage:
    python run_two_datasets_assessment.py
    python run_two_datasets_assessment.py --utility-weight 0.4 --privacy-weight 0.3 --fidelity-weight 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.two_datasets import run_two_dataset_assessment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cancer & Mushroom two-dataset case study — publication figures and tables",
    )
    parser.add_argument("--utility-weight", type=float, default=0.40)
    parser.add_argument("--privacy-weight", type=float, default=0.30)
    parser.add_argument("--fidelity-weight", type=float, default=0.30)
    args = parser.parse_args()

    run_two_dataset_assessment(
        utility_weight=args.utility_weight,
        privacy_weight=args.privacy_weight,
        fidelity_weight=args.fidelity_weight,
    )


if __name__ == "__main__":
    main()
