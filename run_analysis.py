#!/usr/bin/env python3
"""Run the SYNTH benchmark journal paper analysis pipeline.

Automatically discovers all experiment Excel files, merges fidelity/utility/privacy
metrics, computes cumulative category scores, runs statistical tests, and generates
Figures 1–14 plus supplementary analyses at 300 DPI (PNG/PDF/SVG/EPS).

Usage:
    python run_analysis.py
    python run_analysis.py --utility-weight 0.4 --privacy-weight 0.3 --fidelity-weight 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path when run directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SYNTH Benchmark — automated publication analysis pipeline",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "Results"),
        help="Output directory (default: Results/)",
    )
    parser.add_argument("--utility-weight", type=float, default=0.40)
    parser.add_argument("--privacy-weight", type=float, default=0.30)
    parser.add_argument("--fidelity-weight", type=float, default=0.30)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Also build GitHub Pages dashboard (off by default)",
    )
    args = parser.parse_args()

    total = args.utility_weight + args.privacy_weight + args.fidelity_weight
    if abs(total - 1.0) > 0.01:
        print(f"Warning: weights sum to {total:.2f}, not 1.0")

    run_pipeline(
        output_dir=args.output,
        utility_weight=args.utility_weight,
        privacy_weight=args.privacy_weight,
        fidelity_weight=args.fidelity_weight,
        build_dashboard=args.dashboard,
    )


if __name__ == "__main__":
    main()
