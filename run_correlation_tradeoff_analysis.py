#!/usr/bin/env python3
"""Run Correlation Trade-off Analysis → Results/Correlation_Tradeoff_Analysis/.

Also mirrors outputs to Results/Two_Datasets_Assessment/Correlation_Tradeoff_Analysis/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.correlation_tradeoff.pipeline import run_correlation_tradeoff_analysis


def main() -> None:
    run_correlation_tradeoff_analysis()


if __name__ == "__main__":
    main()
