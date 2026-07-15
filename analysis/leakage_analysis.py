"""Data leakage experiment analysis."""

from __future__ import annotations

import pandas as pd

from analysis.config import LEAKAGE_LEVELS, PipelineConfig
from analysis.metrics_utils import aggregate_stats


def analyze_leakage(master_utility: pd.DataFrame, config: PipelineConfig) -> dict[str, pd.DataFrame]:
    if master_utility.empty:
        return {
            "leakage_utility": pd.DataFrame(),
            "leakage_privacy": pd.DataFrame(),
            "leakage_fidelity": pd.DataFrame(),
            "leakage_available": pd.DataFrame(),
        }

    if "Leakage" not in master_utility.columns:
        master_utility = master_utility.copy()
        master_utility["Leakage"] = 0

    leakage_levels = sorted(master_utility["Leakage"].dropna().unique())
    available = pd.DataFrame({"LeakageLevel": leakage_levels, "Count": [len(master_utility[master_utility["Leakage"] == lv]) for lv in leakage_levels]})

    if len(leakage_levels) <= 1:
        # Framework ready when per-leakage exports are added
        placeholder = pd.DataFrame(
            {
                "LeakageLevel": LEAKAGE_LEVELS,
                "Status": ["pending_export"] * len(LEAKAGE_LEVELS),
            }
        )
        return {
            "leakage_utility": pd.DataFrame(),
            "leakage_privacy": pd.DataFrame(),
            "leakage_fidelity": pd.DataFrame(),
            "leakage_available": pd.concat([available, placeholder], ignore_index=True),
        }

    leakage_utility = aggregate_stats(
        master_utility,
        ["Dataset", "Generator", "Leakage", "Metric", "EvaluationType"],
        n_seeds=config.n_seeds,
    )

    return {
        "leakage_utility": leakage_utility,
        "leakage_privacy": pd.DataFrame(),
        "leakage_fidelity": pd.DataFrame(),
        "leakage_available": available,
    }
