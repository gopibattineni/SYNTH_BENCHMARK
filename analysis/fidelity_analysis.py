"""Fidelity metric analysis."""

from __future__ import annotations

import pandas as pd

from analysis.config import FIDELITY_METRICS, PipelineConfig
from analysis.metrics_utils import aggregate_stats, normalize_scores, rank_generators


def analyze_fidelity(master_fidelity: pd.DataFrame, config: PipelineConfig) -> dict[str, pd.DataFrame]:
    if master_fidelity.empty:
        return {
            "fidelity_stats": pd.DataFrame(),
            "fidelity_ranks": pd.DataFrame(),
            "overall_fidelity": pd.DataFrame(),
        }

    stats = aggregate_stats(
        master_fidelity,
        ["Dataset", "Generator", "Metric"],
        n_seeds=config.n_seeds,
    )
    stats = normalize_scores(stats)
    ranks = rank_generators(stats, group_cols=["Dataset", "Metric"])

    overall = (
        stats.groupby("Generator", dropna=False)["NormalizedScore"]
        .agg(Mean="mean", Std="std")
        .reset_index()
        .sort_values("Mean", ascending=False)
    )
    overall["OverallRank"] = overall["Mean"].rank(ascending=False, method="average")

    return {
        "fidelity_stats": stats,
        "fidelity_ranks": ranks,
        "overall_fidelity": overall,
    }
