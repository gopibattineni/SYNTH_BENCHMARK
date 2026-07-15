"""Privacy metric analysis."""

from __future__ import annotations

import pandas as pd

from analysis.config import PipelineConfig
from analysis.metrics_utils import aggregate_stats, normalize_scores, rank_generators


def analyze_privacy(
    master_privacy: pd.DataFrame,
    privacy_detail: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    if master_privacy.empty and privacy_detail.empty:
        return {
            "privacy_stats": pd.DataFrame(),
            "privacy_ranks": pd.DataFrame(),
            "overall_privacy": pd.DataFrame(),
            "mahalanobis_distribution": pd.DataFrame(),
        }

    stats = aggregate_stats(
        master_privacy,
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

    mahalanobis = pd.DataFrame()
    if not privacy_detail.empty:
        mahalanobis = (
            privacy_detail.groupby(["Dataset", "Generator"], dropna=False)["Mahalanobis_Distance"]
            .agg(Mean="mean", Std="std", Median="median", Count="count")
            .reset_index()
        )

    return {
        "privacy_stats": stats,
        "privacy_ranks": ranks,
        "overall_privacy": overall,
        "mahalanobis_distribution": mahalanobis,
    }
