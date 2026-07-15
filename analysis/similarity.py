"""Statistical similarity analysis from available distance/matching data."""

from __future__ import annotations

import pandas as pd


def analyze_similarity(privacy_detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if privacy_detail.empty:
        return {
            "mahalanobis_by_generator": pd.DataFrame(),
            "mahalanobis_summary": pd.DataFrame(),
        }

    by_gen = privacy_detail.groupby(["Dataset", "Generator"], dropna=False)["Mahalanobis_Distance"].apply(list).reset_index()
    summary = (
        privacy_detail.groupby(["Dataset", "Generator"], dropna=False)["Mahalanobis_Distance"]
        .describe()
        .reset_index()
    )
    return {
        "mahalanobis_by_generator": by_gen,
        "mahalanobis_summary": summary,
    }
