"""Correlation analysis across metrics."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def metric_correlation_matrix(
    summary_df: pd.DataFrame,
    index_cols: list[str] | None = None,
    value_col: str = "Mean",
) -> dict[str, pd.DataFrame]:
    if summary_df.empty:
        return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame(), "kendall": pd.DataFrame()}

    index_cols = index_cols or ["Dataset", "Generator"]
    wide = summary_df.pivot_table(
        index=index_cols,
        columns="Metric",
        values=value_col,
        aggfunc="mean",
    )
    wide = wide.dropna(axis=1, how="all").fillna(wide.mean())

    if wide.shape[1] < 2:
        return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame(), "kendall": pd.DataFrame()}

    return {
        "pearson": wide.corr(method="pearson"),
        "spearman": wide.corr(method="spearman"),
        "kendall": wide.corr(method="kendall"),
    }


def hierarchical_cluster_order(corr: pd.DataFrame) -> list[str]:
    if corr.empty:
        return []
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        dist = 1 - corr.abs()
        link = linkage(squareform(dist.values, checks=False), method="average")
        order = leaves_list(link)
        return list(corr.columns[order])
    except Exception:
        return list(corr.columns)
