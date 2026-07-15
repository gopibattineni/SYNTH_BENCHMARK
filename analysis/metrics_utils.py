"""Shared aggregation and normalization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.config import LOWER_IS_BETTER, PipelineConfig


def confidence_interval(mean: float, std: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 1 or np.isnan(std) or std == 0:
        return mean, mean
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def aggregate_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "Value",
    std_col: str = "Std",
    n_seeds: int = 10,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            Mean=(value_col, "mean"),
            Std=(value_col, "std"),
            Count=(value_col, "count"),
        )
        .reset_index()
    )

    if std_col in df.columns:
        std_agg = df.groupby(group_cols, dropna=False)[std_col].mean().reset_index(name="SeedStd")
        agg = agg.merge(std_agg, on=group_cols, how="left")
        agg["Std"] = agg["SeedStd"].where(agg["SeedStd"].notna(), agg["Std"])
        agg = agg.drop(columns=["SeedStd"])

    ci_low, ci_high = [], []
    for _, row in agg.iterrows():
        lo, hi = confidence_interval(row["Mean"], row["Std"], n_seeds)
        ci_low.append(lo)
        ci_high.append(hi)
    agg["CI_Low"] = ci_low
    agg["CI_High"] = ci_high
    return agg


def normalize_metric(value: float, metric: str, vmin: float, vmax: float) -> float:
    if vmax == vmin or np.isnan(value):
        return np.nan
    norm = (value - vmin) / (vmax - vmin)
    if metric in LOWER_IS_BETTER or any(k in metric for k in ("Drop", "Increase", "Distance", "JS", "Wasserstein")):
        norm = 1.0 - norm
    return float(np.clip(norm, 0, 1))


def normalize_scores(df: pd.DataFrame, metric_col: str = "Metric", value_col: str = "Mean") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["NormalizedScore"] = np.nan
    for metric, group in out.groupby(metric_col):
        vals = group[value_col]
        vmin, vmax = vals.min(), vals.max()
        idx = group.index
        out.loc[idx, "NormalizedScore"] = [
            normalize_metric(v, str(metric), vmin, vmax) for v in vals
        ]
    return out


def rank_generators(
    df: pd.DataFrame,
    score_col: str = "NormalizedScore",
    group_cols: list[str] | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = group_cols or ["Metric"]
    ranked = df.copy()
    ranked["Rank"] = (
        ranked.groupby(group_cols, dropna=False)[score_col]
        .rank(ascending=ascending, method="average")
    )
    return ranked
