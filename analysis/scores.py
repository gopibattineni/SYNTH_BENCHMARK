"""Cumulative Utility, Fidelity, Privacy, and Overall scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import (
    CLASSIFICATION_METRICS,
    LOWER_IS_BETTER,
    REGRESSION_METRICS,
    PipelineConfig,
)
from analysis.data_loader import normalize_dataset_name
from analysis.metrics_utils import confidence_interval, normalize_metric


def _normalize_group(values: pd.Series, metric: str) -> pd.Series:
    vmin, vmax = values.min(), values.max()
    return values.apply(lambda v: normalize_metric(v, metric, vmin, vmax))


def compute_utility_scores(
    utility_long: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Normalized Utility Score (0–1) per Dataset × Generator × LeakageLevel."""
    if utility_long.empty:
        return pd.DataFrame()

    df = utility_long.copy()
    if "LeakageLevel" not in df.columns:
        df["LeakageLevel"] = df.get("Leakage", 0)
    df["Dataset"] = df["Dataset"].map(normalize_dataset_name)

    tstr = df[df["EvaluationType"] == "TSTR"].copy()
    rows: list[dict] = []

    for (dataset, generator, leakage, task), grp in tstr.groupby(
        ["Dataset", "Generator", "LeakageLevel", "TaskType"], dropna=False
    ):
        metric_scores: list[float] = []
        for metric, mgrp in grp.groupby("Metric"):
            metric_str = str(metric)
            if task == "classification" and metric_str.replace("-", "") not in [
                m.replace("-", "") for m in CLASSIFICATION_METRICS
            ]:
                if metric_str not in CLASSIFICATION_METRICS:
                    continue
            if task == "regression" and metric_str not in REGRESSION_METRICS:
                continue
            mean_val = mgrp["Mean"].mean() if "Mean" in mgrp.columns else mgrp["MetricValue"].mean()
            std_val = mgrp["Std"].mean() if "Std" in mgrp.columns and mgrp["Std"].notna().any() else np.nan
            if pd.isna(mean_val):
                continue
            metric_scores.append((metric_str, mean_val, std_val))

        if not metric_scores:
            continue

        # Normalize each metric within dataset×leakage×task then average
        norm_vals = []
        for metric_str, mean_val, std_val in metric_scores:
            pool = tstr[
                (tstr["Dataset"] == dataset)
                & (tstr["LeakageLevel"] == leakage)
                & (tstr["TaskType"] == task)
                & (tstr["Metric"] == metric_str)
            ]
            pool_vals = pool["Mean"] if "Mean" in pool.columns else pool["MetricValue"]
            vmin, vmax = pool_vals.min(), pool_vals.max()
            norm_vals.append(normalize_metric(mean_val, metric_str, vmin, vmax))

        utility_score = float(np.nanmean(norm_vals))
        std_agg = float(np.nanmean([s for _, _, s in metric_scores if pd.notna(s)])) if metric_scores else np.nan
        ci_lo, ci_hi = confidence_interval(utility_score, std_agg or 0, config.n_seeds)

        rows.append(
            {
                "Dataset": dataset,
                "Generator": generator,
                "LeakageLevel": leakage,
                "TaskType": task,
                "UtilityScore": utility_score,
                "UtilityStd": std_agg,
                "Utility_CI_Low": ci_lo,
                "Utility_CI_High": ci_hi,
                "N_Metrics": len(norm_vals),
            }
        )

    return pd.DataFrame(rows)


def compute_category_scores(
    metric_long: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    """Cumulative Fidelity or Privacy score per Dataset × Generator × LeakageLevel."""
    if metric_long.empty:
        return pd.DataFrame()

    df = metric_long.copy()
    if "LeakageLevel" not in df.columns:
        df["LeakageLevel"] = df.get("Leakage", 0)
    df["Dataset"] = df["Dataset"].map(normalize_dataset_name)
    value_col = "MetricValue" if "MetricValue" in df.columns else "Mean"

    rows: list[dict] = []
    score_col = f"{category}Score"

    for (dataset, generator, leakage), grp in df.groupby(
        ["Dataset", "Generator", "LeakageLevel"], dropna=False
    ):
        norm_vals = []
        for metric, mgrp in grp.groupby("Metric"):
            val = mgrp[value_col].mean()
            if pd.isna(val):
                continue
            pool = df[(df["Dataset"] == dataset) & (df["LeakageLevel"] == leakage) & (df["Metric"] == metric)]
            pool_vals = pool[value_col]
            vmin, vmax = pool_vals.min(), pool_vals.max()
            norm_vals.append(normalize_metric(val, str(metric), vmin, vmax))

        if not norm_vals:
            continue
        rows.append(
            {
                "Dataset": dataset,
                "Generator": generator,
                "LeakageLevel": leakage,
                score_col: float(np.nanmean(norm_vals)),
                f"{category}Std": float(np.nanstd(norm_vals)),
                f"N_{category}_Metrics": len(norm_vals),
            }
        )

    return pd.DataFrame(rows)


def merge_overall_scores(
    utility_scores: pd.DataFrame,
    fidelity_scores: pd.DataFrame,
    privacy_scores: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Dataset × Generator × LeakageLevel with Utility, Fidelity, Privacy, Overall."""
    frames = []
    for df, col in [
        (utility_scores, "UtilityScore"),
        (fidelity_scores, "FidelityScore"),
        (privacy_scores, "PrivacyScore"),
    ]:
        if df.empty:
            continue
        key_cols = ["Dataset", "Generator", "LeakageLevel"]
        sub = df[key_cols + [col]].copy()
        frames.append(sub.set_index(key_cols))

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    merged = merged.reset_index()

    w_u, w_p, w_f = config.utility_weight, config.privacy_weight, config.fidelity_weight
    for col in ("UtilityScore", "FidelityScore", "PrivacyScore"):
        if col in merged.columns:
            merged[col] = merged.groupby(["Dataset", "LeakageLevel"])[col].transform(
                lambda s: s.fillna(s.mean()) if s.notna().any() else s
            )
            merged[col] = merged[col].fillna(merged[col].mean())

    merged["OverallScore"] = (
        merged.get("UtilityScore", 0) * w_u
        + merged.get("FidelityScore", 0) * w_f
        + merged.get("PrivacyScore", 0) * w_p
    )
    merged["OverallRank"] = merged.groupby(["Dataset", "LeakageLevel"])["OverallScore"].rank(
        ascending=False, method="average"
    )
    return merged.sort_values(["Dataset", "LeakageLevel", "OverallRank"])


def build_processed_scores(
    utility_long: pd.DataFrame,
    fidelity_long: pd.DataFrame,
    privacy_long: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Run Steps 2–5: cumulative category and overall scores."""
    utility_scores = compute_utility_scores(utility_long, config)
    fidelity_scores = compute_category_scores(fidelity_long, "Fidelity")
    privacy_scores = compute_category_scores(privacy_long, "Privacy")
    overall = merge_overall_scores(utility_scores, fidelity_scores, privacy_scores, config)

    generator_ranking = (
        overall.groupby("Generator", dropna=False)
        .agg(
            UtilityScore=("UtilityScore", "mean"),
            FidelityScore=("FidelityScore", "mean"),
            PrivacyScore=("PrivacyScore", "mean"),
            OverallScore=("OverallScore", "mean"),
        )
        .reset_index()
        .sort_values("OverallScore", ascending=False)
    )
    generator_ranking["Rank"] = range(1, len(generator_ranking) + 1)

    return {
        "utility_scores": utility_scores,
        "fidelity_scores": fidelity_scores,
        "privacy_scores": privacy_scores,
        "overall_scores": overall,
        "generator_ranking": generator_ranking,
    }
