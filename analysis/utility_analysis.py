"""Utility (TRTR/TSTR) analysis."""

from __future__ import annotations

import pandas as pd

from analysis.config import CLASSIFICATION_METRICS, REGRESSION_METRICS, PipelineConfig
from analysis.metrics_utils import aggregate_stats, normalize_scores, rank_generators


def analyze_utility(master_utility: pd.DataFrame, config: PipelineConfig) -> dict[str, pd.DataFrame]:
    if master_utility.empty:
        return {
            "classification_stats": pd.DataFrame(),
            "regression_stats": pd.DataFrame(),
            "classification_gaps": pd.DataFrame(),
            "regression_gaps": pd.DataFrame(),
            "generator_summary": pd.DataFrame(),
        }

    clf = master_utility[master_utility["TaskType"] == "classification"].copy()
    reg = master_utility[master_utility["TaskType"] == "regression"].copy()

    clf_metrics = [m for m in CLASSIFICATION_METRICS if m.replace("-", "") in clf["Metric"].unique() or m in clf["Metric"].unique()]
    reg_metrics = [m for m in REGRESSION_METRICS if m in reg["Metric"].unique()]

    clf_stats = aggregate_stats(
        clf[clf["EvaluationType"].isin(["TRTR", "TSTR"])],
        ["Dataset", "Generator", "Classifier", "Metric", "EvaluationType"],
        n_seeds=config.n_seeds,
    )
    reg_stats = aggregate_stats(
        reg[reg["EvaluationType"].isin(["TRTR", "TSTR"])],
        ["Dataset", "Generator", "RegressionModel", "Metric", "EvaluationType"],
        n_seeds=config.n_seeds,
    )

    clf_gaps = aggregate_stats(
        clf[(clf["EvaluationType"] == "Gap") | clf["Metric"].str.endswith("_Drop", na=False)],
        ["Dataset", "Generator", "Classifier", "Metric"],
        n_seeds=config.n_seeds,
    )
    reg_gaps = aggregate_stats(
        reg[(reg["EvaluationType"] == "Gap") | reg["Metric"].str.contains("Drop|Increase", na=False)],
        ["Dataset", "Generator", "RegressionModel", "Metric"],
        n_seeds=config.n_seeds,
    )

    gen_clf = (
        clf_stats.groupby(["Generator", "Metric", "EvaluationType"], dropna=False)["Mean"]
        .mean()
        .reset_index()
        if not clf_stats.empty
        else pd.DataFrame()
    )
    gen_reg = (
        reg_stats.groupby(["Generator", "Metric", "EvaluationType"], dropna=False)["Mean"]
        .mean()
        .reset_index()
        if not reg_stats.empty
        else pd.DataFrame()
    )
    generator_summary = pd.concat([gen_clf, gen_reg], ignore_index=True)
    generator_summary = normalize_scores(generator_summary.rename(columns={"Mean": "Mean"}))
    generator_summary = rank_generators(generator_summary, group_cols=["Metric", "EvaluationType"])

    return {
        "classification_stats": clf_stats,
        "regression_stats": reg_stats,
        "classification_gaps": clf_gaps,
        "regression_gaps": reg_gaps,
        "generator_summary": generator_summary,
    }
