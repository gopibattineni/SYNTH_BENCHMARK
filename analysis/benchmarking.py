"""Advanced benchmarking analyses for publication-ready comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.config import GENERATORS, LOWER_IS_BETTER, PipelineConfig
from analysis.data_loader import normalize_dataset_name
from analysis.metrics_utils import normalize_metric, normalize_scores
from analysis.rankings import weighted_overall_ranking

# Nemenyi critical values q_alpha (alpha=0.05), Demšar 2006 Table 5.
_NEMENYI_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
}


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Critical difference for average-rank comparison after Friedman test."""
    if k < 2 or n_blocks < 1:
        return np.nan
    q = _NEMENYI_Q05.get(k, 3.031)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def _primary_utility_metric(task_type: str) -> str:
    return "R2" if task_type == "regression" else "Accuracy"


def build_dataset_generator_scores(
    utility_results: dict,
    privacy_results: dict,
    fidelity_results: dict,
) -> pd.DataFrame:
    """Per-dataset × generator normalized scores for utility, privacy, fidelity."""
    rows: list[dict] = []

    clf = utility_results.get("classification_stats", pd.DataFrame())
    reg = utility_results.get("regression_stats", pd.DataFrame())
    util_stats = pd.concat([clf, reg], ignore_index=True)
    util_tstr = util_stats[util_stats["EvaluationType"] == "TSTR"].copy()

    priv_stats = privacy_results.get("privacy_stats", pd.DataFrame())
    if not priv_stats.empty and "Dataset" in priv_stats.columns:
        priv_stats = priv_stats.copy()
        priv_stats["Dataset"] = priv_stats["Dataset"].map(normalize_dataset_name)

    fid_stats = fidelity_results.get("fidelity_stats", pd.DataFrame())

    datasets = sorted(
        set(util_tstr["Dataset"].dropna().unique())
        | set(priv_stats.get("Dataset", pd.Series(dtype=str)).dropna().unique())
        | set(fid_stats.get("Dataset", pd.Series(dtype=str)).dropna().unique())
    )

    for dataset in datasets:
        ds_util = util_tstr[util_tstr["Dataset"] == dataset]
        task_type = (
            ds_util["TaskType"].iloc[0]
            if "TaskType" in ds_util.columns and not ds_util.empty
            else ("regression" if _dataset_num(dataset) >= 10 else "classification")
        )
        util_metric = _primary_utility_metric(str(task_type))

        util_pivot = (
            ds_util[ds_util["Metric"].astype(str).str.replace("-", "") == util_metric.replace("-", "")]
            .groupby("Generator")["Mean"]
            .mean()
        )
        ds_priv = priv_stats[priv_stats["Dataset"] == dataset] if not priv_stats.empty else pd.DataFrame()
        priv_pivot = (
            ds_priv.groupby("Generator")["Mean"].mean()
            if not ds_priv.empty
            else pd.Series(dtype=float)
        )
        if priv_pivot.empty and not ds_priv.empty:
            priv_pivot = ds_priv.groupby("Generator")["NormalizedScore"].mean()

        ds_fid = fid_stats[fid_stats["Dataset"] == dataset] if not fid_stats.empty else pd.DataFrame()
        fid_pivot = (
            ds_fid.groupby("Generator")["NormalizedScore"].mean()
            if not ds_fid.empty and "NormalizedScore" in ds_fid.columns
            else pd.Series(dtype=float)
        )
        if fid_pivot.empty and not ds_fid.empty:
            fid_pivot = ds_fid.groupby("Generator")["Mean"].mean()

        generators = sorted(
            set(util_pivot.index)
            | set(priv_pivot.index)
            | set(fid_pivot.index)
            | set(GENERATORS)
        )

        def _norm_series(series: pd.Series, lower_better: bool = False) -> pd.Series:
            if series.empty:
                return series
            vmin, vmax = series.min(), series.max()
            out = series.copy()
            for gen in out.index:
                val = out[gen]
                metric = "Distance" if lower_better else "Score"
                if lower_better:
                    out[gen] = normalize_metric(val, metric, vmin, vmax)
                else:
                    out[gen] = normalize_metric(val, metric, vmin, vmax)
            return out

        util_norm = _norm_series(util_pivot, lower_better=False)
        priv_norm = _norm_series(priv_pivot, lower_better=True)
        fid_norm = _norm_series(fid_pivot, lower_better=False)

        for gen in generators:
            rows.append(
                {
                    "Dataset": dataset,
                    "Generator": gen,
                    "TaskType": task_type,
                    "Utility": util_norm.get(gen, np.nan),
                    "Privacy": priv_norm.get(gen, np.nan),
                    "Fidelity": fid_norm.get(gen, np.nan),
                    "UtilityRaw": util_pivot.get(gen, np.nan),
                    "PrivacyRaw": priv_pivot.get(gen, np.nan),
                    "FidelityRaw": fid_pivot.get(gen, np.nan),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ("Utility", "Privacy", "Fidelity"):
        global_mean = df[col].mean()
        df[col] = df.groupby("Dataset")[col].transform(
            lambda s: s.fillna(s.mean()) if s.notna().any() else s
        )
        df[col] = df[col].fillna(global_mean)
    return df


def _dataset_num(dataset: str) -> int:
    import re

    match = re.match(r"(\d+)", str(dataset))
    return int(match.group(1)) if match else 0


def is_pareto_optimal(row: pd.Series, points: pd.DataFrame, cols: tuple[str, ...]) -> bool:
    """True if no other point dominates this row (maximize all objectives)."""
    for _, other in points.iterrows():
        if other["Generator"] == row["Generator"]:
            continue
        better_or_equal = all(other[c] >= row[c] for c in cols)
        strictly_better = any(other[c] > row[c] for c in cols)
        if better_or_equal and strictly_better:
            return False
    return True


def pareto_analysis(scores_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """3D Pareto frontier per dataset and aggregate optimal counts."""
    cols = ("Utility", "Privacy", "Fidelity")
    if scores_df.empty or not set(cols).issubset(scores_df.columns):
        return {
            "pareto_by_dataset": pd.DataFrame(),
            "pareto_counts": pd.DataFrame(),
            "pareto_fraction": pd.DataFrame(),
        }

    frontier_rows: list[dict] = []
    for dataset, grp in scores_df.groupby("Dataset"):
        valid = grp.dropna(subset=list(cols))
        if valid.empty:
            continue
        for _, row in valid.iterrows():
            if is_pareto_optimal(row, valid, cols):
                frontier_rows.append(
                    {
                        "Dataset": dataset,
                        "Generator": row["Generator"],
                        **{c: row[c] for c in cols},
                        "ParetoOptimal": True,
                    }
                )

    pareto_by_dataset = pd.DataFrame(frontier_rows)
    n_datasets = scores_df["Dataset"].nunique()

    if pareto_by_dataset.empty:
        counts = pd.DataFrame({"Generator": GENERATORS, "ParetoCount": 0, "ParetoFraction": 0.0})
    else:
        counts = (
            pareto_by_dataset.groupby("Generator")
            .size()
            .reset_index(name="ParetoCount")
        )
        counts["ParetoFraction"] = counts["ParetoCount"] / max(n_datasets, 1)
        for gen in GENERATORS:
            if gen not in counts["Generator"].values:
                counts = pd.concat(
                    [counts, pd.DataFrame({"Generator": [gen], "ParetoCount": [0], "ParetoFraction": [0.0]})],
                    ignore_index=True,
                )
        counts = counts.sort_values("ParetoCount", ascending=False)

    return {
        "pareto_by_dataset": pareto_by_dataset,
        "pareto_counts": counts,
        "pareto_fraction": counts[["Generator", "ParetoFraction"]].copy(),
    }


def friedman_average_ranks(
    summary_df: pd.DataFrame,
    value_col: str = "Mean",
    block_col: str = "Dataset",
    treatment_col: str = "Generator",
) -> pd.DataFrame:
    """Compute Friedman average ranks across blocks (datasets)."""
    if summary_df.empty:
        return pd.DataFrame()

    pivot = summary_df.pivot_table(
        index=block_col,
        columns=treatment_col,
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return pd.DataFrame()

    rank_matrix = pivot.rank(axis=1, ascending=False, method="average")
    avg_ranks = rank_matrix.mean().reset_index()
    avg_ranks.columns = ["Generator", "AverageRank"]
    avg_ranks = avg_ranks.sort_values("AverageRank")
    avg_ranks["N_Datasets"] = pivot.shape[0]
    avg_ranks["N_Generators"] = pivot.shape[1]
    avg_ranks["CriticalDifference"] = nemenyi_critical_difference(
        pivot.shape[1], pivot.shape[0]
    )
    return avg_ranks


def nemenyi_significance_groups(avg_ranks: pd.DataFrame) -> pd.DataFrame:
    """Assign Nemenyi CD groups (generators not significantly different share a group)."""
    if avg_ranks.empty or "CriticalDifference" not in avg_ranks.columns:
        return pd.DataFrame()

    cd = avg_ranks["CriticalDifference"].iloc[0]
    sorted_df = avg_ranks.sort_values("AverageRank").reset_index(drop=True)
    groups: list[str] = []
    group_id = 1
    for i, row in sorted_df.iterrows():
        if i == 0:
            groups.append(str(group_id))
            continue
        if row["AverageRank"] - sorted_df.loc[i - 1, "AverageRank"] <= cd:
            groups.append(groups[-1])
        else:
            group_id += 1
            groups.append(str(group_id))
    sorted_df["CD_Group"] = groups
    return sorted_df


def composite_score_analysis(
    scores_df: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Normalized composite score with configurable weights (default 40/30/30)."""
    if scores_df.empty:
        return {"composite_overall": pd.DataFrame(), "composite_by_dataset": pd.DataFrame()}

    w_u, w_p, w_f = config.utility_weight, config.privacy_weight, config.fidelity_weight

    by_dataset = scores_df.copy()
    by_dataset["CompositeScore"] = (
        by_dataset["Utility"].fillna(by_dataset["Utility"].mean()) * w_u
        + by_dataset["Privacy"].fillna(by_dataset["Privacy"].mean()) * w_p
        + by_dataset["Fidelity"].fillna(by_dataset["Fidelity"].mean()) * w_f
    )
    by_dataset["CompositeRank"] = by_dataset.groupby("Dataset")["CompositeScore"].rank(
        ascending=False, method="average"
    )

    util_scores = (
        by_dataset.groupby("Generator")["Utility"].mean().reset_index().rename(columns={"Utility": "NormalizedScore"})
    )
    priv_scores = (
        by_dataset.groupby("Generator")["Privacy"].mean().reset_index().rename(columns={"Privacy": "NormalizedScore"})
    )
    fid_scores = (
        by_dataset.groupby("Generator")["Fidelity"].mean().reset_index().rename(columns={"Fidelity": "NormalizedScore"})
    )
    overall = weighted_overall_ranking(util_scores, priv_scores, fid_scores, config)
    overall = overall.rename(columns={"OverallScore": "CompositeScore", "OverallRank": "CompositeRank"})

    util_map = util_scores.set_index("Generator")["NormalizedScore"]
    priv_map = priv_scores.set_index("Generator")["NormalizedScore"] if not priv_scores.empty else pd.Series(dtype=float)
    fid_map = fid_scores.set_index("Generator")["NormalizedScore"] if not fid_scores.empty else pd.Series(dtype=float)

    component_breakdown = pd.DataFrame({"Generator": overall["Generator"]})
    component_breakdown["UtilityComponent"] = component_breakdown["Generator"].map(util_map).fillna(0) * w_u
    component_breakdown["PrivacyComponent"] = component_breakdown["Generator"].map(priv_map).fillna(0) * w_p
    component_breakdown["FidelityComponent"] = component_breakdown["Generator"].map(fid_map).fillna(0) * w_f

    overall = overall.merge(component_breakdown, on="Generator", how="left")
    overall["Weights"] = f"{w_u:.0%}/{w_p:.0%}/{w_f:.0%}"

    return {
        "composite_overall": overall.sort_values("CompositeRank"),
        "composite_by_dataset": by_dataset.sort_values(["Dataset", "CompositeRank"]),
    }


def domain_correlation_analysis(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Answer: does higher fidelity → better utility? Does stronger privacy reduce utility?"""
    if scores_df.empty:
        return pd.DataFrame()

    pairs = [
        ("Fidelity", "Utility", "Higher fidelity → better utility?"),
        ("Privacy", "Utility", "Stronger privacy reduces utility?"),
        ("Fidelity", "Privacy", "Fidelity–privacy trade-off"),
        ("Utility", "Privacy", "Utility–privacy trade-off"),
    ]
    rows: list[dict] = []
    for x, y, question in pairs:
        valid = scores_df[[x, y]].dropna()
        if len(valid) < 3:
            continue
        pr, pp = stats.pearsonr(valid[x], valid[y])
        sr, sp = stats.spearmanr(valid[x], valid[y])
        rows.append(
            {
                "Question": question,
                "X": x,
                "Y": y,
                "N": len(valid),
                "Pearson_r": pr,
                "Pearson_p": pp,
                "Spearman_r": sr,
                "Spearman_p": sp,
                "Interpretation": _interpret_correlation(x, y, sr),
            }
        )
    return pd.DataFrame(rows)


def _interpret_correlation(x: str, y: str, rho: float) -> str:
    strength = "strong" if abs(rho) > 0.6 else "moderate" if abs(rho) > 0.3 else "weak"
    direction = "positive" if rho > 0 else "negative"
    if x == "Fidelity" and y == "Utility":
        return f"{strength} {direction} association — {'yes' if rho > 0.3 else 'limited evidence'} that fidelity predicts utility"
    if x == "Privacy" and y == "Utility":
        return f"{strength} {direction} association — privacy {'costs' if rho < -0.3 else 'may not strongly reduce'} utility"
    return f"{strength} {direction} correlation"


def seed_stability_analysis(
    utility_long: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Coefficient of variation (Std/Mean) across seeds — lower is more stable."""
    if utility_long.empty or "Std" not in utility_long.columns:
        return {"seed_cv": pd.DataFrame(), "seed_stability_ranks": pd.DataFrame()}

    tstr = utility_long[utility_long["EvaluationType"] == "TSTR"].copy()
    tstr = tstr[tstr["Std"].notna() & tstr["Mean"].notna() & (tstr["Mean"] != 0)]
    if tstr.empty:
        return {"seed_cv": pd.DataFrame(), "seed_stability_ranks": pd.DataFrame()}

    tstr["CV"] = (tstr["Std"].abs() / tstr["Mean"].abs()).clip(0, 5)
    cv = (
        tstr.groupby(["Dataset", "Generator", "Metric"], dropna=False)
        .agg(Mean_CV=("CV", "mean"), Std_CV=("CV", "std"), N=("CV", "count"))
        .reset_index()
    )

    stability = (
        cv.groupby("Generator", dropna=False)["Mean_CV"]
        .agg(Mean_CV="mean", Std_CV="std", N_Metrics="count")
        .reset_index()
        .sort_values("Mean_CV")
    )
    stability["StabilityRank"] = stability["Mean_CV"].rank(method="average")
    stability["StabilityScore"] = 1.0 / (1.0 + stability["Mean_CV"])

    return {"seed_cv": cv, "seed_stability_ranks": stability}


def leakage_sensitivity_analysis(
    utility_long: pd.DataFrame,
    scores_df: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Leakage sensitivity curves when multi-level Leakage exists; else utility-retention proxy."""
    if utility_long.empty:
        return {"leakage_curves": pd.DataFrame(), "leakage_proxy": pd.DataFrame()}

    if "Leakage" not in utility_long.columns:
        utility_long = utility_long.copy()
        utility_long["Leakage"] = 0

    levels = sorted(utility_long["Leakage"].dropna().unique())
    if len(levels) > 1:
        curves = []
        for leakage in levels:
            sub = utility_long[utility_long["Leakage"] == leakage]
            util = sub.groupby(["Generator", "Dataset"])["Mean"].mean().reset_index()
            util["Domain"] = "Utility"
            util["Leakage"] = leakage
            curves.append(util)
        return {"leakage_curves": pd.concat(curves, ignore_index=True), "leakage_proxy": pd.DataFrame()}

    # Proxy: utility retention (TSTR/TRTR) vs composite scores — memorization / information loss
    proxy_rows: list[dict] = []
    clf = utility_long[utility_long["TaskType"] == "classification"]
    reg = utility_long[utility_long["TaskType"] == "regression"]

    for task_df, metric in [(clf, "Accuracy"), (reg, "R2")]:
        if task_df.empty:
            continue
        trtr = task_df[
            (task_df["EvaluationType"] == "TRTR") & (task_df["Metric"] == metric)
        ]
        tstr = task_df[
            (task_df["EvaluationType"] == "TSTR") & (task_df["Metric"] == metric)
        ]
        merged = trtr.merge(
            tstr,
            on=["Dataset", "Generator", "Classifier", "RegressionModel"],
            suffixes=("_TRTR", "_TSTR"),
        )
        if merged.empty:
            continue
        merged["Retention"] = merged["Mean_TSTR"] / merged["Mean_TRTR"].replace(0, np.nan)
        retention = merged.groupby(["Dataset", "Generator"])["Retention"].mean().reset_index()

        if not scores_df.empty:
            retention = retention.merge(
                scores_df[["Dataset", "Generator", "Utility", "Privacy", "Fidelity"]],
                on=["Dataset", "Generator"],
                how="left",
            )
        retention["Metric"] = metric
        proxy_rows.extend(retention.to_dict("records"))

    proxy = pd.DataFrame(proxy_rows)
    return {"leakage_curves": pd.DataFrame(), "leakage_proxy": proxy}


def run_benchmarking(
    utility_results: dict,
    privacy_results: dict,
    fidelity_results: dict,
    utility_long: pd.DataFrame,
    statistics: dict,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Run all advanced benchmarking analyses."""
    scores = build_dataset_generator_scores(utility_results, privacy_results, fidelity_results)
    pareto = pareto_analysis(scores)
    composite = composite_score_analysis(scores, config)
    correlations = domain_correlation_analysis(scores)
    seed = seed_stability_analysis(utility_long, config)
    leakage = leakage_sensitivity_analysis(utility_long, scores, config)

    # Friedman average ranks on primary utility metric (TSTR)
    util_stats = utility_results.get("classification_stats", pd.DataFrame())
    reg_stats = utility_results.get("regression_stats", pd.DataFrame())
    util_tstr = pd.concat([util_stats, reg_stats], ignore_index=True)
    util_tstr = util_tstr[util_tstr["EvaluationType"] == "TSTR"]

    friedman_ranks: dict[str, pd.DataFrame] = {}
    cd_groups: dict[str, pd.DataFrame] = {}

    for label, df, metric_filter in [
        ("utility_accuracy", util_tstr, lambda d: d[d["Metric"].astype(str).str.replace("-", "") == "Accuracy"]),
        ("utility_r2", util_tstr, lambda d: d[d["Metric"] == "R2"]),
        ("utility_f1", util_tstr, lambda d: d[d["Metric"] == "F1"]),
    ]:
        sub = metric_filter(df)
        if sub.empty:
            continue
        agg = sub.groupby(["Dataset", "Generator"], dropna=False)["Mean"].mean().reset_index()
        ranks = friedman_average_ranks(agg)
        if not ranks.empty:
            friedman_ranks[label] = ranks
            cd_groups[label] = nemenyi_significance_groups(ranks)

    # Fidelity & privacy Friedman ranks
    fid_stats = fidelity_results.get("fidelity_stats", pd.DataFrame())
    if not fid_stats.empty:
        fid_agg = fid_stats.groupby(["Dataset", "Generator"], dropna=False)["NormalizedScore"].mean().reset_index()
        fid_agg = fid_agg.rename(columns={"NormalizedScore": "Mean"})
        ranks = friedman_average_ranks(fid_agg)
        if not ranks.empty:
            friedman_ranks["fidelity"] = ranks
            cd_groups["fidelity"] = nemenyi_significance_groups(ranks)

    priv_stats = privacy_results.get("privacy_stats", pd.DataFrame())
    if not priv_stats.empty:
        priv_agg = priv_stats.groupby(["Dataset", "Generator"], dropna=False)["NormalizedScore"].mean().reset_index()
        priv_agg = priv_agg.rename(columns={"NormalizedScore": "Mean"})
        ranks = friedman_average_ranks(priv_agg)
        if not ranks.empty:
            friedman_ranks["privacy"] = ranks
            cd_groups["privacy"] = nemenyi_significance_groups(ranks)

    # Composite score ranks for Friedman
    comp_by_ds = composite.get("composite_by_dataset", pd.DataFrame())
    if not comp_by_ds.empty:
        comp_agg = comp_by_ds.groupby(["Dataset", "Generator"], dropna=False)["CompositeScore"].mean().reset_index()
        comp_agg = comp_agg.rename(columns={"CompositeScore": "Mean"})
        ranks = friedman_average_ranks(comp_agg)
        if not ranks.empty:
            friedman_ranks["composite"] = ranks
            cd_groups["composite"] = nemenyi_significance_groups(ranks)

    return {
        "dataset_generator_scores": scores,
        "pareto_by_dataset": pareto["pareto_by_dataset"],
        "pareto_counts": pareto["pareto_counts"],
        "composite_overall": composite["composite_overall"],
        "composite_by_dataset": composite["composite_by_dataset"],
        "domain_correlations": correlations,
        "seed_cv": seed["seed_cv"],
        "seed_stability_ranks": seed["seed_stability_ranks"],
        "leakage_curves": leakage["leakage_curves"],
        "leakage_proxy": leakage["leakage_proxy"],
        **{f"friedman_ranks_{k}": v for k, v in friedman_ranks.items()},
        **{f"cd_groups_{k}": v for k, v in cd_groups.items()},
    }
