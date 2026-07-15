"""Publication-quality figure generation (300 dpi, multi-format export)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from analysis.config import GENERATORS, PLOT_RC, PipelineConfig


def _apply_style():
    for key, val in PLOT_RC.items():
        plt.rcParams[key] = val


def save_figure(fig: Figure, path_stem: Path, formats: list[str], dpi: int) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        try:
            fig.savefig(f"{path_stem}.{fmt}", dpi=dpi, bbox_inches="tight")
        except Exception:
            pass
    plt.close(fig)


def _barplot(df: pd.DataFrame, x: str, y: str, hue: str | None, title: str, path: Path, config: PipelineConfig, **kwargs):
    if df.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    if hue:
        sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, errorbar="sd", **kwargs)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        sns.barplot(data=df, x=x, y=y, ax=ax, errorbar="sd", **kwargs)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _boxplot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, config: PipelineConfig):
    if df.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x=x, y=y, ax=ax, order=sorted(df[x].dropna().unique()))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _heatmap(data: pd.DataFrame, title: str, path: Path, config: PipelineConfig, annot: bool = True):
    if data.empty or data.shape[0] < 2:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, data.shape[1] * 0.6), max(6, data.shape[0] * 0.4)))
    sns.heatmap(data, annot=annot, fmt=".2f", cmap="RdYlGn", ax=ax, cbar_kws={"label": "Score"})
    ax.set_title(title)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _violinplot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, config: PipelineConfig):
    if df.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x=x, y=y, ax=ax, order=sorted(df[x].dropna().unique()), inner="box")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _ci_plot(df: pd.DataFrame, x: str, title: str, path: Path, config: PipelineConfig):
    if df.empty or "CI_Low" not in df.columns:
        return
    _apply_style()
    plot_df = df.groupby(x, dropna=False).agg({"Mean": "mean", "CI_Low": "mean", "CI_High": "mean"}).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = range(len(plot_df))
    ax.errorbar(
        x_pos,
        plot_df["Mean"],
        yerr=[plot_df["Mean"] - plot_df["CI_Low"], plot_df["CI_High"] - plot_df["Mean"]],
        fmt="o",
        capsize=4,
    )
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(plot_df[x], rotation=45, ha="right")
    ax.set_title(title)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _qq_plot(series: pd.Series, title: str, path: Path, config: PipelineConfig):
    if series.empty:
        return
    _apply_style()
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(6, 6))
    sp_stats.probplot(series.dropna().values, dist="norm", plot=ax)
    ax.set_title(title)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _scatter(df: pd.DataFrame, x: str, y: str, hue: str, title: str, path: Path, config: PipelineConfig, size: str | None = None):
    if df.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, ax=ax, s=100)
    ax.set_title(title)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _radar(scores: pd.DataFrame, categories: list[str], title: str, path: Path, config: PipelineConfig):
    if scores.empty or not categories:
        return
    _apply_style()
    labels = categories
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for _, row in scores.iterrows():
        vals = [row.get(c, np.nan) for c in labels]
        vals = [float(v) if pd.notna(v) else 0 for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, label=str(row.get("Generator", "")))
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left", fontsize=8)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def _critical_difference_diagram(avg_ranks: pd.DataFrame, path: Path, config: PipelineConfig):
    if avg_ranks.empty or "AverageRank" not in avg_ranks.columns:
        return
    _apply_style()
    df = avg_ranks.sort_values("AverageRank")
    fig, ax = plt.subplots(figsize=(10, 3))
    y = 0.5
    ax.hlines(y, df["AverageRank"].min(), df["AverageRank"].max(), colors="black", linewidth=2)
    for _, row in df.iterrows():
        ax.plot(row["AverageRank"], y, "o", markersize=10)
        ax.text(row["AverageRank"], y + 0.08, str(row["Generator"]), ha="center", fontsize=8)
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title("Critical Difference Diagram (Average Ranks)")
    ax.set_yticks([])
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def generate_all_figures(
    results: dict[str, Any],
    dirs: dict[str, Path],
    config: PipelineConfig,
) -> list[str]:
    saved: list[str] = []

    def _track(name: str) -> None:
        saved.append(name)
    util_dir = dirs["figures_utility"]
    priv_dir = dirs["figures_privacy"]
    fid_dir = dirs["figures_fidelity"]
    trade_dir = dirs["figures_tradeoff"]
    stat_dir = dirs["figures_statistical"]
    leak_dir = dirs["figures_leakage"]

    clf = results.get("utility", {}).get("classification_stats", pd.DataFrame())
    reg = results.get("utility", {}).get("regression_stats", pd.DataFrame())
    fid_stats = results.get("fidelity", {}).get("fidelity_stats", pd.DataFrame())
    priv_stats = results.get("privacy", {}).get("privacy_stats", pd.DataFrame())
    trade = results.get("tradeoff", {}).get("tradeoff_points", pd.DataFrame())
    rankings = results.get("rankings", {})
    stats_res = results.get("statistics", {})
    master = results.get("master")

    # --- Utility figures ---
    if not clf.empty:
        acc = clf[(clf["Metric"] == "Accuracy") & (clf["EvaluationType"].isin(["TRTR", "TSTR"]))]
        _barplot(acc, "Generator", "Mean", "EvaluationType", "Accuracy: TRTR vs TSTR", util_dir / "accuracy_trtr_tstr", config)
        _track("accuracy_trtr_tstr")

        f1 = clf[(clf["Metric"] == "F1") & (clf["EvaluationType"].isin(["TRTR", "TSTR"]))]
        _barplot(f1, "Generator", "Mean", "EvaluationType", "F1: TRTR vs TSTR", util_dir / "f1_trtr_tstr", config)
        _track("f1_trtr_tstr")

        for metric in ["Accuracy", "F1", "Precision", "Recall"]:
            mdf = clf[(clf["Metric"] == metric) & (clf["EvaluationType"] == "TSTR")]
            if not mdf.empty:
                _boxplot(mdf, "Generator", "Mean", f"{metric} TSTR by Generator", util_dir / f"{metric.lower()}_boxplot", config)
                _violinplot(mdf, "Generator", "Mean", f"{metric} TSTR Violin", util_dir / f"{metric.lower()}_violin", config)
                _track(f"{metric.lower()}_distribution")

        acc_tstr = clf[(clf["Metric"] == "Accuracy") & (clf["EvaluationType"] == "TSTR")]
        _ci_plot(acc_tstr, "Generator", "Accuracy TSTR 95% CI", util_dir / "accuracy_ci", config)

        clf_heat = clf[(clf["EvaluationType"] == "TSTR") & (clf["Classifier"].notna())]
        if not clf_heat.empty:
            clf_pivot = clf_heat.pivot_table(index="Classifier", columns="Generator", values="Mean", aggfunc="mean")
            _heatmap(clf_pivot, "Classifier × Generator TSTR Heatmap", util_dir / "classifier_ranking_heatmap", config)

        pivot = acc.pivot_table(index="Generator", columns="EvaluationType", values="Mean", aggfunc="mean")
        if not pivot.empty:
            _heatmap(pivot, "Accuracy TRTR vs TSTR Heatmap", util_dir / "accuracy_heatmap", config)

        gap = results.get("utility", {}).get("classification_gaps", pd.DataFrame())
        if not gap.empty:
            _barplot(gap.groupby("Generator")["Mean"].mean().reset_index(), "Generator", "Mean", None,
                     "Utility Gap (TRTR − TSTR)", util_dir / "utility_gap", config)

    if not reg.empty:
        r2 = reg[(reg["Metric"] == "R2") & (reg["EvaluationType"].isin(["TRTR", "TSTR"]))]
        _barplot(r2, "Generator", "Mean", "EvaluationType", "R²: TRTR vs TSTR", util_dir / "r2_trtr_tstr", config)
        for metric in ["R2", "RMSE", "MAE"]:
            mdf = reg[(reg["Metric"] == metric) & (reg["EvaluationType"] == "TSTR")]
            if not mdf.empty:
                _barplot(mdf.groupby("Generator")["Mean"].mean().reset_index(), "Generator", "Mean", None,
                         f"{metric} TSTR Comparison", util_dir / f"{metric.lower()}_comparison", config)

    # --- Fidelity figures ---
    if not fid_stats.empty:
        qs = fid_stats[fid_stats["Metric"].str.contains("Quality", na=False)]
        if not qs.empty:
            _barplot(qs.groupby("Generator")["Mean"].mean().reset_index(), "Generator", "Mean", None,
                     "SDV Quality Score by Generator", fid_dir / "quality_score_bar", config)
            _boxplot(qs, "Generator", "Mean", "Quality Score Variability", fid_dir / "quality_boxplot", config)

        rank_pivot = fid_stats.pivot_table(index="Dataset", columns="Generator", values="NormalizedScore", aggfunc="mean")
        _heatmap(rank_pivot, "Fidelity Normalized Scores", fid_dir / "fidelity_heatmap", config)

    # --- Privacy figures ---
    if not priv_stats.empty:
        md = priv_stats[priv_stats["Metric"] == "Mean_Distance"]
        if not md.empty:
            _barplot(md.groupby("Generator")["Mean"].mean().reset_index(), "Generator", "Mean", None,
                     "Mahalanobis Mean Distance", priv_dir / "mahalanobis_comparison", config)
            _boxplot(md, "Generator", "Mean", "Mahalanobis Distance Distribution", priv_dir / "mahalanobis_boxplot", config)

        priv_pivot = priv_stats.pivot_table(index="Dataset", columns="Generator", values="NormalizedScore", aggfunc="mean")
        _heatmap(priv_pivot, "Privacy Normalized Scores", priv_dir / "privacy_heatmap", config)

    # --- Trade-off figures ---
    if not trade.empty and {"Utility", "Privacy"}.issubset(trade.columns):
        _scatter(trade, "Utility", "Privacy", "Generator", "Privacy vs Utility", trade_dir / "privacy_vs_utility", config)
    if not trade.empty and {"Fidelity", "Utility"}.issubset(trade.columns):
        _scatter(trade, "Fidelity", "Utility", "Generator", "Fidelity vs Utility", trade_dir / "fidelity_vs_utility", config)
    if not trade.empty and {"Fidelity", "Privacy"}.issubset(trade.columns):
        _scatter(trade, "Fidelity", "Privacy", "Generator", "Fidelity vs Privacy", trade_dir / "fidelity_vs_privacy", config)
    if not trade.empty and {"Utility", "Privacy", "Fidelity"}.issubset(trade.columns):
        _scatter(trade, "Utility", "Privacy", "Generator", "Bubble: Utility vs Privacy (size=Fidelity)",
                 trade_dir / "bubble_tradeoff", config, size="Fidelity")

        # 3D scatter saved as PNG only via mpl
        _apply_style()
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for gen in trade["Generator"].unique():
            sub = trade[trade["Generator"] == gen]
            ax.scatter(sub["Utility"], sub["Privacy"], sub["Fidelity"], label=gen, s=60)
        ax.set_xlabel("Utility")
        ax.set_ylabel("Privacy")
        ax.set_zlabel("Fidelity")
        ax.set_title("3D Trade-off")
        ax.legend(fontsize=7)
        save_figure(fig, trade_dir / "tradeoff_3d", config.figure_formats, config.figure_dpi)

        # Parallel coordinates via matplotlib
        _apply_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        cols = ["Utility", "Privacy", "Fidelity"]
        x_pos = range(len(cols))
        for _, row in trade.iterrows():
            vals = [row[c] for c in cols]
            ax.plot(x_pos, vals, marker="o", label=row["Generator"], alpha=0.8)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(cols)
        ax.set_title("Parallel Coordinates")
        ax.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
        save_figure(fig, trade_dir / "parallel_coordinates", config.figure_formats, config.figure_dpi)

    # --- Statistical figures ---
    nemenyi = stats_res.get("nemenyi", pd.DataFrame())
    if not nemenyi.empty and "Metric" in nemenyi.columns:
        for metric in nemenyi["Metric"].unique()[:3]:
            sub = nemenyi[nemenyi["Metric"] == metric]
            num_cols = [c for c in sub.columns if c not in ("Metric", "Generator_A")]
            if num_cols:
                mat = sub.set_index("Generator_A")[num_cols[: len(GENERATORS)]]
                _heatmap(mat.astype(float, errors="ignore"), f"Nemenyi p-matrix: {metric}",
                         stat_dir / f"nemenyi_{metric}", config, annot=False)

    wilcoxon = stats_res.get("pairwise_wilcoxon", pd.DataFrame())
    if not wilcoxon.empty:
        for metric in wilcoxon["Metric"].unique()[:5]:
            sub = wilcoxon[wilcoxon["Metric"] == metric]
            mat = sub.pivot(index="Generator_A", columns="Generator_B", values="p_bh")
            _heatmap(mat, f"Wilcoxon p-values (BH): {metric}", stat_dir / f"significance_{metric}", config)

    weighted = rankings.get("weighted_overall", pd.DataFrame())
    if not weighted.empty:
        _barplot(weighted, "Generator", "OverallScore", None, "Weighted Overall Ranking", stat_dir / "weighted_ranking", config)

    borda = rankings.get("borda", pd.DataFrame())
    if not borda.empty:
        _barplot(borda, "Generator", "BordaScore", None, "Borda Count Ranking", stat_dir / "borda_ranking", config)

    for name, avg in rankings.items():
        if "avg_ranks" in name and not avg.empty:
            _critical_difference_diagram(avg, stat_dir / name, config)

    # Radar plots
    if not trade.empty:
        _radar(trade, ["Utility", "Privacy", "Fidelity"], "Generator Radar (Trade-off)", trade_dir / "radar_tradeoff", config)

    # Correlation heatmaps
    corr = results.get("correlation", {})
    for method, mat in corr.items():
        if isinstance(mat, pd.DataFrame) and not mat.empty:
            _heatmap(mat, f"{method.capitalize()} Correlation", stat_dir / f"corr_{method}", config)

    # Leakage placeholder
    leak = results.get("leakage", {}).get("leakage_available", pd.DataFrame())
    if not leak.empty:
        _apply_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(leak["LeakageLevel"].astype(str), leak.get("Count", [1] * len(leak)))
        ax.set_title("Leakage Levels in Master Data")
        ax.set_xlabel("Leakage (%)")
        save_figure(fig, leak_dir / "leakage_coverage", config.figure_formats, config.figure_dpi)

    # ECDF from Mahalanobis detail
    if master is not None and not master.privacy_detail.empty:
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        for gen, grp in master.privacy_detail.groupby("Generator"):
            sorted_vals = np.sort(grp["Mahalanobis_Distance"].values)
            y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, y, label=str(gen))
        ax.set_xlabel("Mahalanobis Distance")
        ax.set_ylabel("ECDF")
        ax.set_title("ECDF of Mahalanobis Distances")
        ax.legend(fontsize=7)
        save_figure(fig, stat_dir / "mahalanobis_ecdf", config.figure_formats, config.figure_dpi)
        _track("mahalanobis_ecdf")

        for gen, grp in master.privacy_detail.groupby("Generator"):
            if len(grp) >= 10:
                _qq_plot(grp["Mahalanobis_Distance"], f"QQ Plot: {gen} Mahalanobis", stat_dir / f"qq_mahalanobis_{gen}", config)
                break

    return saved


def generate_benchmark_figures(
    results: dict[str, Any],
    dirs: dict[str, Path],
    config: PipelineConfig,
) -> list[str]:
    """Publication figures for advanced benchmarking analyses."""
    saved: list[str] = []
    bench_dir = dirs.get("figures_benchmark", dirs["figures_tradeoff"])
    bench_dir.mkdir(parents=True, exist_ok=True)

    benchmarking = results.get("benchmarking", {})
    feature_fid = results.get("feature_fidelity", pd.DataFrame())

    def _track(name: str) -> None:
        saved.append(name)

    # --- Pareto frontier ---
    pareto = benchmarking.get("pareto_by_dataset", pd.DataFrame())
    pareto_counts = benchmarking.get("pareto_counts", pd.DataFrame())
    scores = benchmarking.get("dataset_generator_scores", pd.DataFrame())

    if not scores.empty and {"Utility", "Privacy", "Fidelity"}.issubset(scores.columns):
        _scatter(
            scores,
            "Utility",
            "Privacy",
            "Generator",
            "Pareto: Utility vs Privacy (all datasets)",
            bench_dir / "pareto_utility_privacy",
            config,
            size="Fidelity",
        )
        _track("pareto_utility_privacy")

        if not pareto.empty:
            _apply_style()
            fig, ax = plt.subplots(figsize=(9, 7))
            for dataset, grp in scores.groupby("Dataset"):
                pareto_gens = set(pareto[pareto["Dataset"] == dataset]["Generator"])
                for _, row in grp.iterrows():
                    marker = "*" if row["Generator"] in pareto_gens else "o"
                    size = 120 if row["Generator"] in pareto_gens else 40
                    ax.scatter(row["Utility"], row["Fidelity"], marker=marker, s=size, alpha=0.7)
            ax.set_xlabel("Utility")
            ax.set_ylabel("Fidelity")
            ax.set_title("Pareto-optimal generators (* = on frontier)")
            save_figure(fig, bench_dir / "pareto_utility_fidelity", config.figure_formats, config.figure_dpi)
            _track("pareto_utility_fidelity")

    if not pareto_counts.empty:
        _barplot(
            pareto_counts,
            "Generator",
            "ParetoCount",
            None,
            "Pareto-Optimal Count Across Datasets",
            bench_dir / "pareto_optimal_counts",
            config,
        )
        _track("pareto_optimal_counts")

    # --- Composite score ---
    composite = benchmarking.get("composite_overall", pd.DataFrame())
    if not composite.empty:
        _barplot(
            composite,
            "Generator",
            "CompositeScore",
            None,
            f"Composite Score (weights: {config.utility_weight:.0%}/{config.privacy_weight:.0%}/{config.fidelity_weight:.0%})",
            bench_dir / "composite_score",
            config,
        )
        _track("composite_score")

        if {"UtilityComponent", "PrivacyComponent", "FidelityComponent"}.issubset(composite.columns):
            comp_melt = composite.melt(
                id_vars=["Generator"],
                value_vars=["UtilityComponent", "PrivacyComponent", "FidelityComponent"],
                var_name="Component",
                value_name="Score",
            )
            _barplot(
                comp_melt,
                "Generator",
                "Score",
                "Component",
                "Composite Score Breakdown",
                bench_dir / "composite_breakdown",
                config,
            )
            _track("composite_breakdown")

    # --- Critical Difference diagrams ---
    for key in sorted(benchmarking.keys()):
        if not key.startswith("friedman_ranks_"):
            continue
        label = key.replace("friedman_ranks_", "")
        avg = benchmarking[key]
        cd = benchmarking.get(f"cd_groups_{label}", pd.DataFrame())
        if avg.empty:
            continue
        _critical_difference_diagram(avg, bench_dir / f"cd_{label}", config)
        _track(f"cd_{label}")

        if not cd.empty and "CD_Group" in cd.columns:
            _apply_style()
            fig, ax = plt.subplots(figsize=(10, 3))
            y = 0.5
            cd_val = cd["CriticalDifference"].iloc[0] if "CriticalDifference" in cd.columns else np.nan
            ax.hlines(y, cd["AverageRank"].min(), cd["AverageRank"].max(), colors="black", linewidth=2)
            if cd_val == cd_val:
                ax.axvspan(
                    cd["AverageRank"].min(),
                    cd["AverageRank"].min() + cd_val,
                    alpha=0.15,
                    color="steelblue",
                    label=f"CD = {cd_val:.2f}",
                )
            colors = plt.cm.tab10(np.linspace(0, 1, cd["CD_Group"].nunique()))
            group_colors = {g: colors[i] for i, g in enumerate(sorted(cd["CD_Group"].unique()))}
            for _, row in cd.iterrows():
                color = group_colors.get(row["CD_Group"], "gray")
                ax.plot(row["AverageRank"], y, "o", markersize=10, color=color)
                ax.text(row["AverageRank"], y + 0.12, str(row["Generator"]), ha="center", fontsize=7)
            ax.set_xlabel("Average Rank (lower is better)")
            ax.set_title(f"Nemenyi CD Diagram: {label.replace('_', ' ').title()}")
            ax.set_yticks([])
            ax.legend(loc="upper right", fontsize=8)
            save_figure(fig, bench_dir / f"cd_nemenyi_{label}", config.figure_formats, config.figure_dpi)
            _track(f"cd_nemenyi_{label}")

    # --- Domain correlations ---
    domain_corr = benchmarking.get("domain_correlations", pd.DataFrame())
    if not domain_corr.empty:
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = domain_corr["Question"].str.wrap(30)
        colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in domain_corr["Spearman_r"]]
        ax.barh(range(len(domain_corr)), domain_corr["Spearman_r"], color=colors)
        ax.set_yticks(range(len(domain_corr)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Spearman ρ")
        ax.set_title("Cross-Domain Correlations")
        save_figure(fig, bench_dir / "domain_correlations", config.figure_formats, config.figure_dpi)
        _track("domain_correlations")

    if not scores.empty:
        for x, y, name in [
            ("Fidelity", "Utility", "fidelity_vs_utility"),
            ("Privacy", "Utility", "privacy_vs_utility"),
        ]:
            valid = scores[[x, y, "Generator", "Dataset"]].dropna()
            if len(valid) >= 3:
                _scatter(valid, x, y, "Generator", f"{x} vs {y}", bench_dir / name, config)
                _track(name)

    # --- Feature-level fidelity heatmaps ---
    if isinstance(feature_fid, pd.DataFrame) and not feature_fid.empty:
        from analysis.feature_fidelity import feature_fidelity_matrix, feature_fidelity_by_dataset

        ks_matrix = feature_fidelity_matrix(feature_fid, "KS_Complement")
        if not ks_matrix.empty:
            _heatmap(ks_matrix, "Feature-Level KS Complement (avg across datasets)", bench_dir / "feature_ks_heatmap", config, annot=False)
            _track("feature_ks_heatmap")

        js_matrix = feature_fidelity_matrix(feature_fid, "JS_Divergence")
        if not js_matrix.empty:
            _apply_style()
            js_norm = js_matrix.copy()
            for col in js_norm.columns:
                vals = js_norm[col]
                vmin, vmax = vals.min(), vals.max()
                js_norm[col] = 1 - (vals - vmin) / (vmax - vmin + 1e-9)
            _heatmap(js_norm, "Feature-Level JS Divergence (inverted: greener=better)", bench_dir / "feature_js_heatmap", config, annot=False)
            _track("feature_js_heatmap")

        datasets = sorted(feature_fid["Dataset"].unique())[:6]
        for ds in datasets:
            ds_mat = feature_fidelity_by_dataset(feature_fid, ds, "KS_Complement")
            if not ds_mat.empty and ds_mat.shape[1] >= 2:
                safe = re.sub(r"[^\w]+", "_", ds)[:40]
                _heatmap(ds_mat, f"KS Complement: {ds}", bench_dir / f"feature_ks_{safe}", config, annot=False)
                _track(f"feature_ks_{safe}")

    # --- Seed stability ---
    seed_ranks = benchmarking.get("seed_stability_ranks", pd.DataFrame())
    if not seed_ranks.empty:
        _barplot(
            seed_ranks,
            "Generator",
            "StabilityScore",
            None,
            "Seed Stability (higher = lower CV across 10 seeds)",
            bench_dir / "seed_stability",
            config,
        )
        _track("seed_stability")

        cv_detail = benchmarking.get("seed_cv", pd.DataFrame())
        if not cv_detail.empty:
            cv_pivot = cv_detail.pivot_table(
                index="Generator", columns="Metric", values="Mean_CV", aggfunc="mean"
            )
            if not cv_pivot.empty:
                _heatmap(cv_pivot, "Coefficient of Variation by Metric", bench_dir / "seed_cv_heatmap", config)
                _track("seed_cv_heatmap")

    # --- Leakage sensitivity ---
    leakage_curves = benchmarking.get("leakage_curves", pd.DataFrame())
    leakage_proxy = benchmarking.get("leakage_proxy", pd.DataFrame())

    if not leakage_curves.empty:
        agg = leakage_curves.groupby(["Leakage", "Generator"])["Mean"].mean().reset_index()
        _lineplot_multi(agg, "Leakage", "Mean", "Generator", "Utility vs Leakage Level", bench_dir / "leakage_sensitivity_utility", config)
        _track("leakage_sensitivity_utility")
    elif not leakage_proxy.empty and "Retention" in leakage_proxy.columns:
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        for domain, col in [("Utility", "Utility"), ("Privacy", "Privacy"), ("Fidelity", "Fidelity")]:
            valid = leakage_proxy[[col, "Retention"]].dropna()
            if len(valid) >= 3:
                ax.scatter(valid["Retention"], valid[col], alpha=0.5, label=domain, s=30)
        ax.set_xlabel("Utility Retention (TSTR/TRTR)")
        ax.set_ylabel("Normalized Score")
        ax.set_title("Information Retention vs Evaluation Domains")
        ax.legend()
        save_figure(fig, bench_dir / "leakage_proxy_retention", config.figure_formats, config.figure_dpi)
        _track("leakage_proxy_retention")

    return saved


def _lineplot_multi(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    path: Path,
    config: PipelineConfig,
):
    if df.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, grp in df.groupby(hue):
        grp = grp.sort_values(x)
        ax.plot(grp[x], grp[y], marker="o", label=str(name))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)

