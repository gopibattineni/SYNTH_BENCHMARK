"""Publication figures 1–14 for journal paper (300 dpi, multi-format)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysis.benchmarking import is_pareto_optimal, nemenyi_critical_difference
from analysis.config import GENERATORS, PLOT_RC, PipelineConfig
from analysis.figures import save_figure

# Consistent generator colors and leakage markers
GENERATOR_COLORS = {
    "CTGAN": "#1f77b4",
    "CopulaGAN": "#ff7f0e",
    "TVAE": "#2ca02c",
    "GaussianCopula": "#d62728",
    "WGAN_GP": "#9467bd",
    "CTABGAN": "#8c564b",
    "TabDDPM": "#e377c2",
    "ForestDiffusion": "#7f7f7f",
}

LEAKAGE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _apply_style() -> None:
    for key, val in PLOT_RC.items():
        plt.rcParams[key] = val


def _pareto_frontier_2d(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Non-dominated points (maximize x and y)."""
    if df.empty:
        return df
    frontier = []
    for _, row in df.iterrows():
        if is_pareto_optimal(row, df, (x, y)):
            frontier.append(row)
    return pd.DataFrame(frontier)


def _draw_tradeoff_scatter(
    scores: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
    config: PipelineConfig,
    fig_num: int,
) -> None:
    if scores.empty or x not in scores.columns or y not in scores.columns:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    leak_levels = sorted(scores["LeakageLevel"].dropna().unique())
    x_std = x.replace("Score", "Std")
    y_std = y.replace("Score", "Std")
    agg_cols = [x, y]
    if x_std in scores.columns:
        agg_cols.append(x_std)
    if y_std in scores.columns:
        agg_cols.append(y_std)
    std_agg = scores.groupby(["Generator", "LeakageLevel"], dropna=False)[agg_cols].mean().reset_index()

    for i, leakage in enumerate(leak_levels):
        sub = std_agg[std_agg["LeakageLevel"] == leakage]
        marker = LEAKAGE_MARKERS[i % len(LEAKAGE_MARKERS)]
        for _, row in sub.iterrows():
            gen = row["Generator"]
            color = GENERATOR_COLORS.get(gen, "gray")
            xerr = row.get(x_std, 0) if pd.notna(row.get(x_std)) else 0
            yerr = row.get(y_std, 0) if pd.notna(row.get(y_std)) else 0
            ax.errorbar(
                row[x], row[y], xerr=xerr, yerr=yerr,
                fmt=marker, color=color, markersize=8, capsize=3, alpha=0.85,
            )
            ax.annotate(
                gen, (row[x], row[y]), textcoords="offset points",
                xytext=(4, 4), fontsize=6, color=color,
            )

    # Pareto frontier (aggregate across leakage per generator)
    gen_agg = scores.groupby("Generator")[[x, y]].mean().reset_index()
    frontier = _pareto_frontier_2d(gen_agg, x, y)
    if len(frontier) >= 2:
        frontier = frontier.sort_values(x)
        ax.plot(frontier[x], frontier[y], "k--", linewidth=1.5, alpha=0.7, label="Pareto frontier")

    ax.set_xlabel(x.replace("Score", " Score"))
    ax.set_ylabel(y.replace("Score", " Score"))
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    gen_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENERATOR_COLORS.get(g, "gray"),
               markersize=8, label=g)
        for g in GENERATORS if g in scores["Generator"].values
    ]
    leak_handles = [
        Line2D([0], [0], marker=LEAKAGE_MARKERS[i % len(LEAKAGE_MARKERS)], color="gray",
               linestyle="None", markersize=8, label=f"Leakage {lv}%")
        for i, lv in enumerate(leak_levels)
    ]
    ax.legend(handles=gen_handles + leak_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
    save_figure(fig, path, config.figure_formats, config.figure_dpi)


def generate_publication_figures(
    results: dict[str, Any],
    dirs: dict[str, Path],
    config: PipelineConfig,
) -> list[str]:
    """Generate all 14 publication figures."""
    saved: list[str] = []
    scores = results.get("processed", {}).get("overall_scores", pd.DataFrame())
    ranking = results.get("processed", {}).get("generator_ranking", pd.DataFrame())
    utility = results.get("utility", {})
    clf = utility.get("classification_stats", pd.DataFrame())
    reg = utility.get("regression_stats", pd.DataFrame())
    correlation = results.get("correlation", {})
    benchmarking = results.get("benchmarking", {})
    utility_long = results.get("master").utility_long if results.get("master") else pd.DataFrame()

    trade_dir = dirs["figures_tradeoff"]
    util_dir = dirs["figures_utility"]
    stat_dir = dirs["figures_statistical"]
    leak_dir = dirs["figures_leakage"]
    fid_dir = dirs["figures_fidelity"]
    priv_dir = dirs["figures_privacy"]

    # Figure 1–3: Trade-off scatter plots
    if not scores.empty:
        _draw_tradeoff_scatter(
            scores, "UtilityScore", "PrivacyScore",
            "Utility vs Privacy", trade_dir / "Figure01_utility_vs_privacy", config, 1,
        )
        saved.append("Figure01_utility_vs_privacy")
        _draw_tradeoff_scatter(
            scores, "UtilityScore", "FidelityScore",
            "Utility vs Fidelity", trade_dir / "Figure02_utility_vs_fidelity", config, 2,
        )
        saved.append("Figure02_utility_vs_fidelity")
        _draw_tradeoff_scatter(
            scores, "PrivacyScore", "FidelityScore",
            "Privacy vs Fidelity", trade_dir / "Figure03_privacy_vs_fidelity", config, 3,
        )
        saved.append("Figure03_privacy_vs_fidelity")

        # Figure 4: Bubble chart
        _apply_style()
        fig, ax = plt.subplots(figsize=(9, 7))
        bubble = scores.groupby("Generator").agg(
            UtilityScore=("UtilityScore", "mean"),
            PrivacyScore=("PrivacyScore", "mean"),
            FidelityScore=("FidelityScore", "mean"),
        ).reset_index()
        for _, row in bubble.iterrows():
            ax.scatter(
                row["UtilityScore"], row["PrivacyScore"],
                s=300 * row["FidelityScore"] + 50,
                c=GENERATOR_COLORS.get(row["Generator"], "gray"),
                alpha=0.7, edgecolors="black", linewidth=0.5,
            )
            ax.annotate(row["Generator"], (row["UtilityScore"], row["PrivacyScore"]), fontsize=8)
        ax.set_xlabel("Utility Score")
        ax.set_ylabel("Privacy Score")
        ax.set_title("Bubble Chart (size = Fidelity Score)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        save_figure(fig, trade_dir / "Figure04_bubble_chart", config.figure_formats, config.figure_dpi)
        saved.append("Figure04_bubble_chart")

        # Figure 5: Radar charts
        radar_cols = ["UtilityScore", "FidelityScore", "PrivacyScore"]
        labels = ["Utility", "Fidelity", "Privacy"]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        if not ranking.empty:
            for _, row in ranking.iterrows():
                gen = row["Generator"]
                vals = [row["UtilityScore"], row["FidelityScore"], row["PrivacyScore"]]
                vals = [float(v) if pd.notna(v) else 0 for v in vals]
                vals += vals[:1]
                _apply_style()
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
                ax.plot(angles, vals, color=GENERATOR_COLORS.get(gen, "gray"), linewidth=2)
                ax.fill(angles, vals, alpha=0.2, color=GENERATOR_COLORS.get(gen, "gray"))
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels)
                ax.set_title(f"{gen}")
                save_figure(fig, trade_dir / f"Figure05_radar_{gen}", config.figure_formats, config.figure_dpi)
            saved.append("Figure05_radar_per_generator")

            # Combined radar
            _apply_style()
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
            for _, row in ranking.iterrows():
                gen = row["Generator"]
                vals = [row["UtilityScore"], row["FidelityScore"], row["PrivacyScore"]]
                vals = [float(v) if pd.notna(v) else 0 for v in vals]
                vals += vals[:1]
                ax.plot(angles, vals, label=gen, color=GENERATOR_COLORS.get(gen, "gray"))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title("All Generators")
            ax.legend(bbox_to_anchor=(1.2, 1), fontsize=7)
            save_figure(fig, trade_dir / "Figure05_radar_combined", config.figure_formats, config.figure_dpi)
            saved.append("Figure05_radar_combined")

        # Figure 6: Overall ranking horizontal bar
        if not ranking.empty:
            _apply_style()
            fig, ax = plt.subplots(figsize=(8, 5))
            r = ranking.dropna(subset=["Generator", "OverallScore"]).copy()
            r["Generator"] = r["Generator"].astype(str)
            r = r.sort_values("OverallScore")
            colors = [GENERATOR_COLORS.get(g, "gray") for g in r["Generator"]]
            ax.barh(r["Generator"].tolist(), r["OverallScore"].tolist(), color=colors)
            ax.set_xlabel("Overall Score")
            ax.set_title("Overall Generator Ranking")
            ax.set_xlim(0, 1)
            save_figure(fig, stat_dir / "Figure06_overall_ranking", config.figure_formats, config.figure_dpi)
            saved.append("Figure06_overall_ranking")

        # Figure 7: Heatmap
        heat = ranking.set_index("Generator")[
            ["UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"]
        ]
        heat.columns = ["Utility", "Fidelity", "Privacy", "Overall"]
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
        ax.set_title("Generator Score Heatmap")
        save_figure(fig, stat_dir / "Figure07_score_heatmap", config.figure_formats, config.figure_dpi)
        saved.append("Figure07_score_heatmap")

    # Figure 8: Leakage line plots
    if not scores.empty and scores["LeakageLevel"].nunique() > 1:
        for domain, col in [("Utility", "UtilityScore"), ("Privacy", "PrivacyScore"), ("Fidelity", "FidelityScore")]:
            _apply_style()
            fig, ax = plt.subplots(figsize=(9, 5))
            for gen in GENERATORS:
                sub = scores[scores["Generator"] == gen].sort_values("LeakageLevel")
                if sub.empty:
                    continue
                ax.plot(sub["LeakageLevel"], sub[col], marker="o", label=gen,
                        color=GENERATOR_COLORS.get(gen, "gray"))
            ax.set_xlabel("Leakage Level (%)")
            ax.set_ylabel(f"{domain} Score")
            ax.set_title(f"Leakage vs {domain}")
            ax.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
            save_figure(fig, leak_dir / f"Figure08_leakage_vs_{domain.lower()}", config.figure_formats, config.figure_dpi)
            saved.append(f"Figure08_leakage_vs_{domain.lower()}")
    elif not scores.empty:
        # Single leakage level — still export placeholder structure
        for domain, col in [("Utility", "UtilityScore"), ("Privacy", "PrivacyScore"), ("Fidelity", "FidelityScore")]:
            _apply_style()
            fig, ax = plt.subplots(figsize=(9, 5))
            agg = scores.groupby("Generator")[col].mean().reset_index()
            ax.bar(agg["Generator"], agg[col], color=[GENERATOR_COLORS.get(g, "gray") for g in agg["Generator"]])
            ax.set_ylabel(f"{domain} Score")
            ax.set_title(f"{domain} by Generator (Leakage=0% baseline)")
            ax.tick_params(axis="x", rotation=45)
            save_figure(fig, leak_dir / f"Figure08_{domain.lower()}_baseline", config.figure_formats, config.figure_dpi)
            saved.append(f"Figure08_{domain.lower()}_baseline")

    # Figure 9: TRTR vs TSTR
    if not clf.empty:
        for metric in ["Accuracy", "F1"]:
            mdf = clf[clf["Metric"] == metric]
            trtr = mdf[mdf["EvaluationType"] == "TRTR"].groupby("Generator")["Mean"].mean().reset_index()
            tstr = mdf[mdf["EvaluationType"] == "TSTR"].groupby("Generator")["Mean"].mean().reset_index()
            plot = trtr.merge(tstr, on="Generator", suffixes=("_TRTR", "_TSTR"))
            if plot.empty:
                continue
            _apply_style()
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(plot))
            w = 0.35
            ax.bar(x - w / 2, plot["Mean_TRTR"], w, label="TRTR")
            ax.bar(x + w / 2, plot["Mean_TSTR"], w, label="TSTR")
            ax.set_xticks(x)
            ax.set_xticklabels(plot["Generator"], rotation=45, ha="right")
            ax.set_title(f"TRTR vs TSTR ({metric})")
            ax.legend()
            save_figure(fig, util_dir / f"Figure09_trtr_tstr_{metric.lower()}", config.figure_formats, config.figure_dpi)
            saved.append(f"Figure09_trtr_tstr_{metric.lower()}")

        gap = clf[(clf["Metric"] == "Accuracy") & (clf["EvaluationType"] == "Gap")]
        if gap.empty:
            trtr = clf[(clf["Metric"] == "Accuracy") & (clf["EvaluationType"] == "TRTR")]
            tstr = clf[(clf["Metric"] == "Accuracy") & (clf["EvaluationType"] == "TSTR")]
            gap = trtr.merge(tstr, on=["Dataset", "Generator", "Classifier"], suffixes=("_TRTR", "_TSTR"))
            gap["Mean"] = gap["Mean_TRTR"] - gap["Mean_TSTR"]
        gap_agg = gap.groupby("Generator")["Mean"].mean().reset_index()
        _apply_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(gap_agg["Generator"], gap_agg["Mean"], color="steelblue")
        ax.set_title("Utility Gap (TRTR − TSTR)")
        ax.tick_params(axis="x", rotation=45)
        save_figure(fig, util_dir / "Figure09_utility_gap", config.figure_formats, config.figure_dpi)
        saved.append("Figure09_utility_gap")

    # Figure 10: Classifier performance
    if not clf.empty:
        for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            mdf = clf[(clf["Metric"].astype(str).str.replace("-", "") == metric.replace("-", "")) & (clf["EvaluationType"] == "TSTR")]
            if mdf.empty:
                continue
            pivot = mdf.pivot_table(index="Classifier", columns="Generator", values="Mean", aggfunc="mean")
            if pivot.empty:
                continue
            _apply_style()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title(f"{metric} by Classifier × Generator")
            save_figure(fig, util_dir / f"Figure10_classifier_{metric.lower().replace('-','')}", config.figure_formats, config.figure_dpi)
            saved.append(f"Figure10_classifier_{metric.lower()}")

    # Figure 11: Distribution boxplots / violins
    if not scores.empty:
        for col, name, out_dir in [
            ("UtilityScore", "Utility", util_dir),
            ("FidelityScore", "Fidelity", fid_dir),
            ("PrivacyScore", "Privacy", priv_dir),
        ]:
            if col not in scores.columns:
                continue
            long = scores[["Generator", col]].rename(columns={col: "Score"})
            _apply_style()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.boxplot(data=long, x="Generator", y="Score", ax=axes[0], order=GENERATORS)
            axes[0].set_title(f"{name} Boxplot")
            axes[0].tick_params(axis="x", rotation=45)
            sns.violinplot(data=long, x="Generator", y="Score", ax=axes[1], order=GENERATORS, inner="box")
            axes[1].set_title(f"{name} Violin")
            axes[1].tick_params(axis="x", rotation=45)
            fig.suptitle(f"{name} Distribution Across Generators")
            save_figure(fig, out_dir / f"Figure11_{name.lower()}_distribution", config.figure_formats, config.figure_dpi)
            saved.append(f"Figure11_{name.lower()}_distribution")

    # Figure 12: Correlation heatmaps
    if not scores.empty:
        corr_cols = ["UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"]
        avail = [c for c in corr_cols if c in scores.columns]
        if len(avail) >= 2:
            sub = scores[avail].dropna()
            for method in ("pearson", "spearman"):
                _apply_style()
                fig, ax = plt.subplots(figsize=(6, 5))
                mat = sub.corr(method=method)
                mat.index = mat.columns = [c.replace("Score", "") for c in mat.columns]
                sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
                ax.set_title(f"{method.capitalize()} Correlation")
                save_figure(fig, stat_dir / f"Figure12_corr_{method}", config.figure_formats, config.figure_dpi)
                saved.append(f"Figure12_corr_{method}")

    # Figure 13: Critical Difference diagram
    cd_ranks = benchmarking.get("friedman_ranks_composite", pd.DataFrame())
    if cd_ranks.empty:
        cd_ranks = benchmarking.get("friedman_ranks_utility_accuracy", pd.DataFrame())
    if not cd_ranks.empty:
        cd = cd_ranks["CriticalDifference"].iloc[0] if "CriticalDifference" in cd_ranks.columns else nemenyi_critical_difference(8, cd_ranks["N_Datasets"].iloc[0] if "N_Datasets" in cd_ranks.columns else 9)
        cd_groups = benchmarking.get("cd_groups_composite", benchmarking.get("cd_groups_utility_accuracy", pd.DataFrame()))
        plot_df = cd_groups if not cd_groups.empty else cd_ranks
        _apply_style()
        fig, ax = plt.subplots(figsize=(11, 3.5))
        y = 0.5
        plot_df = plot_df.sort_values("AverageRank")
        ax.hlines(y, plot_df["AverageRank"].min() - 0.5, plot_df["AverageRank"].max() + 0.5, colors="black", lw=2)
        if cd == cd:
            ax.axvspan(plot_df["AverageRank"].min(), plot_df["AverageRank"].min() + cd, alpha=0.12, color="steelblue")
        for _, row in plot_df.iterrows():
            ax.plot(row["AverageRank"], y, "o", ms=12, color=GENERATOR_COLORS.get(row["Generator"], "gray"))
            ax.text(row["AverageRank"], y + 0.15, row["Generator"], ha="center", fontsize=8, rotation=45)
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title("Critical Difference Diagram (Friedman + Nemenyi)")
        ax.set_yticks([])
        save_figure(fig, stat_dir / "Figure13_critical_difference", config.figure_formats, config.figure_dpi)
        saved.append("Figure13_critical_difference")

    # Figure 14: Parallel coordinates
    if not ranking.empty:
        _apply_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        cols = ["UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"]
        labels = ["Utility", "Fidelity", "Privacy", "Overall"]
        x_pos = range(len(labels))
        for _, row in ranking.iterrows():
            vals = [row[c] for c in cols]
            ax.plot(x_pos, vals, marker="o", label=row["Generator"],
                    color=GENERATOR_COLORS.get(row["Generator"], "gray"), alpha=0.85)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title("Parallel Coordinates")
        ax.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
        ax.grid(True, alpha=0.3)
        save_figure(fig, trade_dir / "Figure14_parallel_coordinates", config.figure_formats, config.figure_dpi)
        saved.append("Figure14_parallel_coordinates")

    return saved
