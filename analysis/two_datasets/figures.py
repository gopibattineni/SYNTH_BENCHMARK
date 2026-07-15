"""Publication figures 1–18 for Cancer & Mushroom case study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysis.benchmarking import is_pareto_optimal, nemenyi_critical_difference
from analysis.config import GENERATORS, PLOT_RC
from analysis.figures import save_figure
from analysis.two_datasets.config import DISPLAY_NAMES, FIDELITY_PLOT_METRICS, PRIVACY_PLOT_METRICS, TwoDatasetConfig

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


def _apply_style() -> None:
    for k, v in PLOT_RC.items():
        plt.rcParams[k] = v


def _pareto_line(ax, df: pd.DataFrame, x: str, y: str) -> None:
    frontier = []
    for _, row in df.iterrows():
        if is_pareto_optimal(row, df, (x, y)):
            frontier.append(row)
    if len(frontier) >= 2:
        fdf = pd.DataFrame(frontier).sort_values(x)
        ax.plot(fdf[x], fdf[y], "k--", lw=1.5, alpha=0.7, label="Pareto frontier")


def _scatter_tradeoff(
    scores: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
    tdc: TwoDatasetConfig,
    x_std: str | None = None,
    y_std: str | None = None,
) -> None:
    if scores.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    x_std = x_std or x.replace("Score", "Std")
    y_std = y_std or y.replace("Score", "Std")
    agg = scores.groupby("Generator")[[x, y] + [c for c in (x_std, y_std) if c in scores.columns]].mean().reset_index()
    for _, row in agg.iterrows():
        gen = row["Generator"]
        xe = row.get(x_std, 0) if pd.notna(row.get(x_std)) else 0
        ye = row.get(y_std, 0) if pd.notna(row.get(y_std)) else 0
        ax.errorbar(
            row[x], row[y], xerr=xe, yerr=ye,
            fmt="o", color=GENERATOR_COLORS.get(gen, "gray"), ms=9, capsize=3,
        )
        ax.annotate(gen, (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    _pareto_line(ax, agg, x, y)
    ax.set_xlabel(x.replace("Score", " Score"))
    ax.set_ylabel(y.replace("Score", " Score"))
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    save_figure(fig, path, tdc.figure_formats, tdc.figure_dpi)


def generate_dataset_figures(
    ds_key: str,
    processed: dict,
    utility: dict,
    master: dict,
    stats: dict,
    dirs: dict[str, Path],
    tdc: TwoDatasetConfig,
) -> list[str]:
    saved: list[str] = []
    fig_dir = dirs["figures"]
    display = DISPLAY_NAMES[ds_key]
    scores = processed.get("overall_scores", pd.DataFrame())
    ranking = processed.get("generator_ranking", pd.DataFrame())
    clf = utility.get("classification_stats", pd.DataFrame())
    fid_long = master.get("fidelity_long", pd.DataFrame())
    priv_long = master.get("privacy_long", pd.DataFrame())
    util_long = master.get("utility_long", pd.DataFrame())

    # Figure 1–3: Trade-off scatters
    for num, x, y, name in [
        (1, "UtilityScore", "PrivacyScore", "utility_vs_privacy"),
        (2, "UtilityScore", "FidelityScore", "utility_vs_fidelity"),
        (3, "PrivacyScore", "FidelityScore", "privacy_vs_fidelity"),
    ]:
        _scatter_tradeoff(scores, x, y, f"Figure {num}: {display} — {name.replace('_', ' ').title()}", fig_dir / f"Figure{num:02d}_{name}", tdc)
        saved.append(f"Figure{num:02d}_{name}")

    # Figure 4: Bubble
    if not scores.empty:
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        bubble = scores.groupby("Generator").agg(
            UtilityScore=("UtilityScore", "mean"),
            PrivacyScore=("PrivacyScore", "mean"),
            FidelityScore=("FidelityScore", "mean"),
        ).reset_index()
        for _, row in bubble.iterrows():
            ax.scatter(row["UtilityScore"], row["PrivacyScore"], s=300 * row["FidelityScore"] + 50,
                       c=GENERATOR_COLORS.get(row["Generator"], "gray"), edgecolors="k", lw=0.5, alpha=0.75)
            ax.annotate(row["Generator"], (row["UtilityScore"], row["PrivacyScore"]), fontsize=8)
        ax.set_xlabel("Utility Score")
        ax.set_ylabel("Privacy Score")
        ax.set_title(f"Figure 4: {display} — Bubble Chart (size = Fidelity)")
        save_figure(fig, fig_dir / "Figure04_bubble_chart", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure04_bubble_chart")

    # Figure 5: Radar
    if not ranking.empty:
        labels = ["Utility", "Fidelity", "Privacy"]
        cols = ["UtilityScore", "FidelityScore", "PrivacyScore"]
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist() + [0]
        for _, row in ranking.iterrows():
            gen = row["Generator"]
            vals = [float(row[c]) for c in cols] + [float(row[cols[0]])]
            _apply_style()
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
            ax.plot(angles, vals, color=GENERATOR_COLORS.get(gen, "gray"), lw=2)
            ax.fill(angles, vals, alpha=0.15, color=GENERATOR_COLORS.get(gen, "gray"))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title(f"Figure 5: {gen}")
            save_figure(fig, fig_dir / f"Figure05_radar_{gen}", tdc.figure_formats, tdc.figure_dpi)
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        for _, row in ranking.iterrows():
            gen = row["Generator"]
            vals = [float(row[c]) for c in cols]
            vals = vals + [vals[0]]
            ax.plot(angles, vals, label=gen, color=GENERATOR_COLORS.get(gen, "gray"))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f"Figure 5: {display} — All Generators")
        ax.legend(bbox_to_anchor=(1.2, 1), fontsize=7)
        save_figure(fig, fig_dir / "Figure05_radar_combined", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure05_radar")

    # Figure 6: Overall ranking
    if not ranking.empty:
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        r = ranking.sort_values("OverallScore")
        ax.barh(r["Generator"].astype(str), r["OverallScore"],
                color=[GENERATOR_COLORS.get(g, "gray") for g in r["Generator"]])
        ax.set_xlabel("Overall Score")
        ax.set_title(f"Figure 6: {display} — Generator Ranking (40/30/30)")
        ax.set_xlim(0, 1)
        save_figure(fig, fig_dir / "Figure06_overall_ranking", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure06_overall_ranking")

    # Figure 7: Heatmap
    if not ranking.empty:
        heat = ranking.set_index("Generator")[["UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"]]
        heat.columns = ["Utility", "Fidelity", "Privacy", "Overall"]
        _apply_style()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Figure 7: {display} — Score Heatmap")
        save_figure(fig, fig_dir / "Figure07_heatmap", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure07_heatmap")

    # Figure 8: TRTR vs TSTR
    if not clf.empty:
        trtr_acc = clf[(clf["Metric"] == "Accuracy") & (clf["EvaluationType"] == "TRTR")]
        tstr_acc = clf[(clf["Metric"] == "Accuracy") & (clf["EvaluationType"] == "TSTR")]
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
            ax.set_title(f"Figure 8: {display} — TRTR vs TSTR ({metric})")
            ax.legend()
            save_figure(fig, fig_dir / f"Figure08_trtr_tstr_{metric.lower()}", tdc.figure_formats, tdc.figure_dpi)
        if not trtr_acc.empty and not tstr_acc.empty:
            gap = (
                trtr_acc.groupby("Generator")["Mean"].mean().reset_index()
                .merge(tstr_acc.groupby("Generator")["Mean"].mean().reset_index(), on="Generator", suffixes=("_TRTR", "_TSTR"))
            )
            gap["Gap"] = gap["Mean_TRTR"] - gap["Mean_TSTR"]
            _apply_style()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(gap["Generator"], gap["Gap"], color="steelblue")
            ax.set_title(f"Figure 8: {display} — Utility Gap (TRTR − TSTR)")
            ax.tick_params(axis="x", rotation=45)
            save_figure(fig, fig_dir / "Figure08_utility_gap", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure08_trtr_tstr")

    # Figure 9: Classifier comparison
    if not clf.empty:
        for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            mdf = clf[(clf["Metric"].astype(str).str.replace("-", "") == metric.replace("-", "")) & (clf["EvaluationType"] == "TSTR")]
            if mdf.empty:
                continue
            pivot = mdf.pivot_table(index="Classifier", columns="Generator", values="Mean", aggfunc="mean")
            _apply_style()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title(f"Figure 9: {display} — {metric} by Classifier × Generator")
            save_figure(fig, fig_dir / f"Figure09_classifier_{metric.lower().replace('-','')}", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure09_classifier")

    # Figure 10: Privacy metrics
    if not priv_long.empty:
        for metric in PRIVACY_PLOT_METRICS:
            sub = priv_long[priv_long["Metric"].astype(str).str.contains(metric.split("_")[0], case=False, na=False)]
            if sub.empty:
                sub = priv_long[priv_long["Metric"] == metric]
            if sub.empty:
                continue
            agg = sub.groupby("Generator")["MetricValue" if "MetricValue" in sub.columns else "Mean"].mean().reset_index()
            agg.columns = ["Generator", "Value"]
            _apply_style()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(agg["Generator"], agg["Value"], color=[GENERATOR_COLORS.get(g, "gray") for g in agg["Generator"]])
            ax.set_title(f"Figure 10: {display} — {metric}")
            ax.tick_params(axis="x", rotation=45)
            save_figure(fig, fig_dir / f"Figure10_privacy_{metric.lower()}", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure10_privacy")

    # Figure 11: Fidelity metrics
    if not fid_long.empty:
        for metric in FIDELITY_PLOT_METRICS:
            sub = fid_long[fid_long["Metric"].astype(str).str.contains(metric.split("_")[0], case=False, na=False)]
            if sub.empty:
                sub = fid_long[fid_long["Metric"] == metric]
            if sub.empty:
                continue
            val_col = "MetricValue" if "MetricValue" in sub.columns else "Mean"
            agg = sub.groupby("Generator")[val_col].mean().reset_index(name="Value")
            _apply_style()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(agg["Generator"], agg["Value"], color=[GENERATOR_COLORS.get(g, "gray") for g in agg["Generator"]])
            ax.set_title(f"Figure 11: {display} — {metric}")
            ax.tick_params(axis="x", rotation=45)
            save_figure(fig, fig_dir / f"Figure11_fidelity_{metric.lower()}", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure11_fidelity")

    # Figure 12: Leakage
    if not scores.empty and scores["LeakageLevel"].nunique() > 1:
        for domain, col in [("Utility", "UtilityScore"), ("Privacy", "PrivacyScore"), ("Fidelity", "FidelityScore")]:
            _apply_style()
            fig, ax = plt.subplots(figsize=(9, 5))
            for gen in GENERATORS:
                sub = scores[scores["Generator"] == gen].sort_values("LeakageLevel")
                if sub.empty:
                    continue
                ax.plot(sub["LeakageLevel"], sub[col], marker="o", label=gen, color=GENERATOR_COLORS.get(gen, "gray"))
            ax.set_xlabel("Leakage Level (%)")
            ax.set_ylabel(f"{domain} Score")
            ax.set_title(f"Figure 12: {display} — Leakage vs {domain}")
            ax.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
            save_figure(fig, fig_dir / f"Figure12_leakage_{domain.lower()}", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure12_leakage")
    elif not scores.empty:
        for domain, col in [("Utility", "UtilityScore"), ("Privacy", "PrivacyScore"), ("Fidelity", "FidelityScore")]:
            _apply_style()
            fig, ax = plt.subplots(figsize=(9, 5))
            agg = scores.groupby("Generator")[col].mean().reset_index()
            ax.bar(agg["Generator"], agg[col], color=[GENERATOR_COLORS.get(g, "gray") for g in agg["Generator"]])
            ax.set_title(f"Figure 12: {display} — {domain} (baseline Leakage=0%)")
            ax.tick_params(axis="x", rotation=45)
            save_figure(fig, fig_dir / f"Figure12_{domain.lower()}_baseline", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure12_baseline")

    # Figure 13: Distributions
    if not scores.empty:
        for col, name in [("UtilityScore", "Utility"), ("FidelityScore", "Fidelity"), ("PrivacyScore", "Privacy")]:
            long = scores[["Generator", col]].rename(columns={col: "Score"})
            _apply_style()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.boxplot(data=long, x="Generator", y="Score", ax=axes[0], order=GENERATORS)
            axes[0].set_title(f"{name} Boxplot")
            axes[0].tick_params(axis="x", rotation=45)
            sns.violinplot(data=long, x="Generator", y="Score", ax=axes[1], order=GENERATORS, inner="box")
            axes[1].set_title(f"{name} Violin")
            axes[1].tick_params(axis="x", rotation=45)
            fig.suptitle(f"Figure 13: {display} — {name} Distribution")
            save_figure(fig, fig_dir / f"Figure13_{name.lower()}_distribution", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure13_distribution")

    # Figure 14: Correlations
    if not scores.empty:
        cols = ["UtilityScore", "FidelityScore", "PrivacyScore"]
        sub = scores[cols].dropna()
        if len(sub) >= 3:
            for method in ("pearson", "spearman"):
                _apply_style()
                fig, ax = plt.subplots(figsize=(5, 4))
                mat = sub.corr(method=method)
                mat.index = mat.columns = ["Utility", "Fidelity", "Privacy"]
                sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
                ax.set_title(f"Figure 14: {display} — {method.capitalize()}")
                save_figure(fig, fig_dir / f"Figure14_corr_{method}", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure14_correlation")

    # Figure 15: CD diagram
    cd_ranks = stats.get("friedman_ranks", pd.DataFrame())
    cd_groups = stats.get("cd_groups", pd.DataFrame())
    if not cd_ranks.empty:
        cd = cd_ranks["CriticalDifference"].iloc[0] if "CriticalDifference" in cd_ranks.columns else np.nan
        plot_df = cd_groups if not cd_groups.empty else cd_ranks
        _apply_style()
        fig, ax = plt.subplots(figsize=(11, 3.5))
        y = 0.5
        plot_df = plot_df.sort_values("AverageRank")
        ax.hlines(y, plot_df["AverageRank"].min() - 0.5, plot_df["AverageRank"].max() + 0.5, colors="k", lw=2)
        if cd == cd:
            ax.axvspan(plot_df["AverageRank"].min(), plot_df["AverageRank"].min() + cd, alpha=0.12, color="steelblue")
        for _, row in plot_df.iterrows():
            ax.plot(row["AverageRank"], y, "o", ms=12, color=GENERATOR_COLORS.get(row["Generator"], "gray"))
            ax.text(row["AverageRank"], y + 0.15, row["Generator"], ha="center", fontsize=8, rotation=45)
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title(f"Figure 15: {display} — Critical Difference (Friedman + Nemenyi)")
        ax.set_yticks([])
        save_figure(fig, fig_dir / "Figure15_critical_difference", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure15_cd")

    # Figure 16: Parallel coordinates
    if not ranking.empty:
        _apply_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ["Utility", "Fidelity", "Privacy", "Overall"]
        cols = ["UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"]
        x_pos = range(len(labels))
        for _, row in ranking.iterrows():
            vals = [row[c] for c in cols]
            ax.plot(x_pos, vals, marker="o", label=row["Generator"],
                    color=GENERATOR_COLORS.get(row["Generator"], "gray"), alpha=0.85)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"Figure 16: {display} — Parallel Coordinates")
        ax.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
        ax.grid(alpha=0.3)
        save_figure(fig, fig_dir / "Figure16_parallel_coordinates", tdc.figure_formats, tdc.figure_dpi)
        saved.append("Figure16_parallel")

    # Figure 18: Seed stability
    if not util_long.empty and "Std" in util_long.columns:
        tstr = util_long[util_long["EvaluationType"] == "TSTR"].copy()
        tstr = tstr[tstr["Std"].notna() & tstr["Mean"].notna() & (tstr["Mean"] != 0)]
        if not tstr.empty:
            tstr["CV"] = (tstr["Std"].abs() / tstr["Mean"].abs()).clip(0, 2)
            cv = tstr.groupby("Generator")["CV"].agg(["mean", "std"]).reset_index()
            _apply_style()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(cv["Generator"], cv["mean"], yerr=cv["std"], capsize=4,
                   color=[GENERATOR_COLORS.get(g, "gray") for g in cv["Generator"]])
            ax.set_ylabel("Coefficient of Variation (across seeds)")
            ax.set_title(f"Figure 18: {display} — Generator Stability (10 seeds)")
            ax.tick_params(axis="x", rotation=45)
            save_figure(fig, fig_dir / "Figure18_seed_stability_cv", tdc.figure_formats, tdc.figure_dpi)
            saved.append("Figure18_stability")

    return saved


def generate_comparison_figures(
    all_processed: dict[str, dict],
    dirs: dict[str, Path],
    tdc: TwoDatasetConfig,
) -> list[str]:
    """Figure 17 and cross-dataset comparison plots."""
    saved: list[str] = []
    fig_dir = dirs["figures"]

    rows = []
    for ds_key, proc in all_processed.items():
        rank = proc.get("generator_ranking", pd.DataFrame())
        if rank.empty:
            continue
        chunk = rank.copy()
        chunk["DatasetLabel"] = DISPLAY_NAMES[ds_key]
        chunk["DatasetKey"] = ds_key
        rows.append(chunk)
    if not rows:
        return saved

    combined = pd.concat(rows, ignore_index=True)

    # Figure 17: Side-by-side grouped bars
    for score_col, title in [
        ("UtilityScore", "Utility"),
        ("PrivacyScore", "Privacy"),
        ("FidelityScore", "Fidelity"),
        ("OverallScore", "Overall"),
    ]:
        _apply_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot = combined.pivot(index="Generator", columns="DatasetLabel", values=score_col)
        pivot = pivot.reindex(GENERATORS)
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel(f"{title} Score")
        ax.set_title(f"Figure 17: {title} — Cancer vs Mushroom")
        ax.set_ylim(0, 1)
        ax.legend(title="Dataset")
        ax.tick_params(axis="x", rotation=45)
        save_figure(fig, fig_dir / f"Figure17_{title.lower()}_comparison", tdc.figure_formats, tdc.figure_dpi)
        saved.append(f"Figure17_{title.lower()}")

    # Combined ranking comparison
    trade_dir = dirs.get("tradeoff", fig_dir)
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds_key in all_processed:
        rank = all_processed[ds_key].get("generator_ranking", pd.DataFrame())
        if rank.empty:
            continue
        ax.scatter(rank["UtilityScore"], rank["PrivacyScore"], label=DISPLAY_NAMES[ds_key], s=80, alpha=0.5)
        for _, row in rank.iterrows():
            ax.annotate(row["Generator"], (row["UtilityScore"], row["PrivacyScore"]), fontsize=6)
    ax.set_xlabel("Utility Score")
    ax.set_ylabel("Privacy Score")
    ax.set_title("Cancer vs Mushroom — Utility–Privacy Landscape")
    ax.legend()
    save_figure(fig, trade_dir / "comparison_utility_privacy", tdc.figure_formats, tdc.figure_dpi)
    saved.append("comparison_utility_privacy")

    return saved
