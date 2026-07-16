"""Comparative publication figures for Cancer vs Mushroom (all generators together)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from analysis.config import GENERATORS, PLOT_RC
from analysis.figures import save_figure
from analysis.two_datasets.config import TwoDatasetConfig

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

DATASET_MARKERS = {
    "Cancer": "o",
    "Mushroom": "s",
}

DATASET_LABELS = {
    "Cancer": "Cancer",
    "Mushroom": "Mushroom",
}


def _apply_style() -> None:
    for k, v in PLOT_RC.items():
        plt.rcParams[k] = v


def _combined_ranking(all_processed: dict[str, dict]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ds_key, proc in all_processed.items():
        rank = proc.get("generator_ranking", pd.DataFrame())
        if rank.empty:
            continue
        chunk = rank.copy()
        chunk["DatasetKey"] = ds_key
        chunk["DatasetLabel"] = DATASET_LABELS.get(ds_key, ds_key)
        rows.append(chunk)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for col in ("UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"):
        if col in out.columns:
            out[col] = out[col].clip(0, 1)
    return out


def _pareto_frontier_xy(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Pareto frontier maximizing both x and y (all points compared)."""
    if df.empty:
        return pd.DataFrame()
    pts = df[[x, y]].dropna().copy()
    pts = pts.reset_index(drop=False).rename(columns={"index": "_idx"})
    keep = []
    for i, row in pts.iterrows():
        dominated = False
        for j, other in pts.iterrows():
            if i == j:
                continue
            if other[x] >= row[x] and other[y] >= row[y] and (
                other[x] > row[x] or other[y] > row[y]
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    frontier = pts.loc[keep].sort_values(x)
    return frontier


def _confidence_ellipse(ax, x: float, y: float, x_std: float, y_std: float, color: str) -> None:
    """Axis-aligned 1-sigma ellipse from seed-derived stds (≈10 seeds)."""
    if not (np.isfinite(x_std) and np.isfinite(y_std)):
        return
    if x_std <= 0 and y_std <= 0:
        return
    width = max(2.0 * float(x_std), 0.02)
    height = max(2.0 * float(y_std), 0.02)
    ell = Ellipse(
        (x, y),
        width=width,
        height=height,
        facecolor=color,
        edgecolor=color,
        alpha=0.12,
        lw=0.8,
        zorder=1,
    )
    ax.add_patch(ell)


def figure01_tradeoff_scatter(
    combined: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
) -> str | None:
    if combined.empty:
        return None
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in combined.iterrows():
        gen = row["Generator"]
        ds = row["DatasetKey"]
        color = GENERATOR_COLORS.get(gen, "gray")
        marker = DATASET_MARKERS.get(ds, "o")
        x_std = float(row["UtilityStd"]) if pd.notna(row.get("UtilityStd")) else 0.0
        y_std = float(row["PrivacyStd"]) if pd.notna(row.get("PrivacyStd")) else 0.0
        _confidence_ellipse(ax, row["UtilityScore"], row["PrivacyScore"], x_std, y_std, color)
        ax.scatter(
            row["UtilityScore"],
            row["PrivacyScore"],
            c=color,
            marker=marker,
            s=110,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        u = float(row["UtilityScore"])
        p = float(row["PrivacyScore"])
        ax.annotate(
            f"{gen}\n({u:.2f}, {p:.2f})",
            (u, p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=color,
            ha="left",
            va="bottom",
        )

    frontier = _pareto_frontier_xy(combined, "UtilityScore", "PrivacyScore")
    if len(frontier) >= 2:
        ax.plot(
            frontier["UtilityScore"],
            frontier["PrivacyScore"],
            "k--",
            lw=1.6,
            alpha=0.75,
            label="Pareto frontier",
            zorder=2,
        )

    gen_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENERATOR_COLORS[g],
               markeredgecolor="k", markersize=8, label=g)
        for g in GENERATORS
        if g in set(combined["Generator"])
    ]
    ds_handles = [
        Line2D([0], [0], marker=DATASET_MARKERS[d], color="k", linestyle="None",
               markersize=8, label=DATASET_LABELS[d])
        for d in ("Cancer", "Mushroom")
        if d in set(combined["DatasetKey"])
    ]
    leg1 = ax.legend(handles=gen_handles, title="Generator", loc="lower left", fontsize=7, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles + ([Line2D([0], [0], color="k", ls="--", label="Pareto frontier")] if len(frontier) >= 2 else []),
              title="Dataset", loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Normalized Utility Score")
    ax.set_ylabel("Normalized Privacy Score")
    ax.set_title("Utility–Privacy Trade-off (Cancer vs Mushroom)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    save_figure(fig, out_dir / "Figure01_utility_privacy_scatter", tdc.figure_formats, tdc.figure_dpi)
    return "Figure01_utility_privacy_scatter"


def figure02_bubble(
    combined: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
) -> str | None:
    if combined.empty:
        return None
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in combined.iterrows():
        gen = row["Generator"]
        ds = row["DatasetKey"]
        size = 80 + 420 * float(row["FidelityScore"])
        ax.scatter(
            row["UtilityScore"],
            row["PrivacyScore"],
            s=size,
            c=GENERATOR_COLORS.get(gen, "gray"),
            marker=DATASET_MARKERS.get(ds, "o"),
            alpha=0.75,
            edgecolors="black",
            linewidths=0.6,
        )
        u = float(row["UtilityScore"])
        p = float(row["PrivacyScore"])
        f = float(row["FidelityScore"])
        ax.annotate(
            f"{gen} ({DATASET_LABELS.get(ds, ds)})\n"
            f"U={u:.2f}  P={p:.2f}  F={f:.2f}",
            (u, p),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            ha="left",
            va="bottom",
        )

    gen_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENERATOR_COLORS[g],
               markeredgecolor="k", markersize=8, label=g)
        for g in GENERATORS if g in set(combined["Generator"])
    ]
    ds_handles = [
        Line2D([0], [0], marker=DATASET_MARKERS[d], color="k", linestyle="None",
               markersize=8, label=DATASET_LABELS[d])
        for d in ("Cancer", "Mushroom") if d in set(combined["DatasetKey"])
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=ms, label=lab)
        for ms, lab in ((6, "Fidelity ≈ 0.2"), (10, "Fidelity ≈ 0.5"), (14, "Fidelity ≈ 0.8"))
    ]
    leg1 = ax.legend(handles=gen_handles, title="Generator", loc="lower left", fontsize=7)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=ds_handles, title="Dataset", loc="upper left", fontsize=8)
    ax.add_artist(leg2)
    ax.legend(handles=size_handles, title="Bubble size", loc="lower right", fontsize=7)

    ax.set_xlabel("Utility Score")
    ax.set_ylabel("Privacy Score")
    ax.set_title("Bubble Trade-off (size = Fidelity)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    save_figure(fig, out_dir / "Figure02_bubble_tradeoff", tdc.figure_formats, tdc.figure_dpi)
    return "Figure02_bubble_tradeoff"


def figure03_radar(
    combined: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
) -> str | None:
    if combined.empty:
        return None
    gens = [g for g in GENERATORS if g in set(combined["Generator"])]
    if not gens:
        return None

    labels = ["Utility", "Fidelity", "Privacy"]
    cols = ["UtilityScore", "FidelityScore", "PrivacyScore"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    n = len(gens)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    _apply_style()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), subplot_kw={"polar": True}
    )
    axes = np.atleast_1d(axes).ravel()

    ds_styles = {
        "Cancer": {"ls": "-", "alpha": 0.35},
        "Mushroom": {"ls": "--", "alpha": 0.25},
    }

    for i, gen in enumerate(gens):
        ax = axes[i]
        color = GENERATOR_COLORS.get(gen, "gray")
        for ds_key in ("Cancer", "Mushroom"):
            sub = combined[(combined["Generator"] == gen) & (combined["DatasetKey"] == ds_key)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            vals = [float(row[c]) for c in cols] + [float(row[cols[0]])]
            style = ds_styles[ds_key]
            ax.plot(angles, vals, color=color, lw=2, ls=style["ls"], label=DATASET_LABELS[ds_key])
            ax.fill(angles, vals, color=color, alpha=style["alpha"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(gen, color=color, pad=12, fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Radar Comparison — Cancer vs Mushroom per Generator", y=1.02, fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir / "Figure03_radar_comparison", tdc.figure_formats, tdc.figure_dpi)
    return "Figure03_radar_comparison"


def figure04_ranking(
    combined: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
) -> str | None:
    if combined.empty:
        return None
    pivot = combined.pivot(index="Generator", columns="DatasetKey", values="OverallScore")
    for col in ("Cancer", "Mushroom"):
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["Average"] = pivot[["Cancer", "Mushroom"]].mean(axis=1)
    pivot = pivot.reindex([g for g in GENERATORS if g in pivot.index])
    pivot = pivot.sort_values("Average", ascending=True)

    # Rank numbers (1 = best) by average overall score
    rank_map = {
        gen: i
        for i, gen in enumerate(
            pivot.sort_values("Average", ascending=False).index.tolist(), start=1
        )
    }

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(pivot))
    h = 0.35
    ax.barh(y - h / 2, pivot["Cancer"], h, label="Cancer", color="#4C78A8", edgecolor="k", lw=0.4)
    ax.barh(y + h / 2, pivot["Mushroom"], h, label="Mushroom", color="#F58518", edgecolor="k", lw=0.4)

    for i, gen in enumerate(pivot.index):
        c_val = pivot.loc[gen, "Cancer"]
        m_val = pivot.loc[gen, "Mushroom"]
        if pd.notna(c_val):
            ax.text(c_val + 0.01, i - h / 2, f"{c_val:.2f}", va="center", fontsize=7)
        if pd.notna(m_val):
            ax.text(m_val + 0.01, i + h / 2, f"{m_val:.2f}", va="center", fontsize=7)
        ax.text(
            -0.02,
            i,
            f"#{rank_map[gen]}",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            transform=ax.get_yaxis_transform(),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Overall Score (0.4·Utility + 0.3·Fidelity + 0.3·Privacy)")
    ax.set_xlim(0, 1.15)
    ax.set_title("Overall Ranking Comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, out_dir / "Figure04_overall_ranking", tdc.figure_formats, tdc.figure_dpi)
    return "Figure04_overall_ranking"


def figure05_heatmap(
    combined: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
) -> str | None:
    if combined.empty:
        return None

    rows = []
    for gen in GENERATORS:
        if gen not in set(combined["Generator"]):
            continue
        rec: dict[str, float] = {"Generator": gen}
        for ds_key, prefix in (("Cancer", "Cancer"), ("Mushroom", "Mushroom")):
            sub = combined[(combined["Generator"] == gen) & (combined["DatasetKey"] == ds_key)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rec[f"{prefix} Utility"] = float(r["UtilityScore"])
            rec[f"{prefix} Fidelity"] = float(r["FidelityScore"])
            rec[f"{prefix} Privacy"] = float(r["PrivacyScore"])
        rows.append(rec)

    heat = pd.DataFrame(rows).set_index("Generator")
    col_order = [
        "Cancer Utility",
        "Cancer Fidelity",
        "Cancer Privacy",
        "Mushroom Utility",
        "Mushroom Fidelity",
        "Mushroom Privacy",
    ]
    heat = heat[[c for c in col_order if c in heat.columns]].clip(0, 1)

    if len(heat) >= 2:
        try:
            Z = linkage(pdist(heat.fillna(heat.mean()).values, metric="euclidean"), method="average")
            order = leaves_list(Z)
            heat = heat.iloc[order]
        except Exception:
            pass

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.4,
        cbar_kws={"label": "Normalized score"},
    )
    ax.set_title("Score Heatmap (rows clustered)")
    ax.set_xlabel("")
    ax.set_ylabel("Generator")
    plt.xticks(rotation=30, ha="right")
    save_figure(fig, out_dir / "Figure05_heatmap", tdc.figure_formats, tdc.figure_dpi)
    return "Figure05_heatmap"


def figure06_parallel(
    combined: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
) -> str | None:
    if combined.empty:
        return None

    axes_labels = [
        "Cancer Utility",
        "Cancer Fidelity",
        "Cancer Privacy",
        "Mushroom Utility",
        "Mushroom Fidelity",
        "Mushroom Privacy",
    ]
    records = []
    for gen in GENERATORS:
        if gen not in set(combined["Generator"]):
            continue
        vals = []
        ok = True
        for ds_key, metric in (
            ("Cancer", "UtilityScore"),
            ("Cancer", "FidelityScore"),
            ("Cancer", "PrivacyScore"),
            ("Mushroom", "UtilityScore"),
            ("Mushroom", "FidelityScore"),
            ("Mushroom", "PrivacyScore"),
        ):
            sub = combined[(combined["Generator"] == gen) & (combined["DatasetKey"] == ds_key)]
            if sub.empty:
                ok = False
                break
            vals.append(float(sub.iloc[0][metric]))
        if ok:
            records.append((gen, vals))

    if not records:
        return None

    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(axes_labels))
    for gen, vals in records:
        ax.plot(
            x,
            vals,
            marker="o",
            color=GENERATOR_COLORS.get(gen, "gray"),
            label=gen,
            lw=2,
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels, rotation=25, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Normalized score")
    ax.set_title("Parallel Coordinates — Consistency across Datasets")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    # Vertical separator between datasets
    ax.axvline(2.5, color="gray", ls=":", lw=1.2, alpha=0.8)
    save_figure(fig, out_dir / "Figure06_parallel_coordinates", tdc.figure_formats, tdc.figure_dpi)
    return "Figure06_parallel_coordinates"


def figure07_stability(
    util_long: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
    dataset_ids: dict[str, str],
) -> tuple[str | None, pd.DataFrame]:
    """Error-bar stability plot from seed Mean/Std (≈10 seeds)."""
    if util_long.empty or "Std" not in util_long.columns:
        return None, pd.DataFrame()

    tstr = util_long[
        (util_long["EvaluationType"] == "TSTR")
        & util_long["Std"].notna()
        & util_long["Mean"].notna()
        & (util_long["Mean"] != 0)
    ].copy()
    if tstr.empty:
        return None, pd.DataFrame()

    # Prefer Accuracy for classification datasets
    if "Metric" in tstr.columns:
        acc = tstr[tstr["Metric"].astype(str).str.replace("-", "") == "Accuracy"]
        if not acc.empty:
            tstr = acc

    id_to_key = {v: k for k, v in dataset_ids.items()}
    tstr["DatasetKey"] = tstr["Dataset"].map(id_to_key)
    tstr = tstr[tstr["DatasetKey"].notna()].copy()
    tstr["CV"] = (tstr["Std"].abs() / tstr["Mean"].abs()).clip(0, 2)

    stats = (
        tstr.groupby(["DatasetKey", "Generator"], dropna=False)
        .agg(
            Mean=("Mean", "mean"),
            Std=("Std", "mean"),
            CV=("CV", "mean"),
        )
        .reset_index()
    )
    if stats.empty:
        return None, pd.DataFrame()

    gens = [g for g in GENERATORS if g in set(stats["Generator"])]
    x = np.arange(len(gens))
    width = 0.35

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Mean ± Std
    ax = axes[0]
    for offset, ds_key, color in ((-width / 2, "Cancer", "#4C78A8"), (width / 2, "Mushroom", "#F58518")):
        sub = stats[stats["DatasetKey"] == ds_key].set_index("Generator").reindex(gens)
        means = sub["Mean"].fillna(0).values
        stds = sub["Std"].fillna(0).values
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=DATASET_LABELS[ds_key],
            color=color,
            edgecolor="k",
            lw=0.4,
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=40, ha="right")
    ax.set_ylabel("TSTR Accuracy (mean ± std across seeds)")
    ax.set_title("Seed Stability — Mean ± Std")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: Coefficient of variation
    ax = axes[1]
    for offset, ds_key, color in ((-width / 2, "Cancer", "#4C78A8"), (width / 2, "Mushroom", "#F58518")):
        sub = stats[stats["DatasetKey"] == ds_key].set_index("Generator").reindex(gens)
        cvs = sub["CV"].fillna(0).values
        ax.bar(
            x + offset,
            cvs,
            width,
            label=DATASET_LABELS[ds_key],
            color=color,
            edgecolor="k",
            lw=0.4,
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=40, ha="right")
    ax.set_ylabel("Coefficient of Variation (Std / Mean)")
    ax.set_title("Seed Stability — CV (lower = more robust)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Stability Analysis (10 random seeds)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir / "Figure07_stability_errorbars", tdc.figure_formats, tdc.figure_dpi)
    return "Figure07_stability_errorbars", stats


def figure08_critical_difference(
    combined: pd.DataFrame,
    out_dir: Path,
    tdc: TwoDatasetConfig,
    util_long: pd.DataFrame | None = None,
) -> tuple[str | None, pd.DataFrame, pd.DataFrame]:
    """Friedman + Nemenyi CD diagram using both datasets together.

    Prefer Dataset×Classifier blocks on TSTR Accuracy (more statistical power).
    Fall back to OverallScore with the two datasets as blocks.
    """
    from analysis.benchmarking import friedman_average_ranks, nemenyi_significance_groups

    ranks = pd.DataFrame()
    if util_long is not None and not util_long.empty:
        tstr = util_long[
            (util_long["EvaluationType"] == "TSTR")
            & (util_long["Metric"].astype(str).str.replace("-", "") == "Accuracy")
            & util_long["Classifier"].notna()
        ].copy()
        if not tstr.empty:
            tstr["Block"] = tstr["Dataset"].astype(str) + "|" + tstr["Classifier"].astype(str)
            agg = (
                tstr.groupby(["Block", "Generator"], dropna=False)["Mean"]
                .mean()
                .reset_index()
            )
            ranks = friedman_average_ranks(
                agg, value_col="Mean", block_col="Block", treatment_col="Generator"
            )

    if ranks.empty and not combined.empty and "OverallScore" in combined.columns:
        ranks = friedman_average_ranks(
            combined,
            value_col="OverallScore",
            block_col="DatasetKey",
            treatment_col="Generator",
        )

    if ranks.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    cd_groups = nemenyi_significance_groups(ranks)
    plot_df = cd_groups if not cd_groups.empty else ranks
    cd = float(ranks["CriticalDifference"].iloc[0]) if "CriticalDifference" in ranks.columns else np.nan
    n_blocks = int(ranks["N_Datasets"].iloc[0]) if "N_Datasets" in ranks.columns else 0

    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = 0.5
    plot_df = plot_df.sort_values("AverageRank")
    ax.hlines(
        y,
        plot_df["AverageRank"].min() - 0.4,
        plot_df["AverageRank"].max() + 0.4,
        colors="black",
        lw=2,
    )
    if np.isfinite(cd):
        xmin = float(plot_df["AverageRank"].min())
        ax.axvspan(xmin, xmin + cd, alpha=0.12, color="steelblue")
        ax.annotate(
            "",
            xy=(xmin + cd, y - 0.22),
            xytext=(xmin, y - 0.22),
            arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.5),
        )
        ax.text(xmin + cd / 2, y - 0.32, f"CD = {cd:.2f}", ha="center", fontsize=8, color="steelblue")

    if "CD_Group" in plot_df.columns and np.isfinite(cd):
        for _, grp in plot_df.groupby("CD_Group"):
            if len(grp) < 2:
                continue
            ax.plot(
                [grp["AverageRank"].min(), grp["AverageRank"].max()],
                [y + 0.28, y + 0.28],
                color="0.3",
                lw=3,
                solid_capstyle="round",
            )

    for _, row in plot_df.iterrows():
        gen = row["Generator"]
        ax.plot(
            row["AverageRank"],
            y,
            "o",
            ms=12,
            color=GENERATOR_COLORS.get(gen, "gray"),
            markeredgecolor="k",
            markeredgewidth=0.6,
        )
        ax.text(row["AverageRank"], y + 0.12, gen, ha="center", fontsize=8, rotation=35)

    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(
        f"Critical Difference (Friedman + Nemenyi; {n_blocks} blocks across Cancer & Mushroom)"
    )
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    save_figure(fig, out_dir / "Figure08_critical_difference", tdc.figure_formats, tdc.figure_dpi)
    return "Figure08_critical_difference", ranks, cd_groups


def generate_comparison_figures(
    all_processed: dict[str, dict],
    dirs: dict[str, Path],
    tdc: TwoDatasetConfig,
    util_long: pd.DataFrame | None = None,
    dataset_ids: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Generate comparative Figures 1–8 (Cancer & Mushroom together) + side-by-side bars."""
    saved: list[str] = []
    extras: dict[str, pd.DataFrame] = {}
    fig_dir = dirs["figures"]
    combined = _combined_ranking(all_processed)
    if combined.empty:
        return saved, extras

    for fn in (
        figure01_tradeoff_scatter,
        figure02_bubble,
        figure03_radar,
        figure04_ranking,
        figure05_heatmap,
        figure06_parallel,
    ):
        name = fn(combined, fig_dir, tdc)
        if name:
            saved.append(name)

    if util_long is not None and dataset_ids is not None:
        name, stab = figure07_stability(util_long, fig_dir, tdc, dataset_ids)
        if name:
            saved.append(name)
        if not stab.empty:
            extras["stability_stats"] = stab

    name, ranks, cd_groups = figure08_critical_difference(
        combined, fig_dir, tdc, util_long=util_long
    )
    if name:
        saved.append(name)
    if not ranks.empty:
        extras["friedman_ranks"] = ranks
    if not cd_groups.empty:
        extras["cd_groups"] = cd_groups

    # Side-by-side Cancer vs Mushroom grouped bars (former Figure 17)
    # Keep legacy Comparison/Figures path used in the case-study tree
    legacy_fig_dir = tdc.output_root / "Comparison" / "Figures"
    legacy_fig_dir.mkdir(parents=True, exist_ok=True)
    short_labels = {"Cancer": "Wisconsin", "Mushroom": "Secondary"}
    for score_col, title in (
        ("UtilityScore", "Utility"),
        ("PrivacyScore", "Privacy"),
        ("FidelityScore", "Fidelity"),
        ("OverallScore", "Overall"),
    ):
        if score_col not in combined.columns or "DatasetKey" not in combined.columns:
            continue
        _apply_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_df = combined.copy()
        plot_df["DatasetLabel"] = plot_df["DatasetKey"].map(short_labels).fillna(plot_df["DatasetKey"])
        pivot = plot_df.pivot_table(
            index="Generator", columns="DatasetLabel", values=score_col, aggfunc="mean"
        )
        pivot = pivot.reindex([g for g in GENERATORS if g in pivot.index])
        cols = [c for c in ("Wisconsin", "Secondary") if c in pivot.columns] + [
            c for c in pivot.columns if c not in ("Wisconsin", "Secondary")
        ]
        pivot = pivot[cols]
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel(f"{title} Score")
        ax.set_xlabel("Generator")
        ax.set_title(f"{title} — Cancer vs Mushroom")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Dataset")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        stem = f"Figure17_{title.lower()}_comparison"
        save_figure(fig, legacy_fig_dir / stem, tdc.figure_formats, tdc.figure_dpi)
        # Mirror into primary Figures/ folder
        for ext in tdc.figure_formats:
            src = legacy_fig_dir / f"{stem}.{ext}"
            dst = fig_dir / f"{stem}.{ext}"
            if src.is_file():
                dst.write_bytes(src.read_bytes())
        saved.append(stem)

    extras["combined_scores"] = combined
    return saved, extras


def generate_dataset_figures(
    ds_key: str,
    processed: dict,
    utility: dict,
    master: dict,
    stats: dict,
    dirs: dict[str, Path],
    tdc: TwoDatasetConfig,
) -> list[str]:
    """Deprecated: per-dataset plots are not generated (comparative figures only)."""
    return []
