"""Publication figures for representative-metrics dual trade-off analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysis.benchmarking import friedman_average_ranks, nemenyi_significance_groups
from analysis.config import GENERATORS, PLOT_RC
from analysis.figure_tables import (
    add_side_values_table,
    draw_numbered_marker,
    export_values_csv,
    make_plot_with_table,
    order_generator_dataset,
)
from analysis.figures import save_figure
from analysis.representative_tradeoff.config import DATASET_LABELS, RepresentativeTradeoffConfig

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

DATASET_MARKERS = {"Cancer": "o", "Mushroom": "s"}


def _apply_style() -> None:
    for k, v in PLOT_RC.items():
        plt.rcParams[k] = v


def _pareto_xy(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    pts = df[[x, y]].dropna().copy().reset_index(drop=True)
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
    return pts.loc[keep].sort_values(x)


def _caption_suffix(analysis: str) -> str:
    if analysis == "Best_Classifier":
        return " (Supplementary — best classifier)"
    return " (Primary — classifier-independent)"


def _rep_side_table(
    ax_tab,
    ordered: pd.DataFrame,
    metric_cols: list[tuple[str, str]],
) -> None:
    cell_text: list[list[str]] = []
    for _, row in ordered.iterrows():
        entry = [
            str(int(row["PointId"])),
            str(row["Generator"]),
            DATASET_LABELS.get(str(row["DatasetKey"]), str(row["DatasetKey"])),
        ]
        for col, _short in metric_cols:
            val = row.get(col)
            entry.append(f"{float(val):.3f}" if pd.notna(val) else "—")
        cell_text.append(entry)
    col_labels = ["#", "Generator", "Dataset"] + [s for _, s in metric_cols]
    add_side_values_table(ax_tab, cell_text, col_labels, generator_colors=GENERATOR_COLORS)


def figure01_utility_privacy(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty:
        return None
    _apply_style()
    ordered = order_generator_dataset(scores, generators=GENERATORS)
    fig, ax, ax_tab = make_plot_with_table()
    for _, row in ordered.iterrows():
        gen, ds = str(row["Generator"]), str(row["DatasetKey"])
        color = GENERATOR_COLORS.get(gen, "gray")
        x, y = float(row["F1"]), float(row["NNDR"])
        if pd.notna(row.get("F1_CI_Low")):
            xerr = [[x - float(row["F1_CI_Low"])], [float(row["F1_CI_High"]) - x]]
            ax.errorbar(
                x, y, xerr=xerr, fmt="none", ecolor=color, capsize=3, lw=1.0, zorder=2,
            )
        draw_numbered_marker(
            ax, x, y, int(row["PointId"]),
            color=color, marker=DATASET_MARKERS.get(ds, "o"), size=200,
        )
    frontier = _pareto_xy(ordered, "F1", "NNDR")
    if len(frontier) >= 2:
        ax.plot(frontier["F1"], frontier["NNDR"], "k--", lw=1.5, alpha=0.75, label="Pareto frontier")
    _rep_side_table(ax_tab, ordered, [("F1", "F1"), ("NNDR", "NNDR")])
    ax.set_xlabel("Utility — Mean F1-score")
    ax.set_ylabel("Privacy — Mean NNDR (Avg NN distance; higher = more private)")
    ax.set_title(f"Utility vs Privacy{_caption_suffix(analysis)}")
    ax.grid(alpha=0.3)
    _dual_legend(ax, ordered, show_pareto=len(frontier) >= 2)
    export_values_csv(
        ordered[["PointId", "Generator", "DatasetKey", "F1", "NNDR"]].rename(
            columns={"PointId": "#", "DatasetKey": "Dataset"}
        ),
        out.parent / "Tables" / "Figure01_utility_vs_privacy_values.csv",
        round_cols=["F1", "NNDR"],
    )
    save_figure(fig, out / "Figure01_utility_vs_privacy", cfg.figure_formats, cfg.figure_dpi)
    return "Figure01_utility_vs_privacy"


def figure02_utility_fidelity(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty:
        return None
    _apply_style()
    ordered = order_generator_dataset(scores, generators=GENERATORS)
    fig, ax, ax_tab = make_plot_with_table()
    for _, row in ordered.iterrows():
        gen, ds = str(row["Generator"]), str(row["DatasetKey"])
        draw_numbered_marker(
            ax, float(row["F1"]), float(row["Quality"]), int(row["PointId"]),
            color=GENERATOR_COLORS.get(gen, "gray"),
            marker=DATASET_MARKERS.get(ds, "o"), size=200,
        )
    frontier = _pareto_xy(ordered.dropna(subset=["F1", "Quality"]), "F1", "Quality")
    if len(frontier) >= 2:
        ax.plot(frontier["F1"], frontier["Quality"], "k--", lw=1.5, alpha=0.75, label="Pareto frontier")
    _rep_side_table(ax_tab, ordered, [("F1", "F1"), ("Quality", "Q")])
    ax.set_xlabel("Utility — Mean F1-score")
    ax.set_ylabel("Fidelity — SDMetrics Quality Score")
    ax.set_title(f"Utility vs Fidelity{_caption_suffix(analysis)}")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    _dual_legend(ax, ordered, show_pareto=len(frontier) >= 2)
    export_values_csv(
        ordered[["PointId", "Generator", "DatasetKey", "F1", "Quality"]].rename(
            columns={"PointId": "#", "DatasetKey": "Dataset"}
        ),
        out.parent / "Tables" / "Figure02_utility_vs_fidelity_values.csv",
        round_cols=["F1", "Quality"],
    )
    save_figure(fig, out / "Figure02_utility_vs_fidelity", cfg.figure_formats, cfg.figure_dpi)
    return "Figure02_utility_vs_fidelity"


def figure03_privacy_fidelity(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty:
        return None
    _apply_style()
    ordered = order_generator_dataset(scores, generators=GENERATORS)
    fig, ax, ax_tab = make_plot_with_table()
    for _, row in ordered.iterrows():
        gen, ds = str(row["Generator"]), str(row["DatasetKey"])
        draw_numbered_marker(
            ax, float(row["NNDR"]), float(row["Quality"]), int(row["PointId"]),
            color=GENERATOR_COLORS.get(gen, "gray"),
            marker=DATASET_MARKERS.get(ds, "o"), size=200,
        )
    frontier = _pareto_xy(ordered.dropna(subset=["NNDR", "Quality"]), "NNDR", "Quality")
    if len(frontier) >= 2:
        ax.plot(frontier["NNDR"], frontier["Quality"], "k--", lw=1.5, alpha=0.75, label="Pareto frontier")
    _rep_side_table(ax_tab, ordered, [("NNDR", "NNDR"), ("Quality", "Q")])
    ax.set_xlabel("Privacy — Mean NNDR (higher = more private)")
    ax.set_ylabel("Fidelity — SDMetrics Quality Score")
    ax.set_title(f"Privacy vs Fidelity{_caption_suffix(analysis)}")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    _dual_legend(ax, ordered, show_pareto=len(frontier) >= 2)
    export_values_csv(
        ordered[["PointId", "Generator", "DatasetKey", "NNDR", "Quality"]].rename(
            columns={"PointId": "#", "DatasetKey": "Dataset"}
        ),
        out.parent / "Tables" / "Figure03_privacy_vs_fidelity_values.csv",
        round_cols=["NNDR", "Quality"],
    )
    save_figure(fig, out / "Figure03_privacy_vs_fidelity", cfg.figure_formats, cfg.figure_dpi)
    return "Figure03_privacy_vs_fidelity"


def figure04_bubble(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty:
        return None
    _apply_style()
    ordered = order_generator_dataset(scores, generators=GENERATORS)
    fig, ax, ax_tab = make_plot_with_table()
    for _, row in ordered.iterrows():
        gen, ds = str(row["Generator"]), str(row["DatasetKey"])
        q = float(row["Quality"]) if pd.notna(row["Quality"]) else 0.5
        size = 120 + 380 * q
        draw_numbered_marker(
            ax, float(row["F1"]), float(row["NNDR"]), int(row["PointId"]),
            color=GENERATOR_COLORS.get(gen, "gray"),
            marker=DATASET_MARKERS.get(ds, "o"), size=size, alpha=0.82,
        )
    _rep_side_table(ax_tab, ordered, [("F1", "F1"), ("NNDR", "NNDR"), ("Quality", "Q")])
    ax.set_xlabel("Mean F1-score")
    ax.set_ylabel("Mean NNDR")
    ax.set_title(f"Bubble Trade-off (size = Quality){_caption_suffix(analysis)}")
    ax.grid(alpha=0.3)
    _dual_legend(ax, ordered, size_note=True)
    export_values_csv(
        ordered[["PointId", "Generator", "DatasetKey", "F1", "NNDR", "Quality"]].rename(
            columns={"PointId": "#", "DatasetKey": "Dataset"}
        ),
        out.parent / "Tables" / "Figure04_bubble_chart_values.csv",
        round_cols=["F1", "NNDR", "Quality"],
    )
    save_figure(fig, out / "Figure04_bubble_chart", cfg.figure_formats, cfg.figure_dpi)
    return "Figure04_bubble_chart"


def figure05_radar(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty or "NNDR_Norm" not in scores.columns:
        return None
    gens = [g for g in GENERATORS if g in set(scores["Generator"])]
    labels = ["F1", "NNDR", "Quality"]
    cols = ["F1_Norm", "NNDR_Norm", "Quality_Norm"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]
    ncols, nrows = 4, int(np.ceil(len(gens) / 4))
    _apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.3 * nrows), subplot_kw={"polar": True})
    axes = np.atleast_1d(axes).ravel()
    styles = {"Cancer": "-", "Mushroom": "--"}
    alphas = {"Cancer": 0.30, "Mushroom": 0.18}

    for i, gen in enumerate(gens):
        ax = axes[i]
        color = GENERATOR_COLORS.get(gen, "gray")
        for ds in ("Cancer", "Mushroom"):
            sub = scores[(scores["Generator"] == gen) & (scores["DatasetKey"] == ds)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            vals = [float(row[c]) for c in cols] + [float(row[cols[0]])]
            ax.plot(angles, vals, color=color, lw=2, ls=styles[ds], label=ds)
            ax.fill(angles, vals, color=color, alpha=alphas[ds])
            # numeric labels near axes
            for ang, val, lab in zip(angles[:-1], vals[:-1], labels):
                ax.text(ang, min(val + 0.08, 1.05), f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(gen, color=color, pad=14, fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2), fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Radar — F1 / NNDR / Quality{_caption_suffix(analysis)}", y=1.02)
    fig.tight_layout()
    save_figure(fig, out / "Figure05_radar", cfg.figure_formats, cfg.figure_dpi)
    return "Figure05_radar"


def figure06_ranking(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty:
        return None
    metrics = [("F1", "Mean F1"), ("NNDR", "Mean NNDR"), ("Quality", "Quality Score")]
    gens = [g for g in GENERATORS if g in set(scores["Generator"])]
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    y = np.arange(len(gens))
    h = 0.28

    for ax, (col, title) in zip(axes, metrics):
        cancer = scores[scores["DatasetKey"] == "Cancer"].set_index("Generator").reindex(gens)
        mush = scores[scores["DatasetKey"] == "Mushroom"].set_index("Generator").reindex(gens)
        c_vals = cancer[col].astype(float).values
        m_vals = mush[col].astype(float).values
        avg = np.nanmean(np.vstack([c_vals, m_vals]), axis=0)

        ax.barh(y - h, c_vals, h, label="Cancer", color="#4C78A8", edgecolor="k", lw=0.3)
        ax.barh(y, m_vals, h, label="Mushroom", color="#F58518", edgecolor="k", lw=0.3)
        ax.barh(y + h, avg, h, label="Average", color="#54A24B", edgecolor="k", lw=0.3)

        for i in range(len(gens)):
            for offset, val in ((-h, c_vals[i]), (0, m_vals[i]), (h, avg[i])):
                if np.isfinite(val):
                    ax.text(val + 0.01 * (np.nanmax(np.concatenate([c_vals, m_vals])) or 1),
                            i + offset, f"{val:.3f}", va="center", fontsize=6)
        ax.set_yticks(y)
        ax.set_yticklabels(gens)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Overall Ranking — Cancer / Mushroom / Average{_caption_suffix(analysis)}")
    fig.tight_layout()
    save_figure(fig, out / "Figure06_overall_ranking", cfg.figure_formats, cfg.figure_dpi)
    return "Figure06_overall_ranking"


def figure07_heatmap(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty:
        return None
    rows = []
    for gen in GENERATORS:
        if gen not in set(scores["Generator"]):
            continue
        rec = {"Generator": gen}
        for ds, prefix in (("Cancer", "Cancer"), ("Mushroom", "Mushroom")):
            sub = scores[(scores["Generator"] == gen) & (scores["DatasetKey"] == ds)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rec[f"{prefix} F1"] = float(r["F1"])
            rec[f"{prefix} NNDR"] = float(r["NNDR"]) if pd.notna(r["NNDR"]) else np.nan
            rec[f"{prefix} Quality"] = float(r["Quality"]) if pd.notna(r["Quality"]) else np.nan
        rows.append(rec)
    heat = pd.DataFrame(rows).set_index("Generator")
    cols = [c for c in (
        "Cancer F1", "Cancer NNDR", "Cancer Quality",
        "Mushroom F1", "Mushroom NNDR", "Mushroom Quality",
    ) if c in heat.columns]
    heat = heat[cols]

    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Scale NNDR columns separately for color; annotate with raw values
    display = heat.copy()
    for c in display.columns:
        if "NNDR" in c:
            v = display[c]
            display[c] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0.5
    sns.heatmap(display, annot=heat, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                linewidths=0.4, cbar_kws={"label": "Normalized (NNDR min–max within column)"})
    ax.set_title(f"Heatmap — F1 / NNDR / Quality{_caption_suffix(analysis)}")
    plt.xticks(rotation=30, ha="right")
    save_figure(fig, out / "Figure07_heatmap", cfg.figure_formats, cfg.figure_dpi)
    return "Figure07_heatmap"


def figure08_parallel(scores: pd.DataFrame, out: Path, cfg: RepresentativeTradeoffConfig, analysis: str) -> str | None:
    if scores.empty or "NNDR_Norm" not in scores.columns:
        return None
    axes_labels = [
        "Cancer F1", "Cancer NNDR", "Cancer Quality",
        "Mushroom F1", "Mushroom NNDR", "Mushroom Quality",
    ]
    records = []
    for gen in GENERATORS:
        if gen not in set(scores["Generator"]):
            continue
        vals = []
        ok = True
        for ds, col in (
            ("Cancer", "F1_Norm"), ("Cancer", "NNDR_Norm"), ("Cancer", "Quality_Norm"),
            ("Mushroom", "F1_Norm"), ("Mushroom", "NNDR_Norm"), ("Mushroom", "Quality_Norm"),
        ):
            sub = scores[(scores["Generator"] == gen) & (scores["DatasetKey"] == ds)]
            if sub.empty:
                ok = False
                break
            vals.append(float(sub.iloc[0][col]))
        if ok:
            records.append((gen, vals))
    if not records:
        return None

    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(axes_labels))
    for gen, vals in records:
        ax.plot(x, vals, marker="o", color=GENERATOR_COLORS.get(gen, "gray"), label=gen, lw=2)
        for xi, vi in zip(x, vals):
            ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=6, color=GENERATOR_COLORS.get(gen, "gray"))
    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels, rotation=25, ha="right")
    ax.set_ylim(-0.02, 1.12)
    ax.axvline(2.5, color="gray", ls=":", lw=1.2)
    ax.set_ylabel("Normalized score")
    ax.set_title(f"Parallel Coordinates{_caption_suffix(analysis)}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out / "Figure08_parallel_coordinates", cfg.figure_formats, cfg.figure_dpi)
    return "Figure08_parallel_coordinates"


def figure_cd(
    util_long: pd.DataFrame,
    scores: pd.DataFrame,
    out: Path,
    cfg: RepresentativeTradeoffConfig,
    analysis: str,
) -> tuple[str | None, pd.DataFrame, pd.DataFrame]:
    """Friedman + Nemenyi CD using Dataset×Classifier blocks (or generators on F1)."""
    ranks = pd.DataFrame()
    if util_long is not None and not util_long.empty and analysis == "Classifier_Independent":
        tstr = util_long[
            (util_long["EvaluationType"] == "TSTR")
            & (util_long["Metric"] == "F1")
            & util_long["Classifier"].notna()
        ].copy()
        if not tstr.empty:
            tstr["Block"] = tstr["DatasetKey"].astype(str) + "|" + tstr["Classifier"].astype(str)
            # For classifier-independent: average F1 already across classifiers in scores;
            # CD uses raw per-classifier F1 so rankings reflect classifier robustness.
            agg = tstr.groupby(["Block", "Generator"])["Mean"].mean().reset_index()
            ranks = friedman_average_ranks(agg, value_col="Mean", block_col="Block", treatment_col="Generator")
    elif not scores.empty:
        # Best-classifier: use F1 with Dataset as block
        ranks = friedman_average_ranks(
            scores, value_col="F1", block_col="DatasetKey", treatment_col="Generator"
        )

    if ranks.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    cd_groups = nemenyi_significance_groups(ranks)
    plot_df = cd_groups if not cd_groups.empty else ranks
    cd = float(ranks["CriticalDifference"].iloc[0])
    n_blocks = int(ranks["N_Datasets"].iloc[0])

    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = 0.5
    plot_df = plot_df.sort_values("AverageRank")
    ax.hlines(y, plot_df["AverageRank"].min() - 0.4, plot_df["AverageRank"].max() + 0.4, colors="k", lw=2)
    xmin = float(plot_df["AverageRank"].min())
    if np.isfinite(cd):
        ax.axvspan(xmin, xmin + cd, alpha=0.12, color="steelblue")
        ax.annotate("", xy=(xmin + cd, y - 0.22), xytext=(xmin, y - 0.22),
                    arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.5))
        ax.text(xmin + cd / 2, y - 0.32, f"CD = {cd:.2f}", ha="center", fontsize=8, color="steelblue")
    if "CD_Group" in plot_df.columns:
        for _, grp in plot_df.groupby("CD_Group"):
            if len(grp) >= 2:
                ax.plot([grp["AverageRank"].min(), grp["AverageRank"].max()],
                        [y + 0.28, y + 0.28], color="0.3", lw=3, solid_capstyle="round")
    for _, row in plot_df.iterrows():
        gen = row["Generator"]
        ax.plot(row["AverageRank"], y, "o", ms=12, color=GENERATOR_COLORS.get(gen, "gray"),
                markeredgecolor="k", markeredgewidth=0.5)
        ax.text(row["AverageRank"], y + 0.12, gen, ha="center", fontsize=8, rotation=35)
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(f"Critical Difference Diagram{_caption_suffix(analysis)} — {n_blocks} blocks")
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    save_figure(fig, out / "Figure_CD_critical_difference", cfg.figure_formats, cfg.figure_dpi)
    return "Figure_CD_critical_difference", ranks, cd_groups


def _dual_legend(ax, scores: pd.DataFrame, size_note: bool = False, show_pareto: bool = False) -> None:
    gens = [g for g in GENERATORS if g in set(scores["Generator"])]
    gen_h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GENERATOR_COLORS[g],
               markeredgecolor="k", markersize=8, label=g)
        for g in gens
    ]
    ds_h = [
        Line2D([0], [0], marker=DATASET_MARKERS[d], color="k", linestyle="None", markersize=8, label=DATASET_LABELS[d])
        for d in ("Cancer", "Mushroom") if d in set(scores["DatasetKey"])
    ]
    if show_pareto:
        ds_h.append(
            Line2D([0], [0], color="k", linestyle="--", linewidth=1.8, label="Pareto frontier")
        )
    leg1 = ax.legend(handles=gen_h, title="Generator", loc="lower left", fontsize=7, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=ds_h, title="Dataset / frontier", loc="lower right", fontsize=8, framealpha=0.9)


def generate_analysis_figures(
    scores: pd.DataFrame,
    dirs: dict[str, Path],
    cfg: RepresentativeTradeoffConfig,
    analysis: str,
    util_long: pd.DataFrame | None = None,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    saved: list[str] = []
    extras: dict[str, pd.DataFrame] = {}
    fig_dir = dirs["figures"]
    for fn in (
        figure01_utility_privacy,
        figure02_utility_fidelity,
        figure03_privacy_fidelity,
        figure04_bubble,
        figure05_radar,
        figure06_ranking,
        figure07_heatmap,
        figure08_parallel,
    ):
        name = fn(scores, fig_dir, cfg, analysis)
        if name:
            saved.append(name)

    name, ranks, cd = figure_cd(util_long if util_long is not None else pd.DataFrame(),
                                scores, fig_dir, cfg, analysis)
    if name:
        saved.append(name)
    if not ranks.empty:
        extras["friedman_ranks"] = ranks
    if not cd.empty:
        extras["cd_groups"] = cd
    return saved, extras
