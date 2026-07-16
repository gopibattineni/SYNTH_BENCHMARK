"""Publication-quality correlation scatter figures (600 dpi)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.config import GENERATORS
from analysis.correlation_tradeoff.config import (
    DATASET_LABELS,
    DATASET_MARKERS,
    GENERATOR_COLORS,
    PLOT_RC,
    CorrelationTradeoffConfig,
)
from analysis.correlation_tradeoff.stats import (
    CorrResult,
    compute_correlation,
    regression_band,
    stats_box_text,
)
from analysis.figures import save_figure

# Distinct line colours for optional per-dataset fits (not generator colours)
_DATASET_LINE_COLORS = {
    "Cancer": "#4C4C4C",
    "Mushroom": "#8B0000",
}


def _apply_style() -> None:
    for k, v in PLOT_RC.items():
        plt.rcParams[k] = v


def _legend_handles(panel: pd.DataFrame) -> list[Line2D]:
    """Generator = colour (stable); Dataset = marker shape (○ Cancer, □ Mushroom)."""
    handles = []
    present = set(panel["Generator"].dropna().unique())
    for gen in GENERATORS:
        if gen not in present:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=GENERATOR_COLORS.get(gen, "gray"),
                markeredgecolor="0.2",
                markersize=8,
                label=gen,
            )
        )
    for ds in ("Cancer", "Mushroom"):
        if ds not in set(panel.get("DatasetKey", pd.Series(dtype=str)).dropna().unique()):
            continue
        shape = "circle" if DATASET_MARKERS[ds] == "o" else "square"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=DATASET_MARKERS[ds],
                color="0.25",
                linestyle="None",
                markersize=8,
                markerfacecolor="0.75",
                markeredgecolor="0.25",
                label=f"{DATASET_LABELS.get(ds, ds)} ({shape})",
            )
        )
    return handles


def _add_stats_box(ax, res: CorrResult) -> None:
    text = "Overall fit\n" + stats_box_text(res)
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="0.55",
            alpha=0.92,
            linewidth=0.8,
        ),
        zorder=10,
    )


def _scatter_with_regression(
    panel: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_stem: Path,
    cfg: CorrelationTradeoffConfig,
) -> CorrResult:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    x = panel[x_col].to_numpy(dtype=float)
    y = panel[y_col].to_numpy(dtype=float)
    res = compute_correlation(x, y)

    counts = panel.groupby("Generator").size()
    base_size = 42

    # Generator colour (shared across datasets) × dataset marker (○ / □)
    for gen in GENERATORS:
        for ds, marker in DATASET_MARKERS.items():
            sub = panel[(panel["Generator"] == gen) & (panel["DatasetKey"] == ds)]
            if sub.empty:
                continue
            n_pts = int(counts.get(gen, cfg.n_seeds))
            ax.scatter(
                sub[x_col],
                sub[y_col],
                c=GENERATOR_COLORS.get(gen, "gray"),
                marker=marker,
                s=base_size + 2.5 * n_pts,
                alpha=0.70,
                edgecolors="0.25",
                linewidths=0.35,
                zorder=3,
                label="_nolegend_",
            )

    # Overall regression (solid) + 95% CI across all points
    x_grid, y_hat, half = regression_band(x, y)
    if np.isfinite(y_hat).any():
        ax.plot(x_grid, y_hat, color="0.10", lw=2.0, zorder=5, label="Overall fit")
        ax.fill_between(
            x_grid,
            y_hat - half,
            y_hat + half,
            color="0.45",
            alpha=0.18,
            zorder=2,
            label="95% CI",
        )

    # Optional dataset-specific dashed regression lines
    ds_line_handles: list[Line2D] = []
    if cfg.dataset_specific_regression and "DatasetKey" in panel.columns:
        for ds in ("Cancer", "Mushroom"):
            sub = panel[panel["DatasetKey"] == ds]
            if len(sub) < 3:
                continue
            xs = sub[x_col].to_numpy(dtype=float)
            ys = sub[y_col].to_numpy(dtype=float)
            grid, y_ds, _ = regression_band(
                xs, ys, x_grid=x_grid if np.isfinite(y_hat).any() else None
            )
            if not np.isfinite(y_ds).any():
                continue
            color = _DATASET_LINE_COLORS.get(ds, "0.4")
            ax.plot(grid, y_ds, color=color, lw=1.5, ls="--", zorder=4, label=f"{ds} fit")
            ds_line_handles.append(
                Line2D([0], [0], color=color, lw=1.5, ls="--", label=f"{ds} fit")
            )

    _add_stats_box(ax, res)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)

    handles = _legend_handles(panel)
    if np.isfinite(y_hat).any():
        handles.extend(
            [
                Line2D([0], [0], color="0.10", lw=2.0, label="Overall fit"),
                Line2D([0], [0], color="0.45", lw=6, alpha=0.35, label="95% CI"),
            ]
        )
    handles.extend(ds_line_handles)

    # Horizontal legend below the axes — does not cover data points
    n_handles = len(handles)
    ncol = 4 if n_handles >= 8 else min(3, n_handles)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        framealpha=0.95,
        fontsize=7.5,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.28, left=0.12, right=0.96, top=0.92)

    save_figure(fig, out_stem, cfg.figure_formats, cfg.figure_dpi)
    return res


def generate_supplementary_figures(
    panel: pd.DataFrame,
    out_dir: Path,
    cfg: CorrelationTradeoffConfig,
) -> dict[str, CorrResult]:
    """Seed-/dataset-level scatters (variability & robustness)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, CorrResult] = {}
    if panel.empty:
        return results

    metric = panel["UtilityMetric"].iloc[0] if not panel.empty else "Accuracy"
    fid_m = panel["FidelityMetric"].iloc[0] if not panel.empty else "Quality_Score"
    fid_label = {
        "Quality_Score": "SDMetrics Quality Score",
        "Column_Shapes": "Column Shapes Score",
        "KS_Complement": "Column Shapes (KS Complement)",
    }.get(fid_m, fid_m)
    priv_label = panel["PrivacyLabel"].iloc[0] if not panel.empty else "Privacy"
    higher = bool(panel["HigherIsPrivate"].iloc[0]) if not panel.empty else True
    note = "higher is more private" if higher else "lower is more private"

    specs = [
        (
            f"FigureS01_fidelity_vs_utility_{metric}",
            "Fidelity",
            "Utility",
            f"Fidelity ({fid_label})",
            f"Mean TSTR {metric}",
            f"Supplementary: Fidelity vs Utility ({metric}; seed-level)",
        ),
        (
            f"FigureS02_privacy_vs_utility_{metric}",
            "Privacy",
            "Utility",
            f"Privacy ({priv_label}; {note})",
            f"Mean TSTR {metric}",
            f"Supplementary: Privacy vs Utility ({metric}, {priv_label}; seed-level)",
        ),
        (
            f"FigureS03_fidelity_vs_privacy_{metric}",
            "Privacy",
            "Fidelity",
            f"Privacy ({priv_label}; {note})",
            f"Fidelity ({fid_label})",
            f"Supplementary: Fidelity vs Privacy ({priv_label}; seed-level)",
        ),
    ]
    for stem_name, x_col, y_col, xlabel, ylabel, title in specs:
        res = _scatter_with_regression(
            panel,
            x_col=x_col,
            y_col=y_col,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            out_stem=out_dir / stem_name,
            cfg=cfg,
        )
        results[stem_name] = res
    return results


def _primary_label(row: pd.Series, x_col: str, y_col: str) -> str:
    """Short on-plot label: generator name only (values go in the side table)."""
    return str(row["Generator"])


def _label_contrast_color(hex_color: str) -> str:
    """Darken pale marker colours so text stays readable on white."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        # Relative luminance
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        if lum > 0.55:
            return "#222222"
        return hex_color
    except Exception:
        return "#222222"


def _fmt_metric(val: float) -> str:
    v = float(val)
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _add_values_table(ax_tab, gen_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Values table on a dedicated axes — full generator names, no clipping."""
    ax_tab.set_axis_off()
    x_tag = {"Fidelity": "F", "Utility": "U", "Privacy": "P"}.get(x_col, "X")
    y_tag = {"Fidelity": "F", "Utility": "U", "Privacy": "P"}.get(y_col, "Y")
    cell_text = []
    for gen in GENERATORS:
        sub = gen_df[gen_df["Generator"] == gen]
        if sub.empty:
            continue
        row = sub.iloc[0]
        cell_text.append(
            [str(len(cell_text) + 1), gen, _fmt_metric(row[x_col]), _fmt_metric(row[y_col])]
        )
    if not cell_text:
        return

    table = ax_tab.table(
        cellText=cell_text,
        colLabels=["#", "Generator", x_tag, y_tag],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)

    # Column width fractions within the table axes (# | name | x | y)
    col_widths = {0: 0.10, 1: 0.50, 2: 0.20, 3: 0.20}
    for (r, c), cell in table.get_celld().items():
        cell.set_width(col_widths.get(c, 0.25))
        cell.set_edgecolor("0.70")
        cell.set_linewidth(0.5)
        cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(weight="bold", color="0.15", ha="center")
        else:
            gen = cell_text[r - 1][1]
            color = GENERATOR_COLORS.get(gen, "#222")
            cell.set_facecolor("white")
            if c == 0:
                cell.set_text_props(
                    color=_label_contrast_color(color), weight="bold", ha="center"
                )
            elif c == 1:
                cell.set_text_props(
                    color=_label_contrast_color(color), ha="left", weight="medium"
                )
            else:
                cell.set_text_props(color="0.15", ha="center")


def _scatter_primary_generator(
    gen_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_stem: Path,
    cfg: CorrelationTradeoffConfig,
) -> CorrResult:
    """
    One point per generator — primary publication scatter.

    Labels are *not* drawn on the axes (they overlap with 8 nearby points).
    Instead: colour-coded markers + a side values table + bottom legend.
    """
    _apply_style()
    fig = plt.figure(figsize=(10.6, 6.2))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.15, 1.55],
        wspace=0.10,
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.20,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_tab = fig.add_subplot(gs[0, 1])

    ordered = gen_df.copy()
    ordered["Generator"] = pd.Categorical(ordered["Generator"], categories=GENERATORS, ordered=True)
    ordered = ordered.sort_values("Generator").reset_index(drop=True)

    x = ordered[x_col].to_numpy(dtype=float)
    y = ordered[y_col].to_numpy(dtype=float)
    res = compute_correlation(x, y)

    # Draw markers with index drawn *inside* the marker (no overlapping callouts)
    for i, row in ordered.iterrows():
        gen = row["Generator"]
        xi, yi = float(row[x_col]), float(row[y_col])
        color = GENERATOR_COLORS.get(gen, "gray")
        ax.scatter(
            xi,
            yi,
            c=color,
            marker="o",
            s=280,
            alpha=0.92,
            edgecolors="0.1",
            linewidths=1.0,
            zorder=3,
        )
        # White text on dark markers, dark text on pale ones
        try:
            h = color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        except Exception:
            lum = 0.4
        txt_color = "#111111" if lum > 0.55 else "#FFFFFF"
        ax.text(
            xi,
            yi,
            str(int(i) + 1),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=txt_color,
            zorder=4,
        )

    x_grid, y_hat, half = regression_band(x, y)
    if np.isfinite(y_hat).any():
        ax.plot(x_grid, y_hat, color="0.10", lw=2.0, zorder=4, label="OLS fit")
        ax.fill_between(
            x_grid,
            y_hat - half,
            y_hat + half,
            color="0.45",
            alpha=0.18,
            zorder=2,
            label="95% CI",
        )

    _add_stats_box(ax, res)
    _add_values_table(ax_tab, ordered, x_col, y_col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)

    xpad = 0.05 * (np.nanmax(x) - np.nanmin(x) + 1e-9)
    ypad = 0.07 * (np.nanmax(y) - np.nanmin(y) + 1e-9)
    ax.set_xlim(np.nanmin(x) - xpad, np.nanmax(x) + xpad)
    ax.set_ylim(np.nanmin(y) - ypad, np.nanmax(y) + ypad)

    present = [g for g in GENERATORS if g in set(ordered["Generator"])]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GENERATOR_COLORS.get(g, "gray"),
            markeredgecolor="0.2",
            markersize=8,
            label=f"{i+1}. {g}",
        )
        for i, g in enumerate(present)
    ]
    if np.isfinite(y_hat).any():
        handles.extend(
            [
                Line2D([0], [0], color="0.10", lw=2.0, label="OLS fit"),
                Line2D([0], [0], color="0.45", lw=6, alpha=0.35, label="95% CI"),
            ]
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        framealpha=0.95,
        fontsize=7.5,
    )

    save_figure(fig, out_stem, cfg.figure_formats, cfg.figure_dpi)
    return res


def generate_primary_figures(
    gen_df: pd.DataFrame,
    out_dir: Path,
    cfg: CorrelationTradeoffConfig,
) -> dict[str, CorrResult]:
    """Eight-point generator-mean scatters — primary publication figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, CorrResult] = {}
    if gen_df.empty:
        return results

    metric = gen_df["UtilityMetric"].iloc[0]
    fid_m = gen_df["FidelityMetric"].iloc[0]
    fid_label = {
        "Quality_Score": "SDMetrics Quality Score",
        "Column_Shapes": "Column Shapes Score",
        "KS_Complement": "Column Shapes (KS Complement)",
    }.get(fid_m, fid_m)
    priv_label = gen_df["PrivacyLabel"].iloc[0]
    higher = bool(gen_df["HigherIsPrivate"].iloc[0])
    note = "higher is more private" if higher else "lower is more private"

    specs = [
        (
            f"Figure01_fidelity_vs_utility_{metric}",
            "Fidelity",
            "Utility",
            f"Fidelity ({fid_label})",
            f"Mean TSTR {metric}",
            f"Fidelity vs Utility ({metric})",
        ),
        (
            f"Figure02_privacy_vs_utility_{metric}",
            "Privacy",
            "Utility",
            f"Privacy ({priv_label}; {note})",
            f"Mean TSTR {metric}",
            f"Privacy vs Utility ({metric}, {priv_label})",
        ),
        (
            f"Figure03_fidelity_vs_privacy_{metric}",
            "Privacy",
            "Fidelity",
            f"Privacy ({priv_label}; {note})",
            f"Fidelity ({fid_label})",
            f"Fidelity vs Privacy ({priv_label})",
        ),
    ]
    for stem_name, x_col, y_col, xlabel, ylabel, title in specs:
        res = _scatter_primary_generator(
            gen_df,
            x_col=x_col,
            y_col=y_col,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            out_stem=out_dir / stem_name,
            cfg=cfg,
        )
        results[stem_name] = res
    return results


# Backwards-compatible alias
def generate_figures(
    panel: pd.DataFrame,
    out_dir: Path,
    cfg: CorrelationTradeoffConfig,
) -> dict[str, CorrResult]:
    return generate_supplementary_figures(panel, out_dir, cfg)
