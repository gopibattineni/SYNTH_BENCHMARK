"""Shared helpers: numbered markers + side values table (no overlapping callouts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def marker_text_color(hex_color: str) -> str:
    """White text on dark markers, dark text on pale ones."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    except Exception:
        lum = 0.4
    return "#111111" if lum > 0.55 else "#FFFFFF"


def table_text_color(hex_color: str) -> str:
    """Keep generator colour readable on a white table cell."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        if lum > 0.55:
            return "#222222"
        return hex_color
    except Exception:
        return "#222222"


def order_generator_dataset(
    df: pd.DataFrame,
    *,
    generators: list[str],
    generator_col: str = "Generator",
    dataset_col: str = "DatasetKey",
    dataset_order: tuple[str, ...] = ("Cancer", "Mushroom"),
) -> pd.DataFrame:
    """Stable point order: generator list × dataset order, with PointId 1..N."""
    out = df.copy()
    present_ds = [d for d in dataset_order if d in set(out[dataset_col].astype(str))]
    out[generator_col] = pd.Categorical(
        out[generator_col], categories=generators, ordered=True
    )
    if present_ds:
        out[dataset_col] = pd.Categorical(
            out[dataset_col], categories=present_ds, ordered=True
        )
        out = out.sort_values([generator_col, dataset_col]).reset_index(drop=True)
    else:
        out = out.sort_values([generator_col]).reset_index(drop=True)
    out["PointId"] = np.arange(1, len(out) + 1)
    return out


def add_side_values_table(
    ax_tab,
    cell_text: list[list[str]],
    col_labels: list[str],
    *,
    generator_col_idx: int = 1,
    generator_colors: dict[str, str] | None = None,
    fontsize: float = 7,
    scale_y: float = 1.28,
) -> None:
    """Draw a values table on a dedicated axes (full names, no clipping)."""
    ax_tab.set_axis_off()
    if not cell_text:
        return
    n_cols = len(col_labels)
    table = ax_tab.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, scale_y)

    # Heuristic widths: # narrow, name wider, metrics equal
    if n_cols <= 4:
        widths = {0: 0.10, 1: 0.40}
        rem = (1.0 - 0.50) / max(n_cols - 2, 1)
        for c in range(2, n_cols):
            widths[c] = rem
    elif n_cols == 5:
        widths = {0: 0.09, 1: 0.34, 2: 0.19, 3: 0.19, 4: 0.19}
    elif n_cols == 6:
        widths = {0: 0.08, 1: 0.30, 2: 0.16, 3: 0.15, 4: 0.15, 5: 0.16}
    else:
        name_w = 0.28
        rem = (1.0 - 0.08 - name_w) / max(n_cols - 2, 1)
        widths = {0: 0.08, 1: name_w}
        for c in range(2, n_cols):
            widths[c] = rem

    colors = generator_colors or {}
    for (r, c), cell in table.get_celld().items():
        cell.set_width(widths.get(c, 1.0 / n_cols))
        cell.set_edgecolor("0.70")
        cell.set_linewidth(0.45)
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(weight="bold", color="0.15", ha="center")
        else:
            gen = cell_text[r - 1][generator_col_idx] if generator_col_idx < n_cols else ""
            # Strip dataset suffix if present in name cell
            gen_key = gen.split(" (")[0] if isinstance(gen, str) else gen
            color = table_text_color(colors.get(gen_key, "#222"))
            cell.set_facecolor("white")
            if c in (0, generator_col_idx):
                cell.set_text_props(
                    color=color,
                    weight="bold" if c == 0 else "medium",
                    ha="center" if c == 0 else "left",
                )
            else:
                cell.set_text_props(color="0.15", ha="center")


def make_plot_with_table(
    *,
    figsize: tuple[float, float] = (11.2, 7.0),
    width_ratios: tuple[float, float] = (2.9, 1.55),
    bottom: float = 0.20,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Figure with main axes + dedicated side table axes."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=list(width_ratios),
        wspace=0.08,
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=bottom,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_tab = fig.add_subplot(gs[0, 1])
    return fig, ax, ax_tab


def export_values_csv(df: pd.DataFrame, path: Path, round_cols: list[str] | None = None) -> None:
    out = df.copy()
    for c in round_cols or []:
        if c in out.columns:
            out[c] = out[c].astype(float).round(3)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def draw_numbered_marker(
    ax,
    x: float,
    y: float,
    point_id: int,
    *,
    color: str,
    marker: str = "o",
    size: float = 200,
    alpha: float = 0.92,
    zorder: int = 3,
) -> None:
    ax.scatter(
        x,
        y,
        c=color,
        marker=marker,
        s=size,
        alpha=alpha,
        edgecolors="black",
        linewidths=0.7,
        zorder=zorder,
    )
    ax.text(
        x,
        y,
        str(int(point_id)),
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color=marker_text_color(color),
        zorder=zorder + 1,
    )
