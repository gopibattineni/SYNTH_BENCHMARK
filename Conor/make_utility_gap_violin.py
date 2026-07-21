"""Publication-quality violin plots of Utility Gap by generator.

Utility Gap = TRTR − TSTR (F1), one value per classification dataset.
Median is overlaid on each violin. Generators ordered best → worst
(lowest median gap).

Output: Conor/utility_gap_violin.png
"""

from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
GAPS_JSON = ROOT / "docs" / "data" / "utility_gaps.json"

CLASSIFICATION_DATASETS = [
    "1. Cancer",
    "2. Alzhimers",
    "3. Adult",
    "4. Forest cover dataset",
    "5. Bank Markting",
    "6. Wine dataset",
    "7. CDC diabetes dataset",
    "8. Mushroom dataset",
    "9. MAGIC Gamma Telescope",
]

DISPLAY = {
    "ForestDiffusion": "ForestDiffusion",
    "TVAE": "TVAE",
    "CTABGAN": "CTABGAN",
    "WGAN_GP": "WGAN-GP",
    "GaussianCopula": "GaussianCopula",
    "CopulaGAN": "CopulaGAN",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
}

NAVY = "#1f3a5f"
MEDIAN = "#c0392b"
POINT = "#1c1f24"
VIOLIN = "#5b8fc7"


def load_gaps() -> pd.DataFrame:
    raw = pd.DataFrame(json.loads(GAPS_JSON.read_text()))
    df = raw[
        (raw["Dataset"].isin(CLASSIFICATION_DATASETS))
        & (raw["Metric"] == "F1_Gap")
        & (raw["TaskType"] == "classification")
    ].copy()
    df = df.rename(columns={"Mean": "UtilityGap"})
    df["Generator"] = df["Generator"].map(lambda g: DISPLAY.get(g, g))
    # Order by median gap (best = lowest)
    order = (
        df.groupby("Generator")["UtilityGap"]
        .median()
        .sort_values()
        .index.tolist()
    )
    df["Generator"] = pd.Categorical(df["Generator"], categories=order, ordered=True)
    return df.sort_values("Generator")


def render(df: pd.DataFrame) -> plt.Figure:
    from conor_fonts import apply_font_to_figure, configure_times_font
    font_name = configure_times_font()
    sns.set_style("whitegrid", {"axes.edgecolor": NAVY, "grid.color": "#dce3eb"})

    order = list(df["Generator"].cat.categories)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    palette = sns.color_palette("Blues", n_colors=len(order) + 2)[2:]

    sns.violinplot(
        data=df,
        x="Generator",
        y="UtilityGap",
        hue="Generator",
        order=order,
        hue_order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.1,
        saturation=0.9,
        legend=False,
        ax=ax,
        zorder=2,
    )

    # Soften violin faces
    for coll in ax.collections:
        coll.set_alpha(0.78)
        coll.set_edgecolor(NAVY)
        coll.set_linewidth(1.0)

    # Overlay individual dataset points
    sns.stripplot(
        data=df,
        x="Generator",
        y="UtilityGap",
        order=order,
        color=POINT,
        size=4.5,
        alpha=0.55,
        jitter=0.08,
        ax=ax,
        zorder=3,
        legend=False,
    )

    # Median overlay (horizontal ticks + connecting markers)
    medians = df.groupby("Generator", observed=True)["UtilityGap"].median().reindex(order)
    xs = np.arange(len(order))
    ax.scatter(
        xs, medians.values,
        s=70, color=MEDIAN, zorder=5, marker="D",
        edgecolors="white", linewidths=0.8, label="Median",
    )
    for x, m in zip(xs, medians.values):
        ax.hlines(
            m, x - 0.28, x + 0.28,
            colors=MEDIAN, linewidths=2.0, zorder=4,
        )

    ax.axhline(0, color="#7a8490", linestyle="--", linewidth=1.0, zorder=1, alpha=0.85)

    ax.set_xlabel("")
    ax.set_ylabel("Utility Gap  (TRTR − TSTR, F1)", fontsize=11.5, color=NAVY)
    ax.set_title(
        "Distribution of Utility Gap Across Datasets",
        fontsize=14.5, fontweight="bold", color=NAVY, pad=14,
    )
    ax.text(
        0.0, 1.02,
        "One point per classification dataset (n = 9)  ·  lower gap is better  ·  dashed line = zero gap",
        transform=ax.transAxes, fontsize=9, color="#5a5f66", style="italic",
        va="bottom", ha="left",
    )

    ax.tick_params(axis="x", labelsize=10.5, colors=NAVY, rotation=18)
    ax.tick_params(axis="y", labelsize=10, colors=NAVY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)

    ax.legend(
        loc="upper left", frameon=True, fontsize=9.5,
        edgecolor="#c5d0dc", fancybox=False,
    )

    fig.tight_layout()
    apply_font_to_figure(fig, font_name)
    return fig


def main() -> None:
    df = load_gaps()
    csv_path = OUT_DIR / "utility_gap_violin_data.csv"
    df.to_csv(csv_path, index=False)

    fig = render(df)
    png_path = OUT_DIR / "utility_gap_violin.png"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)

    summary = (
        df.groupby("Generator", observed=True)["UtilityGap"]
        .agg(median="median", mean="mean", std="std", min="min", max="max")
        .round(4)
    )
    print(summary.to_string())
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
