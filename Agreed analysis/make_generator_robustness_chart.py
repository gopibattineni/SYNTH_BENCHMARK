"""Horizontal bar chart: generator robustness via average rank.

For each generator, rank is taken from OverallRank on each of the 9
classification datasets (no-leakage). The chart shows mean rank ± SD
across datasets, sorted best → worst (lower average rank is better).

Output: Conor/generator_robustness_average_rank.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
ROOT = SCRIPT_DIR.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORES = ROOT / "Results" / "Processed_Data" / "overall_scores.csv"

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

DISPLAY_NAMES = {
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
BAR = "#2f6fb5"
ERR = "#1c1f24"
GRID = "#d5dde6"


def compute_rank_stats() -> pd.DataFrame:
    df = pd.read_csv(SCORES)
    sub = df[
        df["Dataset"].isin(CLASSIFICATION_DATASETS) & (df["LeakageLevel"] == 0.0)
    ].copy()
    piv = sub.pivot_table(index="Generator", columns="Dataset", values="OverallRank")
    stats = pd.DataFrame({
        "AverageRank": piv.mean(axis=1),
        "RankStd": piv.std(axis=1, ddof=1),
        "N_Datasets": piv.notna().sum(axis=1),
    }).sort_values("AverageRank", ascending=True)
    stats["Display"] = [DISPLAY_NAMES.get(g, g) for g in stats.index]
    return stats


def render(stats: pd.DataFrame) -> plt.Figure:
    from latex_fonts import apply_font_to_figure, configure_times_font
    font_name = configure_times_font()
    n = len(stats)
    fig_h = max(4.8, 0.55 * n + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    y = np.arange(n)
    means = stats["AverageRank"].to_numpy()
    stds = stats["RankStd"].to_numpy()
    labels = stats["Display"].tolist()

    # Best at top → reverse y tick order via invert_yaxis after plotting
    bars = ax.barh(
        y, means, height=0.62, color=BAR, edgecolor=NAVY, linewidth=0.8,
        zorder=3, label="Average rank",
    )
    ax.errorbar(
        means, y, xerr=stds, fmt="none", ecolor=ERR, elinewidth=1.4,
        capsize=4, capthick=1.3, zorder=4, label="SD across datasets",
    )

    # Annotate mean values just past the error bar
    for yi, m, s in zip(y, means, stds):
        ax.text(
            m + s + 0.12, yi, f"{m:.2f}",
            va="center", ha="left", fontsize=9.5, color=NAVY, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()  # best (lowest rank) at top
    ax.set_xlabel("Average Rank  (lower is better)", fontsize=11, color=NAVY)
    ax.set_xlim(0, max(means + stds) + 1.2)
    ax.set_title(
        "Generator Robustness Across Datasets",
        fontsize=14, fontweight="bold", color=NAVY, pad=12,
    )
    ax.text(
        0.0, 1.02,
        "Mean OverallRank over 9 classification datasets  ·  error bars = SD of ranks",
        transform=ax.transAxes, fontsize=9, color="#5a5f66", style="italic",
        va="bottom", ha="left",
    )

    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.tick_params(colors=NAVY)

    ax.legend(
        loc="lower right", frameon=True, fontsize=9,
        edgecolor=GRID, fancybox=False,
    )

    fig.tight_layout()
    apply_font_to_figure(fig, font_name)
    return fig


def main() -> None:
    stats = compute_rank_stats()
    csv_path = OUT_DIR / "generator_robustness_average_rank.csv"
    stats.reset_index().to_csv(csv_path, index=False)

    fig = render(stats)
    png_path = OUT_DIR / "generator_robustness_average_rank.png"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    print(stats[["Display", "AverageRank", "RankStd", "N_Datasets"]].to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
