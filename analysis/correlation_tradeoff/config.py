"""Configuration for Correlation Trade-off Analysis (Utility × Fidelity × Privacy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analysis.config import FIGURE_FORMATS, N_SEEDS, REPO_ROOT, RESULTS_DIR

CANCER = "1. Cancer"
MUSHROOM = "8. Mushroom dataset"
FOCUS_DATASETS = {"Cancer": CANCER, "Mushroom": MUSHROOM}
DATASET_LABELS = {"Cancer": "Cancer", "Mushroom": "Mushroom"}
DATASET_MARKERS = {"Cancer": "o", "Mushroom": "s"}

# Okabe–Ito color-blind friendly palette (stable across all figures)
GENERATOR_COLORS = {
    "CTGAN": "#0072B2",
    "CopulaGAN": "#E69F00",
    "TVAE": "#009E73",
    "GaussianCopula": "#D55E00",
    "WGAN_GP": "#CC79A7",
    "CTABGAN": "#56B4E9",
    "TabDDPM": "#F0E442",
    "ForestDiffusion": "#000000",
}

UTILITY_METRICS = ("Accuracy", "F1")
PRIVACY_ANALYSES = {
    "MIA": {
        "metric": "MIA_AUC",
        "label": "MIA AUC",
        "higher_is_private": False,
        "folder": "Analysis_A_MIA",
    },
    "NNDR": {
        "metric": "NNDR",
        "label": "NNDR",
        "higher_is_private": True,
        "folder": "Analysis_B_NNDR",
    },
}

FIDELITY_PREFERRED = ("Quality_Score", "Column_Shapes", "KS_Complement")

FIGURE_DPI = 600

PLOT_RC = {
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


@dataclass
class CorrelationTradeoffConfig:
    repo_root: Path = REPO_ROOT
    output_root: Path = RESULTS_DIR / "Correlation_Tradeoff_Analysis"
    two_datasets_mirror: Path = RESULTS_DIR / "Two_Datasets_Assessment" / "Correlation_Tradeoff_Analysis"
    figure_dpi: int = FIGURE_DPI
    figure_formats: list[str] = field(default_factory=lambda: FIGURE_FORMATS.copy())
    n_seeds: int = N_SEEDS
    utility_metrics: tuple[str, ...] = UTILITY_METRICS
    seed: int = 42
    # Solid overall OLS across all points; dashed per-dataset OLS overlays
    dataset_specific_regression: bool = True

    def dirs(self) -> dict[str, Path]:
        mapping = {
            "root": self.output_root,
            "figures": self.output_root / "Figures",
            "figures_primary": self.output_root / "Figures" / "Primary",
            "figures_supplementary": self.output_root / "Figures" / "Supplementary",
            "tables": self.output_root / "Tables",
            "statistics": self.output_root / "Statistics",
            "summaries": self.output_root / "Summaries",
            "processed": self.output_root / "Processed_Data",
        }
        for p in mapping.values():
            p.mkdir(parents=True, exist_ok=True)
        return mapping

    def analysis_dirs(self, utility_metric: str, privacy_key: str) -> dict[str, Path]:
        priv = PRIVACY_ANALYSES[privacy_key]
        base = self.output_root / f"Utility_{utility_metric}" / priv["folder"]
        mapping = {
            "root": base,
            "primary": base / "Primary",
            "supplementary": base / "Supplementary",
            "figures_primary": base / "Primary" / "Figures",
            "figures_supplementary": base / "Supplementary" / "Figures",
            "statistics": base / "Statistics",
            "summaries": base / "Summaries",
        }
        for p in mapping.values():
            p.mkdir(parents=True, exist_ok=True)
        return mapping
