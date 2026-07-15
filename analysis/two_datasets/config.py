"""Configuration for Wisconsin Breast Cancer & Mushroom case study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analysis.config import FIGURE_DPI, FIGURE_FORMATS, REPO_ROOT, RESULTS_DIR

# Canonical folder names in experiment outputs
CANCER_DATASET = "1. Cancer"
MUSHROOM_DATASET = "8. Mushroom dataset"

DATASETS = {
    "Cancer": CANCER_DATASET,
    "Mushroom": MUSHROOM_DATASET,
}

DISPLAY_NAMES = {
    "Cancer": "Wisconsin Breast Cancer",
    "Mushroom": "Secondary Mushroom",
}

FIDELITY_PLOT_METRICS = [
    "KS_Complement",
    "TV_Complement",
    "JS_Divergence",
    "Wasserstein_Distance",
    "Correlation_Similarity",
    "Quality_Score",
]

PRIVACY_PLOT_METRICS = [
    "NNDR",
    "MIA_AUC",
    "Mahalanobis_Distance",
    "Mean_Distance",
    "Cosine_Similarity",
    "Hungarian_Cosine_Similarity",
]


@dataclass
class TwoDatasetConfig:
    repo_root: Path = REPO_ROOT
    output_root: Path = RESULTS_DIR / "Two_Datasets_Assessment"
    utility_weight: float = 0.40
    fidelity_weight: float = 0.30
    privacy_weight: float = 0.30
    figure_dpi: int = FIGURE_DPI
    figure_formats: list[str] = field(default_factory=lambda: FIGURE_FORMATS.copy())

    def dataset_dirs(self, key: str) -> dict[str, Path]:
        base = self.output_root / key
        mapping = {
            "root": base,
            "figures": base / "Figures",
            "tables": base / "Tables",
            "statistics": base / "Statistics",
            "processed": base / "Processed_Data",
        }
        for p in mapping.values():
            p.mkdir(parents=True, exist_ok=True)
        return mapping

    def comparison_dirs(self) -> dict[str, Path]:
        base = self.output_root / "Comparison"
        mapping = {
            "root": base,
            "figures": base / "Figures",
            "tables": base / "Tables",
            "tradeoff": base / "Tradeoff",
            "correlations": base / "Correlations",
            "rankings": base / "Rankings",
        }
        for p in mapping.values():
            p.mkdir(parents=True, exist_ok=True)
        return mapping

    def summary_dir(self) -> Path:
        p = self.output_root / "Summary_Report"
        p.mkdir(parents=True, exist_ok=True)
        return p
