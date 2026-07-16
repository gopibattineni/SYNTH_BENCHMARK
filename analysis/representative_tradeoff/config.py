"""Configuration for representative-metrics dual trade-off analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analysis.config import FIGURE_DPI, FIGURE_FORMATS, REPO_ROOT, RESULTS_DIR

CANCER_DATASET = "1. Cancer"
MUSHROOM_DATASET = "8. Mushroom dataset"

DATASETS = {
    "Cancer": CANCER_DATASET,
    "Mushroom": MUSHROOM_DATASET,
}

DATASET_LABELS = {
    "Cancer": "Cancer",
    "Mushroom": "Mushroom",
}

# Primary utility metric for ranking / best-classifier selection
PRIMARY_UTILITY_METRIC = "F1"
UTILITY_METRICS = ["Accuracy", "Precision", "Recall", "F1"]  # ROC-AUC absent in sources

N_SEEDS = 10


@dataclass
class RepresentativeTradeoffConfig:
    repo_root: Path = REPO_ROOT
    output_root: Path = RESULTS_DIR / "Representative_Metrics_Tradeoff"
    figure_dpi: int = FIGURE_DPI
    figure_formats: list[str] = field(default_factory=lambda: FIGURE_FORMATS.copy())
    primary_utility_metric: str = PRIMARY_UTILITY_METRIC
    n_seeds: int = N_SEEDS

    def analysis_dirs(self, analysis: str) -> dict[str, Path]:
        """analysis: 'Classifier_Independent' | 'Best_Classifier'"""
        base = self.output_root / analysis
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
            "statistics": base / "Statistics",
        }
        for p in mapping.values():
            p.mkdir(parents=True, exist_ok=True)
        return mapping
