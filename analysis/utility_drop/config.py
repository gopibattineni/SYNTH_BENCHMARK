"""Configuration for Utility Drop Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analysis.config import FIGURE_DPI, FIGURE_FORMATS, LEAKAGE_LEVELS, REPO_ROOT, RESULTS_DIR

CANCER = "1. Cancer"
MUSHROOM = "8. Mushroom dataset"
FOCUS_DATASETS = {"Cancer": CANCER, "Mushroom": MUSHROOM}
DATASET_LABELS = {"Cancer": "Cancer", "Mushroom": "Mushroom"}

PRIMARY_METRIC = "Accuracy"
SUPPLEMENTARY_METRICS = ["F1", "ROC-AUC", "Precision", "Recall"]
CLASSIFICATION_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

N_SEEDS = 10


@dataclass
class UtilityDropConfig:
    repo_root: Path = REPO_ROOT
    output_root: Path = RESULTS_DIR / "Utility_Drop_Analysis"
    figure_dpi: int = FIGURE_DPI
    figure_formats: list[str] = field(default_factory=lambda: FIGURE_FORMATS.copy())
    primary_metric: str = PRIMARY_METRIC
    n_seeds: int = N_SEEDS
    leakage_levels: list[int] = field(default_factory=lambda: list(LEAKAGE_LEVELS))

    def dirs(self) -> dict[str, Path]:
        mapping = {
            "root": self.output_root,
            "figures": self.output_root / "Figures",
            "tables": self.output_root / "Tables",
            "statistics": self.output_root / "Statistics",
            "processed": self.output_root / "Processed_Data",
        }
        for p in mapping.values():
            p.mkdir(parents=True, exist_ok=True)
        return mapping
