"""Load fidelity and privacy metrics from generator notebook HTML tables."""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.config import GENERATORS, REPO_ROOT, PipelineConfig
from analysis.data_loader import normalize_dataset_name, normalize_generator

SDV_DIR = REPO_ROOT / "Generators" / "SDV models"
OTHER_DIR = REPO_ROOT / "Generators" / "Other GANS"
DIFFUSION_DIR = REPO_ROOT / "Generators" / "Diffusion GANs"

SDV_GENERATORS = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
OTHER_GENERATORS = ["CTABGAN", "WGAN_GP"]
DIFFUSION_GENERATORS = ["TabDDPM", "ForestDiffusion"]
NOTEBOOK_GROUPS = [
    (SDV_DIR, SDV_GENERATORS),
    (OTHER_DIR, OTHER_GENERATORS),
    (DIFFUSION_DIR, DIFFUSION_GENERATORS),
]

FIDELITY_METRIC_MAP = {
    "quality_overall": "Quality_Score",
    "ks_column_shapes": "KS_Complement",
    "js_divergence": "JS_Divergence",
    "wasserstein_summary": "Wasserstein_Distance",
    "gower_summary": "Gower_Distance",
    "mmd_summary": "MMD",
    "multivariate_mmd": "MMD_Multivariate",
    "cosine_summary": "Cosine_Similarity",
    "pca_mean_errors": "PCA_Mean_Error",
    "pca_projection": "PCA_Correlation_Diff",
}

PRIVACY_METRIC_MAP = {
    "mia": "MIA_AUC",
    "nearest_neighbors": "NNDR",
    "mahalanobis_2d": "Mahalanobis_Distance",
    "hungarian_matching": "Hungarian_Cosine_Similarity",
}


def _dataset_number(name: str) -> int:
    match = re.match(r"(\d+)", name)
    return int(match.group(1)) if match else 999


def _find_notebook(folder: Path, number: int) -> Path | None:
    if not folder.is_dir():
        return None
    matches = [p for p in folder.glob("*.ipynb") if _dataset_number(p.name) == number]
    return sorted(matches)[0] if matches else None


def _extract_tables(nb_path: Path) -> list[pd.DataFrame]:
    if not nb_path.is_file():
        return []
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    tables: list[pd.DataFrame] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            html = out.get("data", {}).get("text/html")
            if not html:
                continue
            if isinstance(html, list):
                html = "".join(html)
            try:
                tables.extend(pd.read_html(StringIO(html), flavor="lxml"))
            except Exception:
                continue
    return tables


def _cols(df: pd.DataFrame) -> set[str]:
    return {str(c) for c in df.columns}


def classify_table(df: pd.DataFrame) -> str | None:
    c = _cols(df)
    if {"Model", "Quality Score"}.issubset(c) or {"Model", "Diagnostic Score", "Quality Score"}.issubset(c):
        return "quality_overall"
    if {"Column", "Score", "Model"}.issubset(c) or {"Column", "Metric", "Score", "Model"}.issubset(c):
        return "ks_column_shapes"
    if {"Feature", "JS_Divergence", "Model"}.issubset(c):
        return "js_divergence"
    if {"Model", "Mean_Wasserstein"}.issubset(c):
        return "wasserstein_summary"
    if {"Model", "Avg_Intra_Real", "Avg_Cross_Real_vs_Synth"}.issubset(c):
        return "gower_summary"
    if {"Model", "MMD_Score"}.issubset(c):
        return "mmd_summary"
    if {"Model", "Global_MMD_RBF"}.issubset(c):
        return "multivariate_mmd"
    if {"Model", "Average_Cosine_Similarity"}.issubset(c):
        return "cosine_summary"
    if {"Model", "Mean Error %"}.issubset(c):
        return "pca_mean_errors"
    if {"Model", "TopCorrDiff"}.issubset(c) or {"Model", "MeanTop10CorrDiff"}.issubset(c):
        return "pca_projection"
    if {"Model", "AUC", "Advantage"}.issubset(c) and "Accuracy_TRTR" not in c:
        return "mia"
    if {"Model", "Real_Mean_MD_2D"}.issubset(c):
        return "mahalanobis_2d"
    if {"Model", "Num_Matches", "Avg_Cosine_Similarity"}.issubset(c):
        return "hungarian_matching"
    if {"Model", "Avg_NN_Distance"}.issubset(c):
        return "nearest_neighbors"
    return None


def _metric_value(row: pd.Series, kind: str) -> float | None:
    if kind == "quality_overall":
        for col in ("Quality Score", "Quality_Score", "Diagnostic Score"):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
    if kind == "ks_column_shapes":
        return float(row["Score"]) if pd.notna(row.get("Score")) else None
    if kind == "js_divergence":
        return float(row["JS_Divergence"]) if pd.notna(row.get("JS_Divergence")) else None
    if kind == "wasserstein_summary":
        return float(row["Mean_Wasserstein"]) if pd.notna(row.get("Mean_Wasserstein")) else None
    if kind == "gower_summary":
        return float(row["Avg_Cross_Real_vs_Synth"]) if pd.notna(row.get("Avg_Cross_Real_vs_Synth")) else None
    if kind == "mmd_summary":
        return float(row["MMD_Score"]) if pd.notna(row.get("MMD_Score")) else None
    if kind == "multivariate_mmd":
        return float(row["Global_MMD_RBF"]) if pd.notna(row.get("Global_MMD_RBF")) else None
    if kind == "cosine_summary":
        return float(row["Average_Cosine_Similarity"]) if pd.notna(row.get("Average_Cosine_Similarity")) else None
    if kind == "pca_mean_errors":
        return float(row["Mean Error %"]) if pd.notna(row.get("Mean Error %")) else None
    if kind == "pca_projection":
        col = "TopCorrDiff" if "TopCorrDiff" in row.index else "MeanTop10CorrDiff"
        return float(row[col]) if pd.notna(row.get(col)) else None
    if kind == "mia":
        return float(row["AUC"]) if pd.notna(row.get("AUC")) else None
    if kind == "mahalanobis_2d":
        return float(row["Synth_Mean_MD_2D"]) if pd.notna(row.get("Synth_Mean_MD_2D")) else None
    if kind == "hungarian_matching":
        return float(row["Avg_Cosine_Similarity"]) if pd.notna(row.get("Avg_Cosine_Similarity")) else None
    if kind == "nearest_neighbors":
        return float(row["Avg_NN_Distance"]) if pd.notna(row.get("Avg_NN_Distance")) else None
    return None


def load_notebook_metrics_long(config: PipelineConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fidelity_long, privacy_long) parsed from all generator notebooks."""
    config = config or PipelineConfig()
    datasets_json = (
        config.repo_root
        / "Generators"
        / "Experiment with utility data leak"
        / "python_scripts"
        / "hive"
        / "datasets.json"
    )
    if not datasets_json.is_file():
        return pd.DataFrame(), pd.DataFrame()

    entries = json.loads(datasets_json.read_text(encoding="utf-8"))
    fid_rows: list[dict[str, Any]] = []
    priv_rows: list[dict[str, Any]] = []

    for entry in entries:
        dataset = entry["notebook_dir"]
        num = _dataset_number(dataset)
        seen: set[tuple] = set()

        for folder, allowed in NOTEBOOK_GROUPS:
            nb = _find_notebook(folder, num)
            if nb is None:
                continue
            for table in _extract_tables(nb):
                kind = classify_table(table)
                if kind is None:
                    continue
                gen_col = "Model" if "Model" in table.columns else None
                is_privacy = kind in PRIVACY_METRIC_MAP
                metric_name = PRIVACY_METRIC_MAP.get(kind) or FIDELITY_METRIC_MAP.get(kind)
                if not metric_name:
                    continue

                for _, row in table.iterrows():
                    generator = normalize_generator(row.get(gen_col)) if gen_col else None
                    if generator not in allowed:
                        continue
                    val = _metric_value(row, kind)
                    if val is None:
                        continue
                    key = (dataset, generator, metric_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    record = {
                        "Dataset": dataset,
                        "Generator": generator,
                        "Seed": None,
                        "LeakageLevel": 0,
                        "Metric": metric_name,
                        "MetricValue": val,
                        "Mean": val,
                        "Std": None,
                        "Classifier": None,
                        "Regressor": None,
                        "Category": "Privacy" if is_privacy else "Fidelity",
                        "EvaluationType": "Score",
                        "SourceFile": str(nb.relative_to(config.repo_root)),
                    }
                    if is_privacy:
                        priv_rows.append(record)
                    else:
                        fid_rows.append(record)

    return pd.DataFrame(fid_rows), pd.DataFrame(priv_rows)
