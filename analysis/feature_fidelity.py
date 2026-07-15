"""Feature-level fidelity metrics (KS, JS) from generator notebooks."""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from analysis.config import GENERATORS, REPO_ROOT, PipelineConfig

SDV_DIR = REPO_ROOT / "Generators" / "SDV models"
OTHER_DIR = REPO_ROOT / "Generators" / "Other GANS"
DIFFUSION_DIR = REPO_ROOT / "Generators" / "Diffusion GANs"

SDV_GENERATORS = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
OTHER_GENERATORS = ["CTABGAN", "WGAN_GP"]
DIFFUSION_GENERATORS = ["TabDDPM", "ForestDiffusion"]
NOTEBOOK_GROUPS = [
    (SDV_DIR, "SDV", SDV_GENERATORS),
    (OTHER_DIR, "Other_GAN", OTHER_GENERATORS),
    (DIFFUSION_DIR, "Diffusion", DIFFUSION_GENERATORS),
]


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


def _classify_ks_js(df: pd.DataFrame) -> str | None:
    cols = {str(c) for c in df.columns}
    if {"Column", "Score", "Model"}.issubset(cols) or {"Column", "Metric", "Score", "Model"}.issubset(cols):
        return "ks"
    if {"Feature", "JS_Divergence", "Model"}.issubset(cols):
        return "js"
    return None


def _normalize_generator(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    for gen in GENERATORS:
        if text.lower() == gen.lower():
            return gen
    return text if text in GENERATORS else None


def load_quality_from_notebooks(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Extract overall Quality Score tables from generator notebooks."""
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
        return pd.DataFrame()

    entries = json.loads(datasets_json.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for entry in entries:
        dataset = entry["notebook_dir"]
        num = _dataset_number(dataset)
        seen: set[tuple] = set()

        for folder, _group, allowed in NOTEBOOK_GROUPS:
            nb = _find_notebook(folder, num)
            if nb is None:
                continue
            for table in _extract_tables(nb):
                cols = {str(c) for c in table.columns}
                score_col = None
                if {"Model", "Quality Score"}.issubset(cols) or {
                    "Model",
                    "Diagnostic Score",
                    "Quality Score",
                }.issubset(cols):
                    score_col = "Quality Score" if "Quality Score" in table.columns else "Diagnostic Score"
                elif {"Model", "Average_Column_Shapes_Score"}.issubset(cols):
                    # Diffusion notebooks report mean KS complement instead of SDMetrics Quality Score.
                    score_col = "Average_Column_Shapes_Score"
                else:
                    continue

                for _, row in table.iterrows():
                    generator = _normalize_generator(row.get("Model"))
                    if generator not in allowed:
                        continue
                    val = row.get(score_col)
                    if pd.isna(val):
                        continue
                    key = (dataset, generator)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Generator": generator,
                            "Metric": "Quality_Score",
                            "Value": float(val),
                            "Mean": float(val),
                            "Std": None,
                            "Experiment": "fidelity",
                            "Category": "Fidelity",
                            "LeakageLevel": 0,
                            "MetricValue": float(val),
                            "SourceFile": str(nb.relative_to(config.repo_root)),
                        }
                    )

    return pd.DataFrame(rows)


def load_feature_fidelity(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Parse KS / JS tables from all generator notebooks into long format."""
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
        return pd.DataFrame()

    entries = json.loads(datasets_json.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for entry in entries:
        dataset = entry["notebook_dir"]
        num = _dataset_number(dataset)
        seen: set[tuple] = set()

        for folder, _group, allowed in NOTEBOOK_GROUPS:
            nb = _find_notebook(folder, num)
            if nb is None:
                continue
            for table in _extract_tables(nb):
                kind = _classify_ks_js(table)
                if kind is None:
                    continue

                gen_col = "Model" if "Model" in table.columns else None
                feature_col = "Column" if "Column" in table.columns else "Feature"
                score_col = "Score" if "Score" in table.columns else "JS_Divergence"
                metric_name = "KS_Complement" if kind == "ks" else "JS_Divergence"

                for _, row in table.iterrows():
                    generator = _normalize_generator(row.get(gen_col)) if gen_col else None
                    if generator not in allowed:
                        continue
                    feature = row.get(feature_col)
                    score = row.get(score_col)
                    if pd.isna(feature) or pd.isna(score):
                        continue
                    key = (dataset, generator, str(feature), metric_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Generator": generator,
                            "Feature": str(feature),
                            "Metric": metric_name,
                            "Value": float(score),
                            "Mean": float(score),
                            "Std": None,
                            "Experiment": "feature_fidelity",
                            "SourceFile": str(nb.relative_to(config.repo_root)),
                        }
                    )

    return pd.DataFrame(rows)


def feature_fidelity_matrix(
    feature_df: pd.DataFrame,
    metric: str = "KS_Complement",
) -> pd.DataFrame:
    """Pivot to generator × feature matrix (averaged across datasets)."""
    if feature_df.empty:
        return pd.DataFrame()
    sub = feature_df[feature_df["Metric"] == metric].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="Generator",
        columns="Feature",
        values="Mean",
        aggfunc="mean",
    )


def feature_fidelity_by_dataset(
    feature_df: pd.DataFrame,
    dataset: str,
    metric: str = "KS_Complement",
) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()
    sub = feature_df[
        (feature_df["Dataset"] == dataset) & (feature_df["Metric"] == metric)
    ]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="Generator",
        columns="Feature",
        values="Mean",
        aggfunc="mean",
    )
