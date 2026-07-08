"""Load TRTR/TSTR Excel results from the utility data-leak benchmark."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "Generators" / "Experiment with utility data leak"
DIFFUSION_ROOT = RESULTS_ROOT / "diffusion_dataleak"
DATASETS_JSON = RESULTS_ROOT / "python_scripts" / "hive" / "datasets.json"

# Six generators in the main utility-leak notebooks; two diffusion models in diffusion_dataleak.
GENERATORS_MAIN = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN",
]
GENERATORS_DIFFUSION = ["TabDDPM", "ForestDiffusion"]
ALL_GENERATORS = GENERATORS_MAIN + GENERATORS_DIFFUSION

CLASSIFICATION_METRICS = {
    "Accuracy_Drop": {"label": "Accuracy drop (TRTR − TSTR)", "better": "lower"},
    "F1_Drop": {"label": "F1 drop (TRTR − TSTR)", "better": "lower"},
    "Precision_Drop": {"label": "Precision drop", "better": "lower"},
    "Recall_Drop": {"label": "Recall drop", "better": "lower"},
}
REGRESSION_METRICS = {
    "R2_Drop": {"label": "R² drop (TRTR − TSTR)", "better": "lower"},
    "RMSE_Increase": {"label": "RMSE increase (TSTR − TRTR)", "better": "lower"},
    "MAE_Increase": {"label": "MAE increase (TSTR − TRTR)", "better": "lower"},
    "MSE_Increase": {"label": "MSE increase (TSTR − TRTR)", "better": "lower"},
}


@dataclass
class DatasetResults:
    dataset_id: str
    name: str
    number: int
    task_type: str
    notebook_dir: str
    excel_path: Optional[Path]
    trtr: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    generators: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _dataset_number(notebook_dir: str) -> int:
    match = re.match(r"(\d+)", notebook_dir)
    return int(match.group(1)) if match else 0


def _task_type(number: int) -> str:
    return "regression" if number >= 10 else "classification"


def _find_excel(folder: Path, preferred: str) -> Optional[Path]:
    if not folder.is_dir():
        return None
    preferred_path = folder / preferred
    if preferred_path.is_file():
        return preferred_path
    matches = sorted(folder.glob("TRTR_TSTR*.xlsx"))
    return matches[0] if matches else None


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        xl = pd.ExcelFile(path)
        if sheet not in xl.sheet_names:
            return pd.DataFrame()
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Synthetic_Model" in out.columns:
        out = out.rename(columns={"Synthetic_Model": "Generator"})
    return out


def _normalize_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Synthetic_Model" in out.columns:
        out = out.rename(columns={"Synthetic_Model": "Generator"})
    return out


def _load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    trtr_key = "TRTR_Results" if "TRTR_Results" in sheets else "TRTR_Results"
    return {
        "trtr": sheets.get(trtr_key, sheets.get("TRTR_Results", pd.DataFrame())),
        "summary": _normalize_summary(sheets.get("Summary", pd.DataFrame())),
        "comparisons": _normalize_comparisons(sheets.get("All_Comparisons", pd.DataFrame())),
        "quality": sheets.get("Quality_Metrics", pd.DataFrame()),
    }


def load_datasets_metadata() -> List[dict]:
    if DATASETS_JSON.is_file():
        return json.loads(DATASETS_JSON.read_text(encoding="utf-8"))
    return []


def load_all_results() -> Dict[str, DatasetResults]:
    metadata = load_datasets_metadata()
    results: Dict[str, DatasetResults] = {}

    for entry in metadata:
        ds_id = entry["id"]
        notebook_dir = entry["notebook_dir"]
        number = _dataset_number(notebook_dir)
        folder = RESULTS_ROOT / notebook_dir
        excel_path = _find_excel(folder, entry.get("output_file", "TRTR_TSTR_results.xlsx"))

        ds = DatasetResults(
            dataset_id=ds_id,
            name=entry.get("name", ds_id),
            number=number,
            task_type=_task_type(number),
            notebook_dir=notebook_dir,
            excel_path=excel_path,
            generators=list(GENERATORS_MAIN),
        )

        if excel_path is None:
            ds.error = f"No TRTR/TSTR Excel file in {notebook_dir}"
            results[ds_id] = ds
            continue

        try:
            data = _load_workbook(excel_path)
            ds.trtr = data["trtr"]
            ds.summary = data["summary"]
            ds.comparisons = data["comparisons"]
            ds.quality = data["quality"]
            if not ds.summary.empty and "Generator" in ds.summary.columns:
                ds.generators = [
                    g for g in ALL_GENERATORS if g in ds.summary["Generator"].astype(str).tolist()
                ]
        except Exception as exc:
            ds.error = str(exc)

        # Merge TabDDPM / ForestDiffusion from diffusion_dataleak when available.
        diffusion_folder = DIFFUSION_ROOT / notebook_dir
        diffusion_excel = _find_excel(diffusion_folder, "TRTR_TSTR_results.xlsx")
        if diffusion_excel is not None:
            try:
                diff = _load_workbook(diffusion_excel)
                if not diff["summary"].empty:
                    ds.summary = pd.concat([ds.summary, diff["summary"]], ignore_index=True)
                if not diff["comparisons"].empty:
                    ds.comparisons = pd.concat(
                        [ds.comparisons, diff["comparisons"]], ignore_index=True
                    )
                for gen in GENERATORS_DIFFUSION:
                    if gen not in ds.generators and (
                        diff["summary"].empty
                        or gen in diff["summary"].get("Generator", pd.Series(dtype=str)).astype(str).tolist()
                    ):
                        ds.generators.append(gen)
            except Exception:
                pass

        results[ds_id] = ds

    return results


def summary_long_frame(results: Dict[str, DatasetResults]) -> pd.DataFrame:
    rows = []
    for ds_id, ds in results.items():
        if ds.summary.empty or "Generator" not in ds.summary.columns:
            continue
        for _, row in ds.summary.iterrows():
            base = {
                "dataset_id": ds_id,
                "dataset": ds.name,
                "dataset_number": ds.number,
                "task_type": ds.task_type,
                "generator": row.get("Generator"),
            }
            for col in ds.summary.columns:
                if col == "Generator":
                    continue
                base[col] = row[col]
            rows.append(base)
    return pd.DataFrame(rows)


def comparisons_long_frame(results: Dict[str, DatasetResults]) -> pd.DataFrame:
    rows = []
    for ds_id, ds in results.items():
        if ds.comparisons.empty:
            continue
        chunk = ds.comparisons.copy()
        chunk["dataset_id"] = ds_id
        chunk["dataset"] = ds.name
        chunk["dataset_number"] = ds.number
        chunk["task_type"] = ds.task_type
        if "Generator" not in chunk.columns and "Synthetic_Model" in chunk.columns:
            chunk = chunk.rename(columns={"Synthetic_Model": "Generator"})
        rows.append(chunk)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def primary_metric(task_type: str) -> str:
    return "Accuracy_Drop" if task_type == "classification" else "R2_Drop"


def metrics_for_task(task_type: str) -> Dict[str, dict]:
    return CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
