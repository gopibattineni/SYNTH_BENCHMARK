"""Automatic discovery, classification, and merging of experiment Excel files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.config import (
    FIDELITY_METRICS,
    GENERATOR_ALIASES,
    GENERATORS,
    MODEL_ALIASES,
    REPO_ROOT,
    PipelineConfig,
)

UTILITY_SHEETS = {
    "TRTR_Results",
    "All_Comparisons",
    "Summary",
    "Quality_Metrics",
}
UTILITY_GENERATOR_SHEETS = set(GENERATORS)
UTILITY_FILENAME_HINTS = ("TRTR_TSTR", "TRTR_TSTR_results")
PRIVACY_FILENAME_HINTS = ("Mahalanobis", "Matching", "matching", "Hungarian")
DATASET_DIR_PATTERN = re.compile(r"^(\d+)\.\s*(.+)$")

# Map privacy / Mahalanobis workbook labels to numbered dataset folders.
DATASET_ALIASES: dict[str, str] = {
    "adult_census": "3. Adult",
    "adult": "3. Adult",
    "heart": "7. CDC diabetes dataset",
    "cdc_diabetes": "7. CDC diabetes dataset",
    "secondary_mushroom": "8. Mushroom dataset",
    "mushroom": "8. Mushroom dataset",
    "winequality": "6. Wine dataset",
    "wine": "6. Wine dataset",
    "forestcover": "4. Forest cover dataset",
    "forest_cover": "4. Forest cover dataset",
    "metro": "10. Metro interstate",
    "metro_interstate": "10. Metro interstate",
    "magic": "9. MAGIC Gamma Telescope",
    "bank_marketing": "5. Bank Markting",
    "bank": "5. Bank Markting",
    "airquality": "12. Air Quality",
    "air_quality": "12. Air Quality",
    "concrete": "13. Concrete Compressive Strength",
    "energyefficiency": "14. Energy Efficiency",
    "energy_efficiency": "14. Energy Efficiency",
    "real_estate": "15. Real Estate Valuation",
    "realestate": "15. Real Estate Valuation",
    "cancer": "1. Cancer",
    "alzhimers": "2. Alzhimers",
    "alzheimer": "2. Alzhimers",
    "online_shopping": "11. online shopping",
    "shopping": "11. online shopping",
}


def normalize_dataset_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if DATASET_DIR_PATTERN.match(text):
        return text
    key = re.sub(r"^\d+\.\s*", "", text).lower()
    key = re.sub(r"[_\s]+", "_", key).strip("_")
    for alias, canonical in DATASET_ALIASES.items():
        if alias in key.replace(" ", "_"):
            return canonical
    # Match by substring against known numbered dirs from datasets.json
    datasets_json = REPO_ROOT / "Generators" / "Experiment with utility data leak" / "python_scripts" / "hive" / "datasets.json"
    if datasets_json.is_file():
        import json

        for entry in json.loads(datasets_json.read_text(encoding="utf-8")):
            folder = entry["notebook_dir"]
            folder_key = re.sub(r"^\d+\.\s*", "", folder).lower().replace(" ", "")
            if folder_key and folder_key in key.replace(" ", "").replace("_", ""):
                return folder
            if entry.get("id", "").replace("_", "") in key.replace("_", ""):
                return folder
    return text



@dataclass
class ExcelFileRecord:
    path: Path
    file_type: str
    dataset: str | None = None
    priority: int = 0


@dataclass
class MasterData:
    utility_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    utility_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    fidelity_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    privacy_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    privacy_detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    file_inventory: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_sheets: dict[str, pd.DataFrame] = field(default_factory=dict)


def normalize_generator(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    key = text.lower().replace("-", "_").replace(" ", "_")
    if key in GENERATOR_ALIASES:
        return GENERATOR_ALIASES[key]
    for gen in GENERATORS:
        if text.lower() == gen.lower():
            return gen
    return text


def normalize_model(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    key = text.lower().replace("-", " ").replace("_", " ")
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if key.replace(" ", "") in MODEL_ALIASES:
        return MODEL_ALIASES[key.replace(" ", "")]
    return text


def infer_dataset_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = DATASET_DIR_PATTERN.match(part)
        if match:
            return part
    return None


def classify_excel_file(path: Path, config: PipelineConfig) -> ExcelFileRecord | None:
    rel = path.relative_to(config.repo_root)
    name = path.name
    lower = name.lower()

    if any(part in config.exclude_dirs for part in rel.parts):
        return None
    if not lower.endswith((".xlsx", ".xls", ".xlsm")):
        return None

    dataset = infer_dataset_from_path(path)

    if any(h.lower() in lower for h in UTILITY_FILENAME_HINTS):
        priority = 0
        if "utility results" in str(path).lower():
            priority = 3
        elif "merged" in lower or "cancer" in lower or "alzhimers" in lower:
            priority = 2
        elif "diffusion_dataleak" in str(path):
            priority = 1
        return ExcelFileRecord(path=path, file_type="utility", dataset=dataset, priority=priority)

    if any(h in name for h in PRIVACY_FILENAME_HINTS):
        return ExcelFileRecord(path=path, file_type="privacy", dataset=dataset, priority=1)

    return ExcelFileRecord(path=path, file_type="other", dataset=dataset, priority=0)


def discover_excel_files(config: PipelineConfig) -> list[ExcelFileRecord]:
    records: list[ExcelFileRecord] = []
    for path in sorted(config.repo_root.rglob("*")):
        if not path.is_file():
            continue
        record = classify_excel_file(path, config)
        if record is not None:
            records.append(record)
    return records


def select_utility_files(records: list[ExcelFileRecord]) -> list[ExcelFileRecord]:
    """Pick highest-priority utility workbook per dataset folder name."""
    by_dataset: dict[str, ExcelFileRecord] = {}

    for record in records:
        if record.file_type != "utility":
            continue
        key = record.dataset or str(record.path.parent)
        current = by_dataset.get(key)
        if current is None or record.priority > current.priority:
            by_dataset[key] = record
        elif (
            record.priority == current.priority
            and record.path.stat().st_mtime > current.path.stat().st_mtime
        ):
            by_dataset[key] = record

    return list(by_dataset.values())


def _read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}


def _task_type_from_dataset(dataset: str | None) -> str:
    if not dataset:
        return "unknown"
    match = DATASET_DIR_PATTERN.match(dataset)
    if match and int(match.group(1)) >= 10:
        return "regression"
    return "classification"


def _utility_from_comparisons(
    df: pd.DataFrame,
    dataset: str,
    task_type: str,
    source_file: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)

    gen_col = next((c for c in ("Synthetic_Model", "Generator") if c in df.columns), None)
    model_col = "Model" if "Model" in df.columns else None

    metric_bases: dict[str, list[str]] = {}
    for col in df.columns:
        col_str = str(col)
        if " Mean_TRTR" in col_str:
            base = col_str.replace(" Mean_TRTR", "").strip()
            metric_bases.setdefault(base, []).append("TRTR")
        elif " Mean_TSTR" in col_str:
            base = col_str.replace(" Mean_TSTR", "").strip()
            metric_bases.setdefault(base, []).append("TSTR")

    for _, row in df.iterrows():
        generator = normalize_generator(row[gen_col]) if gen_col else None
        model = normalize_model(row[model_col]) if model_col else None

        for base, evals in metric_bases.items():
            for eval_type in evals:
                mean_col = f"{base} Mean_{eval_type}"
                std_col = f"{base} Std_{eval_type}"
                if mean_col not in df.columns:
                    continue
                mean_val = row.get(mean_col)
                std_val = row.get(std_col, float("nan"))
                metric_name = base.replace(" ", "")
                entry = {
                    "Dataset": dataset,
                    "Generator": generator,
                    "Seed": None,
                    "Experiment": "utility",
                    "Metric": metric_name,
                    "Value": float(mean_val) if pd.notna(mean_val) else None,
                    "Mean": float(mean_val) if pd.notna(mean_val) else None,
                    "Std": float(std_val) if pd.notna(std_val) else None,
                    "Classifier": model if task_type == "classification" else None,
                    "RegressionModel": model if task_type == "regression" else None,
                    "Regressor": model if task_type == "regression" else None,
                    "Leakage": 0,
                    "LeakageLevel": 0,
                    "TrainingPercentage": 100,
                    "EvaluationType": eval_type,
                    "TaskType": task_type,
                    "Category": "Utility",
                    "MetricValue": float(mean_val) if pd.notna(mean_val) else None,
                    "SourceFile": source_file,
                }
                rows.append(entry)

                if eval_type == "TSTR":
                    trtr_col = f"{base} Mean_TRTR"
                    if trtr_col in df.columns and pd.notna(row.get(trtr_col)) and pd.notna(mean_val):
                        gap = float(row[trtr_col]) - float(mean_val)
                        if metric_name in ("RMSE", "MAE", "MSE"):
                            gap = float(mean_val) - float(row[trtr_col])
                        gap_row = entry.copy()
                        gap_row["EvaluationType"] = "Gap"
                        gap_row["Metric"] = f"{metric_name}_Gap"
                        gap_row["Value"] = gap
                        gap_row["Mean"] = gap
                        rows.append(gap_row)

        for gap_col in (
            "Accuracy_Drop",
            "F1_Drop",
            "Precision_Drop",
            "Recall_Drop",
            "R2_Drop",
            "MSE_Increase",
            "RMSE_Increase",
            "MAE_Increase",
        ):
            if gap_col in df.columns and pd.notna(row.get(gap_col)):
                rows.append(
                    {
                        "Dataset": dataset,
                        "Generator": generator,
                        "Seed": None,
                        "Experiment": "utility",
                        "Metric": gap_col,
                        "Value": float(row[gap_col]),
                        "Mean": float(row[gap_col]),
                        "Std": None,
                        "Classifier": model if task_type == "classification" else None,
                        "RegressionModel": model if task_type == "regression" else None,
                        "Leakage": 0,
                        "TrainingPercentage": 100,
                        "EvaluationType": "Gap",
                        "TaskType": task_type,
                        "SourceFile": source_file,
                    }
                )

    return pd.DataFrame(rows)


def _fidelity_from_quality(df: pd.DataFrame, dataset: str, source_file: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)

    gen_col = next(
        (c for c in ("Generator", "Synthetic Model", "Synthetic_Model", "Model") if c in df.columns),
        None,
    )
    score_cols = [c for c in df.columns if "quality" in str(c).lower() or "score" in str(c).lower()]

    for _, row in df.iterrows():
        generator = normalize_generator(row.get(gen_col)) if gen_col else None
        for col in score_cols + [c for c in df.columns if c in FIDELITY_METRICS]:
            val = row.get(col)
            if pd.isna(val):
                continue
            metric = str(col).replace(" ", "_")
            if metric.lower() == "quality_score" or col == "Quality Score":
                metric = "Quality_Score"
            rows.append(
                {
                    "Dataset": dataset,
                    "Generator": generator,
                    "Seed": None,
                    "Experiment": "fidelity",
                    "Metric": metric,
                    "Value": float(val),
                    "Mean": float(val),
                    "Std": None,
                    "Classifier": None,
                    "RegressionModel": None,
                    "Regressor": None,
                    "Leakage": 0,
                    "LeakageLevel": 0,
                    "TrainingPercentage": 100,
                    "EvaluationType": "Score",
                    "Category": "Fidelity",
                    "MetricValue": float(val),
                    "SourceFile": source_file,
                }
            )
    return pd.DataFrame(rows)


def _privacy_from_workbook(path: Path, dataset: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    try:
        sheets = _read_workbook(path)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    ds_name = dataset or path.stem

    for sheet_name, df in sheets.items():
        if df.empty:
            continue

        lower = sheet_name.lower()
        if "summary" in lower:
            gen_col = next((c for c in ("Model", "Generator", "Synthetic_Model") if c in df.columns), None)
            for _, row in df.iterrows():
                generator = normalize_generator(row.get(gen_col)) if gen_col else None
                for metric_col in (
                    "Mean_Distance",
                    "Median_Distance",
                    "Std_Distance",
                    "Min_Distance",
                    "Max_Distance",
                    "Num_Matches",
                ):
                    if metric_col not in df.columns:
                        continue
                    val = row.get(metric_col)
                    if pd.isna(val):
                        continue
                    summary_rows.append(
                        {
                            "Dataset": normalize_dataset_name(row.get("Dataset", ds_name)),
                            "Generator": generator,
                            "Seed": None,
                            "Experiment": "privacy",
                            "Metric": metric_col,
                            "Value": float(val),
                            "Mean": float(val),
                            "Std": None,
                            "Classifier": None,
                            "RegressionModel": None,
                            "Regressor": None,
                            "Leakage": 0,
                            "LeakageLevel": 0,
                            "TrainingPercentage": 100,
                            "EvaluationType": "Score",
                            "Category": "Privacy",
                            "MetricValue": float(val),
                            "SourceFile": str(path),
                        }
                    )
        elif "Mahalanobis_Distance" in df.columns:
            gen_col = "Model" if "Model" in df.columns else None
            for _, row in df.iterrows():
                generator = normalize_generator(row.get(gen_col)) if gen_col else None
                dist = row.get("Mahalanobis_Distance")
                if pd.notna(dist):
                    detail_rows.append(
                        {
                            "Dataset": normalize_dataset_name(row.get("Dataset", ds_name)),
                            "Generator": generator,
                            "Mahalanobis_Distance": float(dist),
                            "SourceFile": str(path),
                            "Sheet": sheet_name,
                        }
                    )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def _summary_from_sheet(df: pd.DataFrame, dataset: str, task_type: str, source_file: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    gen_col = next((c for c in ("Synthetic_Model", "Generator") if c in df.columns), None)
    rows = []
    for _, row in df.iterrows():
        generator = normalize_generator(row.get(gen_col)) if gen_col else None
        for col in df.columns:
            if col == gen_col:
                continue
            val = row.get(col)
            if pd.notna(val):
                rows.append(
                    {
                        "Dataset": dataset,
                        "Generator": generator,
                        "TaskType": task_type,
                        "Metric": col,
                        "Mean": float(val),
                        "SourceFile": source_file,
                    }
                )
    return pd.DataFrame(rows)


def load_master_data(config: PipelineConfig) -> MasterData:
    records = discover_excel_files(config)
    inventory = pd.DataFrame(
        [
            {
                "Path": str(r.path),
                "FileType": r.file_type,
                "Dataset": r.dataset,
                "Priority": r.priority,
            }
            for r in records
        ]
    )

    utility_files = select_utility_files(records)
    privacy_files = [r for r in records if r.file_type == "privacy"]

    utility_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    fidelity_frames: list[pd.DataFrame] = []
    privacy_frames: list[pd.DataFrame] = []
    privacy_detail_frames: list[pd.DataFrame] = []
    raw_sheets: dict[str, pd.DataFrame] = {}

    for record in utility_files:
        path = record.path
        dataset = record.dataset or path.parent.name
        task_type = _task_type_from_dataset(dataset)
        source = str(path.relative_to(config.repo_root))

        try:
            sheets = _read_workbook(path)
        except Exception:
            continue

        for sheet_name, df in sheets.items():
            key = f"{source}::{sheet_name}"
            raw_sheets[key] = df.copy()

            if sheet_name == "All_Comparisons":
                utility_frames.append(_utility_from_comparisons(df, dataset, task_type, source))
            elif sheet_name == "Quality_Metrics":
                fidelity_frames.append(_fidelity_from_quality(df, dataset, source))
            elif sheet_name == "Summary":
                summary_frames.append(_summary_from_sheet(df, dataset, task_type, source))
            elif sheet_name in UTILITY_GENERATOR_SHEETS:
                df_copy = df.copy()
                if "Synthetic_Model" not in df_copy.columns:
                    df_copy["Synthetic_Model"] = sheet_name
                utility_frames.append(_utility_from_comparisons(df_copy, dataset, task_type, source))

    for record in privacy_files:
        dataset = record.dataset or record.path.stem
        summary, detail = _privacy_from_workbook(record.path, dataset)
        if not summary.empty:
            privacy_frames.append(summary)
        if not detail.empty:
            privacy_detail_frames.append(detail)

    return MasterData(
        utility_long=pd.concat(utility_frames, ignore_index=True) if utility_frames else pd.DataFrame(),
        utility_summary=pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        fidelity_long=pd.concat(fidelity_frames, ignore_index=True) if fidelity_frames else pd.DataFrame(),
        privacy_long=pd.concat(privacy_frames, ignore_index=True) if privacy_frames else pd.DataFrame(),
        privacy_detail=pd.concat(privacy_detail_frames, ignore_index=True) if privacy_detail_frames else pd.DataFrame(),
        file_inventory=inventory,
        raw_sheets=raw_sheets,
    )


def build_unified_master(master: MasterData) -> pd.DataFrame:
    """Single standardized long-format DataFrame (Step 1 output)."""
    frames = []
    for df, default_cat in [
        (master.utility_long, "Utility"),
        (master.fidelity_long, "Fidelity"),
        (master.privacy_long, "Privacy"),
    ]:
        if df.empty:
            continue
        chunk = df.copy()
        if "Category" not in chunk.columns:
            chunk["Category"] = default_cat
        if "LeakageLevel" not in chunk.columns:
            chunk["LeakageLevel"] = chunk.get("Leakage", 0)
        if "MetricValue" not in chunk.columns:
            chunk["MetricValue"] = chunk.get("Value", chunk.get("Mean"))
        if "Regressor" not in chunk.columns:
            chunk["Regressor"] = chunk.get("RegressionModel")
        chunk["Dataset"] = chunk["Dataset"].map(normalize_dataset_name)
        frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    unified = pd.concat(frames, ignore_index=True)
    std_cols = [
        "Dataset", "Generator", "Seed", "LeakageLevel", "Metric", "MetricValue",
        "Classifier", "Regressor", "Category", "EvaluationType", "Mean", "Std",
        "TaskType", "SourceFile",
    ]
    for col in std_cols:
        if col not in unified.columns:
            unified[col] = None
    return unified[std_cols + [c for c in unified.columns if c not in std_cols]]


def save_master_data(master: MasterData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    master.utility_long.to_csv(output_dir / "utility_long.csv", index=False)
    master.utility_summary.to_csv(output_dir / "utility_summary.csv", index=False)
    master.fidelity_long.to_csv(output_dir / "fidelity_long.csv", index=False)
    master.privacy_long.to_csv(output_dir / "privacy_long.csv", index=False)
    master.privacy_detail.to_csv(output_dir / "privacy_detail.csv", index=False)
    master.file_inventory.to_csv(output_dir / "file_inventory.csv", index=False)

    unified = build_unified_master(master)
    if not unified.empty:
        unified.to_csv(output_dir / "master_unified.csv", index=False)

    with pd.ExcelWriter(output_dir / "master_merged.xlsx", engine="openpyxl") as writer:
        for name, df in [
            ("Utility", master.utility_long),
            ("UtilitySummary", master.utility_summary),
            ("Fidelity", master.fidelity_long),
            ("Privacy", master.privacy_long),
            ("Unified", unified if not unified.empty else pd.DataFrame()),
            ("Inventory", master.file_inventory),
        ]:
            if not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)
