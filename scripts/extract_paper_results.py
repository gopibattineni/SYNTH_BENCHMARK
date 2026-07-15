"""Extract fidelity, privacy, and utility metrics into paper_results/ Excel workbooks."""

from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "paper results"
DATASETS_JSON = (
    REPO_ROOT
    / "Generators"
    / "Experiment with utility data leak"
    / "python_scripts"
    / "hive"
    / "datasets.json"
)

SDV_DIR = REPO_ROOT / "Generators" / "SDV models"
OTHER_DIR = REPO_ROOT / "Generators" / "Other GANS"
DIFFUSION_DIR = REPO_ROOT / "Generators" / "Diffusion GANs"
UTILITY_DIR = REPO_ROOT / "Generators" / "Experiment with utility data leak" / "utility results"

SDV_GENERATORS = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
OTHER_GENERATORS = ["CTABGAN", "WGAN_GP"]
DIFFUSION_GENERATORS = ["TabDDPM", "ForestDiffusion"]
ALL_GENERATORS = SDV_GENERATORS + OTHER_GENERATORS + DIFFUSION_GENERATORS

GENERATOR_SOURCE = {
    **{g: "SDV" for g in SDV_GENERATORS},
    **{g: "Other_GAN" for g in OTHER_GENERATORS},
    **{g: "Diffusion" for g in DIFFUSION_GENERATORS},
}


@dataclass
class DatasetSources:
    dataset_dir: str
    name: str
    dataset_id: str = ""
    utility_output_file: str = "TRTR_TSTR_results_merged.xlsx"
    sdv_nb: Path | None = None
    other_nb: Path | None = None
    diffusion_nb: Path | None = None
    utility_xlsx: Path | None = None
    mahalanobis_xlsx: list[Path] = field(default_factory=list)


def _dataset_number(name: str) -> int:
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 999


def _find_notebook(folder: Path, number: int) -> Path | None:
    if not folder.is_dir():
        return None
    matches = []
    for p in folder.glob("*.ipynb"):
        n = _dataset_number(p.name)
        if n == number:
            matches.append(p)
    return sorted(matches)[0] if matches else None


def _find_utility_xlsx(dataset_dir: str) -> Path | None:
    folder = UTILITY_DIR / dataset_dir
    if not folder.is_dir():
        return None
    for p in sorted(folder.glob("*.xlsx")):
        return p
    return None


def _find_mahalanobis_files(dataset_dir: str, name: str) -> list[Path]:
    """Locate Mahalanobis / matching Excel files across generator folders."""
    keywords = [
        re.sub(r"^\d+\.\s*", "", dataset_dir).lower().replace(" ", ""),
        name.lower().replace(" ", ""),
    ]
    alias = {
        "cdc diabetes dataset": ["cdc", "diabetes", "heart"],
        "forest cover dataset": ["forest", "cover", "forestcover"],
        "bank markting": ["bank", "marketing"],
        "online shopping": ["shopping", "online"],
        "metro interstate": ["metro", "interstate", "traffic"],
        "concrete compressive strength": ["concrete"],
        "energy efficiency": ["energy"],
        "real estate valuation": ["realestate", "estate"],
        "air quality": ["airquality", "air"],
        "magic gamma telescope": ["magic"],
    }
    ds_key = re.sub(r"^\d+\.\s*", "", dataset_dir).lower()
    keywords.extend(alias.get(ds_key, []))

    found: list[Path] = []
    for root in (SDV_DIR, OTHER_DIR, DIFFUSION_DIR, SDV_DIR / "Excel sheets"):
        if not root.is_dir():
            continue
        for p in root.rglob("*.xlsx"):
            stem = p.stem.lower().replace(" ", "").replace("_", "")
            if "hungarian" in stem or "mahalanobis" in stem or "matching" in stem:
                if any(k.replace(" ", "") in stem for k in keywords if k):
                    found.append(p)
    return sorted(set(found))


def load_dataset_sources() -> list[DatasetSources]:
    entries = json.loads(DATASETS_JSON.read_text(encoding="utf-8"))
    sources: list[DatasetSources] = []
    for entry in entries:
        num = _dataset_number(entry["notebook_dir"])
        ds = DatasetSources(
            dataset_dir=entry["notebook_dir"],
            name=entry["name"],
            dataset_id=entry.get("id", ""),
            utility_output_file=entry.get("output_file", "TRTR_TSTR_results_merged.xlsx"),
            sdv_nb=_find_notebook(SDV_DIR, num),
            other_nb=_find_notebook(OTHER_DIR, num),
            diffusion_nb=_find_notebook(DIFFUSION_DIR, num),
            utility_xlsx=_find_utility_xlsx(entry["notebook_dir"]),
            mahalanobis_xlsx=_find_mahalanobis_files(entry["notebook_dir"], entry["name"]),
        )
        sources.append(ds)
    return sources


def extract_tables_from_notebook(nb_path: Path) -> list[pd.DataFrame]:
    if not nb_path or not nb_path.is_file():
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


def extract_tsne_images(nb_path: Path, out_dir: Path, prefix: str) -> list[str]:
    if not nb_path or not nb_path.is_file():
        return []
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    saved: list[str] = []
    fig_dir = out_dir / "Figures" / "tSNE"
    fig_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "t-SNE" not in src and "tsne" not in src.lower():
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            png = data.get("image/png")
            if not png:
                continue
            idx += 1
            path = fig_dir / f"{prefix}_tsne_{idx}.png"
            path.write_bytes(base64.b64decode(png))
            saved.append(str(path.relative_to(out_dir)))
    return saved


def _cols_set(df: pd.DataFrame) -> set[str]:
    return {str(c) for c in df.columns}


def classify_table(df: pd.DataFrame) -> str | None:
    c = _cols_set(df)
    if {"Model", "Quality Score"}.issubset(c):
        return "quality_overall"
    if {"Model", "Diagnostic Score", "Quality Score"}.issubset(c):
        return "quality_overall"
    if {"Column", "Score", "Model"}.issubset(c) or {"Column", "Metric", "Score", "Model"}.issubset(c):
        return "ks_column_shapes"
    if {"Feature", "JS_Divergence", "Model"}.issubset(c):
        return "js_divergence"
    if {"Feature", "Wasserstein_Distance"}.issubset(c):
        return "wasserstein_detail"
    if {"Model", "Mean_Wasserstein"}.issubset(c):
        return "wasserstein_summary"
    if {"Model", "Avg_Intra_Real", "Avg_Cross_Real_vs_Synth"}.issubset(c):
        return "gower_summary"
    if {"Model", "MMD_Score"}.issubset(c):
        return "mmd_summary"
    if {"Model", "Average_Cosine_Similarity"}.issubset(c):
        return "cosine_summary"
    if {"Model", "Avg_NN_Distance"}.issubset(c):
        return "nearest_neighbors"
    if {"Model", "Avg_Abs_Diff_Outlier_Count"}.issubset(c):
        return "outlier_diff"
    if {"Model", "Mean Error %"}.issubset(c):
        return "pca_mean_errors"
    if {"Model", "TopCorrDiff"}.issubset(c) or {"Model", "MeanTop10CorrDiff"}.issubset(c):
        return "pca_projection"
    if {"Model", "Global_MMD_RBF"}.issubset(c):
        return "multivariate_mmd"
    if {"Model", "AUC", "Advantage"}.issubset(c) and "Accuracy_TRTR" not in c:
        return "mia"
    if {"Model", "Real_Mean_MD_2D"}.issubset(c):
        return "mahalanobis_2d"
    if {"Model", "Num_Matches", "Avg_Cosine_Similarity"}.issubset(c):
        return "hungarian_matching"
    if {"Synthetic_Model", "Accuracy_TRTR", "Accuracy_TSTR"}.issubset(c):
        return "utility_leakage_comparison"
    if {"Model", "Accuracy_TRTR", "Accuracy_TSTR"}.issubset(c):
        return "utility_leakage_comparison"
    if {"Model", "Accuracy", "F1"}.issubset(c) and "Synthetic_Model" not in c and "Accuracy_TRTR" not in c:
        return "utility_classifier_scores"
    return None


def _tag_df(df: pd.DataFrame, source_group: str) -> pd.DataFrame:
    out = df.copy()
    out["Source_Group"] = source_group
    if "Synthetic_Model" in out.columns and "Model" not in out.columns:
        out = out.rename(columns={"Synthetic_Model": "Generator"})
    elif "Model" in out.columns and "Generator" not in out.columns:
        # Keep Model for classifier rows; duplicate as Generator when it's a generator name
        if out["Model"].astype(str).isin(ALL_GENERATORS).any():
            out["Generator"] = out["Model"]
    return out


def parse_notebook_metrics(nb_path: Path, source_group: str, allowed_generators: list[str]) -> dict[str, pd.DataFrame]:
    buckets: dict[str, list[pd.DataFrame]] = {}
    for df in extract_tables_from_notebook(nb_path):
        kind = classify_table(df)
        if not kind:
            continue
        tagged = _tag_df(df, source_group)
        if kind in ("utility_leakage_comparison", "utility_classifier_scores"):
            buckets.setdefault(kind, []).append(tagged)
            continue
        if "Generator" in tagged.columns:
            tagged = tagged[tagged["Generator"].astype(str).isin(allowed_generators)]
        if tagged.empty:
            continue
        buckets.setdefault(kind, []).append(tagged)

    return {k: pd.concat(v, ignore_index=True).drop_duplicates() for k, v in buckets.items()}


def load_mahalanobis_excel(path: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return result
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if df.empty:
            continue
        lower = sheet.lower()
        if "summary" in lower:
            result["mahalanobis_summary"] = df
        elif "mahalanobis_distance" in str(df.columns).lower():
            gen = df["Model"].iloc[0] if "Model" in df.columns else sheet
            result.setdefault("mahalanobis_detail", []).append(df.assign(Sheet=sheet, Generator=gen))
    if "mahalanobis_detail" in result:
        result["mahalanobis_detail"] = pd.concat(result["mahalanobis_detail"], ignore_index=True)
    return result


def load_utility_workbook(path: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    if not path or not path.is_file():
        return result
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return result
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if df.empty:
            continue
        if sheet == "Quality_Metrics":
            result["quality_from_utility"] = df
        elif sheet == "All_Comparisons":
            result["utility_all_comparisons"] = df
        elif sheet == "Summary":
            result["utility_summary"] = df
        elif sheet == "TRTR_Results":
            result["utility_trtr_baseline"] = df
        elif sheet in ALL_GENERATORS:
            result.setdefault("utility_by_generator", []).append(df.assign(Generator=sheet))
    if "utility_by_generator" in result:
        result["utility_by_generator"] = pd.concat(result["utility_by_generator"], ignore_index=True)
    return result


def merge_metric_buckets(all_buckets: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    merged: dict[str, list[pd.DataFrame]] = {}
    for buckets in all_buckets:
        for key, df in buckets.items():
            if df is not None and not df.empty:
                merged.setdefault(key, []).append(df)
    return {k: pd.concat(v, ignore_index=True).drop_duplicates() for k, v in merged.items()}


FIDELITY_PRIVACY_KEYS = {
    "quality_overall",
    "quality_from_utility",
    "ks_column_shapes",
    "js_divergence",
    "wasserstein_summary",
    "wasserstein_detail",
    "gower_summary",
    "mmd_summary",
    "multivariate_mmd",
    "cosine_summary",
    "nearest_neighbors",
    "outlier_diff",
    "pca_mean_errors",
    "pca_projection",
    "mia",
    "mahalanobis_2d",
    "mahalanobis_summary",
    "mahalanobis_detail",
    "hungarian_matching",
    "Fidelity_Summary",
    "Privacy_Summary",
}

SHEET_ORDER = [
    ("quality_overall", "Quality_Scores"),
    ("quality_from_utility", "Quality_Metrics"),
    ("ks_column_shapes", "KS_ColumnShapes"),
    ("js_divergence", "JS_Divergence"),
    ("wasserstein_summary", "Wasserstein_Summary"),
    ("wasserstein_detail", "Wasserstein_Detail"),
    ("gower_summary", "Gower_Distance"),
    ("mmd_summary", "MMD"),
    ("multivariate_mmd", "MMD_Multivariate"),
    ("cosine_summary", "Cosine_Similarity"),
    ("outlier_diff", "Outlier_Count_Diff"),
    ("pca_mean_errors", "PCA_Mean_Errors"),
    ("pca_projection", "PCA_Projections"),
    ("nearest_neighbors", "Nearest_Neighbors"),
    ("mia", "MIA"),
    ("mahalanobis_2d", "Mahalanobis_2D"),
    ("mahalanobis_summary", "Mahalanobis_Summary"),
    ("mahalanobis_detail", "Mahalanobis_Detail"),
    ("hungarian_matching", "Hungarian_Matching"),
    ("Fidelity_Summary", "Fidelity_Summary"),
    ("Privacy_Summary", "Privacy_Summary"),
]

UTILITY_SHEET_ORDER = [
    ("utility_trtr_baseline", "TRTR_Results"),
    ("utility_all_comparisons", "All_Comparisons"),
    ("utility_summary", "Summary"),
    ("utility_by_generator", "By_Generator"),
]


def build_fidelity_summary(metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Merge quality from SDV notebooks + utility-run Quality_Metrics
    quality_frames = []
    for key in ("quality_overall", "quality_from_utility"):
        df = metrics.get(key)
        if df is not None and not df.empty:
            q = df.copy()
            gen_col = "Generator" if "Generator" in q.columns else "Model"
            if "Quality Score" in q.columns:
                quality_frames.append(q[[gen_col, "Quality Score"]].rename(columns={gen_col: "Generator"}))
            elif "Quality_Score" in q.columns:
                quality_frames.append(q[[gen_col, "Quality_Score"]].rename(columns={gen_col: "Generator", "Quality_Score": "Quality Score"}))
    quality_all = pd.concat(quality_frames, ignore_index=True).drop_duplicates(subset=["Generator"]) if quality_frames else pd.DataFrame()

    for gen in ALL_GENERATORS:
        row: dict[str, Any] = {"Generator": gen, "Source": GENERATOR_SOURCE.get(gen, "")}
        if not quality_all.empty:
            qrow = quality_all[quality_all["Generator"].astype(str) == gen]
            if not qrow.empty:
                row["Quality_Score"] = qrow["Quality Score"].iloc[0]
        for key, col in [
            ("wasserstein_summary", "Mean_Wasserstein"),
            ("mmd_summary", "MMD_Score"),
            ("gower_summary", "Avg_Cross_Real_vs_Synth"),
            ("cosine_summary", "Average_Cosine_Similarity"),
            ("outlier_diff", "Avg_Abs_Diff_Outlier_Count"),
            ("pca_mean_errors", "Mean Error %"),
        ]:
            df = metrics.get(key)
            if df is None or df.empty:
                continue
            gen_col = "Generator" if "Generator" in df.columns else "Model"
            sub = df[df[gen_col].astype(str) == gen]
            if sub.empty:
                continue
            if col in sub.columns:
                row[col.replace(" ", "_").replace("%", "pct")] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def build_privacy_summary(metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gen in ALL_GENERATORS:
        row: dict[str, Any] = {"Generator": gen}
        for key, col in [
            ("mia", "AUC"),
            ("nearest_neighbors", "Avg_NN_Distance"),
            ("mahalanobis_2d", "Synth_Mean_MD_2D"),
            ("hungarian_matching", "Avg_Cosine_Similarity"),
        ]:
            df = metrics.get(key)
            if df is None or df.empty:
                continue
            gen_col = "Generator" if "Generator" in df.columns else "Model"
            sub = df[df[gen_col].astype(str) == gen]
            if sub.empty or col not in sub.columns:
                continue
            row[col.replace(" ", "_")] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _write_metrics_workbook(
    path: Path,
    metrics: dict[str, pd.DataFrame],
    sheet_order: list[tuple[str, str]],
    allowed_keys: set[str] | None = None,
) -> list[str]:
    written_sheets: list[str] = []
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        written_keys: set[str] = set()
        for key, sheet_name in sheet_order:
            if allowed_keys is not None and key not in allowed_keys:
                continue
            df = metrics.get(key)
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            written_keys.add(key)
            written_sheets.append(sheet_name)

        for key, df in metrics.items():
            if key in written_keys or df is None or df.empty:
                continue
            if allowed_keys is not None and key not in allowed_keys:
                continue
            sheet_name = key[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            written_keys.add(key)
            written_sheets.append(sheet_name)
    return written_sheets


def copy_utility_workbook(ds: DatasetSources, out_dir: Path) -> Path | None:
    if not ds.utility_xlsx or not ds.utility_xlsx.is_file():
        return None
    target = out_dir / ds.utility_output_file
    shutil.copy2(ds.utility_xlsx, target)
    return target


def export_dataset(ds: DatasetSources) -> dict[str, Any]:
    out_dir = PAPER_ROOT / ds.dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy = out_dir / f"{ds.dataset_dir}_paper_metrics.xlsx"
    if legacy.exists():
        legacy.unlink()

    buckets_list: list[dict[str, pd.DataFrame]] = []
    tsne_saved: list[str] = []

    if ds.sdv_nb:
        buckets_list.append(parse_notebook_metrics(ds.sdv_nb, "SDV", SDV_GENERATORS))
        tsne_saved += extract_tsne_images(ds.sdv_nb, out_dir, "SDV")
    if ds.other_nb:
        buckets_list.append(parse_notebook_metrics(ds.other_nb, "Other_GAN", OTHER_GENERATORS))
        tsne_saved += extract_tsne_images(ds.other_nb, out_dir, "OtherGAN")
    if ds.diffusion_nb:
        buckets_list.append(parse_notebook_metrics(ds.diffusion_nb, "Diffusion", DIFFUSION_GENERATORS))
        tsne_saved += extract_tsne_images(ds.diffusion_nb, out_dir, "Diffusion")

    metrics = merge_metric_buckets(buckets_list)

    for xlsx in ds.mahalanobis_xlsx:
        for k, df in load_mahalanobis_excel(xlsx).items():
            if k in metrics and not metrics[k].empty:
                metrics[k] = pd.concat([metrics[k], df], ignore_index=True).drop_duplicates()
            else:
                metrics[k] = df

    utility = load_utility_workbook(ds.utility_xlsx) if ds.utility_xlsx else {}
    metrics.update(utility)

    metrics["Fidelity_Summary"] = build_fidelity_summary(metrics)
    metrics["Privacy_Summary"] = build_privacy_summary(metrics)

    fidelity_metrics = {k: v for k, v in metrics.items() if k in FIDELITY_PRIVACY_KEYS}
    fidelity_metrics["Fidelity_Summary"] = metrics["Fidelity_Summary"]
    fidelity_metrics["Privacy_Summary"] = metrics["Privacy_Summary"]

    utility_path = copy_utility_workbook(ds, out_dir)
    fidelity_path = out_dir / "fidelity_privacy_metrics.xlsx"
    fidelity_sheets = _write_metrics_workbook(
        fidelity_path,
        fidelity_metrics,
        SHEET_ORDER,
        allowed_keys=FIDELITY_PRIVACY_KEYS,
    )

    # Extraction log
    log = {
        "dataset": ds.dataset_dir,
        "name": ds.name,
        "dataset_id": ds.dataset_id,
        "sdv_notebook": str(ds.sdv_nb) if ds.sdv_nb else None,
        "other_gan_notebook": str(ds.other_nb) if ds.other_nb else None,
        "diffusion_notebook": str(ds.diffusion_nb) if ds.diffusion_nb else None,
        "utility_source": str(ds.utility_xlsx) if ds.utility_xlsx else None,
        "utility_output": str(utility_path) if utility_path else None,
        "fidelity_privacy_output": str(fidelity_path),
        "mahalanobis_files": [str(p) for p in ds.mahalanobis_xlsx],
        "fidelity_sheets_written": fidelity_sheets,
        "tsne_figures": tsne_saved,
    }
    return log


def build_all_paper_results() -> pd.DataFrame:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    logs = [export_dataset(ds) for ds in load_dataset_sources()]
    log_df = pd.DataFrame(logs)
    log_df.to_csv(PAPER_ROOT / "extraction_log.csv", index=False)
    log_df.to_excel(PAPER_ROOT / "extraction_log.xlsx", index=False)
    return log_df


if __name__ == "__main__":
    df = build_all_paper_results()
    print(f"Extracted metrics for {len(df)} datasets into {PAPER_ROOT}")
    cols = [c for c in ["dataset", "utility_output", "fidelity_privacy_output", "fidelity_sheets_written"] if c in df.columns]
    print(df[cols].to_string())
