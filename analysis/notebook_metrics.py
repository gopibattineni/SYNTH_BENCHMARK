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
    "js_summary": "JS_Divergence",
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
    "hungarian_mean_distance": "Mean_Distance",
}

PRIVACY_METRICS = {
    "MIA_AUC",
    "NNDR",
    "Mahalanobis_Distance",
    "Hungarian_Cosine_Similarity",
    "Num_Matches",
    "Mean_Distance",
}


def _dataset_number(name: str) -> int:
    match = re.match(r"(\d+)", name)
    return int(match.group(1)) if match else 999


def _find_notebook(folder: Path, number: int) -> Path | None:
    if not folder.is_dir():
        return None
    matches = [p for p in folder.glob("*.ipynb") if _dataset_number(p.name) == number]
    # Prefer the live notebook over backups / copies
    primary = [
        p
        for p in matches
        if not re.search(r"backup|copy|old|draft", p.name, re.IGNORECASE)
    ]
    pool = primary or matches
    return sorted(pool)[0] if pool else None


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


def _col_names_lower(df: pd.DataFrame) -> set[str]:
    names: set[str] = set()
    for c in df.columns:
        if isinstance(c, tuple):
            names.add(" ".join(str(x) for x in c if str(x) != "nan").lower())
            names.add(str(c[-1]).lower())
        else:
            names.add(str(c).lower())
    return names


def _flatten_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex HTML table headers to single-level names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    new_cols: list[str] = []
    for col in out.columns:
        parts = [
            str(x)
            for x in col
            if str(x) != "nan" and not str(x).startswith("Unnamed")
        ]
        new_cols.append(parts[-1] if parts else str(col[-1]))
    out.columns = new_cols
    return out


def _row_get_ci(row: pd.Series, *candidates: str):
    """Case-insensitive column lookup on a row."""
    lower_map = {str(i).lower(): i for i in row.index}
    for name in candidates:
        key = lower_map.get(name.lower())
        if key is not None and pd.notna(row.get(key)):
            return row.get(key)
    return None


def _mean_distance_from_row(row: pd.Series) -> tuple[float | None, int]:
    """Return (value, priority) for Hungarian mean matching distance.

    Exact ``Mean_Distance`` (Excel-style) ranks highest. Case-insensitive
    ``mean_distance`` alone is lower priority so total-distance dumps do not
    outrank dedicated summary columns.
    """
    exact_map = {str(i): i for i in row.index}
    if "Mean_Distance" in exact_map and pd.notna(row.get(exact_map["Mean_Distance"])):
        try:
            return float(row.get(exact_map["Mean_Distance"])), 4
        except (TypeError, ValueError):
            pass
    for names, priority in (
        (("mean_distance_per_match",), 3),
        (("mean_distance",), 2),
        (("mean_dist",), 1),
    ):
        val = _row_get_ci(row, *names)
        if val is not None:
            try:
                return float(val), priority
            except (TypeError, ValueError):
                continue
    return None, 0


def classify_table(df: pd.DataFrame) -> str | None:
    c = _col_names_lower(df)
    if {"model", "quality score"}.issubset(c) or {"model", "diagnostic score", "quality score"}.issubset(c):
        return "quality_overall"
    if {"column", "score", "model"}.issubset(c) or {"column", "metric", "score", "model"}.issubset(c):
        return "ks_column_shapes"
    if {"model", "mean_js"}.issubset(c):
        return "js_summary"
    if {"feature", "js_divergence", "model"}.issubset(c):
        return "js_divergence"
    if {"model", "mean_wasserstein"}.issubset(c):
        return "wasserstein_summary"
    if {"model", "avg_intra_real", "avg_cross_real_vs_synth"}.issubset(c):
        return "gower_summary"
    if {"model", "avg_cross_similarity"}.issubset(c) or {"model", "avg_gower_similarity"}.issubset(c):
        return "gower_summary"
    if {"model", "mmd_score"}.issubset(c):
        return "mmd_summary"
    if {"model", "global_mmd_rbf"}.issubset(c):
        return "multivariate_mmd"
    if {"model", "average_cosine_similarity"}.issubset(c):
        return "cosine_summary"
    if {"model", "mean error %"}.issubset(c):
        return "pca_mean_errors"
    if {"model", "topcorrdiff"}.issubset(c) or {"model", "meantop10corrdiff"}.issubset(c):
        return "pca_projection"
    if {"model", "auc", "advantage"}.issubset(c) and "accuracy_trtr" not in c:
        return "mia"
    if {"model", "real_mean_md_2d"}.issubset(c):
        return "mahalanobis_2d"
    if {"model", "num_matches"}.issubset(c) and any("cosine" in col for col in c):
        return "hungarian_matching"
    # SDV online-shopping style summary: Model + "Avg Cosine" (no Num_Matches).
    if "model" in c and any(col in c for col in ("avg cosine", "avg_cosine", "avg_cosine_similarity")):
        if "average_cosine_similarity" not in c:  # not the fidelity pairwise cosine table
            return "hungarian_matching"
    # Hungarian Mahalanobis mean matching distance (Mean_Distance / mean_dist).
    if "model" in c and any(
        col in c
        for col in ("mean_distance", "mean_dist", "mean_distance_per_match")
    ):
        return "hungarian_mean_distance"
    if {"model", "avg_nn_distance"}.issubset(c):
        return "nearest_neighbors"
    return None


_MODEL_NAME_RE = (
    r"CTGAN|CopulaGAN|TVAE|GaussianCopula|CTABGAN|WGAN_GP|TabDDPM|ForestDiffusion"
)


def _metric_value(row: pd.Series, kind: str) -> float | None:
    if kind == "quality_overall":
        for col in ("Quality Score", "Quality_Score", "Diagnostic Score"):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
    if kind == "ks_column_shapes":
        return float(row["Score"]) if pd.notna(row.get("Score")) else None
    if kind == "js_divergence":
        return float(row["JS_Divergence"]) if pd.notna(row.get("JS_Divergence")) else None
    if kind == "js_summary":
        return float(row["Mean_JS"]) if pd.notna(row.get("Mean_JS")) else None
    if kind == "wasserstein_summary":
        return float(row["Mean_Wasserstein"]) if pd.notna(row.get("Mean_Wasserstein")) else None
    if kind == "gower_summary":
        for col in (
            "Avg_Cross_Real_vs_Synth",
            "Avg_Cross_Similarity",
            "Avg_Gower_Similarity",
        ):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        return None
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
        val = _row_get_ci(row, "Avg_Cosine_Similarity", "Avg_cosine_similarity")
        return float(val) if val is not None else None
    if kind == "hungarian_mean_distance":
        val, _prio = _mean_distance_from_row(row)
        return val
    if kind == "nearest_neighbors":
        return float(row["Avg_NN_Distance"]) if pd.notna(row.get("Avg_NN_Distance")) else None
    return None


def _outputs_text(outputs: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for out in outputs:
        text = out.get("text")
        if text is None and isinstance(out.get("data"), dict):
            text = out["data"].get("text/plain")
        if text is None:
            continue
        chunks.append(text if isinstance(text, str) else "".join(text))
    return "\n".join(chunks)


def _model_order_from_source(src: str, allowed: list[str]) -> list[str]:
    """Parse generator order from notebook source (model_order / models dict)."""
    allowed_set = set(allowed)
    ordered: list[str] = []

    match = re.search(r"model_order\s*=\s*\[(.*?)\]", src, re.DOTALL)
    if match:
        names = re.findall(rf'["\']({_MODEL_NAME_RE})["\']', match.group(1))
    else:
        names = re.findall(rf'["\']({_MODEL_NAME_RE})["\']', src)

    for name in names:
        gen = normalize_generator(name)
        if gen in allowed_set and gen not in ordered:
            ordered.append(gen)
    return ordered


def extract_column_shapes_scores(nb_path: Path, allowed: list[str]) -> dict[str, float]:
    """
    Read SDMetrics Column Shapes Score from notebook stdout.

    Notebooks often only display ``combined_details.head(20)``, so per-column KS
    HTML tables miss TVAE / GaussianCopula. The overall Column Shapes Score is
    printed for every model and is the correct KS Complement aggregate.
    """
    if not nb_path.is_file():
        return {}

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    preferred: dict[str, float] = {}
    fallback: dict[str, float] = {}

    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        order = _model_order_from_source(src, allowed)
        if not order:
            continue

        blob = _outputs_text(cell.get("outputs", []))
        scores = [float(x) / 100.0 for x in re.findall(r"Column Shapes Score:\s*([0-9.]+)%", blob)]
        if len(scores) < len(order):
            continue

        mapping = {model: score for model, score in zip(order, scores[: len(order)])}
        is_quality_cell = any(
            token in src
            for token in ("QualityReport", 'get_details("Column Shapes")', "Column Shapes")
        )
        target = preferred if is_quality_cell else fallback
        for model, score in mapping.items():
            target.setdefault(model, score)

    out = dict(fallback)
    out.update(preferred)
    return out


def _ks_means_from_tables(tables: list[pd.DataFrame], allowed: list[str]) -> dict[str, float]:
    """Fallback: mean column-shape Score per model from HTML tables."""
    buckets: dict[str, list[float]] = {}
    for table in tables:
        if classify_table(table) != "ks_column_shapes":
            continue
        if "Model" not in table.columns or "Score" not in table.columns:
            continue
        for _, row in table.iterrows():
            generator = normalize_generator(row.get("Model"))
            if generator not in allowed or pd.isna(row.get("Score")):
                continue
            buckets.setdefault(generator, []).append(float(row["Score"]))
    return {gen: float(sum(vals) / len(vals)) for gen, vals in buckets.items() if vals}


def _metric_record(
    *,
    dataset: str,
    generator: str,
    metric_name: str,
    val: float,
    source: str,
    is_privacy: bool,
) -> dict[str, Any]:
    return {
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
        "SourceFile": source,
    }


def _js_means_from_tables(
    tables: list[pd.DataFrame], allowed: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (mean_js_summary, feature_means) for JS divergence."""
    summary: dict[str, float] = {}
    feature_buckets: dict[str, list[float]] = {}
    for table in tables:
        kind = classify_table(table)
        if kind == "js_summary" and "Model" in table.columns and "Mean_JS" in table.columns:
            for _, row in table.iterrows():
                generator = normalize_generator(row.get("Model"))
                if generator not in allowed or pd.isna(row.get("Mean_JS")):
                    continue
                summary[generator] = float(row["Mean_JS"])
        elif kind == "js_divergence" and "Model" in table.columns and "JS_Divergence" in table.columns:
            for _, row in table.iterrows():
                generator = normalize_generator(row.get("Model"))
                if generator not in allowed or pd.isna(row.get("JS_Divergence")):
                    continue
                feature_buckets.setdefault(generator, []).append(float(row["JS_Divergence"]))
    feature_means = {
        gen: float(sum(vals) / len(vals))
        for gen, vals in feature_buckets.items()
        if vals
    }
    return summary, feature_means


def _load_js_from_plots(config: PipelineConfig) -> dict[tuple[str, str], float]:
    """Optional recovered mean JS values from notebook fidelity plots."""
    path = config.repo_root / "Results" / "Supplementary" / "js_divergence_from_plots.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    out: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        dataset = normalize_dataset_name(row.get("Dataset"))
        generator = normalize_generator(row.get("Generator"))
        val = row.get("MetricValue")
        if not dataset or not generator or pd.isna(val):
            continue
        out[(dataset, generator)] = float(val)
    return out


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
    js_from_plots = _load_js_from_plots(config)
    skip_kinds = {"ks_column_shapes", "js_divergence", "js_summary"}

    for entry in entries:
        dataset = entry["notebook_dir"]
        num = _dataset_number(dataset)
        seen: set[tuple] = set()
        mean_dist_priority: dict[tuple[str, str], int] = {}

        for folder, allowed in NOTEBOOK_GROUPS:
            nb = _find_notebook(folder, num)
            if nb is None:
                continue
            source = str(nb.relative_to(config.repo_root))
            tables = [_flatten_table_columns(t) for t in _extract_tables(nb)]

            for table in tables:
                kind = classify_table(table)
                if kind is None or kind in skip_kinds:
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

                    # Hungarian tables carry both match count and cosine similarity.
                    metric_values: list[tuple[str, float, int]] = []
                    if kind == "hungarian_matching":
                        cos = _row_get_ci(
                            row,
                            "Avg_Cosine_Similarity",
                            "Avg_cosine_similarity",
                            "Avg Cosine",
                            "Avg_Cosine",
                            "avg_cosine",
                        )
                        matches = _row_get_ci(row, "Num_Matches")
                        mean_d, mean_prio = _mean_distance_from_row(row)
                        if cos is not None:
                            metric_values.append(("Hungarian_Cosine_Similarity", float(cos), 0))
                        if matches is not None:
                            metric_values.append(("Num_Matches", float(matches), 0))
                        if mean_d is not None:
                            metric_values.append(("Mean_Distance", mean_d, mean_prio))
                    elif kind == "hungarian_mean_distance":
                        mean_d, mean_prio = _mean_distance_from_row(row)
                        matches = _row_get_ci(row, "Num_Matches", "n_matches")
                        if mean_d is not None:
                            metric_values.append(("Mean_Distance", mean_d, mean_prio))
                        if matches is not None:
                            metric_values.append(("Num_Matches", float(matches), 0))
                    else:
                        val = _metric_value(row, kind)
                        if val is not None and metric_name:
                            metric_values.append((metric_name, val, 0))

                    for out_metric, val, prio in metric_values:
                        key = (dataset, generator, out_metric)
                        if out_metric == "Mean_Distance":
                            prev = mean_dist_priority.get((dataset, generator), -1)
                            if key in seen and prio <= prev:
                                continue
                            mean_dist_priority[(dataset, generator)] = prio
                            if key in seen:
                                priv_rows[:] = [
                                    r
                                    for r in priv_rows
                                    if not (
                                        r["Dataset"] == dataset
                                        and r["Generator"] == generator
                                        and r["Metric"] == "Mean_Distance"
                                    )
                                ]
                        elif key in seen:
                            continue
                        seen.add(key)
                        record = _metric_record(
                            dataset=dataset,
                            generator=generator,
                            metric_name=out_metric,
                            val=val,
                            source=source,
                            is_privacy=out_metric in PRIVACY_METRICS or is_privacy,
                        )
                        if record["Category"] == "Privacy":
                            priv_rows.append(record)
                        else:
                            fid_rows.append(record)

            # Prefer SDMetrics Column Shapes Score (covers TVAE / GaussianCopula).
            ks_values = _ks_means_from_tables(tables, allowed)
            ks_values.update(extract_column_shapes_scores(nb, allowed))
            for generator, val in ks_values.items():
                key = (dataset, generator, "KS_Complement")
                if key in seen:
                    continue
                seen.add(key)
                fid_rows.append(
                    _metric_record(
                        dataset=dataset,
                        generator=generator,
                        metric_name="KS_Complement",
                        val=val,
                        source=source,
                        is_privacy=False,
                    )
                )

            # JS priority: Mean_JS table > plot OCR (full mean) > truncated feature HTML.
            js_summary, js_features = _js_means_from_tables(tables, allowed)
            for generator in allowed:
                plot_val = js_from_plots.get((dataset, generator))
                if generator in js_summary:
                    val = js_summary[generator]
                    source_js = source
                elif plot_val is not None:
                    val = plot_val
                    source_js = "Results/Supplementary/js_divergence_from_plots.csv"
                elif generator in js_features:
                    val = js_features[generator]
                    source_js = source
                else:
                    continue
                key = (dataset, generator, "JS_Divergence")
                if key in seen:
                    continue
                seen.add(key)
                fid_rows.append(
                    _metric_record(
                        dataset=dataset,
                        generator=generator,
                        metric_name="JS_Divergence",
                        val=val,
                        source=source_js,
                        is_privacy=False,
                    )
                )

    # Plot-recovered JS for dataset/generator pairs not present in notebook tables.
    for (dataset, generator), val in js_from_plots.items():
        key = (dataset, generator, "JS_Divergence")
        # Avoid duplicates if already added above
        if any(
            r["Dataset"] == dataset and r["Generator"] == generator and r["Metric"] == "JS_Divergence"
            for r in fid_rows
        ):
            continue
        fid_rows.append(
            _metric_record(
                dataset=dataset,
                generator=generator,
                metric_name="JS_Divergence",
                val=val,
                source="Results/Supplementary/js_divergence_from_plots.csv",
                is_privacy=False,
            )
        )

    return pd.DataFrame(fid_rows), pd.DataFrame(priv_rows)
