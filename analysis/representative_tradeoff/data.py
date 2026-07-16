"""Build classifier-independent and best-classifier score tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import GENERATORS, PipelineConfig
from analysis.data_loader import normalize_dataset_name, normalize_generator
from analysis.feature_fidelity import load_quality_from_notebooks
from analysis.metrics_utils import confidence_interval
from analysis.representative_tradeoff.config import (
    DATASETS,
    DATASET_LABELS,
    UTILITY_METRICS,
    RepresentativeTradeoffConfig,
)


def _load_utility_tstr(config: RepresentativeTradeoffConfig) -> pd.DataFrame:
    """Load TSTR utility rows for Cancer & Mushroom (Mean/Std already seed-averaged)."""
    frames: list[pd.DataFrame] = []
    for ds_key, dataset_id in DATASETS.items():
        path = (
            config.output_root.parent
            / "Two_Datasets_Assessment"
            / ds_key
            / "Processed_Data"
            / "master_unified.csv"
        )
        if not path.is_file():
            # Fallback: Master_Data utility_long
            continue
        df = pd.read_csv(path)
        util = df[df["Category"] == "Utility"].copy()
        util = util[util["EvaluationType"] == "TSTR"]
        util = util[util["Metric"].isin(UTILITY_METRICS)]
        util["Dataset"] = dataset_id
        util["DatasetKey"] = ds_key
        frames.append(util)

    if not frames:
        # Master utility_long fallback
        master = config.repo_root / "Results" / "Master_Data" / "utility_long.csv"
        if master.is_file():
            df = pd.read_csv(master)
            df["Dataset"] = df["Dataset"].map(normalize_dataset_name)
            df = df[df["Dataset"].isin(DATASETS.values())]
            df = df[df["EvaluationType"] == "TSTR"]
            df = df[df["Metric"].isin(UTILITY_METRICS)]
            id_to_key = {v: k for k, v in DATASETS.items()}
            df["DatasetKey"] = df["Dataset"].map(id_to_key)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["Generator"] = out["Generator"].map(normalize_generator)
    out["Mean"] = pd.to_numeric(out["Mean"], errors="coerce")
    out["Std"] = pd.to_numeric(out.get("Std"), errors="coerce")
    return out.dropna(subset=["Mean", "Generator", "Classifier"])


def _load_nndr(config: RepresentativeTradeoffConfig) -> pd.DataFrame:
    path = config.repo_root / "Results" / "Master_Data" / "privacy_long.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["Dataset"] = df["Dataset"].map(normalize_dataset_name)
    df = df[(df["Dataset"].isin(DATASETS.values())) & (df["Metric"] == "NNDR")].copy()
    df["Generator"] = df["Generator"].map(normalize_generator)
    val = df["Mean"] if "Mean" in df.columns else df.get("MetricValue")
    df["NNDR"] = pd.to_numeric(val, errors="coerce")
    id_to_key = {v: k for k, v in DATASETS.items()}
    df["DatasetKey"] = df["Dataset"].map(id_to_key)
    return df.dropna(subset=["NNDR", "Generator"])[["Dataset", "DatasetKey", "Generator", "NNDR"]]


def _load_quality(config: RepresentativeTradeoffConfig) -> pd.DataFrame:
    q = load_quality_from_notebooks(PipelineConfig(repo_root=config.repo_root))
    if q.empty:
        return pd.DataFrame()
    q = q.copy()
    q["Dataset"] = q["Dataset"].map(normalize_dataset_name)
    q = q[q["Dataset"].isin(DATASETS.values())]
    q["Generator"] = q["Generator"].map(normalize_generator)
    q["Quality"] = pd.to_numeric(q["Mean"], errors="coerce")
    id_to_key = {v: k for k, v in DATASETS.items()}
    q["DatasetKey"] = q["Dataset"].map(id_to_key)
    return q.dropna(subset=["Quality", "Generator"])[
        ["Dataset", "DatasetKey", "Generator", "Quality"]
    ]


def _ci_cols(mean: float, std: float, n: int) -> tuple[float, float]:
    lo, hi = confidence_interval(mean, std if pd.notna(std) else 0.0, n)
    return float(lo), float(hi)


def build_classifier_independent_scores(
    config: RepresentativeTradeoffConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analysis A: average utility metrics across all classifiers (seed means already
    aggregated in source Mean/Std), then attach NNDR and Quality Score.

    Returns (scores, utility_by_metric) where scores is one row per Dataset×Generator.
    """
    config = config or RepresentativeTradeoffConfig()
    util = _load_utility_tstr(config)
    nndr = _load_nndr(config)
    quality = _load_quality(config)
    if util.empty:
        return pd.DataFrame(), pd.DataFrame()

    primary = config.primary_utility_metric
    n_seeds = config.n_seeds

    # Per Dataset × Generator × Metric: mean across classifiers
    by_metric = (
        util.groupby(["Dataset", "DatasetKey", "Generator", "Metric"], dropna=False)
        .agg(
            Mean=("Mean", "mean"),
            Std_AcrossClassifiers=("Mean", "std"),
            SeedStd=("Std", "mean"),
            N_Classifiers=("Mean", "count"),
        )
        .reset_index()
    )

    # Primary F1 row for trade-off figures
    primary_df = by_metric[by_metric["Metric"] == primary].copy()
    if primary_df.empty:
        primary_df = by_metric[by_metric["Metric"] == "Accuracy"].copy()
        primary = "Accuracy"

    rows: list[dict] = []
    for _, row in primary_df.iterrows():
        seed_std = float(row["SeedStd"]) if pd.notna(row["SeedStd"]) else float(row["Std_AcrossClassifiers"] or 0)
        clf_std = float(row["Std_AcrossClassifiers"]) if pd.notna(row["Std_AcrossClassifiers"]) else 0.0
        # Prefer seed uncertainty for CI (protocol: 10 seeds), fall back to classifier std
        use_std = seed_std if seed_std > 0 else clf_std
        ci_lo, ci_hi = _ci_cols(float(row["Mean"]), use_std, n_seeds)
        rows.append(
            {
                "Dataset": row["Dataset"],
                "DatasetKey": row["DatasetKey"],
                "DatasetLabel": DATASET_LABELS.get(row["DatasetKey"], row["DatasetKey"]),
                "Generator": row["Generator"],
                "Analysis": "Classifier_Independent",
                "UtilityMetric": primary,
                "F1": float(row["Mean"]),
                "F1_Std": use_std,
                "F1_Std_Classifiers": clf_std,
                "F1_CI_Low": ci_lo,
                "F1_CI_High": ci_hi,
                "N_Classifiers": int(row["N_Classifiers"]),
                "BestClassifier": "ALL",
            }
        )
        # also store under metric name when primary is F1
        if primary == "F1":
            rows[-1]["F1_SeedStd"] = seed_std

    scores = pd.DataFrame(rows)
    scores = scores.merge(nndr, on=["Dataset", "DatasetKey", "Generator"], how="left")
    scores = scores.merge(quality, on=["Dataset", "DatasetKey", "Generator"], how="left")

    # Attach other utility metrics as wide columns for tables
    for metric in UTILITY_METRICS:
        if metric == "F1" and primary == "F1":
            continue  # already stored as F1
        sub = by_metric[by_metric["Metric"] == metric][
            ["DatasetKey", "Generator", "Mean", "Std_AcrossClassifiers", "SeedStd"]
        ].rename(
            columns={
                "Mean": metric,
                "Std_AcrossClassifiers": f"{metric}_Std_Classifiers",
                "SeedStd": f"{metric}_SeedStd",
            }
        )
        scores = scores.merge(sub, on=["DatasetKey", "Generator"], how="left")

    return scores, by_metric


def build_best_classifier_scores(
    config: RepresentativeTradeoffConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analysis B: for each Dataset×Generator pick the classifier with highest mean F1
    (fallback Accuracy), then attach NNDR and Quality.
    """
    config = config or RepresentativeTradeoffConfig()
    util = _load_utility_tstr(config)
    nndr = _load_nndr(config)
    quality = _load_quality(config)
    if util.empty:
        return pd.DataFrame(), pd.DataFrame()

    primary = config.primary_utility_metric
    n_seeds = config.n_seeds

    prim = util[util["Metric"] == primary].copy()
    if prim.empty:
        prim = util[util["Metric"] == "Accuracy"].copy()
        primary = "Accuracy"

    # Best classifier per Dataset × Generator
    idx = prim.groupby(["DatasetKey", "Generator"])["Mean"].idxmax()
    best = prim.loc[idx].copy()
    best_lookup = best.set_index(["DatasetKey", "Generator"])["Classifier"].to_dict()

    # All metrics for the chosen classifier
    detail_rows: list[dict] = []
    score_rows: list[dict] = []
    for (ds_key, gen), clf in best_lookup.items():
        sub = util[(util["DatasetKey"] == ds_key) & (util["Generator"] == gen) & (util["Classifier"] == clf)]
        metric_map = {r.Metric: r for r in sub.itertuples()}
        f1_row = metric_map.get(primary) or metric_map.get("F1") or metric_map.get("Accuracy")
        if f1_row is None:
            continue
        use_std = float(f1_row.Std) if pd.notna(getattr(f1_row, "Std", np.nan)) else 0.0
        ci_lo, ci_hi = _ci_cols(float(f1_row.Mean), use_std, n_seeds)
        ds_id = DATASETS[ds_key]
        rec = {
            "Dataset": ds_id,
            "DatasetKey": ds_key,
            "DatasetLabel": DATASET_LABELS.get(ds_key, ds_key),
            "Generator": gen,
            "Analysis": "Best_Classifier",
            "UtilityMetric": primary,
            "BestClassifier": clf,
            "F1": float(f1_row.Mean),
            "F1_Std": use_std,
            "F1_CI_Low": ci_lo,
            "F1_CI_High": ci_hi,
            "N_Classifiers": 1,
        }
        for metric in UTILITY_METRICS:
            mrow = metric_map.get(metric)
            if mrow is not None:
                rec[metric] = float(mrow.Mean)
                rec[f"{metric}_SeedStd"] = float(mrow.Std) if pd.notna(mrow.Std) else np.nan
                detail_rows.append(
                    {
                        "DatasetKey": ds_key,
                        "Generator": gen,
                        "Classifier": clf,
                        "Metric": metric,
                        "Mean": float(mrow.Mean),
                        "Std": float(mrow.Std) if pd.notna(mrow.Std) else np.nan,
                    }
                )
        score_rows.append(rec)

    scores = pd.DataFrame(score_rows)
    scores = scores.merge(nndr, on=["Dataset", "DatasetKey", "Generator"], how="left")
    scores = scores.merge(quality, on=["Dataset", "DatasetKey", "Generator"], how="left")
    detail = pd.DataFrame(detail_rows)
    return scores, detail


def add_normalized_and_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Add 0–1 normalized NNDR (higher better) and overall rank within each dataset."""
    if scores.empty:
        return scores
    out = scores.copy()
    # Normalize NNDR globally across both datasets for fair radar/heatmap axes
    nndr = out["NNDR"].astype(float)
    vmin, vmax = nndr.min(), nndr.max()
    if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin:
        out["NNDR_Norm"] = ((nndr - vmin) / (vmax - vmin)).clip(0, 1)
    else:
        out["NNDR_Norm"] = 0.5
    out["Quality_Norm"] = out["Quality"].clip(0, 1)
    out["F1_Norm"] = out["F1"].clip(0, 1)
    # Simple equal-weight overall for ranking display
    out["Overall"] = (out["F1_Norm"] + out["NNDR_Norm"] + out["Quality_Norm"]) / 3.0
    out["Rank"] = out.groupby("DatasetKey")["Overall"].rank(ascending=False, method="min").astype(int)
    return out


def save_processed(scores: pd.DataFrame, extra: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> None:
    scores.to_csv(dirs["processed"] / "scores.csv", index=False)
    scores.to_excel(dirs["processed"] / "scores.xlsx", index=False)
    for name, df in extra.items():
        if df is None or df.empty:
            continue
        df.to_csv(dirs["processed"] / f"{name}.csv", index=False)
