"""Load and aggregate utility Excel results for Utility Drop Analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import GENERATORS, PipelineConfig
from analysis.data_loader import (
    load_master_data,
    normalize_dataset_name,
    normalize_generator,
    normalize_model,
)
from analysis.utility_drop.config import (
    CLASSIFICATION_METRICS,
    FOCUS_DATASETS,
    UtilityDropConfig,
)

# Harmonize classifier labels across Excel exports
CLASSIFIER_ALIASES = {
    "LogReg": "Logistic Regression",
    "LogisticRegression": "Logistic Regression",
    "SVM-RBF": "SVM-RBF",
    "SVC-RBF": "SVM-RBF",
    "SVM": "SVM-RBF",
    "ExtraTrees": "Extra Trees",
    "RandomForest": "Random Forest",
    "DecisionTree": "Decision Tree",
    "GradientBoost": "Gradient Boosting",
    "NaiveBayes": "Naive Bayes",
    "GaussianNB": "Naive Bayes",
}


def _normalize_classifier(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in CLASSIFIER_ALIASES:
        return CLASSIFIER_ALIASES[text]
    mapped = normalize_model(text)
    return CLASSIFIER_ALIASES.get(mapped or text, mapped or text)


def load_utility_long(config: UtilityDropConfig | None = None) -> pd.DataFrame:
    """Load classification utility rows from master Excel merge (all datasets)."""
    config = config or UtilityDropConfig()
    pipe = PipelineConfig(repo_root=config.repo_root)
    master = load_master_data(pipe)
    df = master.utility_long.copy()
    if df.empty:
        # Fallback to saved CSV
        path = config.repo_root / "Results" / "Master_Data" / "utility_long.csv"
        if path.is_file():
            df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df

    df["Dataset"] = df["Dataset"].map(normalize_dataset_name)
    df["Generator"] = df["Generator"].map(normalize_generator)
    df["Classifier"] = df["Classifier"].map(_normalize_classifier)
    if "Leakage" not in df.columns:
        df["Leakage"] = 0
    if "LeakageLevel" not in df.columns:
        df["LeakageLevel"] = df["Leakage"]
    df["Leakage"] = pd.to_numeric(df["Leakage"], errors="coerce").fillna(0)
    df["LeakageLevel"] = pd.to_numeric(df["LeakageLevel"], errors="coerce").fillna(df["Leakage"])
    df["Mean"] = pd.to_numeric(df.get("Mean", df.get("MetricValue")), errors="coerce")
    df["Std"] = pd.to_numeric(df.get("Std"), errors="coerce")

    # Classification TSTR + drops
    keep_metrics = set(CLASSIFICATION_METRICS) | {
        "Accuracy_Drop",
        "F1_Drop",
        "Precision_Drop",
        "Recall_Drop",
        "Accuracy_Gap",
        "F1_Gap",
    }
    out = df[
        (df["TaskType"].fillna("classification") == "classification")
        | (df["Classifier"].notna())
    ].copy()
    out = out[out["Metric"].isin(keep_metrics)]
    out = out[out["Generator"].isin(GENERATORS)]
    out = out.dropna(subset=["Mean", "Generator", "Dataset"])
    return out


def build_tstr_table(utility: pd.DataFrame, metric: str = "Accuracy") -> pd.DataFrame:
    """One row per Dataset × Generator × Classifier × Leakage for TSTR metric."""
    tstr = utility[
        (utility["EvaluationType"] == "TSTR") & (utility["Metric"] == metric)
    ].copy()
    if tstr.empty:
        return tstr
    # Aggregate duplicate sources: mean of means
    agg = (
        tstr.groupby(["Dataset", "Generator", "Classifier", "Leakage"], dropna=False)
        .agg(Mean=("Mean", "mean"), Std=("Std", "mean"), N=("Mean", "count"))
        .reset_index()
    )
    return agg


def build_trtr_table(utility: pd.DataFrame, metric: str = "Accuracy") -> pd.DataFrame:
    trtr = utility[
        (utility["EvaluationType"] == "TRTR") & (utility["Metric"] == metric)
    ].copy()
    if trtr.empty:
        return trtr
    return (
        trtr.groupby(["Dataset", "Generator", "Classifier", "Leakage"], dropna=False)
        .agg(Mean=("Mean", "mean"), Std=("Std", "mean"))
        .reset_index()
        .rename(columns={"Mean": "TRTR_Mean", "Std": "TRTR_Std"})
    )


def attach_utility_loss(
    tstr: pd.DataFrame,
    trtr: pd.DataFrame,
) -> pd.DataFrame:
    """Utility Loss = TRTR − TSTR (same leakage). Also % drop vs TRTR."""
    if tstr.empty:
        return tstr
    merged = tstr.merge(
        trtr,
        on=["Dataset", "Generator", "Classifier", "Leakage"],
        how="left",
    )
    if "TRTR_Mean" in merged.columns:
        merged["Utility_Loss"] = merged["TRTR_Mean"] - merged["Mean"]
        merged["Utility_Drop_Pct"] = np.where(
            merged["TRTR_Mean"].abs() > 1e-12,
            100.0 * merged["Utility_Loss"] / merged["TRTR_Mean"],
            np.nan,
        )
    else:
        merged["Utility_Loss"] = np.nan
        merged["Utility_Drop_Pct"] = np.nan
    return merged


def generator_summary(tstr: pd.DataFrame) -> pd.DataFrame:
    """Mean Accuracy ± SD across classifiers (and datasets if multiple)."""
    if tstr.empty:
        return tstr
    return (
        tstr.groupby(["Dataset", "Generator", "Leakage"], dropna=False)
        .agg(
            Mean=("Mean", "mean"),
            Std=("Std", "mean"),
            Std_AcrossClassifiers=("Mean", "std"),
            N_Classifiers=("Classifier", "nunique"),
        )
        .reset_index()
    )


def classifier_summary(tstr: pd.DataFrame) -> pd.DataFrame:
    if tstr.empty:
        return tstr
    return (
        tstr.groupby(["Dataset", "Classifier", "Leakage"], dropna=False)
        .agg(
            Mean=("Mean", "mean"),
            Std=("Std", "mean"),
            Std_AcrossGenerators=("Mean", "std"),
            N_Generators=("Generator", "nunique"),
        )
        .reset_index()
    )


def best_classifier_per_generator(tstr: pd.DataFrame) -> pd.DataFrame:
    if tstr.empty:
        return tstr
    idx = tstr.groupby(["Dataset", "Generator", "Leakage"])["Mean"].idxmax()
    return tstr.loc[idx].reset_index(drop=True)


def best_generator_per_classifier(tstr: pd.DataFrame) -> pd.DataFrame:
    if tstr.empty:
        return tstr
    idx = tstr.groupby(["Dataset", "Classifier", "Leakage"])["Mean"].idxmax()
    return tstr.loc[idx].reset_index(drop=True)


def leakage_levels_present(utility: pd.DataFrame) -> list[float]:
    if utility.empty or "Leakage" not in utility.columns:
        return [0.0]
    return sorted(pd.to_numeric(utility["Leakage"], errors="coerce").dropna().unique())


def focus_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Dataset" not in df.columns:
        return df
    return df[df["Dataset"].isin(FOCUS_DATASETS.values())].copy()


def add_dataset_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    inv = {v: k for k, v in FOCUS_DATASETS.items()}
    out["DatasetKey"] = out["Dataset"].map(inv)
    return out


def reconstruct_seed_samples(mean: float, std: float, n: int = 10, rng: np.random.Generator | None = None) -> np.ndarray:
    """Approximate seed draws from exported Mean±SD (raw seeds not in Excel)."""
    rng = rng or np.random.default_rng(42)
    std = float(std) if pd.notna(std) and std > 0 else 1e-6
    return rng.normal(loc=float(mean), scale=std, size=n)
