"""Load Utility / Fidelity / Privacy and expand to Generator × Seed points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import GENERATORS, PipelineConfig
from analysis.data_loader import load_master_data, normalize_dataset_name, normalize_generator
from analysis.correlation_tradeoff.config import (
    DATASET_LABELS,
    FIDELITY_PREFERRED,
    FOCUS_DATASETS,
    PRIVACY_ANALYSES,
    CorrelationTradeoffConfig,
)
from analysis.utility_drop.data import reconstruct_seed_samples


def _load_long(name: str, config: CorrelationTradeoffConfig) -> pd.DataFrame:
    """Prefer saved MasterData CSVs; fall back to Excel discovery only if needed."""
    for base in (
        config.repo_root / "Results" / "MasterData",
        config.repo_root / "Results" / "Master_Data",
    ):
        path = base / f"{name}.csv"
        if path.is_file():
            return pd.read_csv(path, low_memory=False)

    pipe = PipelineConfig(repo_root=config.repo_root)
    try:
        master = load_master_data(pipe)
        frame = getattr(master, name, pd.DataFrame())
        if frame is not None and not frame.empty:
            return frame.copy()
    except Exception:
        pass
    return pd.DataFrame()


def _focus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Dataset" not in df.columns:
        return df
    out = df.copy()
    out["Dataset"] = out["Dataset"].map(normalize_dataset_name)
    out = out[out["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    inv = {v: k for k, v in FOCUS_DATASETS.items()}
    out["DatasetKey"] = out["Dataset"].map(inv)
    out["DatasetLabel"] = out["DatasetKey"].map(DATASET_LABELS)
    return out


def load_utility_means(config: CorrelationTradeoffConfig, metric: str = "Accuracy") -> pd.DataFrame:
    """Mean TSTR metric (±Std) averaged across classifiers — one row per Dataset×Generator."""
    util = _focus(_load_long("utility_long", config))
    if util.empty:
        return pd.DataFrame()
    util["Generator"] = util["Generator"].map(normalize_generator)
    util["Mean"] = pd.to_numeric(util.get("Mean", util.get("MetricValue")), errors="coerce")
    util["Std"] = pd.to_numeric(util.get("Std"), errors="coerce")
    tstr = util[
        (util["EvaluationType"] == "TSTR")
        & (util["Metric"] == metric)
        & (util["Generator"].isin(GENERATORS))
    ].copy()
    if tstr.empty:
        return pd.DataFrame()
    # Prefer Leakage == 0 when present
    if "Leakage" in tstr.columns:
        leak0 = tstr[pd.to_numeric(tstr["Leakage"], errors="coerce").fillna(0) == 0]
        if not leak0.empty:
            tstr = leak0
    agg = (
        tstr.groupby(["Dataset", "DatasetKey", "DatasetLabel", "Generator"], dropna=False)
        .agg(
            UtilityMean=("Mean", "mean"),
            UtilityStd=("Std", "mean"),
            N_Classifiers=("Mean", "count"),
        )
        .reset_index()
    )
    agg["UtilityMetric"] = metric
    return agg


def load_fidelity(config: CorrelationTradeoffConfig) -> pd.DataFrame:
    fid = _focus(_load_long("fidelity_long", config))
    if fid.empty:
        return pd.DataFrame()
    fid["Generator"] = fid["Generator"].map(normalize_generator)
    fid["Mean"] = pd.to_numeric(fid.get("Mean", fid.get("MetricValue")), errors="coerce")
    fid = fid[fid["Generator"].isin(GENERATORS)]

    chosen_metric = None
    for m in FIDELITY_PREFERRED:
        if (fid["Metric"] == m).any():
            chosen_metric = m
            break
    if chosen_metric is None:
        return pd.DataFrame()

    sub = fid[fid["Metric"] == chosen_metric].copy()
    out = (
        sub.groupby(["Dataset", "DatasetKey", "DatasetLabel", "Generator"], dropna=False)
        .agg(Fidelity=("Mean", "mean"))
        .reset_index()
    )
    out["FidelityMetric"] = chosen_metric
    return out


def load_privacy(config: CorrelationTradeoffConfig, privacy_key: str) -> pd.DataFrame:
    meta = PRIVACY_ANALYSES[privacy_key]
    priv = _focus(_load_long("privacy_long", config))
    if priv.empty:
        return pd.DataFrame()
    priv["Generator"] = priv["Generator"].map(normalize_generator)
    priv["Mean"] = pd.to_numeric(priv.get("Mean", priv.get("MetricValue")), errors="coerce")
    priv = priv[
        (priv["Generator"].isin(GENERATORS)) & (priv["Metric"] == meta["metric"])
    ].copy()
    if priv.empty:
        return pd.DataFrame()
    out = (
        priv.groupby(["Dataset", "DatasetKey", "DatasetLabel", "Generator"], dropna=False)
        .agg(Privacy=("Mean", "mean"))
        .reset_index()
    )
    out["PrivacyMetric"] = meta["metric"]
    out["PrivacyLabel"] = meta["label"]
    out["HigherIsPrivate"] = meta["higher_is_private"]
    return out


def build_seed_level_panel(
    config: CorrelationTradeoffConfig,
    utility_metric: str,
    privacy_key: str,
) -> pd.DataFrame:
    """
    One row per Dataset × Generator × Seed.

    Utility seeds are reconstructed from exported Mean±SD (raw seeds are not
    stored in Excel). Fidelity and privacy are generator-level scalars joined
    onto each seed (no per-seed fidelity/privacy exports available).
    """
    util = load_utility_means(config, utility_metric)
    fid = load_fidelity(config)
    priv = load_privacy(config, privacy_key)
    if util.empty or fid.empty or priv.empty:
        return pd.DataFrame()

    base = util.merge(
        fid[["Dataset", "Generator", "Fidelity", "FidelityMetric"]],
        on=["Dataset", "Generator"],
        how="inner",
    ).merge(
        priv[
            [
                "Dataset",
                "Generator",
                "Privacy",
                "PrivacyMetric",
                "PrivacyLabel",
                "HigherIsPrivate",
            ]
        ],
        on=["Dataset", "Generator"],
        how="inner",
    )
    if base.empty:
        return pd.DataFrame()

    import hashlib

    seeds = list(range(42, 42 + int(config.n_seeds)))
    rows: list[dict] = []
    for _, row in base.iterrows():
        # Deterministic RNG per Dataset×Generator so figures are reproducible
        key = f"{row['Dataset']}|{row['Generator']}|{utility_metric}"
        digest = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(digest ^ int(config.seed))
        samples = reconstruct_seed_samples(
            float(row["UtilityMean"]),
            float(row["UtilityStd"]) if pd.notna(row["UtilityStd"]) else 0.0,
            n=config.n_seeds,
            rng=rng,
        )
        samples = np.clip(samples, 0.0, 1.0)
        for seed, util_val in zip(seeds, samples):
            rows.append(
                {
                    "Dataset": row["Dataset"],
                    "DatasetKey": row["DatasetKey"],
                    "DatasetLabel": row["DatasetLabel"],
                    "Generator": row["Generator"],
                    "Seed": int(seed),
                    "Utility": float(util_val),
                    "UtilityMean": float(row["UtilityMean"]),
                    "UtilityStd": float(row["UtilityStd"]) if pd.notna(row["UtilityStd"]) else np.nan,
                    "UtilityMetric": utility_metric,
                    "Fidelity": float(row["Fidelity"]),
                    "FidelityMetric": row["FidelityMetric"],
                    "Privacy": float(row["Privacy"]),
                    "PrivacyMetric": row["PrivacyMetric"],
                    "PrivacyLabel": row["PrivacyLabel"],
                    "HigherIsPrivate": bool(row["HigherIsPrivate"]),
                    "N_Classifiers": int(row["N_Classifiers"]),
                    "SeedSource": "reconstructed_from_Mean_SD",
                }
            )
    return pd.DataFrame(rows)


def save_processed(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)


def aggregate_to_generator_means(seed_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Primary publication points: one row per generator.

    Averages Utility / Fidelity / Privacy across both datasets and all
    reconstructed seeds (utility already averaged over classifiers upstream).
    """
    if seed_panel.empty:
        return pd.DataFrame()

    meta_cols = [
        c
        for c in (
            "UtilityMetric",
            "FidelityMetric",
            "PrivacyMetric",
            "PrivacyLabel",
            "HigherIsPrivate",
        )
        if c in seed_panel.columns
    ]
    meta = seed_panel.groupby("Generator", as_index=False)[meta_cols].first() if meta_cols else seed_panel[["Generator"]].drop_duplicates()

    agg = (
        seed_panel.groupby("Generator", dropna=False)
        .agg(
            Utility=("Utility", "mean"),
            UtilityStd=("Utility", "std"),
            Fidelity=("Fidelity", "mean"),
            FidelityStd=("Fidelity", "std"),
            Privacy=("Privacy", "mean"),
            PrivacyStd=("Privacy", "std"),
            N_Points=("Utility", "count"),
            N_Datasets=("DatasetKey", "nunique"),
            N_Seeds=("Seed", "nunique"),
        )
        .reset_index()
    )
    out = agg.merge(meta, on="Generator", how="left")
    # Stable generator order
    out["Generator"] = pd.Categorical(out["Generator"], categories=GENERATORS, ordered=True)
    out = out.sort_values("Generator").reset_index(drop=True)
    out["Generator"] = out["Generator"].astype(str)
    return out

