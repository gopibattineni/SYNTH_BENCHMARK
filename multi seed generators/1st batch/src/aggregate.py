"""Aggregate seed-level results into mean ± SD (ddof=1), per task folder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ensure_task_dirs, task_paths


SEEDS = [42, 123, 2024]


def load_raw(task: str) -> pd.DataFrame:
    paths = ensure_task_dirs(task)
    global_raw = paths["raw"] / "multi_seed_raw_results.csv"
    if global_raw.exists():
        return pd.read_csv(global_raw)
    frames = []
    for p in paths["raw"].glob("seed_*/*/*/metrics.csv"):
        frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    key_cols = ["dataset", "generator", "metric_category", "metric_name"]
    if "dataset" not in raw.columns and "dataset_id" in raw.columns:
        raw = raw.rename(columns={"dataset_id": "dataset"})
    rows = []
    for keys, g in raw.groupby(key_cols, dropna=False):
        dataset, generator, cat, name = keys
        by_seed = {}
        for seed in SEEDS:
            sub = g[g["seed"] == seed]["metric_value"]
            by_seed[seed] = float(sub.iloc[0]) if len(sub) else np.nan
        vals = np.array([by_seed[s] for s in SEEDS], dtype=float)
        finite = vals[np.isfinite(vals)]
        mean = float(np.mean(finite)) if len(finite) else np.nan
        sd = (
            float(np.std(finite, ddof=1))
            if len(finite) > 1
            else (0.0 if len(finite) == 1 else np.nan)
        )
        mean_sd = f"{mean:.6g} ± {sd:.6g}" if np.isfinite(mean) and np.isfinite(sd) else ""
        rows.append(
            {
                "dataset": dataset,
                "generator": generator,
                "metric_category": cat,
                "metric_name": name,
                "seed_42": by_seed[42],
                "seed_123": by_seed[123],
                "seed_2024": by_seed[2024],
                "mean": mean,
                "sd": sd,
                "mean_sd": mean_sd,
                "n_seeds": int(np.isfinite(vals).sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate(task: str | None = None) -> pd.DataFrame:
    """Aggregate one task or both. Returns combined frame."""
    tasks = [task] if task else ["classification", "regression"]
    outs = []
    for t in tasks:
        paths = ensure_task_dirs(t)
        raw = load_raw(t)
        out = _aggregate_frame(raw)
        if out.empty:
            continue
        out["task"] = t
        agg = paths["aggregated"]
        out.to_csv(agg / "all_metrics_mean_sd.csv", index=False)
        for cat, fname in [
            ("Fidelity", "fidelity_mean_sd.csv"),
            ("Utility", "utility_mean_sd.csv"),
            ("Privacy", "privacy_mean_sd.csv"),
            ("Compute", "compute_mean_sd.csv"),
        ]:
            out[out["metric_category"] == cat].to_csv(agg / fname, index=False)
        outs.append(out)
    if not outs:
        return pd.DataFrame()
    return pd.concat(outs, ignore_index=True)
