"""Load and filter experiment data for Cancer and Mushroom datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.config import PipelineConfig
from analysis.data_loader import (
    build_unified_master,
    load_master_data,
    normalize_dataset_name,
    save_master_data,
)
from analysis.feature_fidelity import load_feature_fidelity, load_quality_from_notebooks
from analysis.notebook_metrics import load_notebook_metrics_long
from analysis.two_datasets.config import DATASETS, TwoDatasetConfig


def _filter_df(df: pd.DataFrame, dataset_ids: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Dataset"] = out["Dataset"].map(normalize_dataset_name)
    return out[out["Dataset"].isin(dataset_ids)].copy()


def load_two_dataset_master(config: TwoDatasetConfig | None = None) -> dict[str, pd.DataFrame]:
    """Discover, merge, and filter all Excel/notebook data to Cancer + Mushroom only."""
    config = config or TwoDatasetConfig()
    pipe = PipelineConfig(repo_root=config.repo_root)
    master = load_master_data(pipe)

    nb_fid, nb_priv = load_notebook_metrics_long(pipe)
    notebook_quality = load_quality_from_notebooks(pipe)
    for frame, target in [(nb_fid, "fidelity"), (notebook_quality, "fidelity"), (nb_priv, "privacy")]:
        if frame.empty:
            continue
        filtered = _filter_df(frame, set(DATASETS.values()))
        if filtered.empty:
            continue
        if target == "fidelity":
            master.fidelity_long = pd.concat([master.fidelity_long, filtered], ignore_index=True).drop_duplicates(
                subset=["Dataset", "Generator", "Metric"], keep="last"
            )
        else:
            master.privacy_long = pd.concat([master.privacy_long, filtered], ignore_index=True).drop_duplicates(
                subset=["Dataset", "Generator", "Metric"], keep="last"
            )

    dataset_ids = set(DATASETS.values())
    return {
        "utility_long": _filter_df(master.utility_long, dataset_ids),
        "fidelity_long": _filter_df(master.fidelity_long, dataset_ids),
        "privacy_long": _filter_df(master.privacy_long, dataset_ids),
        "privacy_detail": _filter_df(master.privacy_detail, dataset_ids),
        "unified": _filter_df(build_unified_master(master), dataset_ids),
        "file_inventory": master.file_inventory[
            master.file_inventory["Dataset"].isin(dataset_ids)
            | master.file_inventory["Path"].str.contains("Cancer|Mushroom", case=False, na=False)
        ].copy(),
    }


def save_dataset_processed(
    data: dict[str, pd.DataFrame],
    dirs: dict[str, Path],
) -> None:
    for name, df in data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(dirs["processed"] / f"{name}.csv", index=False)
            df.to_excel(dirs["processed"] / f"{name}.xlsx", index=False)
