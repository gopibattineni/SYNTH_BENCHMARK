"""Export compact JSON datasets for the static GitHub Pages dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "Results"
DOCS_DATA = REPO_ROOT / "docs" / "data"


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df.empty:
        return []
    chunk = df.head(limit) if limit else df
    return json.loads(chunk.to_json(orient="records"))


def export_dashboard_data(output_dir: Path | None = None) -> Path:
    out = output_dir or DOCS_DATA
    out.mkdir(parents=True, exist_ok=True)

    utility_long = _read_csv("MasterData/utility_long.csv")
    fidelity = _read_csv("MasterData/fidelity_long.csv")
    privacy = _read_csv("MasterData/privacy_long.csv")
    tradeoff = _read_csv("Supplementary/tradeoff_points.csv")
    inventory = _read_csv("MasterData/file_inventory.csv")

    clf_stats = _read_csv("Supplementary/utility_classification_stats.csv")
    reg_stats = _read_csv("Supplementary/utility_regression_stats.csv")
    fid_stats = _read_csv("Supplementary/fidelity_stats.csv")
    priv_stats = _read_csv("Supplementary/privacy_stats.csv")
    wilcoxon = _read_csv("Supplementary/pairwise_wilcoxon.csv")
    effects = _read_csv("Supplementary/effect_sizes.csv")

    # --- Meta / overview ---
    meta = {
        "generators": [
            "CTGAN", "CopulaGAN", "TVAE", "GaussianCopula",
            "WGAN_GP", "CTABGAN", "TabDDPM", "ForestDiffusion",
        ],
        "n_datasets": int(utility_long["Dataset"].nunique()) if not utility_long.empty else 0,
        "n_utility_rows": len(utility_long),
        "n_fidelity_rows": len(fidelity),
        "n_privacy_rows": len(privacy),
        "n_files": len(inventory),
    }
    summary_path = RESULTS_DIR / "Supplementary" / "pipeline_summary.json"
    if summary_path.exists():
        meta.update(json.loads(summary_path.read_text(encoding="utf-8")))

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- Utility: TRTR/TSTR aggregated ---
    if not utility_long.empty:
        util_agg = (
            utility_long[utility_long["EvaluationType"].isin(["TRTR", "TSTR", "Gap"])]
            .groupby(
                ["Dataset", "Generator", "TaskType", "Metric", "EvaluationType"],
                dropna=False,
            )
            .agg(Mean=("Mean", "mean"), Std=("Std", "mean"), Count=("Value", "count"))
            .reset_index()
        )
        (out / "utility_agg.json").write_text(
            json.dumps(_records(util_agg), indent=2), encoding="utf-8"
        )

        # Per classifier TRTR/TSTR for classification
        clf_detail = utility_long[
            (utility_long["TaskType"] == "classification")
            & (utility_long["EvaluationType"].isin(["TRTR", "TSTR"]))
            & (utility_long["Metric"].isin(["Accuracy", "F1", "Precision", "Recall"]))
        ][["Dataset", "Generator", "Classifier", "Metric", "EvaluationType", "Mean", "Std"]]
        (out / "utility_classifier.json").write_text(
            json.dumps(_records(clf_detail), indent=2), encoding="utf-8"
        )

        # Summary gaps per generator per dataset
        gaps = utility_long[
            utility_long["Metric"].str.contains("Drop|Increase|Gap", na=False, regex=True)
        ].groupby(["Dataset", "Generator", "TaskType", "Metric"], dropna=False)["Mean"].mean().reset_index()
        (out / "utility_gaps.json").write_text(
            json.dumps(_records(gaps), indent=2), encoding="utf-8"
        )

    # --- Fidelity & privacy ---
    fid_src = fid_stats if not fid_stats.empty else fidelity
    priv_src = priv_stats if not priv_stats.empty else privacy
    (out / "fidelity.json").write_text(json.dumps(_records(fid_src), indent=2), encoding="utf-8")
    (out / "privacy.json").write_text(json.dumps(_records(priv_src), indent=2), encoding="utf-8")

    # --- Trade-off & rankings ---
    if not tradeoff.empty:
        (out / "tradeoff.json").write_text(json.dumps(_records(tradeoff), indent=2), encoding="utf-8")

    weighted = _read_csv("Tables/weighted_rankings.csv")
    borda = _read_csv("Tables/borda_rankings.csv")
    if not weighted.empty:
        weighted.index.name = "index"
        (out / "rankings_weighted.json").write_text(
            json.dumps(_records(weighted.reset_index()), indent=2), encoding="utf-8"
        )
    if not borda.empty:
        borda.index.name = "index"
        (out / "rankings_borda.json").write_text(
            json.dumps(_records(borda.reset_index()), indent=2), encoding="utf-8"
        )

    # --- Statistics ---
    stats_payload = {
        "wilcoxon": _records(wilcoxon),
        "effect_sizes": _records(effects),
        "classification_stats": _records(clf_stats, limit=2000),
        "regression_stats": _records(reg_stats, limit=2000),
    }
    (out / "statistics.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    # --- Coverage matrix ---
    if not utility_long.empty:
        coverage = (
            utility_long.groupby(["Dataset", "Generator"], dropna=False)
            .size()
            .reset_index(name="Rows")
        )
        coverage["Available"] = (coverage["Rows"] > 0).astype(int)
        (out / "coverage.json").write_text(
            json.dumps(_records(coverage), indent=2), encoding="utf-8"
        )

    return out


if __name__ == "__main__":
    path = export_dashboard_data()
    print(f"Exported dashboard data to {path}")
