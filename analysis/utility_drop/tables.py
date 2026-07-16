"""Export summary tables for Utility Drop Analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.utility_drop.config import FOCUS_DATASETS, SUPPLEMENTARY_METRICS, UtilityDropConfig
from analysis.utility_drop.data import (
    attach_utility_loss,
    best_classifier_per_generator,
    best_generator_per_classifier,
    build_trtr_table,
    build_tstr_table,
    classifier_summary,
    generator_summary,
)


def _fmt_mean_sd(mean: float, std: float | None) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def export_mean_sd_tables(
    utility: pd.DataFrame,
    tables_dir: Path,
    cfg: UtilityDropConfig,
) -> dict[str, Path]:
    """Write Mean±SD tables for Accuracy (primary) and supplementary metrics."""
    written: dict[str, Path] = {}
    metrics = [cfg.primary_metric] + [m for m in SUPPLEMENTARY_METRICS if m != cfg.primary_metric]

    for metric in metrics:
        tstr = build_tstr_table(utility, metric)
        if tstr.empty:
            continue
        tstr = tstr.copy()
        tstr["Mean_SD"] = [
            _fmt_mean_sd(m, s) for m, s in zip(tstr["Mean"], tstr["Std"])
        ]
        path = tables_dir / f"tstr_{metric.lower().replace('-', '_')}_mean_sd.csv"
        tstr.to_csv(path, index=False)
        written[metric] = path

        # Focus Cancer/Mushroom wide table: Generator × Classifier
        focus = tstr[tstr["Dataset"].isin(FOCUS_DATASETS.values())].copy()
        if focus.empty:
            continue
        leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
        for key, ds_id in FOCUS_DATASETS.items():
            sub = leak0[leak0["Dataset"] == ds_id]
            if sub.empty:
                continue
            piv = sub.pivot_table(
                index="Generator", columns="Classifier", values="Mean_SD", aggfunc="first"
            )
            p = tables_dir / f"{key.lower()}_{metric.lower().replace('-', '_')}_generator_x_classifier.csv"
            piv.to_csv(p)
            written[f"{key}_{metric}"] = p

    return written


def export_rankings(
    tstr_acc: pd.DataFrame,
    loss_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """Generator / classifier rankings and robustness summaries."""
    if tstr_acc.empty:
        return
    leak0 = tstr_acc[tstr_acc["Leakage"] == tstr_acc["Leakage"].min()]
    focus = leak0[leak0["Dataset"].isin(FOCUS_DATASETS.values())]

    # Generator ranking (avg across focus datasets + classifiers)
    gen_rank = (
        focus.groupby("Generator")
        .agg(Mean_Accuracy=("Mean", "mean"), Std=("Std", "mean"), N=("Mean", "count"))
        .sort_values("Mean_Accuracy", ascending=False)
        .reset_index()
    )
    gen_rank["Rank"] = range(1, len(gen_rank) + 1)
    gen_rank.to_csv(tables_dir / "generator_ranking_accuracy.csv", index=False)

    clf_rank = (
        focus.groupby("Classifier")
        .agg(Mean_Accuracy=("Mean", "mean"), Std_Across=("Mean", "std"), Seed_Std=("Std", "mean"))
        .reset_index()
    )
    clf_rank["CV"] = (clf_rank["Std_Across"].abs() / clf_rank["Mean_Accuracy"].abs()).replace(
        [float("inf"), float("-inf")], pd.NA
    )
    clf_rank = clf_rank.sort_values("Mean_Accuracy", ascending=False).reset_index(drop=True)
    clf_rank["Rank"] = range(1, len(clf_rank) + 1)
    clf_rank.to_csv(tables_dir / "classifier_ranking_accuracy.csv", index=False)

    # Robustness: lower mean Utility_Loss / Drop_Pct = more robust
    if not loss_df.empty and "Utility_Loss" in loss_df.columns:
        fl = loss_df[loss_df["Dataset"].isin(FOCUS_DATASETS.values())]
        rob = (
            fl.groupby("Generator")
            .agg(
                Mean_Utility_Loss=("Utility_Loss", "mean"),
                Mean_Drop_Pct=("Utility_Drop_Pct", "mean"),
                Mean_TSTR=("Mean", "mean"),
            )
            .sort_values("Mean_Utility_Loss", ascending=True)
            .reset_index()
        )
        rob["Robustness_Rank"] = range(1, len(rob) + 1)
        rob.to_csv(tables_dir / "generator_robustness_utility_loss.csv", index=False)

        clf_rob = (
            fl.groupby("Classifier")
            .agg(
                Mean_Utility_Loss=("Utility_Loss", "mean"),
                Mean_Drop_Pct=("Utility_Drop_Pct", "mean"),
                Std_Loss=("Utility_Loss", "std"),
            )
            .sort_values("Mean_Utility_Loss", ascending=True)
            .reset_index()
        )
        clf_rob["Least_Affected_Rank"] = range(1, len(clf_rob) + 1)
        clf_rob.to_csv(tables_dir / "classifier_least_affected_by_drop.csv", index=False)

    best_clf = best_classifier_per_generator(focus)
    best_gen = best_generator_per_classifier(focus)
    best_clf.to_csv(tables_dir / "best_classifier_per_generator_raw.csv", index=False)
    best_gen.to_csv(tables_dir / "best_generator_per_classifier_raw.csv", index=False)

    generator_summary(focus).to_csv(tables_dir / "generator_summary_focus.csv", index=False)
    classifier_summary(focus).to_csv(tables_dir / "classifier_summary_focus.csv", index=False)


def export_all_dataset_summary(tstr_acc: pd.DataFrame, tables_dir: Path) -> None:
    """All-datasets generator summary (primary metric)."""
    if tstr_acc.empty:
        return
    leak0 = tstr_acc[tstr_acc["Leakage"] == tstr_acc["Leakage"].min()]
    overall = (
        leak0.groupby("Generator")
        .agg(Mean_Accuracy=("Mean", "mean"), Std=("Std", "mean"), N_Rows=("Mean", "count"))
        .sort_values("Mean_Accuracy", ascending=False)
        .reset_index()
    )
    overall["Rank"] = range(1, len(overall) + 1)
    overall.to_csv(tables_dir / "all_datasets_generator_ranking.csv", index=False)

    by_ds = (
        leak0.groupby(["Dataset", "Generator"])
        .agg(Mean_Accuracy=("Mean", "mean"), Std=("Std", "mean"))
        .reset_index()
    )
    by_ds.to_csv(tables_dir / "all_datasets_generator_by_dataset.csv", index=False)


def write_findings_markdown(
    tstr_acc: pd.DataFrame,
    loss_df: pd.DataFrame,
    multi_leakage: bool,
    out_path: Path,
) -> None:
    focus = tstr_acc[tstr_acc["Dataset"].isin(FOCUS_DATASETS.values())]
    if focus.empty:
        out_path.write_text("# Findings\n\nNo focus-dataset utility rows found.\n", encoding="utf-8")
        return
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    gen = leak0.groupby("Generator")["Mean"].mean().sort_values(ascending=False)
    clf = leak0.groupby("Classifier")["Mean"].mean().sort_values(ascending=False)

    lines = [
        "# Utility Drop Analysis — Key Findings",
        "",
        f"**Primary metric:** Accuracy (Mean ± SD across seeds)",
        f"**Focus datasets:** Cancer, Mushroom",
        f"**Multi-level leakage data present:** {'Yes' if multi_leakage else 'No (Leakage=0% only; drop = TRTR−TSTR)'}",
        "",
        "## Highest utility generator",
        f"1. **{gen.index[0]}** — mean Accuracy {gen.iloc[0]:.3f}",
    ]
    if len(gen) > 1:
        lines.append(f"2. {gen.index[1]} — {gen.iloc[1]:.3f}")
    lines += [
        "",
        "## Best classifier",
        f"1. **{clf.index[0]}** — mean Accuracy {clf.iloc[0]:.3f}",
    ]
    if len(clf) > 1:
        lines.append(f"2. {clf.index[1]} — {clf.iloc[1]:.3f}")

    if not loss_df.empty and "Utility_Loss" in loss_df.columns:
        fl = loss_df[loss_df["Dataset"].isin(FOCUS_DATASETS.values())]
        rob_g = fl.groupby("Generator")["Utility_Loss"].mean().sort_values()
        rob_c = fl.groupby("Classifier")["Utility_Loss"].mean().sort_values()
        lines += [
            "",
            "## Most robust generator (lowest utility loss)",
            f"1. **{rob_g.index[0]}** — mean loss {rob_g.iloc[0]:.3f}",
            "",
            "## Classifier least affected by drop",
            f"1. **{rob_c.index[0]}** — mean loss {rob_c.iloc[0]:.3f}",
        ]
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
