"""Publication figures for Utility Drop Analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.benchmarking import friedman_average_ranks, nemenyi_significance_groups
from analysis.config import GENERATORS, PLOT_RC
from analysis.figure_tables import (
    add_side_values_table,
    export_values_csv,
    make_plot_with_table,
    marker_text_color,
    order_generator_dataset,
)
from analysis.figures import save_figure
from analysis.utility_drop.config import FOCUS_DATASETS, UtilityDropConfig
from analysis.utility_drop.data import reconstruct_seed_samples

GENERATOR_COLORS = {
    "CTGAN": "#1f77b4",
    "CopulaGAN": "#ff7f0e",
    "TVAE": "#2ca02c",
    "GaussianCopula": "#d62728",
    "WGAN_GP": "#9467bd",
    "CTABGAN": "#8c564b",
    "TabDDPM": "#e377c2",
    "ForestDiffusion": "#7f7f7f",
}
DS_STYLE = {"Cancer": "-", "Mushroom": "--"}
DS_MARKER = {"Cancer": "o", "Mushroom": "s"}


def _style() -> None:
    for k, v in PLOT_RC.items():
        plt.rcParams[k] = v


def _gens(df: pd.DataFrame) -> list[str]:
    return [g for g in GENERATORS if g in set(df.get("Generator", []))]


def figure01_utility_vs_leakage(
    gen_sum: pd.DataFrame,
    out: Path,
    cfg: UtilityDropConfig,
    multi_leakage: bool,
) -> str | None:
    """Line plot: Accuracy vs Leakage% (or vs generator at Leakage=0)."""
    focus = gen_sum[gen_sum["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    inv = {v: k for k, v in FOCUS_DATASETS.items()}
    focus["DatasetKey"] = focus["Dataset"].map(inv)
    _style()
    fig, ax, ax_tab = make_plot_with_table(figsize=(12.0, 6.8), width_ratios=(2.7, 1.7))

    table_rows: list[list[str]] = []
    if multi_leakage and focus["Leakage"].nunique() > 1:
        ordered = order_generator_dataset(focus, generators=GENERATORS)
        for _, row in ordered.iterrows():
            gen, ds = str(row["Generator"]), str(row["DatasetKey"])
            ax.errorbar(
                float(row["Leakage"]), float(row["Mean"]), yerr=float(row["Std"]) if pd.notna(row.get("Std")) else 0,
                color=GENERATOR_COLORS.get(gen, "gray"), ls=DS_STYLE.get(ds, "-"),
                marker=DS_MARKER.get(ds, "o"), capsize=3, lw=1.8, label="_nolegend_",
            )
            table_rows.append([
                str(int(row["PointId"])), gen, ds,
                f"{float(row['Leakage']):.0f}",
                f"{float(row['Mean']):.3f}",
                f"{float(row['Std']):.3f}" if pd.notna(row.get("Std")) else "—",
            ])
        ax.set_xlabel("Leakage Percentage")
        ax.set_title("Mean Accuracy ± SD vs Leakage — Cancer & Mushroom")
        col_labels = ["#", "Generator", "Dataset", "Leak%", "Mean", "SD"]
    else:
        gens = _gens(focus)
        x = np.arange(len(gens))
        pid = 1
        for ds, offset in (("Cancer", -0.05), ("Mushroom", 0.05)):
            sub = focus[focus["DatasetKey"] == ds].set_index("Generator").reindex(gens)
            means = sub["Mean"].astype(float).values
            stds = sub["Std"].fillna(0).astype(float).values
            ax.errorbar(
                x + offset, means, yerr=stds, marker=DS_MARKER[ds], ls=DS_STYLE[ds],
                color="#333333", capsize=3, lw=1.5, label=ds, zorder=2,
            )
            for i, gen in enumerate(gens):
                color = GENERATOR_COLORS.get(gen, "gray")
                ax.scatter([x[i] + offset], [means[i]], c=color, s=90, zorder=3, edgecolors="k", lw=0.4)
                if np.isfinite(means[i]):
                    ax.text(
                        x[i] + offset, means[i], str(pid),
                        ha="center", va="center", fontsize=6.5, fontweight="bold",
                        color=marker_text_color(color),
                        zorder=4,
                    )
                    table_rows.append([
                        str(pid), gen, ds,
                        f"{means[i]:.3f}",
                        f"{stds[i]:.3f}",
                    ])
                    pid += 1
        ax.set_xticks(x)
        ax.set_xticklabels(gens, rotation=30, ha="right")
        ax.set_xlabel("Generator")
        ax.set_title("Mean Accuracy ± SD (Leakage = 0% only)")
        col_labels = ["#", "Generator", "Dataset", "Mean", "SD"]

        # Dataset legend only (values live in the table)
        from matplotlib.lines import Line2D
        ds_handles = [
            Line2D([0], [0], marker=DS_MARKER[d], color="k", ls=DS_STYLE[d], markersize=7, label=d)
            for d in ("Cancer", "Mushroom")
        ]
        ax.legend(handles=ds_handles, loc="lower left", fontsize=8)

    add_side_values_table(ax_tab, table_rows, col_labels, generator_colors=GENERATOR_COLORS, fontsize=6.5)
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3, axis="y")
    if multi_leakage and focus["Leakage"].nunique() > 1:
        ax.legend(fontsize=7, ncol=2, loc="lower left")

    export_values_csv(
        pd.DataFrame(table_rows, columns=col_labels),
        out.parent / "Tables" / "Figure01_utility_vs_leakage_values.csv",
    )
    save_figure(fig, out / "Figure01_utility_vs_leakage", cfg.figure_formats, cfg.figure_dpi)
    return "Figure01_utility_vs_leakage"


def figure02_classifier_comparison(tstr: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> str | None:
    focus = tstr[tstr["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    # Average across Cancer+Mushroom at leakage 0 (or mean leakage)
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    pivot = (
        leak0.groupby(["Classifier", "Generator"])["Mean"].mean().reset_index()
    )
    std = leak0.groupby(["Classifier", "Generator"])["Std"].mean().reset_index(name="Std")
    pivot = pivot.merge(std, on=["Classifier", "Generator"], how="left")
    classifiers = sorted(pivot["Classifier"].dropna().unique())
    gens = _gens(pivot)
    if not classifiers or not gens:
        return None

    _style()
    fig, ax, ax_tab = make_plot_with_table(figsize=(14.5, 6.8), width_ratios=(3.2, 1.4))
    x = np.arange(len(classifiers))
    width = 0.8 / max(len(gens), 1)
    for i, gen in enumerate(gens):
        sub = pivot[pivot["Generator"] == gen].set_index("Classifier").reindex(classifiers)
        vals = sub["Mean"].astype(float).values
        errs = sub["Std"].fillna(0).astype(float).values
        ax.bar(
            x + i * width - 0.4 + width / 2, vals, width,
            yerr=errs, capsize=2, label=gen, color=GENERATOR_COLORS.get(gen, "gray"),
            edgecolor="k", lw=0.3, alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=35, ha="right")
    ax.set_ylabel("Mean Accuracy ± SD")
    ax.set_ylim(0, 1.15)
    ax.set_title("Classifier Comparison — Mean Accuracy by Generator (Cancer+Mushroom avg)")
    ax.legend(ncol=2, fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Compact mean table: Classifier rows × Generator columns is too wide;
    # instead list generator means averaged across classifiers.
    cell_text = []
    for i, gen in enumerate(gens):
        sub = pivot[pivot["Generator"] == gen]
        cell_text.append([
            str(i + 1),
            gen,
            f"{sub['Mean'].mean():.3f}",
            f"{sub['Std'].mean():.3f}",
        ])
    add_side_values_table(
        ax_tab, cell_text, ["#", "Generator", "Mean", "SD"],
        generator_colors=GENERATOR_COLORS,
    )
    export_values_csv(
        pivot.rename(columns={"Mean": "Accuracy", "Std": "SD"}),
        out.parent / "Tables" / "Figure02_classifier_comparison_values.csv",
        round_cols=["Accuracy", "SD"],
    )
    save_figure(fig, out / "Figure02_classifier_comparison", cfg.figure_formats, cfg.figure_dpi)
    return "Figure02_classifier_comparison"


def figure03_generator_comparison(gen_sum: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> str | None:
    focus = gen_sum[gen_sum["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    agg = (
        leak0.groupby("Generator")
        .agg(Mean=("Mean", "mean"), Std=("Std", "mean"))
        .reindex(_gens(leak0))
        .dropna(subset=["Mean"])
        .sort_values("Mean", ascending=True)
    )
    if agg.empty:
        return None
    ranks = agg["Mean"].rank(ascending=False)
    _style()
    fig, ax, ax_tab = make_plot_with_table(figsize=(11.0, 5.8), width_ratios=(2.6, 1.5))
    colors = []
    for gen in agg.index:
        if ranks[gen] == 1:
            colors.append("#2ca02c")
        elif ranks[gen] == 2:
            colors.append("#98df8a")
        else:
            colors.append(GENERATOR_COLORS.get(gen, "gray"))
    ax.barh(agg.index.astype(str), agg["Mean"], xerr=agg["Std"].fillna(0),
            color=colors, edgecolor="k", lw=0.4, capsize=3)
    ax.set_xlabel("Mean Accuracy ± SD")
    ax.set_xlim(0, 1.15)
    ax.set_title("Generator Comparison (best=green, second=light green)")
    ax.grid(axis="x", alpha=0.3)

    # Ranked table (best first)
    ranked = agg.sort_values("Mean", ascending=False)
    cell_text = [
        [str(i), gen, f"{row['Mean']:.3f}", f"{row['Std']:.3f}"]
        for i, (gen, row) in enumerate(ranked.iterrows(), start=1)
    ]
    add_side_values_table(
        ax_tab, cell_text, ["#", "Generator", "Mean", "SD"],
        generator_colors=GENERATOR_COLORS,
    )
    export_values_csv(
        ranked.reset_index().rename(columns={"Mean": "Accuracy", "Std": "SD"}),
        out.parent / "Tables" / "Figure03_generator_comparison_values.csv",
        round_cols=["Accuracy", "SD"],
    )
    save_figure(fig, out / "Figure03_generator_comparison", cfg.figure_formats, cfg.figure_dpi)
    return "Figure03_generator_comparison"


def figure04_heatmaps(tstr: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> str | None:
    if tstr.empty:
        return None
    leak0 = tstr[tstr["Leakage"] == tstr["Leakage"].min()]
    panels = []
    for key, ds_id in FOCUS_DATASETS.items():
        sub = leak0[leak0["Dataset"] == ds_id]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="Generator", columns="Classifier", values="Mean", aggfunc="mean")
        piv = piv.reindex([g for g in GENERATORS if g in piv.index])
        panels.append((key, piv))
    # Overall average
    overall = leak0.pivot_table(index="Generator", columns="Classifier", values="Mean", aggfunc="mean")
    overall = overall.reindex([g for g in GENERATORS if g in overall.index])
    panels.append(("Overall Average", overall))
    if not panels:
        return None

    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, piv) in zip(axes, panels):
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax, linewidths=0.3)
        ax.set_title(title)
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Generator")
    fig.suptitle("Mean Accuracy Heatmaps (Generator × Classifier)", y=1.02)
    fig.tight_layout()
    save_figure(fig, out / "Figure04_accuracy_heatmap", cfg.figure_formats, cfg.figure_dpi)
    return "Figure04_accuracy_heatmap"


def figure05_utility_drop_heatmap(
    loss_df: pd.DataFrame,
    out: Path,
    cfg: UtilityDropConfig,
    multi_leakage: bool,
) -> str | None:
    focus = loss_df[loss_df["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty or "Utility_Drop_Pct" not in focus.columns:
        return None
    _style()
    if multi_leakage and focus["Leakage"].nunique() > 1:
        # Generators × Leakage %
        agg = (
            focus.groupby(["Generator", "Leakage"])["Utility_Drop_Pct"].mean().reset_index()
        )
        piv = agg.pivot(index="Generator", columns="Leakage", values="Utility_Drop_Pct")
        piv = piv.reindex([g for g in GENERATORS if g in piv.index])
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.3)
        ax.set_title("Utility Drop (%) vs No-Leakage / by Leakage %")
        ax.set_xlabel("Leakage %")
    else:
        # Generators × Classifiers drop % (at L=0 relative to TRTR)
        leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
        piv = leak0.pivot_table(
            index="Generator", columns="Classifier", values="Utility_Drop_Pct", aggfunc="mean"
        )
        piv = piv.reindex([g for g in GENERATORS if g in piv.index])
        fig, ax = plt.subplots(figsize=(12, 5.5))
        sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.3)
        ax.set_title("Utility Drop (%) = 100×(TRTR−TSTR)/TRTR")
        ax.set_xlabel("Classifier")
    ax.set_ylabel("Generator")
    save_figure(fig, out / "Figure05_utility_drop_heatmap", cfg.figure_formats, cfg.figure_dpi)
    return "Figure05_utility_drop_heatmap"


def figure06_utility_loss(loss_df: pd.DataFrame, out: Path, cfg: UtilityDropConfig, multi_leakage: bool) -> str | None:
    focus = loss_df[loss_df["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty or "Utility_Loss" not in focus.columns:
        return None
    inv = {v: k for k, v in FOCUS_DATASETS.items()}
    focus["DatasetKey"] = focus["Dataset"].map(inv)
    # Aggregate across classifiers
    agg = (
        focus.groupby(["DatasetKey", "Generator", "Leakage"], dropna=False)
        .agg(Mean=("Utility_Loss", "mean"), Std=("Utility_Loss", "std"))
        .reset_index()
    )
    _style()
    fig, ax = plt.subplots(figsize=(11, 6))
    if multi_leakage and agg["Leakage"].nunique() > 1:
        for gen in _gens(agg):
            for ds in ("Cancer", "Mushroom"):
                sub = agg[(agg["Generator"] == gen) & (agg["DatasetKey"] == ds)].sort_values("Leakage")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["Leakage"], sub["Mean"], yerr=sub["Std"].fillna(0),
                    color=GENERATOR_COLORS.get(gen, "gray"), ls=DS_STYLE[ds],
                    marker=DS_MARKER[ds], capsize=3, label=f"{gen} ({ds})",
                )
        ax.set_xlabel("Leakage Percentage")
        ax.set_title("Utility Loss = Acc(No Leak) − Acc(Current)")
    else:
        gens = _gens(agg)
        x = np.arange(len(gens))
        for ds, off in (("Cancer", -0.15), ("Mushroom", 0.15)):
            sub = agg[agg["DatasetKey"] == ds].set_index("Generator").reindex(gens)
            ax.bar(
                x + off, sub["Mean"].fillna(0), 0.28, yerr=sub["Std"].fillna(0),
                capsize=3, label=ds, alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(gens, rotation=30, ha="right")
        ax.set_xlabel("Generator")
        ax.set_title("Utility Loss = Acc(TRTR) − Acc(TSTR)  [Leakage=0%]")
    ax.set_ylabel("Utility Loss (Accuracy)")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Values table (avoid on-bar text overlap)
    tab = agg.copy()
    tab.to_csv(out.parent / "Tables" / "Figure06_utility_loss_values.csv", index=False)
    save_figure(fig, out / "Figure06_utility_loss", cfg.figure_formats, cfg.figure_dpi)
    return "Figure06_utility_loss"


def figure07_classifier_stability(tstr: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> str | None:
    focus = tstr[tstr["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    stats = (
        leak0.groupby("Classifier")
        .agg(Mean=("Mean", "mean"), Std=("Mean", "std"), SeedStd=("Std", "mean"))
        .reset_index()
    )
    stats["CV"] = (stats["Std"].abs() / stats["Mean"].abs()).replace([np.inf, -np.inf], np.nan)
    stats = stats.sort_values("Mean", ascending=True)
    _style()
    fig, ax, ax_tab = make_plot_with_table(figsize=(11.0, 5.8), width_ratios=(2.5, 1.7))
    ax.barh(stats["Classifier"], stats["Mean"], xerr=stats["Std"].fillna(0),
            color="#4C78A8", edgecolor="k", lw=0.3, capsize=3)
    ax.set_xlabel("Mean Accuracy ± SD (across generators/datasets)")
    ax.set_xlim(0, 1.15)
    ax.set_title("Classifier Stability Ranking")
    ax.grid(axis="x", alpha=0.3)
    ranked = stats.sort_values("Mean", ascending=False)
    cell_text = [
        [str(i), str(r["Classifier"]), f"{r['Mean']:.3f}", f"{r['Std']:.3f}", f"{r['CV']:.3f}"]
        for i, (_, r) in enumerate(ranked.iterrows(), start=1)
    ]
    add_side_values_table(
        ax_tab, cell_text, ["#", "Classifier", "Mean", "SD", "CV"],
        generator_col_idx=1, generator_colors={},
    )
    stats.to_csv(out.parent / "Tables" / "classifier_stability.csv", index=False)
    save_figure(fig, out / "Figure07_classifier_stability", cfg.figure_formats, cfg.figure_dpi)
    return "Figure07_classifier_stability"


def figure08_best_classifier_table(
    best_clf: pd.DataFrame,
    utility: pd.DataFrame,
    out: Path,
    cfg: UtilityDropConfig,
) -> str | None:
    focus = best_clf[best_clf["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    rows = []
    for gen in _gens(leak0):
        for ds_key, ds_id in FOCUS_DATASETS.items():
            sub = leak0[(leak0["Generator"] == gen) & (leak0["Dataset"] == ds_id)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rec = {
                "Generator": gen,
                "Dataset": ds_key,
                "Best Classifier": r["Classifier"],
                "Accuracy": f"{r['Mean']:.3f} ± {r['Std']:.3f}" if pd.notna(r["Std"]) else f"{r['Mean']:.3f}",
            }
            # attach F1 / ROC-AUC for same clf if available
            for metric, col in (("F1", "F1"), ("ROC-AUC", "ROC-AUC")):
                m = utility[
                    (utility["EvaluationType"] == "TSTR")
                    & (utility["Metric"] == metric)
                    & (utility["Generator"] == gen)
                    & (utility["Dataset"] == ds_id)
                    & (utility["Classifier"] == r["Classifier"])
                ]
                if not m.empty:
                    mr = m.iloc[0]
                    std = mr["Std"] if pd.notna(mr.get("Std")) else 0
                    rec[col] = f"{mr['Mean']:.3f} ± {std:.3f}"
                else:
                    rec[col] = "—"
            rows.append(rec)
    table = pd.DataFrame(rows)
    if table.empty:
        return None
    table.to_csv(out.parent / "Tables" / "best_classifier_per_generator.csv", index=False)

    _style()
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(table) + 1.8))
    ax.axis("off")
    cell = table.values.tolist()
    tbl = ax.table(cellText=cell, colLabels=list(table.columns), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    # highlight best accuracy rows (max mean)
    means = []
    for r in cell:
        try:
            means.append(float(str(r[3]).split("±")[0].strip()))
        except Exception:
            means.append(-1)
    if means:
        best_i = int(np.argmax(means))
        second = int(np.argsort(means)[-2]) if len(means) > 1 else best_i
        for j in range(len(table.columns)):
            tbl[best_i + 1, j].set_facecolor("#c6efce")
            tbl[second + 1, j].set_facecolor("#ffeb9c")
    ax.set_title("Best Classifier for Every Generator", pad=12, fontsize=12)
    save_figure(fig, out / "Figure08_best_classifier_per_generator", cfg.figure_formats, cfg.figure_dpi)
    return "Figure08_best_classifier_per_generator"


def figure09_best_generator_table(best_gen: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> str | None:
    focus = best_gen[best_gen["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    rows = []
    for clf in sorted(leak0["Classifier"].dropna().unique()):
        for ds_key, ds_id in FOCUS_DATASETS.items():
            sub = leak0[(leak0["Classifier"] == clf) & (leak0["Dataset"] == ds_id)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "Classifier": clf,
                    "Dataset": ds_key,
                    "Best Generator": r["Generator"],
                    "Accuracy": f"{r['Mean']:.3f} ± {r['Std']:.3f}" if pd.notna(r["Std"]) else f"{r['Mean']:.3f}",
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return None
    table.to_csv(out.parent / "Tables" / "best_generator_per_classifier.csv", index=False)

    _style()
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(table) + 1.8))
    ax.axis("off")
    cell = table.values.tolist()
    tbl = ax.table(cellText=cell, colLabels=list(table.columns), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    means = []
    for r in cell:
        try:
            means.append(float(str(r[3]).split("±")[0].strip()))
        except Exception:
            means.append(-1)
    if means:
        best_i = int(np.argmax(means))
        for j in range(len(table.columns)):
            tbl[best_i + 1, j].set_facecolor("#c6efce")
    ax.set_title("Best Generator for Every Classifier", pad=12, fontsize=12)
    save_figure(fig, out / "Figure09_best_generator_per_classifier", cfg.figure_formats, cfg.figure_dpi)
    return "Figure09_best_generator_per_classifier"


def figure10_critical_difference(tstr: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> list[str]:
    saved = []
    focus = tstr[tstr["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return saved
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]

    # Generators: blocks = Dataset × Classifier
    gdf = leak0.copy()
    gdf["Block"] = gdf["Dataset"].astype(str) + "|" + gdf["Classifier"].astype(str)
    ranks_g = friedman_average_ranks(gdf, value_col="Mean", block_col="Block", treatment_col="Generator")
    if not ranks_g.empty:
        _cd_plot(ranks_g, out / "Figure10a_cd_generators", cfg,
                 "Critical Difference — Generators (Accuracy)")
        ranks_g.to_csv(out.parent / "Statistics" / "friedman_ranks_generators.csv", index=False)
        saved.append("Figure10a_cd_generators")

    # Classifiers: blocks = Dataset × Generator
    cdf = leak0.copy()
    cdf["Block"] = cdf["Dataset"].astype(str) + "|" + cdf["Generator"].astype(str)
    ranks_c = friedman_average_ranks(cdf, value_col="Mean", block_col="Block", treatment_col="Classifier")
    if not ranks_c.empty:
        # friedman_average_ranks always labels the treatment column "Generator"
        ranks_c = ranks_c.rename(columns={"Generator": "Classifier"})
        _cd_plot(ranks_c, out / "Figure10b_cd_classifiers", cfg,
                 "Critical Difference — Classifiers (Accuracy)")
        ranks_c.to_csv(out.parent / "Statistics" / "friedman_ranks_classifiers.csv", index=False)
        saved.append("Figure10b_cd_classifiers")
    return saved


def _cd_plot(ranks: pd.DataFrame, path: Path, cfg: UtilityDropConfig, title: str) -> None:
    cd_groups = nemenyi_significance_groups(ranks)
    plot_df = cd_groups if not cd_groups.empty else ranks
    cd = float(ranks["CriticalDifference"].iloc[0])
    treat_col = "Generator" if "Generator" in plot_df.columns else "Classifier"
    _style()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    y = 0.5
    plot_df = plot_df.sort_values("AverageRank")
    ax.hlines(y, plot_df["AverageRank"].min() - 0.4, plot_df["AverageRank"].max() + 0.4, colors="k", lw=2)
    xmin = float(plot_df["AverageRank"].min())
    if np.isfinite(cd):
        ax.axvspan(xmin, xmin + cd, alpha=0.12, color="steelblue")
        ax.text(xmin + cd / 2, y - 0.28, f"CD = {cd:.2f}", ha="center", fontsize=8, color="steelblue")
    if "CD_Group" in plot_df.columns:
        for _, grp in plot_df.groupby("CD_Group"):
            if len(grp) >= 2:
                ax.plot([grp["AverageRank"].min(), grp["AverageRank"].max()], [y + 0.25, y + 0.25],
                        color="0.3", lw=3, solid_capstyle="round")
    for _, row in plot_df.iterrows():
        name = row[treat_col]
        color = GENERATOR_COLORS.get(name, "#4C78A8")
        ax.plot(row["AverageRank"], y, "o", ms=11, color=color, markeredgecolor="k", markeredgewidth=0.5)
        ax.text(row["AverageRank"], y + 0.12, name, ha="center", fontsize=7.5, rotation=30)
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    save_figure(fig, path, cfg.figure_formats, cfg.figure_dpi)


def figure11_boxplots(tstr: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> str | None:
    focus = tstr[tstr["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    rng = np.random.default_rng(42)
    rows = []
    for _, r in leak0.iterrows():
        samples = reconstruct_seed_samples(r["Mean"], r["Std"] if pd.notna(r["Std"]) else 0.01, cfg.n_seeds, rng)
        for s in samples:
            rows.append({"Generator": r["Generator"], "Classifier": r["Classifier"], "Accuracy": float(np.clip(s, 0, 1))})
    long = pd.DataFrame(rows)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    order_g = _gens(long)
    sns.boxplot(
        data=long, x="Generator", y="Accuracy", order=order_g, hue="Generator",
        hue_order=order_g, legend=False, ax=axes[0],
        palette=[GENERATOR_COLORS.get(g, "gray") for g in order_g],
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("By Generator")
    axes[0].set_ylim(0, 1)
    clfs = sorted(long["Classifier"].unique())
    sns.boxplot(data=long, x="Classifier", y="Accuracy", order=clfs, hue="Classifier",
                hue_order=clfs, legend=False, ax=axes[1])
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].set_title("By Classifier")
    axes[1].set_ylim(0, 1)
    fig.suptitle(
        "Accuracy Distributions (approx. from Mean±SD across 10 seeds)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, out / "Figure11_boxplots", cfg.figure_formats, cfg.figure_dpi)
    return "Figure11_boxplots"


def figure12_violins(tstr: pd.DataFrame, out: Path, cfg: UtilityDropConfig) -> str | None:
    focus = tstr[tstr["Dataset"].isin(FOCUS_DATASETS.values())].copy()
    if focus.empty:
        return None
    leak0 = focus[focus["Leakage"] == focus["Leakage"].min()]
    rng = np.random.default_rng(7)
    rows = []
    for _, r in leak0.iterrows():
        samples = reconstruct_seed_samples(r["Mean"], r["Std"] if pd.notna(r["Std"]) else 0.01, cfg.n_seeds, rng)
        for s in samples:
            rows.append({"Generator": r["Generator"], "Accuracy": float(np.clip(s, 0, 1))})
    long = pd.DataFrame(rows)
    _style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    order = _gens(long)
    sns.violinplot(
        data=long, x="Generator", y="Accuracy", order=order, hue="Generator",
        hue_order=order, legend=False, ax=ax, inner="box",
        palette=[GENERATOR_COLORS.get(g, "gray") for g in order],
    )
    # mean labels live in CSV — avoid overlapping on-violin text
    means = (
        long.groupby("Generator")["Accuracy"].mean()
        .reindex(order)
        .reset_index()
    )
    means.to_csv(out.parent / "Tables" / "Figure12_violin_means.csv", index=False)
    ax.set_ylim(0, 1.08)
    ax.set_title("Violin Plots — Accuracy by Generator (approx. seed distribution)")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, out / "Figure12_violin_plots", cfg.figure_formats, cfg.figure_dpi)
    return "Figure12_violin_plots"


def generate_all_figures(
    tstr: pd.DataFrame,
    gen_sum: pd.DataFrame,
    loss_df: pd.DataFrame,
    best_clf: pd.DataFrame,
    best_gen: pd.DataFrame,
    utility: pd.DataFrame,
    dirs: dict[str, Path],
    cfg: UtilityDropConfig,
    multi_leakage: bool,
) -> list[str]:
    saved: list[str] = []
    fig_dir = dirs["figures"]
    for name in (
        figure01_utility_vs_leakage(gen_sum, fig_dir, cfg, multi_leakage),
        figure02_classifier_comparison(tstr, fig_dir, cfg),
        figure03_generator_comparison(gen_sum, fig_dir, cfg),
        figure04_heatmaps(tstr, fig_dir, cfg),
        figure05_utility_drop_heatmap(loss_df, fig_dir, cfg, multi_leakage),
        figure06_utility_loss(loss_df, fig_dir, cfg, multi_leakage),
        figure07_classifier_stability(tstr, fig_dir, cfg),
        figure08_best_classifier_table(best_clf, utility, fig_dir, cfg),
        figure09_best_generator_table(best_gen, fig_dir, cfg),
        figure11_boxplots(tstr, fig_dir, cfg),
        figure12_violins(tstr, fig_dir, cfg),
    ):
        if name:
            saved.append(name)
    saved.extend(figure10_critical_difference(tstr, fig_dir, cfg))
    return saved
