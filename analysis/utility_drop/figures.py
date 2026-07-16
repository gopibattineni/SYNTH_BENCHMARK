"""Publication figures for Utility Drop Analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.benchmarking import friedman_average_ranks, nemenyi_significance_groups
from analysis.config import GENERATORS, PLOT_RC
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
    fig, ax = plt.subplots(figsize=(11, 6.5))

    if multi_leakage and focus["Leakage"].nunique() > 1:
        for gen in _gens(focus):
            for ds in ("Cancer", "Mushroom"):
                sub = focus[(focus["Generator"] == gen) & (focus["DatasetKey"] == ds)].sort_values("Leakage")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["Leakage"], sub["Mean"], yerr=sub["Std"].fillna(0),
                    color=GENERATOR_COLORS.get(gen, "gray"), ls=DS_STYLE[ds],
                    marker=DS_MARKER[ds], capsize=3, lw=1.8, label=f"{gen} ({ds})",
                )
                for _, r in sub.iterrows():
                    ax.text(r["Leakage"], r["Mean"] + 0.01, f"{r['Mean']:.3f}", fontsize=6, ha="center",
                            color=GENERATOR_COLORS.get(gen, "gray"))
        ax.set_xlabel("Leakage Percentage")
        ax.set_title("Mean Accuracy ± SD vs Leakage — Cancer & Mushroom")
    else:
        # Single leakage level: grouped lines by dataset across generators
        gens = _gens(focus)
        x = np.arange(len(gens))
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
                ax.scatter([x[i] + offset], [means[i]], c=color, s=70, zorder=3, edgecolors="k", lw=0.4)
                if np.isfinite(means[i]):
                    ax.text(x[i] + offset, means[i] + 0.015, f"{means[i]:.3f}", fontsize=6.5, ha="center", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(gens, rotation=30, ha="right")
        ax.set_xlabel("Generator")
        ax.set_title("Mean Accuracy ± SD (Leakage = 0% only)")

    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
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
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(classifiers))
    width = 0.8 / max(len(gens), 1)
    for i, gen in enumerate(gens):
        sub = pivot[pivot["Generator"] == gen].set_index("Classifier").reindex(classifiers)
        vals = sub["Mean"].astype(float).values
        errs = sub["Std"].fillna(0).astype(float).values
        bars = ax.bar(
            x + i * width - 0.4 + width / 2, vals, width,
            yerr=errs, capsize=2, label=gen, color=GENERATOR_COLORS.get(gen, "gray"),
            edgecolor="k", lw=0.3, alpha=0.9,
        )
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=35, ha="right")
    ax.set_ylabel("Mean Accuracy ± SD")
    ax.set_ylim(0, 1.15)
    ax.set_title("Classifier Comparison — Mean Accuracy by Generator (Cancer+Mushroom avg)")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
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
    fig, ax = plt.subplots(figsize=(10, 5.5))
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
    for gen, row in agg.iterrows():
        ax.text(row["Mean"] + 0.01, gen, f"{row['Mean']:.3f} ± {row['Std']:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean Accuracy ± SD")
    ax.set_xlim(0, 1.25)
    ax.set_title("Generator Comparison (best=green, second=light green)")
    ax.grid(axis="x", alpha=0.3)
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
            for i, gen in enumerate(gens):
                v = sub.loc[gen, "Mean"] if gen in sub.index else np.nan
                if pd.notna(v):
                    ax.text(i + off, v + 0.01, f"{v:.3f}", ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(gens, rotation=30, ha="right")
        ax.set_xlabel("Generator")
        ax.set_title("Utility Loss = Acc(TRTR) − Acc(TSTR)  [Leakage=0%]")
    ax.set_ylabel("Utility Loss (Accuracy)")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
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
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(stats["Classifier"], stats["Mean"], xerr=stats["Std"].fillna(0),
            color="#4C78A8", edgecolor="k", lw=0.3, capsize=3)
    for _, r in stats.iterrows():
        ax.text(
            r["Mean"] + 0.01, r["Classifier"],
            f"{r['Mean']:.3f} ± {r['Std']:.3f}  (CV={r['CV']:.3f})",
            va="center", fontsize=7.5,
        )
    ax.set_xlabel("Mean Accuracy ± SD (across generators/datasets)")
    ax.set_xlim(0, 1.35)
    ax.set_title("Classifier Stability Ranking")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, out / "Figure07_classifier_stability", cfg.figure_formats, cfg.figure_dpi)
    stats.to_csv(out.parent / "Tables" / "classifier_stability.csv", index=False)
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
    # mean labels
    for i, gen in enumerate(order):
        m = long[long["Generator"] == gen]["Accuracy"].mean()
        ax.text(i, min(m + 0.05, 1.02), f"{m:.3f}", ha="center", fontsize=8)
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
