"""Paper-ready tables for representative dual trade-off analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import GENERATORS


def _fmt_best_second(series: pd.Series, higher_is_better: bool = True) -> list[str]:
    vals = series.astype(float)
    order = vals.sort_values(ascending=not higher_is_better)
    best = order.index[0] if len(order) else None
    second = order.index[1] if len(order) > 1 else None
    out = []
    for idx, v in series.items():
        if pd.isna(v):
            out.append("---")
            continue
        text = f"{float(v):.4f}"
        if idx == best:
            text = f"\\textbf{{{text}}}"
        elif idx == second:
            text = f"\\underline{{{text}}}"
        out.append(text)
    return out


def build_results_table(scores: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """Wide table with Mean±Std, CI, ranks for Cancer / Mushroom / Average."""
    if scores.empty:
        return pd.DataFrame()

    rows = []
    for gen in GENERATORS:
        if gen not in set(scores["Generator"]):
            continue
        rec: dict = {"Generator": gen, "Analysis": analysis}
        f1s, nndrs, qs = [], [], []
        for ds in ("Cancer", "Mushroom"):
            sub = scores[(scores["Generator"] == gen) & (scores["DatasetKey"] == ds)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            prefix = ds
            rec[f"{prefix}_F1"] = r["F1"]
            rec[f"{prefix}_F1_Std"] = r.get("F1_Std")
            rec[f"{prefix}_F1_CI_Low"] = r.get("F1_CI_Low")
            rec[f"{prefix}_F1_CI_High"] = r.get("F1_CI_High")
            rec[f"{prefix}_NNDR"] = r.get("NNDR")
            rec[f"{prefix}_Quality"] = r.get("Quality")
            rec[f"{prefix}_Rank"] = r.get("Rank")
            if analysis == "Best_Classifier":
                rec[f"{prefix}_BestClassifier"] = r.get("BestClassifier")
            f1s.append(r["F1"])
            if pd.notna(r.get("NNDR")):
                nndrs.append(r["NNDR"])
            if pd.notna(r.get("Quality")):
                qs.append(r["Quality"])
            # Formatted Mean ± Std (CI)
            std = r.get("F1_Std")
            ci_lo, ci_hi = r.get("F1_CI_Low"), r.get("F1_CI_High")
            if pd.notna(std):
                rec[f"{prefix}_F1_Display"] = f"{r['F1']:.4f} ± {float(std):.4f}"
            else:
                rec[f"{prefix}_F1_Display"] = f"{r['F1']:.4f}"
            if pd.notna(ci_lo) and pd.notna(ci_hi):
                rec[f"{prefix}_F1_CI"] = f"[{float(ci_lo):.4f}, {float(ci_hi):.4f}]"
        rec["Average_F1"] = float(np.mean(f1s)) if f1s else np.nan
        rec["Average_NNDR"] = float(np.mean(nndrs)) if nndrs else np.nan
        rec["Average_Quality"] = float(np.mean(qs)) if qs else np.nan
        ranks = [rec.get("Cancer_Rank"), rec.get("Mushroom_Rank")]
        ranks = [x for x in ranks if pd.notna(x)]
        rec["Average_Rank"] = float(np.mean(ranks)) if ranks else np.nan
        rows.append(rec)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("Average_F1", ascending=False).reset_index(drop=True)

    for col in [
        "Cancer_F1", "Mushroom_F1", "Average_F1",
        "Cancer_NNDR", "Mushroom_NNDR", "Average_NNDR",
        "Cancer_Quality", "Mushroom_Quality", "Average_Quality",
    ]:
        if col in table.columns:
            table[f"{col}_TeX"] = _fmt_best_second(table[col], higher_is_better=True)
    if "Average_Rank" in table.columns:
        table["Average_Rank_TeX"] = _fmt_best_second(table["Average_Rank"], higher_is_better=False)
    return table


def build_comparison_table(indep: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side Classifier-Independent vs Best-Classifier."""
    if indep.empty or best.empty:
        return pd.DataFrame()
    a = indep.set_index("Generator").add_suffix("_Indep")
    b = best.set_index("Generator").add_suffix("_Best")
    merged = a.join(b, how="outer").reset_index()
    for metric in ("Average_F1", "Average_NNDR", "Average_Quality", "Average_Rank"):
        ic, bc = f"{metric}_Indep", f"{metric}_Best"
        if ic in merged.columns and bc in merged.columns:
            merged[f"{metric}_Delta"] = merged[bc] - merged[ic]
    return merged


def build_latex_table(table: pd.DataFrame, title_note: str) -> str:
    if table.empty:
        return ""
    lines = [
        f"% {title_note}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Generator & Cancer F1 & Mushroom F1 & Avg F1 & Avg NNDR & Avg Quality & Avg Rank \\",
        r"\midrule",
    ]

    def cell(row, col):
        tex = row.get(f"{col}_TeX")
        if isinstance(tex, str) and tex:
            return tex
        v = row.get(col)
        return f"{float(v):.4f}" if pd.notna(v) else "---"

    for _, row in table.iterrows():
        lines.append(
            " & ".join(
                [
                    str(row["Generator"]),
                    cell(row, "Cancer_F1"),
                    cell(row, "Mushroom_F1"),
                    cell(row, "Average_F1"),
                    cell(row, "Average_NNDR"),
                    cell(row, "Average_Quality"),
                    cell(row, "Average_Rank"),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def wilcoxon_rank_consistency(indep: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank on paired Average_F1 / Average_Rank across generators."""
    from scipy import stats

    if indep.empty or best.empty:
        return pd.DataFrame()
    merged = indep[["Generator", "Average_F1", "Average_Rank"]].merge(
        best[["Generator", "Average_F1", "Average_Rank"]],
        on="Generator",
        suffixes=("_Indep", "_Best"),
    )
    rows = []
    for metric in ("Average_F1", "Average_Rank"):
        a = merged[f"{metric}_Indep"].astype(float)
        b = merged[f"{metric}_Best"].astype(float)
        if len(a) < 3:
            continue
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, np.nan
        # Spearman rank correlation of generator orderings
        rho, rp = stats.spearmanr(a.rank(ascending=False), b.rank(ascending=False))
        rows.append(
            {
                "Metric": metric,
                "Wilcoxon_stat": stat,
                "Wilcoxon_p": p,
                "Spearman_rank_rho": rho,
                "Spearman_p": rp,
                "Interpretation": (
                    "Rankings consistent (rho ≥ 0.7)"
                    if pd.notna(rho) and rho >= 0.7
                    else "Rankings diverge"
                ),
            }
        )
    return pd.DataFrame(rows)


def export_tables(
    indep_scores: pd.DataFrame,
    best_scores: pd.DataFrame,
    indep_dirs: dict[str, Path],
    best_dirs: dict[str, Path],
    comp_dirs: dict[str, Path],
    extras_indep: dict[str, pd.DataFrame],
    extras_best: dict[str, pd.DataFrame],
) -> None:
    indep_tab = build_results_table(indep_scores, "Classifier_Independent")
    best_tab = build_results_table(best_scores, "Best_Classifier")
    comp = build_comparison_table(indep_tab, best_tab)
    wilcox = wilcoxon_rank_consistency(indep_tab, best_tab)

    for tab, dirs, name, note in (
        (indep_tab, indep_dirs, "classifier_independent_results", "Primary — classifier-independent"),
        (best_tab, best_dirs, "best_classifier_results", "Supplementary — best classifier"),
    ):
        if tab.empty:
            continue
        tab.to_csv(dirs["tables"] / f"{name}.csv", index=False)
        tab.to_excel(dirs["tables"] / f"{name}.xlsx", index=False)
        (dirs["tables"] / f"{name}.tex").write_text(
            build_latex_table(tab, note) + "\n", encoding="utf-8"
        )

    if not comp.empty:
        comp.to_csv(comp_dirs["tables"] / "comparison_indep_vs_best.csv", index=False)
        comp.to_excel(comp_dirs["tables"] / "comparison_indep_vs_best.xlsx", index=False)
    if not wilcox.empty:
        wilcox.to_csv(comp_dirs["statistics"] / "wilcoxon_rank_consistency.csv", index=False)
        wilcox.to_excel(comp_dirs["statistics"] / "wilcoxon_rank_consistency.xlsx", index=False)

    for extras, dirs in ((extras_indep, indep_dirs), (extras_best, best_dirs)):
        for key, df in extras.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(dirs["statistics"] / f"{key}.csv", index=False)
