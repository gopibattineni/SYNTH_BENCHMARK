"""Paper-ready comparative tables for Cancer vs Mushroom case study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import GENERATORS
from analysis.tables import export_tables


def _fmt_best_second(series: pd.Series, higher_is_better: bool = True) -> list[str]:
    """Format values; bold best, underline second-best (LaTeX)."""
    vals = series.astype(float)
    order = vals.sort_values(ascending=not higher_is_better)
    best_idx = order.index[0] if len(order) else None
    second_idx = order.index[1] if len(order) > 1 else None
    out: list[str] = []
    for idx, v in series.items():
        if pd.isna(v):
            out.append("")
            continue
        text = f"{float(v):.4f}"
        if idx == best_idx:
            text = f"\\textbf{{{text}}}"
        elif idx == second_idx:
            text = f"\\underline{{{text}}}"
        out.append(text)
    return out


def build_generator_summary_table(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    cols = ["Generator", "UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore", "Rank"]
    avail = [c for c in cols if c in ranking.columns]
    table = ranking[avail].copy()
    for col in ["UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"]:
        if col not in table.columns:
            continue
        table[f"{col}_Formatted"] = _fmt_best_second(table[col], higher_is_better=True)
    if "Rank" in table.columns:
        table["Rank_Formatted"] = _fmt_best_second(table["Rank"], higher_is_better=False)
    return table


def build_comparative_table(
    cancer_rank: pd.DataFrame,
    mushroom_rank: pd.DataFrame,
) -> pd.DataFrame:
    if cancer_rank.empty or mushroom_rank.empty:
        return pd.DataFrame()
    c = cancer_rank.set_index("Generator").add_suffix("_Cancer")
    m = mushroom_rank.set_index("Generator").add_suffix("_Mushroom")
    merged = c.join(m, how="outer")
    merged["Utility_Delta"] = merged.get("UtilityScore_Cancer", 0) - merged.get("UtilityScore_Mushroom", 0)
    merged["Privacy_Delta"] = merged.get("PrivacyScore_Cancer", 0) - merged.get("PrivacyScore_Mushroom", 0)
    merged["Fidelity_Delta"] = merged.get("FidelityScore_Cancer", 0) - merged.get("FidelityScore_Mushroom", 0)
    merged["Overall_Delta"] = merged.get("OverallScore_Cancer", 0) - merged.get("OverallScore_Mushroom", 0)
    return merged.reset_index()


def build_paper_score_table(
    cancer_rank: pd.DataFrame,
    mushroom_rank: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paper-ready scores: Utility, Fidelity, Privacy, Overall, Average Rank
    for Cancer, Mushroom, and Overall Average.
    """
    if cancer_rank.empty and mushroom_rank.empty:
        return pd.DataFrame()

    c = cancer_rank.set_index("Generator") if not cancer_rank.empty else pd.DataFrame()
    m = mushroom_rank.set_index("Generator") if not mushroom_rank.empty else pd.DataFrame()
    gens = [g for g in GENERATORS if g in set(c.index) | set(m.index)]
    rows: list[dict] = []

    for gen in gens:
        row: dict = {"Generator": gen}
        for label, df, prefix in (("Cancer", c, "Cancer"), ("Mushroom", m, "Mushroom")):
            if df.empty or gen not in df.index:
                for metric in ("Utility", "Fidelity", "Privacy", "Overall", "Rank"):
                    row[f"{prefix}_{metric}"] = np.nan
                continue
            r = df.loc[gen]
            row[f"{prefix}_Utility"] = r.get("UtilityScore")
            row[f"{prefix}_Fidelity"] = r.get("FidelityScore")
            row[f"{prefix}_Privacy"] = r.get("PrivacyScore")
            row[f"{prefix}_Overall"] = r.get("OverallScore")
            row[f"{prefix}_Rank"] = r.get("Rank")
        # Overall average across datasets
        for metric in ("Utility", "Fidelity", "Privacy", "Overall"):
            vals = [row.get(f"Cancer_{metric}"), row.get(f"Mushroom_{metric}")]
            vals = [v for v in vals if pd.notna(v)]
            row[f"Average_{metric}"] = float(np.mean(vals)) if vals else np.nan
        ranks = [row.get("Cancer_Rank"), row.get("Mushroom_Rank")]
        ranks = [v for v in ranks if pd.notna(v)]
        row["Average_Rank"] = float(np.mean(ranks)) if ranks else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Sort by average overall score
    table = table.sort_values("Average_Overall", ascending=False).reset_index(drop=True)

    # LaTeX-formatted highlight columns (best bold, second underlined) within each score column
    score_cols = [
        c for c in table.columns
        if c.endswith(("_Utility", "_Fidelity", "_Privacy", "_Overall")) or c == "Average_Rank"
    ]
    for col in score_cols:
        higher = col != "Average_Rank" and not col.endswith("_Rank")
        if col.endswith("_Rank"):
            higher = False
        table[f"{col}_TeX"] = _fmt_best_second(table[col], higher_is_better=higher)

    return table


def build_latex_score_table(table: pd.DataFrame) -> str:
    """Compact LaTeX tabular with Cancer / Mushroom / Average blocks."""
    if table.empty:
        return ""

    lines = [
        r"\begin{tabular}{l" + "c" * 9 + "}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Cancer} & \multicolumn{3}{c}{Mushroom} & \multicolumn{3}{c}{Average} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Generator & Util. & Fid. & Priv. & Util. & Fid. & Priv. & Util. & Fid. & Overall \\",
        r"\midrule",
    ]

    def _cell(row: pd.Series, col: str) -> str:
        tex = row.get(f"{col}_TeX")
        if isinstance(tex, str) and tex:
            return tex
        v = row.get(col)
        return f"{float(v):.4f}" if pd.notna(v) else "---"

    for _, row in table.iterrows():
        cells = [
            str(row["Generator"]),
            _cell(row, "Cancer_Utility"),
            _cell(row, "Cancer_Fidelity"),
            _cell(row, "Cancer_Privacy"),
            _cell(row, "Mushroom_Utility"),
            _cell(row, "Mushroom_Fidelity"),
            _cell(row, "Mushroom_Privacy"),
            _cell(row, "Average_Utility"),
            _cell(row, "Average_Fidelity"),
            _cell(row, "Average_Overall"),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def export_dataset_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    latex_dir: Path | None = None,
) -> None:
    latex_dir = latex_dir or output_dir
    export_tables(tables, output_dir, latex_dir)


def export_comparative_tables(
    cancer_rank: pd.DataFrame,
    mushroom_rank: pd.DataFrame,
    extras: dict[str, pd.DataFrame],
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    comp = build_comparative_table(cancer_rank, mushroom_rank)
    paper = build_paper_score_table(cancer_rank, mushroom_rank)

    frames = {
        "cancer_vs_mushroom": comp,
        "paper_scores": paper,
        "cancer_summary": build_generator_summary_table(cancer_rank),
        "mushroom_summary": build_generator_summary_table(mushroom_rank),
    }
    for key, df in extras.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames[key] = df

    for name, df in frames.items():
        if df.empty:
            continue
        df.to_csv(tables_dir / f"{name}.csv", index=False)
        try:
            df.to_excel(tables_dir / f"{name}.xlsx", index=False)
        except Exception:
            pass

    if not paper.empty:
        tex = build_latex_score_table(paper)
        (tables_dir / "paper_scores.tex").write_text(tex + "\n", encoding="utf-8")
        # Also export average-rank focused table
        rank_cols = [
            c for c in (
                "Generator",
                "Cancer_Overall",
                "Mushroom_Overall",
                "Average_Overall",
                "Cancer_Rank",
                "Mushroom_Rank",
                "Average_Rank",
            )
            if c in paper.columns
        ]
        paper[rank_cols].to_csv(tables_dir / "overall_ranks.csv", index=False)
