"""Paper-ready tables for two-dataset case study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.tables import build_comparison_table, export_tables


def build_generator_summary_table(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    cols = ["Generator", "UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore", "Rank"]
    avail = [c for c in cols if c in ranking.columns]
    table = ranking[avail].copy()

    for col in ["UtilityScore", "FidelityScore", "PrivacyScore", "OverallScore"]:
        if col not in table.columns:
            continue
        vals = table[col].dropna().sort_values(ascending=False)
        if vals.empty:
            continue
        best = vals.index[0]
        second = vals.index[1] if len(vals) > 1 else None
        formatted = []
        for idx, row in table.iterrows():
            v = row[col]
            if pd.isna(v):
                formatted.append("")
            else:
                text = f"{v:.4f}"
                if idx == best:
                    text = f"\\textbf{{{text}}}"
                elif second is not None and idx == second:
                    text = f"\\underline{{{text}}}"
                formatted.append(text)
        table[f"{col}_Formatted"] = formatted
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


def export_dataset_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    latex_dir: Path | None = None,
) -> None:
    latex_dir = latex_dir or output_dir
    export_tables(tables, output_dir, latex_dir)
