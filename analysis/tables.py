"""Publication-ready table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _format_mean_std(row: pd.Series) -> str:
    mean = row.get("Mean", row.get("NormalizedScore"))
    std = row.get("Std", 0)
    if pd.isna(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def _significance_marker(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_comparison_table(
    stats_df: pd.DataFrame,
    group_col: str = "Generator",
    metric_col: str = "Metric",
    value_cols: tuple[str, str] = ("Mean", "Std"),
) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()

    rows = []
    for (metric, gen), group in stats_df.groupby([metric_col, group_col]):
        row = group.iloc[0]
        rows.append(
            {
                metric_col: metric,
                group_col: gen,
                "Formatted": _format_mean_std(row),
                value_cols[0]: row.get(value_cols[0]),
                value_cols[1]: row.get(value_cols[1]),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    pivot = table.pivot(index=metric_col, columns=group_col, values="Formatted")
    means = table.pivot(index=metric_col, columns=group_col, values=value_cols[0])

    # Bold best, underline second best (higher is better for scores)
    formatted = pivot.copy()
    for metric in means.index:
        vals = means.loc[metric].dropna().sort_values(ascending=False)
        if len(vals) == 0:
            continue
        best, second = vals.index[0], vals.index[1] if len(vals) > 1 else None
        for col in formatted.columns:
            cell = formatted.loc[metric, col]
            if pd.isna(cell):
                continue
            if col == best:
                formatted.loc[metric, col] = f"\\textbf{{{cell}}}"
            elif second and col == second:
                formatted.loc[metric, col] = f"\\underline{{{cell}}}"

    return formatted


def export_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    latex_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        if df.empty:
            continue
        safe = name.replace(" ", "_").lower()
        df.to_csv(output_dir / f"{safe}.csv", index=True)
        df.to_excel(output_dir / f"{safe}.xlsx", index=True)
        try:
            latex = df.to_latex(index=True, escape=False)
            (latex_dir / f"{safe}.tex").write_text(latex, encoding="utf-8")
        except Exception:
            pass
        try:
            md = df.to_markdown()
            (output_dir / f"{safe}.md").write_text(md, encoding="utf-8")
        except Exception:
            pass
