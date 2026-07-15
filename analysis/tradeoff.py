"""Trade-off analysis: privacy vs utility vs fidelity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _pareto_frontier(points: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if points.empty:
        return points
    sorted_pts = points.sort_values([x_col, y_col], ascending=[False, False])
    frontier = []
    max_y = -np.inf
    for _, row in sorted_pts.iterrows():
        if row[y_col] >= max_y:
            frontier.append(row)
            max_y = row[y_col]
    return pd.DataFrame(frontier)


def build_tradeoff_points(
    utility_scores: pd.DataFrame,
    privacy_scores: pd.DataFrame,
    fidelity_scores: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for name, df in [("Utility", utility_scores), ("Privacy", privacy_scores), ("Fidelity", fidelity_scores)]:
        if df.empty or "Generator" not in df.columns:
            continue
        score_col = "NormalizedScore" if "NormalizedScore" in df.columns else "Mean"
        chunk = df.groupby("Generator", dropna=False)[score_col].mean().reset_index()
        chunk = chunk.rename(columns={score_col: name})
        frames.append(chunk.set_index("Generator"))

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    merged = merged.reset_index().fillna(merged.mean(numeric_only=True))
    return merged


def tradeoff_correlations(tradeoff_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("Utility", "Privacy", "Fidelity") if c in tradeoff_df.columns]
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            x, y = tradeoff_df[c1].values, tradeoff_df[c2].values
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            rows.append({"X": c1, "Y": c2, "Pearson_r": pr, "Pearson_p": pp, "Spearman_r": sr, "Spearman_p": sp})
    return pd.DataFrame(rows)


def analyze_tradeoffs(
    utility_results: dict,
    privacy_results: dict,
    fidelity_results: dict,
) -> dict[str, pd.DataFrame]:
    util = utility_results.get("generator_summary", pd.DataFrame())
    priv = privacy_results.get("overall_privacy", pd.DataFrame())
    fid = fidelity_results.get("overall_fidelity", pd.DataFrame())

    if not priv.empty:
        priv = priv.rename(columns={"Mean": "NormalizedScore"})
    if not fid.empty:
        fid = fid.rename(columns={"Mean": "NormalizedScore"})

    points = build_tradeoff_points(util, priv, fid)
    correlations = tradeoff_correlations(points)

    pareto_up = _pareto_frontier(points, "Utility", "Privacy") if not points.empty else pd.DataFrame()
    pareto_fp = _pareto_frontier(points, "Fidelity", "Privacy") if not points.empty else pd.DataFrame()

    if not points.empty and {"Utility", "Privacy", "Fidelity"}.issubset(points.columns):
        dominated = []
        for _, row in points.iterrows():
            is_dominated = any(
                (other["Utility"] >= row["Utility"])
                and (other["Privacy"] >= row["Privacy"])
                and (other["Fidelity"] >= row["Fidelity"])
                and (
                    (other["Utility"] > row["Utility"])
                    or (other["Privacy"] > row["Privacy"])
                    or (other["Fidelity"] > row["Fidelity"])
                )
                for _, other in points.iterrows()
                if other["Generator"] != row["Generator"]
            )
            dominated.append(is_dominated)
        points = points.copy()
        points["Dominated"] = dominated

    return {
        "tradeoff_points": points,
        "tradeoff_correlations": correlations,
        "pareto_utility_privacy": pareto_up,
        "pareto_fidelity_privacy": pareto_fp,
    }
