"""Generator ranking: average ranks, Borda, weighted composite."""

from __future__ import annotations

import pandas as pd

from analysis.config import PipelineConfig


def average_ranks(rank_df: pd.DataFrame, generator_col: str = "Generator", rank_col: str = "Rank") -> pd.DataFrame:
    if rank_df.empty:
        return pd.DataFrame()
    return (
        rank_df.groupby(generator_col, dropna=False)[rank_col]
        .agg(AverageRank="mean", Std="std", Count="count")
        .reset_index()
        .sort_values("AverageRank")
    )


def borda_count(rankings: list[pd.DataFrame], generator_col: str = "Generator", rank_col: str = "Rank") -> pd.DataFrame:
    scores: dict[str, float] = {}
    for ranking in rankings:
        if ranking.empty:
            continue
        n = ranking[rank_col].max()
        for _, row in ranking.iterrows():
            gen = row[generator_col]
            if pd.isna(gen):
                continue
            scores[gen] = scores.get(gen, 0.0) + (n - row[rank_col] + 1)
    if not scores:
        return pd.DataFrame()
    out = pd.DataFrame({"Generator": list(scores.keys()), "BordaScore": list(scores.values())})
    out["BordaRank"] = out["BordaScore"].rank(ascending=False, method="average")
    return out.sort_values("BordaRank")


def weighted_overall_ranking(
    utility_scores: pd.DataFrame,
    privacy_scores: pd.DataFrame,
    fidelity_scores: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    frames = []
    for name, df, weight in [
        ("Utility", utility_scores, config.utility_weight),
        ("Privacy", privacy_scores, config.privacy_weight),
        ("Fidelity", fidelity_scores, config.fidelity_weight),
    ]:
        if df.empty or "Generator" not in df.columns:
            continue
        score_col = "NormalizedScore" if "NormalizedScore" in df.columns else "Mean"
        chunk = df.groupby("Generator", dropna=False)[score_col].mean().reset_index()
        chunk["Component"] = name
        chunk["Weight"] = weight
        chunk["WeightedScore"] = chunk[score_col] * weight
        frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    overall = (
        combined.groupby("Generator", dropna=False)["WeightedScore"]
        .sum()
        .reset_index()
        .rename(columns={"WeightedScore": "OverallScore"})
    )
    overall["OverallRank"] = overall["OverallScore"].rank(ascending=False, method="average")
    return overall.sort_values("OverallRank")


def compile_rankings(
    utility_results: dict,
    privacy_results: dict,
    fidelity_results: dict,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    util_ranks = utility_results.get("generator_summary", pd.DataFrame())
    priv_ranks = privacy_results.get("privacy_ranks", pd.DataFrame())
    fid_ranks = fidelity_results.get("fidelity_ranks", pd.DataFrame())

    util_avg = average_ranks(util_ranks) if not util_ranks.empty else pd.DataFrame()
    priv_avg = average_ranks(priv_ranks) if not priv_ranks.empty else pd.DataFrame()
    fid_avg = average_ranks(fid_ranks) if not fid_ranks.empty else pd.DataFrame()

    borda = borda_count([r for r in [util_ranks, priv_ranks, fid_ranks] if not r.empty])

    util_scores = util_ranks.groupby("Generator", dropna=False)["NormalizedScore"].mean().reset_index() if not util_ranks.empty else pd.DataFrame()
    priv_scores = privacy_results.get("overall_privacy", pd.DataFrame())
    fid_scores = fidelity_results.get("overall_fidelity", pd.DataFrame())

    if not priv_scores.empty:
        priv_scores = priv_scores.rename(columns={"Mean": "NormalizedScore"})
    if not fid_scores.empty:
        fid_scores = fid_scores.rename(columns={"Mean": "NormalizedScore"})

    weighted = weighted_overall_ranking(util_scores, priv_scores, fid_scores, config)

    return {
        "utility_avg_ranks": util_avg,
        "privacy_avg_ranks": priv_avg,
        "fidelity_avg_ranks": fid_avg,
        "borda": borda,
        "weighted_overall": weighted,
    }
