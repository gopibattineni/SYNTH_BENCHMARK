"""Narrative summaries for correlation figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.correlation_tradeoff.stats import CorrResult, top_generators


def _relation_phrase(res: CorrResult, x_name: str, y_name: str) -> str:
    if res.n < 3 or not pd.notna(res.pearson_r):
        return f"Insufficient points to characterise the {x_name}–{y_name} relationship."
    r = res.pearson_r
    if res.pearson_p >= 0.05:
        return (
            f"No statistically significant association between {x_name.lower()} and "
            f"{y_name.lower()} (Pearson r = {r:.2f}, p ≥ 0.05)."
        )
    if r > 0:
        return (
            f"Higher {x_name.lower()} is associated with higher {y_name.lower()} "
            f"(Pearson r = {r:.2f})."
        )
    return (
        f"Higher {x_name.lower()} is associated with lower {y_name.lower()} "
        f"(Pearson r = {r:.2f})."
    )


def summary_fidelity_utility(panel: pd.DataFrame, res: CorrResult) -> str:
    metric = panel["UtilityMetric"].iloc[0]
    tops = top_generators(panel, "Fidelity", "Utility", k=3)
    tops_txt = ", ".join(tops) if tops else "leading generators"
    lines = [
        f"Fidelity vs Utility ({metric})",
        _relation_phrase(res, "Fidelity", f"predictive utility ({metric})"),
        f"{tops_txt} occupy the upper-right region of the fidelity–utility plane.",
        f"{res.interpretation} This suggests that preserving statistical properties "
        f"of the real data tends to improve downstream machine-learning performance.",
    ]
    return "\n".join(lines)


def summary_privacy_utility(panel: pd.DataFrame, res: CorrResult) -> str:
    metric = panel["UtilityMetric"].iloc[0]
    priv = panel["PrivacyLabel"].iloc[0]
    higher = bool(panel["HigherIsPrivate"].iloc[0])
    # For ranking favorable privacy–utility, flip MIA so higher = better privacy
    tmp = panel.copy()
    if not higher:
        tmp = tmp.assign(PrivacyScore=1.0 - tmp["Privacy"].clip(0, 1))
        x_col = "PrivacyScore"
    else:
        x_col = "Privacy"
        tmp = tmp.assign(PrivacyScore=tmp["Privacy"])
    tops = top_generators(tmp, x_col, "Utility", k=3)
    tops_txt = ", ".join(tops) if tops else "selected generators"

    if higher:
        trade = _relation_phrase(res, priv, f"utility ({metric})")
    else:
        # MIA: higher AUC → weaker privacy; negative r with utility means privacy↑ utility↓
        trade = _relation_phrase(res, f"{priv} (attack success)", f"utility ({metric})")

    lines = [
        f"Privacy vs Utility ({metric}, {priv})",
        trade,
        f"{tops_txt} achieve a comparatively favourable privacy–utility balance.",
        f"{res.interpretation} The pattern is consistent with the expected "
        f"privacy–utility trade-off for synthetic tabular data.",
    ]
    return "\n".join(lines)


def summary_fidelity_privacy(panel: pd.DataFrame, res: CorrResult) -> str:
    priv = panel["PrivacyLabel"].iloc[0]
    lines = [
        f"Fidelity vs Privacy ({priv})",
        _relation_phrase(res, priv, "fidelity"),
        "Some generators preserve statistical fidelity while maintaining acceptable privacy protection.",
        f"{res.interpretation}",
    ]
    return "\n".join(lines)


def write_summaries(
    panel: pd.DataFrame,
    results: dict[str, CorrResult],
    out_dir: Path,
    prefix: str = "",
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "Figure01": summary_fidelity_utility,
        "Figure02": summary_privacy_utility,
        "Figure03": summary_fidelity_privacy,
        "FigureS01": summary_fidelity_utility,
        "FigureS02": summary_privacy_utility,
        "FigureS03": summary_fidelity_privacy,
    }
    written: list[str] = []
    for name, res in results.items():
        key = name.split("_")[0]
        fn = writers.get(key)
        if fn is None:
            continue
        text = fn(panel, res)
        if prefix:
            text = prefix + text
        path = out_dir / f"{name}_summary.txt"
        path.write_text(text + "\n", encoding="utf-8")
        written.append(str(path))
    return written
