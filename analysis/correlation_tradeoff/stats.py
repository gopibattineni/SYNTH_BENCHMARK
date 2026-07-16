"""Correlation / regression statistics and text interpretation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CorrResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    r2: float
    slope: float
    intercept: float
    interpretation: str

    def as_dict(self) -> dict:
        return asdict(self)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.3f}"


def interpret_correlation(r: float, p: float) -> str:
    if not np.isfinite(r) or not np.isfinite(p):
        return "Insufficient data to assess the relationship."
    if p >= 0.05:
        return "No statistically significant relationship (p ≥ 0.05)."
    ar = abs(r)
    strength = (
        "strong" if ar >= 0.7 else "moderate" if ar >= 0.4 else "weak" if ar >= 0.2 else "very weak"
    )
    direction = "positive" if r > 0 else "negative"
    return f"{strength.capitalize()} {direction} correlation."


def compute_correlation(x: np.ndarray, y: np.ndarray) -> CorrResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 3:
        return CorrResult(
            n=n,
            pearson_r=np.nan,
            pearson_p=np.nan,
            spearman_rho=np.nan,
            spearman_p=np.nan,
            r2=np.nan,
            slope=np.nan,
            intercept=np.nan,
            interpretation="Too few points for correlation analysis.",
        )

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    r2 = float(pr**2) if np.isfinite(pr) else np.nan
    return CorrResult(
        n=n,
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        r2=r2,
        slope=float(slope),
        intercept=float(intercept),
        interpretation=interpret_correlation(float(pr), float(pp)),
    )


def stats_box_text(res: CorrResult) -> str:
    if res.n < 3 or not np.isfinite(res.pearson_r):
        return "Insufficient data"
    return (
        f"Pearson r = {res.pearson_r:.2f}\n"
        f"Spearman ρ = {res.spearman_rho:.2f}\n"
        f"R² = {res.r2:.2f}\n"
        f"{_fmt_p(res.pearson_p)}"
    )


def regression_band(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray | None = None,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x_grid, y_hat, 95% CI half-width for mean response."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        g = np.linspace(0, 1, 50) if x_grid is None else x_grid
        return g, np.full_like(g, np.nan, dtype=float), np.full_like(g, np.nan, dtype=float)

    slope, intercept, _, _, _ = stats.linregress(x, y)
    if x_grid is None:
        pad = 0.02 * (np.nanmax(x) - np.nanmin(x) + 1e-9)
        x_grid = np.linspace(np.nanmin(x) - pad, np.nanmax(x) + pad, 200)
    y_hat = intercept + slope * x_grid
    y_fit = intercept + slope * x
    resid = y - y_fit
    dof = max(len(x) - 2, 1)
    s_err = np.sqrt(np.sum(resid**2) / dof)
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean) ** 2)
    if ssx <= 0:
        half = np.full_like(x_grid, np.nan, dtype=float)
    else:
        se = s_err * np.sqrt(1.0 / len(x) + (x_grid - x_mean) ** 2 / ssx)
        tcrit = stats.t.ppf(1 - alpha / 2, dof)
        half = tcrit * se
    return x_grid, y_hat, half


def top_generators(panel: pd.DataFrame, x_col: str, y_col: str, k: int = 3) -> list[str]:
    """Generators closest to the upper-right of the (x,y) plane (after optional privacy flip)."""
    if panel.empty:
        return []
    g = (
        panel.groupby("Generator", dropna=False)
        .agg(x=(x_col, "mean"), y=(y_col, "mean"))
        .reset_index()
    )
    # Rank by sum of min-max normalized coords
    for col in ("x", "y"):
        lo, hi = g[col].min(), g[col].max()
        g[f"{col}_n"] = 0.5 if hi == lo else (g[col] - lo) / (hi - lo)
    g["score"] = g["x_n"] + g["y_n"]
    return list(g.sort_values("score", ascending=False)["Generator"].head(k))


def result_to_row(figure: str, x_name: str, y_name: str, res: CorrResult) -> dict:
    d = res.as_dict()
    d.update({"Figure": figure, "X": x_name, "Y": y_name})
    return d
