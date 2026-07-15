"""Statistical testing: ANOVA, Friedman, post-hoc, effect sizes, corrections."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from analysis.config import GENERATORS, PipelineConfig


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum(a > b for a in x for b in y)
    lt = sum(a < b for a in x for b in y)
    return float((gt - lt) / (len(x) * len(y)))


def _holm(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    _, corrected, _, _ = multipletests(pvals, method="holm")
    return corrected.tolist()


def _bonferroni(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    _, corrected, _, _ = multipletests(pvals, method="bonferroni")
    return corrected.tolist()


def _bh(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    _, corrected, _, _ = multipletests(pvals, method="fdr_bh")
    return corrected.tolist()


def run_generator_comparison_tests(
    summary_df: pd.DataFrame,
    value_col: str = "Mean",
    block_col: str = "Dataset",
    treatment_col: str = "Generator",
    metric_col: str = "Metric",
) -> dict[str, pd.DataFrame]:
    """Run Friedman / RM-ANOVA and pairwise tests per metric."""
    if summary_df.empty:
        return {
            "omnibus": pd.DataFrame(),
            "pairwise_wilcoxon": pd.DataFrame(),
            "pairwise_ttest": pd.DataFrame(),
            "effect_sizes": pd.DataFrame(),
            "nemenyi": pd.DataFrame(),
        }

    omnibus_rows: list[dict] = []
    wilcoxon_rows: list[dict] = []
    ttest_rows: list[dict] = []
    effect_rows: list[dict] = []
    nemenyi_frames: list[pd.DataFrame] = []

    for metric, metric_df in summary_df.groupby(metric_col):
        pivot = metric_df.pivot_table(
            index=block_col,
            columns=treatment_col,
            values=value_col,
            aggfunc="mean",
        )
        pivot = pivot.dropna(axis=0, how="any")
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue

        generators = [g for g in pivot.columns if g in GENERATORS or True]
        data_matrix = pivot[generators].values

        # Friedman test (non-parametric repeated measures)
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot[g].values for g in generators])
            test_name = "Friedman"
            stat_val, p_val = friedman_stat, friedman_p
        except Exception:
            stat_val, p_val, test_name = np.nan, np.nan, "Friedman_failed"

        # Repeated measures ANOVA via pingouin when possible
        long = metric_df[[block_col, treatment_col, value_col]].dropna()
        rm_p = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aov = pg.rm_anova(data=long, dv=value_col, within=treatment_col, subject=block_col, detailed=True)
            if not aov.empty:
                rm_p = float(aov.loc[aov["Source"] == treatment_col, "p-unc"].iloc[0])
                if p_val == p_val and rm_p == rm_p:
                    test_name = "Friedman+RM-ANOVA"
        except Exception:
            pass

        omnibus_rows.append(
            {
                "Metric": metric,
                "Test": test_name,
                "Statistic": stat_val,
                "p_value": p_val,
                "RM_ANOVA_p": rm_p,
                "N_blocks": pivot.shape[0],
                "N_generators": pivot.shape[1],
            }
        )

        # Nemenyi post-hoc
        try:
            nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi.index = list(pivot.columns)
            nemenyi.columns = list(pivot.columns)
            nemenyi["Metric"] = metric
            nemenyi_frames.append(nemenyi.reset_index().rename(columns={"index": "Generator_A"}))
        except Exception:
            pass

        # Pairwise Wilcoxon and t-tests with effect sizes
        raw_p_w, raw_p_t = [], []
        pairs = []
        for i, g1 in enumerate(generators):
            for g2 in generators[i + 1 :]:
                x = pivot[g1].values
                y = pivot[g2].values
                try:
                    _, pw = stats.wilcoxon(x, y)
                except Exception:
                    pw = np.nan
                try:
                    _, pt = stats.ttest_rel(x, y)
                except Exception:
                    pt = np.nan
                raw_p_w.append(pw)
                raw_p_t.append(pt)
                pairs.append((g1, g2, x, y, pw, pt))

        holm_w = _holm([p for p in raw_p_w if p == p])
        holm_t = _holm([p for p in raw_p_t if p == p])
        bh_w = _bh([p for p in raw_p_w if p == p])
        bh_t = _bh([p for p in raw_p_t if p == p])

        wi = ti = 0
        for g1, g2, x, y, pw, pt in pairs:
            wilcoxon_rows.append(
                {
                    "Metric": metric,
                    "Generator_A": g1,
                    "Generator_B": g2,
                    "p_raw": pw,
                    "p_holm": holm_w[wi] if pw == pw and wi < len(holm_w) else np.nan,
                    "p_bh": bh_w[wi] if pw == pw and wi < len(bh_w) else np.nan,
                    "Cliffs_Delta": cliffs_delta(x, y),
                }
            )
            if pw == pw:
                wi += 1

            ttest_rows.append(
                {
                    "Metric": metric,
                    "Generator_A": g1,
                    "Generator_B": g2,
                    "p_raw": pt,
                    "p_holm": holm_t[ti] if pt == pt and ti < len(holm_t) else np.nan,
                    "p_bh": bh_t[ti] if pt == pt and ti < len(bh_t) else np.nan,
                    "Cohens_d": cohens_d(x, y),
                }
            )
            if pt == pt:
                ti += 1

            effect_rows.append(
                {
                    "Metric": metric,
                    "Generator_A": g1,
                    "Generator_B": g2,
                    "Cohens_d": cohens_d(x, y),
                    "Cliffs_Delta": cliffs_delta(x, y),
                }
            )

    return {
        "omnibus": pd.DataFrame(omnibus_rows),
        "pairwise_wilcoxon": pd.DataFrame(wilcoxon_rows),
        "pairwise_ttest": pd.DataFrame(ttest_rows),
        "effect_sizes": pd.DataFrame(effect_rows),
        "nemenyi": pd.concat(nemenyi_frames, ignore_index=True) if nemenyi_frames else pd.DataFrame(),
    }
