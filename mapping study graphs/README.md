# Mapping study graphs — Hungarian improvement over Greedy

**Extract-only analysis.** Mapping experiments were not rerun. Generators were not retrained.

## Research question

How much does **Hungarian matching** improve sample-to-sample matching compared with **Greedy matching**?

## Critical data availability

| Comparison | Available? | N (dataset×generator) | Source |
|---|---|---:|---|
| Greedy (many-to-one) Cosine vs Hungarian Cosine | **No** | 0 | Notebooks printed top-10 only; means not persisted |
| Pairwise Cosine vs Hungarian Cosine | Yes | 108 | `Agreed analysis/mapping_cost_comparison.csv` |
| Greedy (one-to-one) Mahalanobis vs Hungarian | Yes | 106 | Notebook outputs + Hungarian Excel / curated CSV |

**Cosine figures therefore use Pairwise Cosine as the non-Hungarian baseline** and are labeled accordingly (not “Greedy”).
**Mahalanobis figures use One-to-One Greedy**, which is the greedy Mahalanobis procedure implemented in the notebooks.

## Improvement definitions

```text
ΔCosine = Hungarian_Cosine − Pairwise_Cosine     # positive ⇒ Hungarian higher similarity
ΔMahalanobis = Greedy_Maha − Hungarian_Maha      # positive ⇒ Hungarian lower distance
```

Percentage improvements:

```text
Cosine % = (ΔCosine / Pairwise) × 100     # unstable when |Pairwise| ≈ 0 (flagged)
Mahalanobis % = (ΔMahalanobis / Greedy) × 100
```

Near-zero pairwise denominators are flagged in `hungarian_vs_greedy_summary.csv` (`Cosine_Pct_Unstable`).

## Overall results (from extracted pairs)

### Cosine (Pairwise → Hungarian)
- N = 108
- Mean ΔCosine = 0.9707
- Median ΔCosine = 0.9847
- % cases Hungarian higher = 100.0%

### Mahalanobis (Greedy one-to-one → Hungarian)
- N = 106
- Mean ΔMahalanobis = 1.3540
- Median ΔMahalanobis = 0.1292
- % cases Hungarian improved (lower distance) = 91.5%

## Main paper figures

| Figure | Path |
|---|---|
| Figure 1 | `main_paper/figure_1_cosine_improvement/` |
| Figure 2 | `main_paper/figure_2_mahalanobis_improvement/` |
| Figure 3 | `main_paper/figure_3_cosine_scatter/` |
| Figure 4 | `main_paper/figure_4_mahalanobis_scatter/` |
| Figure 5 | `main_paper/figure_5_improvement_heatmaps/` |
| Figure 6 | `main_paper/figure_6_improvement_distribution/` |

Each folder: PNG (300 DPI) + SVG.

## Summary tables

- `extracted_results/hungarian_vs_greedy_summary.csv` — unit-level (up to 15×8)
- `extracted_results/hungarian_vs_greedy_aggregate_stats.csv`
- `extracted_results/hungarian_vs_greedy_paired_tests.csv`
- `extracted_results/validation_checklist.json`

## Reproduce (figures only)

```bash
cd "SYNTH/mapping study graphs"
python create_hungarian_vs_greedy_figures.py
```

## Coverage gaps (honest)

- **Greedy cosine** aggregates were never persisted (top-10 prints only) → Figures 1/3/5a/6a use **Pairwise Cosine**, not Greedy.
- **Mahalanobis pairs = 106 / 120**. Missing comparable pairs where notebook Greedy means were non-comparable (e.g. Wine/Bank/OnlineShop SDV sampled means vs Hungarian) or outputs absent.
- Incomparable Wine SDV “Greedy mean (sampled)” values (~1e7–1e8) were **excluded** (not used).

## Figure 5 note

Panel (a) shows **Δ Cosine** (not %) because Pairwise baselines are near zero, making Cosine % numerically unstable (flagged in the summary CSV). Panel (b) shows Mahalanobis **%** improvement.

## Important findings (extracted results only)

### Cosine (Pairwise → Hungarian, N=108)
- Hungarian cosine is higher in **100%** of dataset×generator cases.
- Mean Δ ≈ **0.971**, median ≈ **0.985**.

### Mahalanobis (One-to-One Greedy → Hungarian, N=106)
- Hungarian lower distance in **91.5%** of cases; worse in **5.7%**; ties **2.8%**.
- Mean Δ ≈ **1.35**, median ≈ **0.13** (positive = Hungarian improvement).
- Wilcoxon signed-rank (paired, Greedy > Hungarian): significant (see `hungarian_vs_greedy_paired_tests.csv`).

### Where Greedy can look better
- A small minority of Mahalanobis pairs have Δ < 0 (Hungarian distance higher than Greedy). Inspect `hungarian_vs_greedy_summary.csv` filtered on `Delta_Mahalanobis < 0`.
