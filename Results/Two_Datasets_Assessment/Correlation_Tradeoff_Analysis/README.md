# Correlation Trade-off Analysis

Publication-quality correlation figures relating **Utility**, **Fidelity**, and **Privacy**
for Cancer and Mushroom across all generators.

## Metrics

| Axis | Metric |
|------|--------|
| Utility | Mean TSTR Accuracy (+ F1 variant) |
| Fidelity | SDMetrics Overall Quality Score (fallback: Column Shapes / KS_Complement) |
| Privacy A | MIA AUC (lower = more private) |
| Privacy B | NNDR (higher = more private) |

## Points

Each scatter point is **Generator × Seed × Dataset**. Raw per-seed Excel rows are not
exported by the generator notebooks; utility seeds are reconstructed from Mean±SD
(`N_SEEDS=10`, seeds 42–51). Fidelity and privacy are single estimates per generator.

## Figures (per Utility × Privacy analysis)

1. Fidelity vs Utility
2. Privacy vs Utility
3. Fidelity vs Privacy

Each figure reports Pearson *r*, Spearman *ρ*, R², and *p*, with an OLS fit and 95% CI band.

## Layout

```
Correlation_Tradeoff_Analysis/
  Utility_Accuracy/Analysis_A_MIA/{Figures,Statistics,Summaries}/
  Utility_Accuracy/Analysis_B_NNDR/...
  Utility_F1/Analysis_A_MIA/...
  Utility_F1/Analysis_B_NNDR/...
  Figures/          # flat copies of all figures
  Summaries/
  Statistics/
  Processed_Data/
  FINDINGS.md
```

A mirror is written to `Results/Two_Datasets_Assessment/Correlation_Tradeoff_Analysis/`.

## Run

```bash
python run_correlation_tradeoff_analysis.py
```
