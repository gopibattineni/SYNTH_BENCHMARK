# Utility Drop Analysis

**Runtime:** 195.48s

## Purpose
Evaluate synthetic-data utility under leakage conditions using existing Excel results (no ML re-runs). Identifies best generator, best classifier, and robustness to utility drop.

## Primary metric
Accuracy — reported as **Mean ± Standard Deviation** across random seeds.

## Supplementary metrics
F1-score, ROC-AUC, Precision, Recall (tables under `Tables/`).

## Data note
Master/Excel utility exports currently contain **Leakage = 0% only**.
Configured levels `[0,5,10,...,50]` were not exported as multi-level tables.

- **Figure 1:** Mean Accuracy by generator (Cancer vs Mushroom line styles).
- **Figures 5–6:** Utility drop/loss = TRTR − TSTR (gap vs real-data baseline), not leakage %.
- **Figures 11–12:** Seed distributions approximated from exported Mean±SD (raw seeds not stored).

## Outputs
- `Figures/` — Figures 1–12 (+ CD diagrams 10a/10b)
- `Tables/` — rankings, Mean±SD matrices, robustness
- `Statistics/` — Friedman / Nemenyi ranks
- `Processed_Data/` — intermediate CSVs
- `FINDINGS.md` — short answers to the four research questions

## How to run
```bash
python run_utility_drop_analysis.py
```