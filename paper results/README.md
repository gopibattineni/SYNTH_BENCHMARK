# Paper Results

Per-dataset publication workbooks extracted from generator notebooks and utility Excel files.

## Quick regenerate

```bash
python scripts/extract_paper_results.py
```

## Each dataset folder contains

| File | Content |
|------|---------|
| `TRTR_TSTR_results_<dataset>.xlsx` | Utility — TRTR vs TSTR, all 8 generators |
| `fidelity_privacy_metrics.xlsx` | Fidelity & privacy metrics (17+ sheets) |
| `Figures/tSNE/` | t-SNE plots from notebooks |

## Workbook sheets

**Utility:** `TRTR_Results`, `All_Comparisons`, `Summary`, plus one sheet per generator

**Fidelity & privacy:** `Quality_Scores`, `KS_ColumnShapes`, `JS_Divergence`, `Wasserstein_Summary`, `Gower_Distance`, `MMD`, `MIA`, `Mahalanobis_2D`, `Hungarian_Matching`, and more

See the [main README](../README.md) for full benchmark documentation.
