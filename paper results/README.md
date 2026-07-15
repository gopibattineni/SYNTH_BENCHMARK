# Paper Results

Per-dataset outputs for the publication, matching the **Cancer sample layout**:

- `TRTR_TSTR_results_<dataset>.xlsx` — utility (TRTR vs TSTR, all 8 generators)
- `fidelity_privacy_metrics.xlsx` — fidelity and privacy metrics (8 generators)
- `Figures/tSNE/` — t-SNE plots extracted from generator notebooks

## Sources

| Folder | Generators |
|--------|------------|
| `Generators/SDV models/` | CTGAN, CopulaGAN, TVAE, GaussianCopula |
| `Generators/Other GANS/` | CTABGAN, WGAN_GP |
| `Generators/Diffusion GANs/` | TabDDPM, ForestDiffusion |
| `Generators/Experiment with utility data leak/utility results/` | TRTR/TSTR utility Excel |

## Utility workbook (`TRTR_TSTR_results_*.xlsx`)

Same structure as the Cancer sample:

| Sheet | Content |
|-------|---------|
| `TRTR_Results` | Baseline classifier scores on real data |
| `All_Comparisons` | TRTR vs TSTR for all generators × classifiers |
| `Summary` | Generator-level utility gaps |
| `CTGAN` … `ForestDiffusion` | Per-generator TRTR/TSTR tables |

## Fidelity & privacy workbook (`fidelity_privacy_metrics.xlsx`)

| Sheet | Content |
|-------|---------|
| `Fidelity_Summary` / `Privacy_Summary` | One row per generator |
| `Quality_Scores` | SDV overall quality scores |
| `KS_ColumnShapes` | KS complement per feature |
| `JS_Divergence` | Jensen–Shannon divergence per feature |
| `Wasserstein_Summary` | Mean/median/max Wasserstein |
| `Gower_Distance` | Gower intra/cross similarity |
| `MMD` / `MMD_Multivariate` | Maximum Mean Discrepancy |
| `Cosine_Similarity` | Cosine similarity summary |
| `Outlier_Count_Diff` | Avg absolute difference in outlier counts |
| `PCA_Mean_Errors` / `PCA_Projections` | PCA fidelity metrics |
| `Nearest_Neighbors` | NNDR / nearest-neighbor distances |
| `MIA` | Membership inference attack |
| `Mahalanobis_2D` | 2D Mahalanobis distances |
| `Hungarian_Matching` | Cosine similarity from Hungarian matchings |

## Regenerate

```bash
python scripts/extract_paper_results.py
```

Requires executed notebooks (HTML table outputs in `.ipynb`) and merged files under `utility results/`.
