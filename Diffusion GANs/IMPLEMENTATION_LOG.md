# Diffusion GANs — Implementation Log

Generated: 2026-07-03 (autonomous session)

## Completed

### Folder structure
- Created `Diffusion GANs/` mirroring `Other GANS/` (CSVs, figures paths, Excel export names updated).
- Added `diffusion_generators.py` — shared generator module used by all notebooks.
- Cloned official implementations under `_vendor/`:
  - `tab-ddpm` (TabDDPM)
  - `CoDi` (CoDi)
  - `goggle` (GOGGLE)
  - ForestDiffusion via `pip install ForestDiffusion`

### Notebooks (11/11)
| # | Notebook |
|---|----------|
| 1 | `1. Cancer_diffusion.ipynb` |
| 2 | `2. Alzhimers_diffusion.ipynb` |
| 3 | `3. Adult census_diffusion.ipynb` |
| 4 | `4_Forest_Cover_diffusion.ipynb` |
| 5 | `5. Bank marketing_diffusion.ipynb` |
| 6 | `6. Winequality_diffusion.ipynb` |
| 7 | `7_CDC_diabetes_diffusion.ipynb` |
| 8 | `8. Metro Interstate_diffusion.ipynb` |
| 9 | `9. Mushroom_diffusion.ipynb` |
| 10 | `10. Online shopping_diffusion.ipynb` |
| 11 | `11_MAGIC_Gamma_Telescope_diffusion.ipynb` |

Each notebook:
- Keeps identical preprocessing, evaluation, fidelity/utility/privacy pipelines, plots, and exports.
- Replaces CTAB-GAN+ / WGAN-GP training with **TabDDPM**, **CoDi**, **GOGGLE**, and **ForestDiffusion**.
- Uses `model_order = ["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]` and `synthetic_outputs` dict (1000 samples, seed 42).

### Generator mapping
| Original | Replacement | Source |
|----------|-------------|--------|
| CTABGAN | TabDDPM | yandex-research/tab-ddpm |
| WGAN_GP | CoDi | ChaejeongLee/CoDi |
| — | GOGGLE | tennisonliu/goggle (graph VAE; included per benchmark request) |
| — | ForestDiffusion | PyPI `ForestDiffusion` |

### Build tooling
- `build_diffusion_notebooks.py` — regenerates notebooks from `Other GANS/` templates.

## Dependencies (installed in notebook setup cell)

```
pip install ForestDiffusion xgboost category-encoders libzero rtdl imbalanced-learn absl-py tensorboardX
```

Also requires PyTorch and the cloned repos in `_vendor/`.

## Known issues / notes

1. **GOGGLE is not a diffusion model** — it is a VAE/GNN method (ICLR 2023). Included because it was explicitly requested alongside TabDDPM, CoDi, and ForestDiffusion for benchmark comparison.

2. **Training time** — CoDi defaults were reduced to 200 epochs (vs 20k in paper) for notebook practicality. TabDDPM uses 1000 training steps. Documented in `diffusion_generators.py`.

3. **Synthcity** — Not used (torch 2.2 / opacus RMSNorm incompatibility on this machine). GOGGLE is invoked directly from the official repo without synthcity metrics.

4. **Full notebook execution not completed** — End-to-end runs were not finished overnight (ForestDiffusion smoke test on Cancer was slow on CPU). Re-run notebooks in order; expect long runtimes especially for TabDDPM and CoDi on CPU.

5. **Metro / Online shopping regression** — ForestDiffusion path handles continuous targets via `label_y`; verify column dtypes after first run.

6. **Online shopping notebook** — Original used an absolute path to an external CSV; that path is unchanged in the diffusion notebook.

## How to regenerate

```bash
python build_diffusion_notebooks.py
```

## How to run one dataset

Open e.g. `Diffusion GANs/1. Cancer_diffusion.ipynb`, run all cells. Ensure `_vendor/` repos exist at repository root.
