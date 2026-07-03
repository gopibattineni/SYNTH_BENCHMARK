# SYNTH — Synthetic Data Quality & Utility Benchmark

Benchmark for comparing **tabular synthetic data generators** on **fidelity**, **downstream utility**, and **privacy**, with a leak-safe TRTR/TSTR protocol across **15 datasets** (9 classification + 6 regression) and **8 generators** spanning GAN, diffusion/graph, and statistical (SDV) families.

Associated with LERO / BDS research on synthetic data auditing.

---

## What this repository measures

| Layer                | Question                                                                   | Where it is evaluated                                                                  |
| --------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Utility**          | Can models trained on synthetic data perform on real held-out data?        | `Single run_Data_leak_Synth_Quality/` and `.../diffusion_dataleak/` → `TRTR_TSTR_results*.xlsx` |
| **Fidelity**         | How closely does synthetic data match real distributions and dependencies? | `SDV models/` notebooks (KS, JS, Wasserstein, Gower, MMD, t-SNE, …)                     |
| **Alternative GANs**  | How do non-SDV GAN generators compare on the same audit pipeline?          | `Other GANS/` (CTAB-GAN+, WGAN-GP)                                                       |
| **Diffusion / graph** | How do diffusion and graph-based generators compare on the same pipeline?  | `Diffusion GANs/` (TabDDPM, CoDi, GOGGLE, ForestDiffusion)                               |
| **Privacy**           | Can an attacker infer training membership from synthetic releases?         | MIA cells in `SDV models/`, `Other GANS/`, and `Diffusion GANs/` notebooks               |

**TRTR** (Train Real, Test Real) is the real-data baseline.
**TSTR** (Train Synthetic, Test Real) measures utility when learning from synthetic data only.
The **utility gap** (TRTR − TSTR) is the main comparative signal: smaller drop = more useful synthetic data.

A full visual overview is available in `figures/SYNTH_workflow.{svg,pdf,png}` (see [Pipeline diagram](#pipeline-diagram)).

---

## Benchmark structure (5 folder families)

The benchmark is organised into five parallel folder families that all share the same **15 datasets** and the same **leak-safe 80/20 split**, but differ in which generators they run and how deep the audit is:

| Folder                                                    | Generators (per dataset)                                          | Audit depth                                          |
| ---------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------- |
| `Single run_Data_leak_Synth_Quality/`                     | **CTAB-GAN+**, **WGAN-GP** + `CTGAN`, `CopulaGAN`, `TVAE`, `GaussianCopula` (6) | TRTR/TSTR utility only — the primary benchmark        |
| `Single run_Data_leak_Synth_Quality/diffusion_dataleak/`  | **TabDDPM**, **CoDi** + `CTGAN`, `CopulaGAN`, `TVAE`, `GaussianCopula` (6)      | TRTR/TSTR utility only — diffusion-generator variant  |
| `SDV models/`                                             | `CTGAN`, `CopulaGAN`, `TVAE`, `GaussianCopula` (4)                   | Full audit: fidelity + utility + privacy (MIA)        |
| `Other GANS/`                                             | **CTAB-GAN+**, **WGAN-GP** (2)                                       | Full audit: fidelity + utility + privacy (MIA)        |
| `Diffusion GANs/`                                         | **TabDDPM**, **CoDi**, **GOGGLE**, **ForestDiffusion** (4)           | Full audit: fidelity + utility + privacy (MIA)        |

Across all five families, the **union of generators used is 8**: `CTAB-GAN+`, `WGAN-GP`, `TabDDPM`, `CoDi`, `CTGAN`, `CopulaGAN`, `TVAE`, `GaussianCopula` (`GOGGLE` and `ForestDiffusion` additionally appear only in the deep-audit `Diffusion GANs/` folder).

`GOGGLE`, `TabDDPM`, and `CoDi` are cloned from their official repositories into `_vendor/` (not tracked in git — see [Setup](#setup--dependencies)); `ForestDiffusion` installs via `pip`.

---

## Experimental protocol

Every dataset notebook follows the same core design:

1. **Load & preprocess** — drop date / time / session-ID / record-ID columns, clean missing values.
2. **Subsample** to `N = 1000` rows (`seed = 42`) for tractable generator training.
3. **Leak-safe split:** stratified 80% train / 20% test (`TEST_SIZE = 0.2`).
4. **Generators fit on `train_real` only** — synthetic rows are never built from the held-out test set.
5. Each generator produces **1,000 synthetic rows**.
6. **Fidelity:** `SDV.evaluate_quality()` compares synthetic data against `train_real` (identity, distribution, and feature similarity).
7. **Utility (TRTR vs TSTR):** 10 downstream models × 10 random seeds (`42…51`); metrics reported as **mean ± SD**.
   - **TRTR** — train and test on real data (baseline).
   - **TSTR** — train on synthetic data, test on the **same held-out real test set** for every generator.
8. **Performance drop** `Δ = TRTR − TSTR` is computed per metric, per generator, per downstream model, and exported to Excel.

### Datasets (15)

| #  | Folder                            | Task           | Target (typical)                    |
| -- | ---------------------------------- | -------------- | ------------------------------------ |
| 1  | Cancer                             | Classification | Diagnosis (M/B)                     |
| 2  | Alzheimer's                        | Classification | Group (Demented / Nondemented)      |
| 3  | Adult                               | Classification | Income                               |
| 4  | Forest Cover                       | Classification | Cover type                          |
| 5  | Bank Marketing                     | Classification | Subscription                        |
| 6  | Wine Quality                       | Classification | Quality (ordinal score)             |
| 7  | CDC Diabetes Health Indicators     | Classification | Diabetes indicator                  |
| 8  | Metro Interstate Traffic Volume    | Regression     | Traffic volume                      |
| 9  | Secondary Mushroom                 | Classification | Edible / poisonous                  |
| 10 | Online Shopping                    | Regression     | Revenue / price                     |
| 11 | MAGIC Gamma Telescope              | Classification | Class (gamma / hadron)              |
| 12 | Air Quality                        | Regression     | `CO(GT)`                             |
| 13 | Concrete Compressive Strength      | Regression     | Compressive strength                |
| 14 | Energy Efficiency                  | Regression     | Heating load (`Y1`)                 |
| 15 | Real Estate Valuation              | Regression     | House price per unit area           |

**9 classification** + **6 regression**. Canonical dataset metadata (UCI IDs, notebook paths, sample sizes) lives in `Single run_Data_leak_Synth_Quality/python_scripts/hive/datasets.json`. Raw / cached CSVs live under `Datasets/` and per-dataset notebook folders.

---

## Results files (`TRTR_TSTR_results*.xlsx`)

Each dataset notebook exports an Excel workbook with sheets such as:

| Sheet                                              | Contents                                                |
| --------------------------------------------------- | --------------------------------------------------------- |
| **TRTR**                                           | Downstream performance training on real data (baseline) |
| **Per-generator sheets** (e.g. `CTGAN`, `TVAE`, …) | TSTR results for that synthetic source                  |
| **Combined comparison**                            | TRTR vs TSTR side-by-side per downstream model          |
| **Summary**                                        | Aggregated drops per generator                          |

### Classification metrics

- **TRTR / TSTR:** Accuracy, F1, Precision, Recall (mean ± SD).
- **Utility drop:** `Accuracy_Drop`, `F1_Drop`, etc. = TRTR − TSTR (positive = synthetic training hurt performance).

### Regression metrics (Metro, Online Shopping, Air Quality, Concrete, Energy Efficiency, Real Estate)

- **TRTR / TSTR:** R², RMSE, MAE (mean ± SD).
- **Utility drop:** `R2_Drop`, `RMSE_Increase`, `MAE_Increase`.

### How to read the results

- **No single generator wins everywhere.** Best synthetic source is **dataset-dependent** — always check the exported per-dataset summary.
- **Large TRTR − TSTR gap** → synthetic data preserves little task-relevant signal for that downstream model.
- **Low drop + stable SD across seeds** → more reliable generator for that setting.
- **High SD on TSTR** → seed-sensitive synthesis or unstable downstream fit; report mean ± SD, not a single run.
- **SDV quality score** (in fidelity notebooks) measures distributional similarity; it does **not** guarantee high TSTR utility — always cross-check with TRTR/TSTR.

---

## Pipeline diagram

A publication-ready workflow diagram (5 stages: Data → Synthesis → Quality → Evaluation → Analysis) is generated from code and kept in sync with the pipeline:

- `figures/SYNTH_workflow.svg` / `.pdf` — vector, for LaTeX / journal submission
- `figures/SYNTH_workflow.png` — 300 DPI raster, for slides / Word
- `Single run_Data_leak_Synth_Quality/forge_paper_workflow.png` — mirrored copy for notebook-adjacent reference

Regenerate after any pipeline change:

```bash
python figures/generate_forge_paper_workflow.py
```

---

## Interactive dashboard

Published view (GitHub Pages):

**[https://gopibattineni.github.io/SYNTH/](https://gopibattineni.github.io/SYNTH/)**

### Dashboard views

**Benchmark overview (all datasets)**

- **Fig. A — Classification utility loss heatmap** — generators × downstream models, averaged over classification datasets; cell values = mean utility drop.
- **Fig. B — Regression utility loss heatmap** — same for regression datasets (R²-oriented loss).
- **Fig. C — Generator win-rate** — how often each generator ranks best on utility per dataset.
- Toggle **Paper theme** and **Export PNG** for figures.

**Per dataset (click a dataset in the sidebar)**

- Generator ranking bar chart (lowest utility loss highlighted).
- **TRTR vs TSTR** grouped bars per downstream model.
- Radar chart across metrics.
- Model-level drop chart.
- Tables: TRTR baseline, summary by generator, full TRTR/TSTR comparison, per-generator sheets.

Data are loaded from exported JSON built from the Excel files under
`Single run_Data_leak_Synth_Quality/*/TRTR_TSTR_results*.xlsx` (and `diffusion_dataleak/*/`).

### Enable / update GitHub Pages

1. Open [https://github.com/gopibattineni/SYNTH/settings/pages](https://github.com/gopibattineni/SYNTH/settings/pages)
2. **Source:** Deploy from branch → `gh-pages` → `/ (root)`
3. Save and wait 2–5 minutes.

To rebuild after new Excel results (from a machine with the full repo + `webapp/`):

```bash
python webapp/scripts/export_dashboard_data.py
python webapp/scripts/build_github_pages.py
# commit docs/ or push to gh-pages
```

### Run locally (FastAPI + live experiments)

If the `webapp/` package is present in your clone:

```bash
cd webapp
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Experiment orchestration: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Results dashboard: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

On Windows: `webapp/start.bat`.

---

## Repository layout

| Path                                                          | Purpose                                                                              |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `Single run_Data_leak_Synth_Quality/`                        | **Primary benchmark** — CTAB-GAN+ / WGAN-GP + 4 SDV generators, TRTR/TSTR, Excel exports |
| `Single run_Data_leak_Synth_Quality/diffusion_dataleak/`     | Same protocol, **TabDDPM / CoDi** + 4 SDV generators instead of GAN pair               |
| `Single run_Data_leak_Synth_Quality/python_scripts/`         | Dataset metadata (`hive/datasets.json`) and cluster/batch run scripts (Slurm / Hive)    |
| `SDV models/`                                                | Deep **fidelity + utility + privacy** audits for 4 SDV synthesizers per dataset         |
| `Other GANS/`                                                | Same audit pipeline for **CTAB-GAN+** and **WGAN-GP**                                  |
| `Diffusion GANs/`                                             | Same audit pipeline for **TabDDPM**, **CoDi**, **GOGGLE**, **ForestDiffusion**          |
| `figures/`                                                    | Diagram-generator scripts + exported pipeline/workflow figures                          |
| `Datasets/`                                                  | Local cached dataset copies used by notebooks                                          |
| `Materials/`                                                  | Paper notes and supplementary documents (Cosine similarity, Mahalanobis notes, preprint) |
| `_vendor/`                                                    | Cloned third-party generator implementations (TabDDPM, CoDi, GOGGLE) — not tracked in git |
| `add_regression_datasets.py`                                  | Adds/patches regression datasets (#12–#15) across all five folder families              |
| `build_diffusion_notebooks.py`                                | Regenerates `Diffusion GANs/` notebooks from `Other GANS/` templates                    |
| `build_diffusion_dataleak_notebooks.py`                       | Regenerates `diffusion_dataleak/` notebooks from `Single run/` templates                |
| `webapp/`                                                     | FastAPI app, dashboard UI, Excel→JSON export (when included in clone)                  |
| `docs/`                                                       | Static GitHub Pages build output                                                        |

---

## Extended audits (`SDV models/`, `Other GANS/`, `Diffusion GANs/`)

These notebooks mirror a full synthetic-data audit per dataset:

1. **Univariate:** KS / column shapes, Jensen–Shannon, Wasserstein, Gower, t-SNE, MMD, cosine similarity, nearest neighbours.
2. **Bivariate:** correlation and class-conditional distribution differences.
3. **Multivariate:** global MMD, PCA overlays, classifier two-sample (C2ST) accuracy.
4. **Utility:** TRTR/TSTR with 10 classifiers.
5. **Record matching:** cosine / Mahalanobis with Hungarian (and greedy) assignment → Excel exports (`Excel sheets/`, `Hungarian_*.xlsx`).
6. **Privacy:** membership inference attack (MIA) — lower attack AUC is generally better.

For large datasets (e.g. **MAGIC**, ~19k rows; **Forest Cover**), pairwise metrics use **subsampled rows** (`METRIC_SAMPLE_SIZE = 2000`) to avoid memory errors.

`Diffusion GANs/` notebooks keep the identical preprocessing, evaluation, and export pipeline as `Other GANS/`, only swapping the generator-training cell (`model_order = ["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]`).

---

## Citation & context

When reporting results, state:

- Dataset name and task type (classification vs regression).
- Generator and downstream model names.
- TRTR baseline and TSTR score as **mean ± SD** over 10 seeds.
- Utility drop (or increase for error metrics).
- That generators were trained on **training real data only** and evaluated on a **held-out real test set**.

For methodology figures, see `figures/SYNTH_workflow.{svg,pdf,png}` and `Materials/` for supplementary notes.

---

## Quick start (reproduce one dataset)

```bash
# Example: open and run
Single run_Data_leak_Synth_Quality/1. Cancer/cancer.ipynb
```

Install typical dependencies:

```bash
pip install pandas numpy scikit-learn sdv torch openpyxl ucimlrepo gower xlsxwriter matplotlib
```

Diffusion/graph generators additionally need:

```bash
pip install ForestDiffusion xgboost category-encoders libzero rtdl imbalanced-learn absl-py tensorboardX
```

### Setup — dependencies

- For **CTAB-GAN+** notebooks, clone [CTAB-GAN-Plus](https://github.com/Team-TUD/CTAB-GAN-Plus) into `Other GANS/CTAB-GAN-Plus/` (or adjust `sys.path` in the notebook).
- For **TabDDPM**, **CoDi**, and **GOGGLE**, clone their official repositories into `_vendor/` at the repository root:

  ```bash
  git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm
  git clone https://github.com/ChaejeongLee/CoDi _vendor/CoDi
  git clone https://github.com/tennisonliu/goggle _vendor/goggle
  ```

  `_vendor/` is intentionally excluded from version control (large third-party code with its own git history).

### Regenerating notebooks / figures

```bash
python add_regression_datasets.py            # (re)patch regression datasets #12-#15
python build_diffusion_notebooks.py           # rebuild Diffusion GANs/ from Other GANS/
python build_diffusion_dataleak_notebooks.py  # rebuild diffusion_dataleak/ from Single run/
python figures/generate_forge_paper_workflow.py  # rebuild the pipeline diagram
```
