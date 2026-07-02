# SYNTH — Synthetic Data Quality & Utility Benchmark

Benchmark for comparing **tabular synthetic data generators** on **fidelity**, **downstream utility**, and **privacy**, with a leak-safe TRTR/TSTR protocol across **10 UCI-style datasets** and **6 generators**.

Associated with LERO / BDS research on synthetic data auditing.

---

## What this repository measures

| Layer | Question | Where it is evaluated |
|-------|----------|------------------------|
| **Utility** | Can models trained on synthetic data perform on real held-out data? | `Single run_Data_leak_Synth_Quality/` → `TRTR_TSTR_results.xlsx` |
| **Fidelity** | How closely does synthetic data match real distributions and dependencies? | `SDV models/` notebooks (KS, JS, Wasserstein, Gower, MMD, t-SNE, …) |
| **Alternative GANs** | How do non-SDV generators compare on the same audit pipeline? | `Other GANS/` (CTAB-GAN+, WGAN-GP) |
| **Privacy** | Can an attacker infer training membership from synthetic releases? | MIA cells in `SDV models/` and `Other GANS/` notebooks |

**TRTR** (Train Real, Test Real) is the real-data baseline.  
**TSTR** (Train Synthetic, Test Real) measures utility when learning from synthetic data only.  
The **utility gap** (TRTR − TSTR) is the main comparative signal: smaller drop = more useful synthetic data.

---

## Experimental protocol (main benchmark)

Defined in `Single run_Data_leak_Synth_Quality/*/`. Each dataset notebook follows the same core design:

1. **Leak-safe split:** 80% train / 20% test (`TEST_SIZE = 0.2`), stratified where applicable.
2. **Generators fit on training real data only** — synthetic rows are never built from the test set.
3. **Six generators:**
   - SDV: `CTGAN`, `CopulaGAN`, `TVAE`, `GaussianCopula`
   - GAN variants: `WGAN_GP`, `CTABGAN`
4. **Ten downstream models** (e.g. Logistic Regression, SVM, Random Forest, Gradient Boosting, MLP, …).
5. **Ten random seeds** `[42 … 51]` — metrics reported as **mean ± SD** across seeds.
6. **TSTR evaluation** always uses the **same held-out real test set** for every generator and downstream model.

### Datasets

| # | Folder | Task | Target (typical) |
|---|--------|------|------------------|
| 1 | Cancer | Classification | Diagnosis (M/B) |
| 2 | MAGIC Gamma Telescope | Classification | class (g/h) |
| 3 | Adult | Classification | income |
| 4 | Forest cover | Classification | cover type |
| 5 | Bank Marketing | Classification | subscription |
| 6 | Wine | Classification | quality |
| 7 | Mushroom | Classification | class |
| 8 | CDC diabetes | Classification | diabetes |
| 9 | Metro interstate | Regression | traffic volume |
| 10 | Online shopping | Regression | price / revenue |

Raw files live under `Datasets/` where referenced by notebooks.

---

## Results files (`TRTR_TSTR_results.xlsx`)

Each dataset notebook exports an Excel workbook with sheets such as:

| Sheet | Contents |
|-------|----------|
| **TRTR** | Downstream performance training on real data (baseline) |
| **Per-generator sheets** (e.g. `CTGAN`, `TVAE`, …) | TSTR results for that synthetic source |
| **Combined comparison** | TRTR vs TSTR side-by-side per downstream model |
| **Summary** | Aggregated drops per generator |

### Classification metrics

- **TRTR / TSTR:** Accuracy, F1, Precision, Recall (mean ± SD).
- **Utility drop:** `Accuracy_Drop`, `F1_Drop`, etc. = TRTR − TSTR (positive = synthetic training hurt performance).

### Regression metrics (Metro, Online Shopping)

- **TRTR / TSTR:** R², MSE, RMSE, MAE (mean ± SD).
- **Utility drop:** `R2_Drop`, `MSE_Increase`, `RMSE_Increase`, `MAE_Increase`.

### How to read the results

- **No single generator wins everywhere.** Best synthetic source is **dataset-dependent** (e.g. WGAN-GP on Cancer/MAGIC, GaussianCopula on Forest, CTGAN on Adult — see your exported summaries).
- **Large TRTR − TSTR gap** → synthetic data preserves little task-relevant signal for that downstream model.
- **Low drop + stable SD across seeds** → more reliable generator for that setting.
- **High SD on TSTR** → seed-sensitive synthesis or unstable downstream fit; report mean ± SD, not a single run.
- **SDV quality score** (in fidelity notebooks) measures distributional similarity; it does **not** guarantee high TSTR utility — always cross-check with TRTR/TSTR.

---

## Interactive dashboard

Published view (GitHub Pages):

**https://gopibattineni.github.io/SYNTH/**

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
`Single run_Data_leak_Synth_Quality/*/TRTR_TSTR_results*.xlsx`.

### Enable / update GitHub Pages

1. Open https://github.com/gopibattineni/SYNTH/settings/pages
2. **Source:** Deploy from branch → **`gh-pages`** → **`/ (root)`**
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

- Experiment orchestration: http://127.0.0.1:8000/
- Results dashboard: http://127.0.0.1:8000/dashboard

On Windows: `webapp/start.bat`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `Single run_Data_leak_Synth_Quality/` | **Primary benchmark** — 6 generators, TRTR/TSTR, Excel exports |
| `SDV models/` | Deep **fidelity + utility + privacy** audits for 4 SDV synthesizers per dataset |
| `Other GANS/` | Same audit pipeline for **CTAB-GAN+** and **WGAN-GP** |
| `Datasets/` | Local dataset copies / paths used by notebooks |
| `Materials/` | Paper notes, workflow figures, supplementary documents |
| `webapp/` | FastAPI app, dashboard UI, Excel→JSON export (when included in clone) |
| `docs/` | Static GitHub Pages build output |

---

## Extended audits (`SDV models/` & `Other GANS/`)

These notebooks mirror a full synthetic-data audit per dataset:

1. **Univariate:** KS / column shapes, Jensen–Shannon, Wasserstein, Gower, t-SNE, MMD, cosine similarity, nearest neighbours.
2. **Bivariate:** correlation and class-conditional distribution differences.
3. **Multivariate:** global MMD, PCA overlays, classifier two-sample (C2ST) accuracy.
4. **Utility:** TRTR/TSTR with 10 classifiers.
5. **Record matching:** cosine / Mahalanobis with Hungarian (and greedy) assignment → Excel exports in `Excel sheets/`.
6. **Privacy:** membership inference attack (MIA) — lower attack AUC is generally better.

For large datasets (e.g. **MAGIC**, ~19k rows), pairwise metrics use **subsampled rows** (`METRIC_SAMPLE_SIZE = 2000`) to avoid memory errors.

---

## Citation & context

When reporting results, state:

- Dataset name and task type (classification vs regression).
- Generator and downstream model names.
- TRTR baseline and TSTR score as **mean ± SD** over 10 seeds.
- Utility drop (or increase for error metrics).
- That generators were trained on **training real data only** and evaluated on a **held-out real test set**.

For methodology figures and notes, see `Materials/forge_paper_workflow.png` and related documents in `Materials/`.

---

## Quick start (reproduce one dataset)

```bash
# Example: open and run
Single run_Data_leak_Synth_Quality/1. Cancer/cancer.ipynb
```

Install typical dependencies: `pandas`, `numpy`, `scikit-learn`, `sdv`, `torch`, `openpyxl`, `ucimlrepo`, `gower`, `xlsxwriter`.

For **CTAB-GAN+** notebooks, clone [CTAB-GAN-Plus](https://github.com/Team-TUD/CTAB-GAN-Plus) into `Other GANS/CTAB-GAN-Plus/` (or adjust `sys.path` in the notebook).
