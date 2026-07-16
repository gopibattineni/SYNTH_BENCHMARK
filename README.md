# SYNTH Benchmark

**A reproducible benchmark for tabular synthetic data** — comparing **8 generators** across **15 datasets** on **utility**, **fidelity**, and **privacy**, with publication-ready analysis and figures.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Research](https://img.shields.io/badge/license-research-lightgrey.svg)](Materials/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://gopibattineni.github.io/SYNTH_BENCHMARK/)
[![Datasets](https://img.shields.io/badge/datasets-15-green.svg)](#datasets)
[![Generators](https://img.shields.io/badge/generators-8-orange.svg)](#generators)

> **TL;DR** — Fit generators on real training data only → evaluate synthetic data with TRTR/TSTR utility, distributional fidelity, and privacy attacks → rank generators with statistical tests and Pareto trade-off analysis.

---

## Table of contents

- [Why this benchmark?](#why-this-benchmark)
- [Benchmark at a glance](#benchmark-at-a-glance)
- [Quick start](#quick-start)
- [Analysis pipelines](#analysis-pipelines)
- [Results & figures](#results--figures)
- [Repository layout](#repository-layout)
- [Experimental protocol](#experimental-protocol)
- [Datasets](#datasets)
- [Generators](#generators)
- [Dashboard](#dashboard)
- [Citation](#citation)
- [Changelog](#changelog)
- [Contributing](#contributing)

---

## Why this benchmark?

Synthetic tabular data is used for privacy-preserving sharing and ML augmentation — but **no single metric tells the full story**. SYNTH evaluates every generator on three complementary axes:

| Pillar | Question | Key metrics |
|--------|----------|-------------|
| **Utility** | Does synthetic data train useful models? | TRTR vs TSTR — Accuracy, F1, R², RMSE (10 seeds × 10 downstream models) |
| **Fidelity** | Does synthetic data look like the real distribution? | KS, JS, Wasserstein, MMD, SDV Quality Score, t-SNE |
| **Privacy** | Can an attacker infer training membership? | MIA AUC, Mahalanobis distance, NNDR, Hungarian matching |

```mermaid
flowchart LR
  A[Real data] --> B[8 Generators]
  B --> C[Synthetic data]
  C --> D[Utility TRTR/TSTR]
  C --> E[Fidelity metrics]
  C --> F[Privacy metrics]
  D --> G[Rankings & trade-offs]
  E --> G
  F --> G
```

---

## Benchmark at a glance

| | |
|---|---|
| **Datasets** | 15 UCI-style tabular datasets (9 classification + 6 regression) |
| **Generators** | CTGAN, CopulaGAN, TVAE, GaussianCopula, CTABGAN, WGAN-GP, TabDDPM, ForestDiffusion |
| **Seeds** | 10 random seeds (`42`–`51`) — reported as mean ± SD |
| **Leakage protocol** | Generators fit on **train only**; TSTR tested on held-out **real test** set |
| **Outputs** | Excel workbooks, paper-ready figures (300 DPI), LaTeX tables, statistical tests |

**Current overall ranking** (composite score: 40% utility / 30% fidelity / 30% privacy):

| Rank | Generator | Composite |
|------|-----------|-----------|
| 1 | ForestDiffusion | 0.66 |
| 2 | TVAE | 0.61 |
| 3 | GaussianCopula | 0.59 |

*See [`Results/Processed_Data/generator_ranking.csv`](Results/Processed_Data/generator_ranking.csv) for full rankings.*

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/gopibattineni/SYNTH_BENCHMARK.git
cd SYNTH_BENCHMARK
pip install -r requirements-analysis.txt
```

### 2. Run the full analysis pipeline (~3 min)

Auto-discovers all Excel files, merges metrics, runs Friedman/Nemenyi tests, and generates **60+ publication figures**:

```bash
python run_analysis.py
```

Outputs land in [`Results/`](Results/).

### 3. Run the Cancer & Mushroom case study (~3 min)

Focused two-dataset assessment with comparative figures and tables:

```bash
python run_two_datasets_assessment.py
```

Outputs land in [`Results/Two_Datasets_Assessment/`](Results/Two_Datasets_Assessment/).

### 4. Correlation trade-off analysis (Cancer & Mushroom)

Publication figures relating **Utility × Fidelity × Privacy** (primary: one point per generator; supplementary: seed-level):

```bash
python run_correlation_tradeoff_analysis.py
```

Outputs: [`Results/Correlation_Tradeoff_Analysis/`](Results/Correlation_Tradeoff_Analysis/)  
(also mirrored under `Results/Two_Datasets_Assessment/Correlation_Tradeoff_Analysis/`).

### 5. Utility drop & representative trade-off (optional)

```bash
python run_utility_drop_analysis.py          # → Results/Utility_Drop_Analysis/
python run_representative_tradeoff.py        # → Results/Representative_Metrics_Tradeoff/
```

### 6. Extract paper-ready workbooks (optional)

```bash
python scripts/extract_paper_results.py
```

Creates per-dataset `TRTR_TSTR_results_*.xlsx` and `fidelity_privacy_metrics.xlsx` under [`paper results/`](paper%20results/).

---

## Analysis pipelines

Automated pipelines turn raw experiment Excel files into journal-ready outputs. **No filenames are hardcoded** — new results are picked up automatically on the next run.

### Full benchmark (`run_analysis.py`)

| Step | What it does |
|------|----------------|
| Merge | Discovers all `.xlsx` files under `Generators/` |
| Score | Normalized Utility / Fidelity / Privacy scores (0–1) per dataset × generator |
| Statistics | Friedman, Nemenyi, Wilcoxon, Cohen's d, Cliff's delta |
| Benchmarking | Pareto frontiers, composite rankings, CD diagrams, seed stability |
| Figures | 14 core paper figures + supplementary plots (PNG/PDF/SVG/EPS @ 300 DPI) |

```bash
python run_analysis.py --utility-weight 0.4 --privacy-weight 0.3 --fidelity-weight 0.3
```

### Two-dataset case study (`run_two_datasets_assessment.py`)

Dedicated module for **Wisconsin Breast Cancer** and **Secondary Mushroom** — side-by-side comparison, rankings, and trade-off figures.

```bash
python run_two_datasets_assessment.py
```

### Correlation trade-off (`run_correlation_tradeoff_analysis.py`)

Investigates relationships between Utility, Fidelity, and Privacy on Cancer & Mushroom:

| Figure set | Points | Metrics |
|-----------|--------|---------|
| **Primary** | 8 generators (mean over datasets × classifiers × seeds) | Accuracy/F1 × Quality Score × MIA or NNDR |
| **Supplementary** | Generator × Seed × Dataset | Same axes; Cancer = ○, Mushroom = □ |

Each scatter reports Pearson *r*, Spearman *ρ*, R², *p*, OLS fit, and 95% CI.

```bash
python run_correlation_tradeoff_analysis.py
```

### Utility drop (`run_utility_drop_analysis.py`)

Generator/classifier utility under data leakage (Cancer & Mushroom focus), heatmaps, CD diagrams, and robustness tables → `Results/Utility_Drop_Analysis/`.

### Representative metrics trade-off (`run_representative_tradeoff.py`)

F1 / NNDR / Quality Score dual trade-offs (classifier-independent and best-classifier views) → `Results/Representative_Metrics_Tradeoff/`.

### Key modules

| Path | Purpose |
|------|---------|
| [`analysis/`](analysis/) | Core pipeline — data loading, scoring, statistics, figures |
| [`analysis/two_datasets/`](analysis/two_datasets/) | Cancer & Mushroom case study |
| [`analysis/correlation_tradeoff/`](analysis/correlation_tradeoff/) | Utility–Fidelity–Privacy correlation figures |
| [`analysis/utility_drop/`](analysis/utility_drop/) | Utility drop / leakage analysis |
| [`analysis/representative_tradeoff/`](analysis/representative_tradeoff/) | Representative-metric trade-offs |
| [`scripts/extract_paper_results.py`](scripts/extract_paper_results.py) | Notebook → paper Excel workbooks |
| [`dashboard/`](dashboard/) | Static GitHub Pages dashboard builder |

---

## Results & figures

After running the pipelines:

```
Results/
├── Master_Data/                     # Merged long-format CSV (utility, fidelity, privacy)
├── Processed_Data/                  # Cumulative scores & generator rankings
├── Figures/                         # Full-benchmark publication figures
├── Tables/                          # CSV, XLSX, LaTeX
├── Supplementary/                   # Statistical test outputs, correlations
├── Correlation_Tradeoff_Analysis/   # Primary + seed-level U/F/P scatters
├── Utility_Drop_Analysis/           # Leakage / utility-drop study
├── Representative_Metrics_Tradeoff/ # F1–NNDR–Quality trade-offs
└── Two_Datasets_Assessment/         # Cancer & Mushroom case study
    ├── Cancer/
    ├── Mushroom/
    ├── Figures/                     # Comparative Figures 1–8
    └── Comparison/
```

**Preview — correlation trade-off (primary):**

<p align="center">
  <img src="Results/Correlation_Tradeoff_Analysis/Figures/Primary/Accuracy_NNDR_Figure01_fidelity_vs_utility_Accuracy.png" alt="Fidelity vs Utility (Accuracy)" width="700"/>
</p>

---

## Repository layout

```
SYNTH_BENCHMARK/
├── Generators/
│   ├── SDV models/                    # CTGAN, CopulaGAN, TVAE, GaussianCopula — full audit
│   ├── Other GANS/                    # CTABGAN, WGAN-GP — full audit
│   ├── Diffusion GANs/                # TabDDPM, ForestDiffusion — full audit
│   └── Experiment with utility data leak/
│       ├── utility results/           # TRTR/TSTR Excel (primary utility source)
│       └── diffusion_dataleak/        # Diffusion + SDV utility variant
├── analysis/                          # Automated publication pipeline
├── Results/                           # Generated analysis outputs
├── paper results/                     # Per-dataset paper workbooks
├── scripts/                           # Extraction & utility scripts
├── dashboard/                         # GitHub Pages dashboard
├── docs/                              # Built static site (GitHub Pages)
├── Datasets/                          # Cached dataset CSVs
├── Materials/                         # Paper notes & supplementary docs
└── figures/                           # Workflow diagram generators
```

---

## Experimental protocol

Every dataset notebook follows the same leak-safe design:

1. **Preprocess** — drop IDs, dates, session columns; handle missing values
2. **Subsample** to N = 1,000 rows (`seed = 42`)
3. **Split** — stratified 80% train / 20% test
4. **Generate** — each generator fits on `train_real` only → 1,000 synthetic rows
5. **Evaluate fidelity** — KS, JS, Wasserstein, MMD, SDV quality, t-SNE
6. **Evaluate utility** — 10 downstream models × 10 seeds; TRTR baseline vs TSTR
7. **Evaluate privacy** — MIA, Mahalanobis matching, nearest-neighbour distance
8. **Export** — Excel workbooks with mean ± SD and utility gaps

**TRTR** = Train on Real, Test on Real (baseline)  
**TSTR** = Train on Synthetic, Test on Real (utility of synthetic data)  
**Utility gap** = TRTR − TSTR (smaller = better synthetic utility)

---

## Datasets

| # | Name | Task | # | Name | Task |
|---|------|------|---|------|------|
| 1 | Wisconsin Breast Cancer | Classification | 9 | MAGIC Gamma Telescope | Classification |
| 2 | Alzheimer's | Classification | 10 | Metro Interstate Traffic | Regression |
| 3 | Adult Census | Classification | 11 | Online Shopping | Regression |
| 4 | Forest Cover | Classification | 12 | Air Quality | Regression |
| 5 | Bank Marketing | Classification | 13 | Concrete Strength | Regression |
| 6 | Wine Quality | Classification | 14 | Energy Efficiency | Regression |
| 7 | CDC Diabetes | Classification | 15 | Real Estate Valuation | Regression |
| 8 | Secondary Mushroom | Classification | | | |

Metadata: [`Generators/Experiment with utility data leak/python_scripts/hive/datasets.json`](Generators/Experiment%20with%20utility%20data%20leak/python_scripts/hive/datasets.json)

---

## Generators

| Generator | Family | Folder |
|-----------|--------|--------|
| CTGAN | GAN (SDV) | `SDV models/` |
| CopulaGAN | GAN (SDV) | `SDV models/` |
| TVAE | VAE (SDV) | `SDV models/` |
| GaussianCopula | Statistical (SDV) | `SDV models/` |
| CTABGAN | GAN | `Other GANS/` |
| WGAN-GP | GAN | `Other GANS/` |
| TabDDPM | Diffusion | `Diffusion GANs/` |
| ForestDiffusion | Diffusion | `Diffusion GANs/` |

### Extra setup for diffusion generators

```bash
git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm
pip install ForestDiffusion xgboost category-encoders imbalanced-learn
pip install "libzero==0.0.8" "rtdl==0.0.13" --no-deps   # torch 2.x compatible
```

---

## Dashboard

Interactive Plotly dashboard (utility, fidelity, privacy, trade-offs, rankings, **statistics**):

**[https://gopibattineni.github.io/SYNTH_BENCHMARK/](https://gopibattineni.github.io/SYNTH_BENCHMARK/)**

| Tab | Contents |
|-----|----------|
| Overview | Utility-gap heatmap + generator coverage |
| Utility | TRTR vs TSTR by task/metric/dataset |
| Fidelity | Quality, KS, Gower, Wasserstein, … by dataset |
| Privacy | MIA, NNDR, Mahalanobis, matching distances |
| Trade-offs | Privacy–utility scatter / 3D |
| Rankings | Weighted score + Borda |
| Statistics | PCA mean/median/std error % (8 gens × 15 datasets) + Wilcoxon heatmaps |

If you see a **404**, enable Pages once: **Settings → Pages → Source → Deploy from branch → `main` → `/docs`**.  
See [dashboard/README.md](dashboard/README.md) for full deployment steps.

Rebuild locally:

```bash
python dashboard/build_pages.py
python -m http.server 8080 --directory docs   # preview at http://localhost:8080
```

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{battineni2026synth,
  title  = {SYNTH Benchmark: A Reproducible Evaluation of Tabular Synthetic Data Generators},
  author = {Battineni, Gopi},
  year   = {2026},
  url    = {https://github.com/gopibattineni/SYNTH_BENCHMARK}
}
```

When reporting results, always state: dataset, generator, downstream model, **mean ± SD over 10 seeds**, TRTR baseline, TSTR score, and that generators were trained on training data only.

---

## Changelog

See [**CHANGELOG.md**](CHANGELOG.md) for release history and recent updates.

---

## Contributing

Contributions are welcome! See [**CONTRIBUTING.md**](CONTRIBUTING.md) for guidelines on running pipelines, adding datasets, and submitting pull requests.

---

<p align="center">
  <sub>Associated with LERO / BDS research on synthetic data auditing · Built for reproducible, publication-ready benchmarking</sub>
</p>
