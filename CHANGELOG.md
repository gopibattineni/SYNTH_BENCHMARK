# Changelog

All notable changes to the SYNTH Benchmark repository are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **Automated publication analysis pipeline** (`analysis/`, `run_analysis.py`)
  - Auto-discovers and merges all Excel files (no hardcoded filenames)
  - Cumulative Utility / Fidelity / Privacy scores (0–1) with configurable weights (default 40/30/30)
  - Friedman + Nemenyi + Wilcoxon statistical tests with effect sizes
  - Pareto frontier analysis, composite rankings, critical difference diagrams
  - Seed stability (coefficient of variation across 10 seeds)
  - 60+ publication figures at 300 DPI (PNG, PDF, SVG, EPS)
- **Two-dataset case study** (`analysis/two_datasets/`, `run_two_datasets_assessment.py`)
  - Dedicated Cancer & Mushroom assessment with 18 figures per dataset
  - Cross-dataset comparison plots and tables
  - Output: `Results/Two_Datasets_Assessment/`
- **Paper results extraction** (`scripts/extract_paper_results.py`)
  - Per-dataset `TRTR_TSTR_results_*.xlsx` and `fidelity_privacy_metrics.xlsx`
  - t-SNE figure extraction from notebooks
- **GitHub Pages dashboard** (`dashboard/`, `.github/workflows/deploy-pages.yml`)
  - Static site built from pipeline outputs in `docs/`
- **Feature-level fidelity heatmaps** from generator notebook HTML tables (KS, JS)

### Changed
- README rewritten for clarity — quick start, badges, mermaid workflow, updated folder paths
- Repository layout documented with current `Generators/` structure
- Privacy dataset name normalization for cross-domain joins

### Fixed
- Pareto frontier analysis (privacy dataset name mapping)
- Empty regression stats crash in utility analysis
- Metro TRTR evaluation for categorical features
- Forest Cover diffusion evaluation and ForestDiffusion speed
- Online Shopping TabDDPM training

---

## [2026-07-09] — Initial benchmark release

### Added
- 15-dataset benchmark across 5 generator folder families
- TRTR/TSTR utility protocol with 10 seeds × 10 downstream models
- Full fidelity + privacy audit notebooks (SDV, Other GANs, Diffusion)
- Leak-safe 80/20 train/test split with N=1000 subsampling
- Excel export pipeline (`TRTR_TSTR_results*.xlsx`)
- Workflow diagram generator (`figures/generate_forge_paper_workflow.py`)
- Dataset registry (`python_scripts/hive/datasets.json`)
- Regression datasets #10–#15 support
