# SYNTH Benchmark Dashboard

Interactive dashboard for **TRTR/TSTR** utility results from the diffusion data-leak
benchmark under `Generators/Experiment with utility data leak/diffusion_dataleak`.

## Data source

- **15 datasets** (classification 1–9, regression 10–15)
- **8 generators** tracked end-to-end:
  - SDV / GAN: `CTGAN`, `CopulaGAN`, `TVAE`, `GaussianCopula`, `WGAN_GP`, `CTABGAN`
  - Diffusion: `TabDDPM`, `ForestDiffusion`
- Results are loaded from `diffusion_dataleak/<dataset>/TRTR_TSTR*.xlsx` when available,
  with missing SDV/GAN rows filled from the parent `Experiment with utility data leak`
  folder when that workbook exists.

Each dataset folder may contain `TRTR_TSTR_results*.xlsx` with sheets:
`TRTR_Results`, `All_Comparisons`, `Summary`, `Quality_Metrics`, and per-generator detail tabs.

## Setup

```bash
cd dashboard
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Or on Windows:

```bash
run.bat
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Views

| Tab | Description |
|-----|-------------|
| **Experiment status** | Coverage heatmap (dataset × generator) and pending/partial runs |
| **Overview** | Heatmap and bar chart of utility gap across datasets and generators |
| **Dataset detail** | TRTR vs TSTR per downstream model, summary tables, quality scores |
| **Generator ranking** | Best generator per dataset and win-count chart |
| **Data tables** | Full summary table with CSV export and coverage matrix |
