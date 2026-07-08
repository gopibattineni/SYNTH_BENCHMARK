# SYNTH Benchmark Dashboard

Interactive dashboard for **TRTR/TSTR** utility results exported from
`Generators/Experiment with utility data leak`.

## Data source

- **15 datasets** (classification 1–9, regression 10–15)
- **6 generators** in the main experiment Excel files:
  `CTGAN`, `CopulaGAN`, `TVAE`, `GaussianCopula`, `WGAN_GP`, `CTABGAN`
- **TabDDPM** and **ForestDiffusion** are merged automatically when present under
  `diffusion_dataleak/` (currently Cancer, Alzheimer's, Adult).

Each dataset folder may contain `TRTR_TSTR_results*.xlsx` with sheets:
`TRTR_Results`, `All_Comparisons`, `Summary`, and per-generator detail tabs.

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
| **Overview** | Heatmap and bar chart of utility gap across datasets and generators |
| **Dataset detail** | TRTR vs TSTR per downstream model, summary tables, SDV quality |
| **Generator ranking** | Best generator per dataset and win-count chart |
| **Data tables** | Full summary table with CSV export |
