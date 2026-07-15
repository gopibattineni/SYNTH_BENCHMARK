# SYNTH Benchmark Dashboard

Two ways to explore benchmark results:

| Mode | Use case | Command |
|------|----------|---------|
| **GitHub Pages** (static) | Public website, no server | `python dashboard/build_pages.py` → deploy `docs/` |
| **Streamlit** (local) | Live Excel reload, deeper drill-down | `streamlit run dashboard/app.py` |

---

## GitHub Pages (publish online)

### One-time GitHub setup

1. Push this repository to GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

The workflow `.github/workflows/deploy-pages.yml` runs on every push to `main`/`master`:

1. Runs `python run_analysis.py` (merges all Excel files)
2. Builds the interactive dashboard into `docs/`
3. Deploys to GitHub Pages

Your site will be available at:

```text
https://<username>.github.io/<repository-name>/
```

### Build locally

```bash
# 1. Generate Results/ (if not already done)
python run_analysis.py

# 2. Build static site
python dashboard/build_pages.py --repo-url https://github.com/YOU/SYNTH_BENCHMARK

# 3. Preview
cd docs && python -m http.server 8080
# Open http://localhost:8080
```

### Dashboard tabs

- **Overview** — utility gap heatmap, generator coverage
- **Utility** — TRTR vs TSTR, filters by dataset/metric/classifier
- **Fidelity** — SDV quality scores
- **Privacy** — Mahalanobis distance comparisons
- **Trade-offs** — privacy vs utility scatter, 3D plot
- **Rankings** — weighted & Borda rankings
- **Statistics** — significance heatmaps

---

## Streamlit (local interactive)

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

Reads Excel files directly from `Generators/Experiment with utility data leak/`.

---

## Data sources

- **Static dashboard (`docs/data/`)** — JSON exported from `Results/` by the analysis pipeline
- **Streamlit** — live Excel workbooks (TRTR/TSTR, Summary, Quality_Metrics)

8 generators: CTGAN, CopulaGAN, TVAE, GaussianCopula, WGAN_GP, CTABGAN, TabDDPM, ForestDiffusion
