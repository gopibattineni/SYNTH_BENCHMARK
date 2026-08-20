# Dashboard

Static GitHub Pages dashboard for browsing SYNTH benchmark results.

## Live site

**https://gopibattineni.github.io/SYNTH_BENCHMARK/**

## Rebuild

```bash
python run_analysis.py --dashboard
python dashboard/build_pages.py
```

Output is written to `docs/` at the repository root.

## What it shows

- Generator rankings across all 15 datasets
- TRTR vs TSTR comparisons per dataset
- Utility loss heatmaps (classification & regression)
- Radar charts and summary tables

## Deployment (fix GitHub Pages 404)

The dashboard files live in `docs/`, but GitHub Pages must be **turned on once** in the repository settings.

### Option A — Fastest (use committed `docs/`)

1. Open **https://github.com/gopibattineni/SYNTH_BENCHMARK/settings/pages**
2. Under **Build and deployment → Source**, choose **Deploy from a branch**
3. Branch: **main**, folder: **/docs**
4. Save — the site should appear within 1–2 minutes at  
   **https://gopibattineni.github.io/SYNTH_BENCHMARK/**

> **Note:** GitHub Pages on free accounts requires the repository to be **public**.

### Option B — GitHub Actions (rebuilds JSON on each push)

1. **Settings → Pages → Source:** select **GitHub Actions**
2. Push to `main`, or run **Actions → Deploy Dashboard to GitHub Pages → Run workflow**
3. The workflow exports fresh data from `Results/` and deploys `docs/`

See the [main README](../README.md) for the full analysis pipeline.
