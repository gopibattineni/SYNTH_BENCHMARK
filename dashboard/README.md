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

## Deployment

GitHub Actions workflow `.github/workflows/deploy-pages.yml` rebuilds and deploys automatically on push to `main`.

Enable Pages: **Settings → Pages → Deploy from GitHub Actions**.

See the [main README](../README.md) for the full analysis pipeline.
