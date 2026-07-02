# SYNTH — TRTR/TSTR Results Dashboard

Interactive dashboard for synthetic data utility results (10 datasets, 6 generators, TRTR vs TSTR).

## Live dashboard (GitHub Pages)

After enabling GitHub Pages, the dashboard is published at:

**https://gopibattineni.github.io/SYNTH/**

### One-time GitHub setup

1. Open https://github.com/gopibattineni/SYNTH/settings/pages
2. Under **Build and deployment → Source**, choose **Deploy from a branch**
3. **Branch:** `gh-pages` · **Folder:** `/ (root)`
4. Click **Save** and wait 2–5 minutes

The dashboard is already on the `gh-pages` branch (pushed). After Pages is enabled, open:

**https://gopibattineni.github.io/SYNTH/**

Alternative: use **GitHub Actions** as source and push the `main` branch workflow (`.github/workflows/deploy-dashboard.yml`).

## Run locally (full web app + live experiments)

```bash
cd webapp
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Experiment app: http://127.0.0.1:8000/
- Dashboard (reads Excel via API): http://127.0.0.1:8000/dashboard

On Windows, double-click `webapp/start.bat`.

## Data source

Results are loaded from:

`Synthetic_Data_Audit/SYNTH/Single run_Data_leak_Synth_Quality/*/TRTR_TSTR_results*.xlsx`

The export script converts these to JSON for the static GitHub Pages site.

## Rebuild dashboard after new Excel results

```bash
python webapp/scripts/export_dashboard_data.py
python webapp/scripts/build_github_pages.py
git add docs webapp/app/static/data/audit
git commit -m "Update dashboard results"
git push
```

Or push to `main` and let GitHub Actions rebuild.

## Repository structure

| Path | Purpose |
|------|---------|
| `webapp/` | FastAPI app + dashboard UI |
| `webapp/scripts/export_dashboard_data.py` | Excel → JSON |
| `webapp/scripts/build_github_pages.py` | Build `docs/` for Pages |
| `docs/` | Static site deployed to GitHub Pages |
| `.github/workflows/deploy-dashboard.yml` | CI deploy workflow |
