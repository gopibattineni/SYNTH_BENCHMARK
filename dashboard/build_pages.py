#!/usr/bin/env python3
"""Build the static GitHub Pages dashboard into docs/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
STATIC_DIR = REPO_ROOT / "dashboard" / "static"

sys.path.insert(0, str(REPO_ROOT))

from dashboard.export_for_web import export_dashboard_data  # noqa: E402


def build_pages(github_repo_url: str | None = None) -> Path:
    """Export data + copy static assets to docs/ for GitHub Pages."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Export JSON from Results/
    results_dir = REPO_ROOT / "Results"
    if not results_dir.exists():
        print("Warning: Results/ not found. Run `python run_analysis.py` first.")
    else:
        export_dashboard_data(DOCS_DIR / "data")
        print(f"Exported data to {DOCS_DIR / 'data'}")

    # Copy static assets
    assets_dst = DOCS_DIR / "assets"
    if assets_dst.exists():
        shutil.rmtree(assets_dst)
    shutil.copytree(STATIC_DIR, assets_dst)

    # Copy index.html
    index_src = STATIC_DIR / "index.html"
    index_dst = DOCS_DIR / "index.html"
    html = index_src.read_text(encoding="utf-8")

    if github_repo_url:
        html = html.replace('href="https://github.com/"', f'href="{github_repo_url}"')

    index_dst.write_text(html, encoding="utf-8")

    # Prevent Jekyll from ignoring JSON/data on GitHub Pages
    (DOCS_DIR / ".nojekyll").touch()

    print(f"Built dashboard at {DOCS_DIR}")
    return DOCS_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GitHub Pages dashboard")
    parser.add_argument("--repo-url", type=str, default="", help="GitHub repository URL for footer link")
    args = parser.parse_args()
    build_pages(args.repo_url or None)
