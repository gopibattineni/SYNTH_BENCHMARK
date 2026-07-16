"""Orchestrator for classifier-independent vs best-classifier trade-off analysis."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analysis.representative_tradeoff.config import RepresentativeTradeoffConfig
from analysis.representative_tradeoff.data import (
    add_normalized_and_ranks,
    build_best_classifier_scores,
    build_classifier_independent_scores,
    save_processed,
    _load_utility_tstr,
)
from analysis.representative_tradeoff.figures import generate_analysis_figures
from analysis.representative_tradeoff.tables import export_tables

log = logging.getLogger(__name__)


@dataclass
class RepresentativeTradeoffAssessment:
    config: RepresentativeTradeoffConfig = field(default_factory=RepresentativeTradeoffConfig)
    results: dict[str, Any] = field(default_factory=dict)

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.config

        print("=" * 64)
        print("Representative Metrics Trade-off")
        print("  A) Classifier-Independent (PRIMARY)")
        print("  B) Best Classifier (SUPPLEMENTARY)")
        print("=" * 64)

        print("\n[1/4] Building Classifier-Independent scores (avg over 10 classifiers)...")
        indep_raw, indep_by_metric = build_classifier_independent_scores(cfg)
        indep = add_normalized_and_ranks(indep_raw)
        indep_dirs = cfg.analysis_dirs("Classifier_Independent")
        save_processed(indep, {"utility_by_metric": indep_by_metric}, indep_dirs)
        print(f"  Rows: {len(indep)} → {indep_dirs['root']}")

        print("\n[2/4] Building Best-Classifier scores...")
        best_raw, best_detail = build_best_classifier_scores(cfg)
        best = add_normalized_and_ranks(best_raw)
        best_dirs = cfg.analysis_dirs("Best_Classifier")
        save_processed(best, {"best_classifier_detail": best_detail}, best_dirs)
        if not best.empty and "BestClassifier" in best.columns:
            print("  Best classifiers:")
            for _, r in best.sort_values(["DatasetKey", "Generator"]).iterrows():
                print(f"    {r['DatasetKey']:10s} {r['Generator']:18s} → {r['BestClassifier']} (F1={r['F1']:.3f})")

        util_long = _load_utility_tstr(cfg)

        print("\n[3/4] Generating figures (both analyses)...")
        indep_figs, indep_extras = generate_analysis_figures(
            indep, indep_dirs, cfg, "Classifier_Independent", util_long=util_long
        )
        best_figs, best_extras = generate_analysis_figures(
            best, best_dirs, cfg, "Best_Classifier", util_long=util_long
        )
        print(f"  Classifier-Independent figures: {len(indep_figs)}")
        print(f"  Best-Classifier figures: {len(best_figs)}")

        print("\n[4/4] Exporting tables + statistical comparison...")
        comp_dirs = cfg.comparison_dirs()
        export_tables(indep, best, indep_dirs, best_dirs, comp_dirs, indep_extras, best_extras)

        # Copy CD stats into comparison folder
        for name, extras in (("indep", indep_extras), ("best", best_extras)):
            for key, df in extras.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_csv(comp_dirs["statistics"] / f"{name}_{key}.csv", index=False)

        summary = {
            "runtime_seconds": round(time.time() - t0, 2),
            "primary_analysis": "Classifier_Independent",
            "utility_metric": cfg.primary_utility_metric,
            "privacy_metric": "NNDR (Avg NN distance)",
            "fidelity_metric": "Quality_Score",
            "n_seeds": cfg.n_seeds,
            "figures": {
                "Classifier_Independent": indep_figs,
                "Best_Classifier": best_figs,
            },
            "output_root": str(cfg.output_root),
        }
        summary_path = cfg.output_root / "README.md"
        cfg.output_root.mkdir(parents=True, exist_ok=True)
        (cfg.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        md = [
            "# Representative Metrics Trade-off Analysis",
            "",
            f"**Runtime:** {summary['runtime_seconds']}s",
            "",
            "## Primary analysis (manuscript)",
            "**Classifier-Independent** — mean utility across all 10 classifiers "
            "(seed means already aggregated), with NNDR privacy and SDMetrics Quality Score.",
            "",
            f"Figures: `{indep_dirs['figures']}`",
            "",
            "## Supplementary analysis",
            "**Best Classifier** — highest mean F1 per generator × dataset.",
            "",
            f"Figures: `{best_dirs['figures']}`",
            "",
            "## Comparison",
            f"`{comp_dirs['root']}` — Indep vs Best tables, Wilcoxon rank consistency, CD ranks.",
            "",
            "## Metrics",
            f"- Utility: Mean {cfg.primary_utility_metric}",
            "- Privacy: Mean NNDR (Avg NN distance; higher = more private)",
            "- Fidelity: SDMetrics Quality Score",
        ]
        summary_path.write_text("\n".join(md), encoding="utf-8")

        print(f"\nComplete in {summary['runtime_seconds']}s")
        print(f"  Output: {cfg.output_root}")
        print("=" * 64)

        self.results = {
            "classifier_independent": indep,
            "best_classifier": best,
            "summary": summary,
        }
        return self.results


def run_representative_tradeoff() -> dict[str, Any]:
    return RepresentativeTradeoffAssessment().run()
