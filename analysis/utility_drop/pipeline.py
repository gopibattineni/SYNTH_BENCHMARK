"""Orchestrator for Utility Drop Analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analysis.utility_drop.config import UtilityDropConfig
from analysis.utility_drop.data import (
    attach_utility_loss,
    best_classifier_per_generator,
    best_generator_per_classifier,
    build_trtr_table,
    build_tstr_table,
    generator_summary,
    leakage_levels_present,
    load_utility_long,
)
from analysis.utility_drop.figures import generate_all_figures
from analysis.utility_drop.tables import (
    export_all_dataset_summary,
    export_mean_sd_tables,
    export_rankings,
    write_findings_markdown,
)


@dataclass
class UtilityDropAssessment:
    config: UtilityDropConfig = field(default_factory=UtilityDropConfig)
    results: dict[str, Any] = field(default_factory=dict)

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.config
        dirs = cfg.dirs()

        print("=" * 64)
        print("Utility Drop Analysis")
        print("  Primary metric: Accuracy (Mean ± SD)")
        print("  Focus: Cancer & Mushroom (+ all-dataset tables)")
        print("=" * 64)

        print("\n[1/5] Loading utility Excel / master data...")
        utility = load_utility_long(cfg)
        if utility.empty:
            raise RuntimeError("No utility data found. Check Master_Data/utility_long.csv")
        levels = leakage_levels_present(utility)
        multi_leakage = len([x for x in levels if float(x) != 0.0]) > 0
        print(f"  Rows: {len(utility):,}")
        print(f"  Datasets: {utility['Dataset'].nunique()}")
        print(f"  Generators: {sorted(utility['Generator'].dropna().unique())}")
        print(f"  Leakage levels: {levels}")
        if not multi_leakage:
            print("  NOTE: Only Leakage=0% in exports; Figs 1/5/6 use TRTR−TSTR drop.")

        print("\n[2/5] Aggregating TSTR / TRTR Accuracy...")
        tstr = build_tstr_table(utility, cfg.primary_metric)
        trtr = build_trtr_table(utility, cfg.primary_metric)
        loss_df = attach_utility_loss(tstr, trtr)
        gen_sum = generator_summary(tstr)
        best_clf = best_classifier_per_generator(tstr)
        best_gen = best_generator_per_classifier(tstr)

        # Persist processed
        tstr.to_csv(dirs["processed"] / "tstr_accuracy.csv", index=False)
        loss_df.to_csv(dirs["processed"] / "utility_loss.csv", index=False)
        gen_sum.to_csv(dirs["processed"] / "generator_summary.csv", index=False)
        utility.to_csv(dirs["processed"] / "utility_long_filtered.csv", index=False)

        print(f"  TSTR Accuracy rows: {len(tstr)}")
        print(f"  With TRTR match: {loss_df['TRTR_Mean'].notna().sum() if 'TRTR_Mean' in loss_df.columns else 0}")

        print("\n[3/5] Exporting tables...")
        export_mean_sd_tables(utility, dirs["tables"], cfg)
        export_rankings(tstr, loss_df, dirs["tables"])
        export_all_dataset_summary(tstr, dirs["tables"])
        write_findings_markdown(tstr, loss_df, multi_leakage, dirs["root"] / "FINDINGS.md")

        print("\n[4/5] Generating Figures 1–12...")
        figs = generate_all_figures(
            tstr=tstr,
            gen_sum=gen_sum,
            loss_df=loss_df,
            best_clf=best_clf,
            best_gen=best_gen,
            utility=utility,
            dirs=dirs,
            cfg=cfg,
            multi_leakage=multi_leakage,
        )
        print(f"  Saved: {len(figs)} figure stems")

        # Supplementary F1 / ROC-AUC heatmaps as CSV already; optional mini notes
        for metric in ("F1", "ROC-AUC"):
            m_tstr = build_tstr_table(utility, metric)
            if not m_tstr.empty:
                m_tstr.to_csv(dirs["processed"] / f"tstr_{metric.lower().replace('-', '_')}.csv", index=False)

        print("\n[5/5] Writing README + summary.json...")
        summary = {
            "runtime_seconds": round(time.time() - t0, 2),
            "primary_metric": cfg.primary_metric,
            "n_seeds": cfg.n_seeds,
            "leakage_levels_in_data": [float(x) for x in levels],
            "multi_leakage": multi_leakage,
            "n_utility_rows": int(len(utility)),
            "n_tstr_accuracy": int(len(tstr)),
            "figures": figs,
            "output_root": str(cfg.output_root),
            "note": (
                None
                if multi_leakage
                else "Excel/master only contain Leakage=0. Utility drop/loss computed as TRTR−TSTR."
            ),
        }
        (dirs["root"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_readme(dirs["root"], summary, multi_leakage)

        print(f"\nComplete in {summary['runtime_seconds']}s")
        print(f"  Output: {cfg.output_root}")
        print("=" * 64)

        self.results = {
            "utility": utility,
            "tstr": tstr,
            "loss": loss_df,
            "summary": summary,
        }
        return self.results


def _write_readme(root, summary: dict, multi_leakage: bool) -> None:
    lines = [
        "# Utility Drop Analysis",
        "",
        f"**Runtime:** {summary['runtime_seconds']}s",
        "",
        "## Purpose",
        "Evaluate synthetic-data utility under leakage conditions using existing Excel results "
        "(no ML re-runs). Identifies best generator, best classifier, and robustness to utility drop.",
        "",
        "## Primary metric",
        "Accuracy — reported as **Mean ± Standard Deviation** across random seeds.",
        "",
        "## Supplementary metrics",
        "F1-score, ROC-AUC, Precision, Recall (tables under `Tables/`).",
        "",
        "## Data note",
    ]
    if multi_leakage:
        lines.append(
            f"Leakage levels found: `{summary['leakage_levels_in_data']}`. "
            "Figures 1, 5, 6 use the leakage axis."
        )
    else:
        lines += [
            "Master/Excel utility exports currently contain **Leakage = 0% only**.",
            "Configured levels `[0,5,10,...,50]` were not exported as multi-level tables.",
            "",
            "- **Figure 1:** Mean Accuracy by generator (Cancer vs Mushroom line styles).",
            "- **Figures 5–6:** Utility drop/loss = TRTR − TSTR (gap vs real-data baseline), not leakage %.",
            "- **Figures 11–12:** Seed distributions approximated from exported Mean±SD (raw seeds not stored).",
        ]
    lines += [
        "",
        "## Outputs",
        "- `Figures/` — Figures 1–12 (+ CD diagrams 10a/10b)",
        "- `Tables/` — rankings, Mean±SD matrices, robustness",
        "- `Statistics/` — Friedman / Nemenyi ranks",
        "- `Processed_Data/` — intermediate CSVs",
        "- `FINDINGS.md` — short answers to the four research questions",
        "",
        "## How to run",
        "```bash",
        "python run_utility_drop_analysis.py",
        "```",
    ]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_utility_drop_analysis() -> dict[str, Any]:
    return UtilityDropAssessment().run()
