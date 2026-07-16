"""Orchestrator for Correlation Trade-off Analysis."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.correlation_tradeoff.config import (
    PRIVACY_ANALYSES,
    CorrelationTradeoffConfig,
)
from analysis.correlation_tradeoff.data import (
    aggregate_to_generator_means,
    build_seed_level_panel,
    save_processed,
)
from analysis.correlation_tradeoff.figures import (
    generate_primary_figures,
    generate_supplementary_figures,
)
from analysis.correlation_tradeoff.stats import result_to_row
from analysis.correlation_tradeoff.summaries import write_summaries


def _axis_names(fig_name: str) -> tuple[str, str]:
    key = fig_name.split("_")[0]
    mapping = {
        "Figure01": ("Fidelity", "Utility"),
        "Figure02": ("Privacy", "Utility"),
        "Figure03": ("Privacy", "Fidelity"),
        "FigureS01": ("Fidelity", "Utility"),
        "FigureS02": ("Privacy", "Utility"),
        "FigureS03": ("Privacy", "Fidelity"),
    }
    return mapping.get(key, ("?", "?"))


@dataclass
class CorrelationTradeoffAssessment:
    config: CorrelationTradeoffConfig = field(default_factory=CorrelationTradeoffConfig)
    results: dict[str, Any] = field(default_factory=dict)

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        cfg = self.config
        dirs = cfg.dirs()

        print("=" * 64)
        print("Correlation Trade-off Analysis")
        print("  Primary: 1 point per generator (mean over datasets × seeds)")
        print("  Supplementary: seed-level points (variability)")
        print("  Datasets: Cancer & Mushroom")
        print("  Fidelity: SDMetrics Quality Score")
        print("  Privacy A: MIA AUC | Privacy B: NNDR")
        print("=" * 64)

        all_stats_rows: list[dict] = []
        all_seed_panels: list[pd.DataFrame] = []
        all_gen_panels: list[pd.DataFrame] = []
        figure_index: list[dict] = []

        for util_metric in cfg.utility_metrics:
            for priv_key in PRIVACY_ANALYSES:
                print(f"\n[{util_metric} × {priv_key}] Building panels...")
                seed_panel = build_seed_level_panel(cfg, util_metric, priv_key)
                if seed_panel.empty:
                    print("  WARNING: empty panel — skipping")
                    continue

                gen_panel = aggregate_to_generator_means(seed_panel)
                adirs = cfg.analysis_dirs(util_metric, priv_key)

                save_processed(seed_panel, dirs["processed"] / f"seed_panel_{util_metric}_{priv_key}.csv")
                save_processed(gen_panel, dirs["processed"] / f"generator_means_{util_metric}_{priv_key}.csv")
                save_processed(seed_panel, adirs["root"] / f"seed_panel_{util_metric}_{priv_key}.csv")
                save_processed(gen_panel, adirs["root"] / f"generator_means_{util_metric}_{priv_key}.csv")
                all_seed_panels.append(seed_panel)
                all_gen_panels.append(gen_panel.assign(PrivacyKey=priv_key))

                print(
                    f"  Primary points: {len(gen_panel)} generators "
                    f"(mean over {gen_panel['N_Datasets'].iloc[0]} datasets × "
                    f"{gen_panel['N_Seeds'].iloc[0]} seeds)"
                )
                print(f"  Supplementary points: {len(seed_panel)} (Generator × Seed × Dataset)")
                print(f"  Fidelity: {gen_panel['FidelityMetric'].iloc[0]} | Privacy: {gen_panel['PrivacyMetric'].iloc[0]}")

                # --- Primary publication figures ---
                print("  Rendering PRIMARY Figures 1–3 (generator means)...")
                primary = generate_primary_figures(gen_panel, adirs["figures_primary"], cfg)
                for name, res in primary.items():
                    self._record_figure(
                        name=name,
                        res=res,
                        util_metric=util_metric,
                        priv_key=priv_key,
                        level="primary",
                        src_dir=adirs["figures_primary"],
                        flat_dir=dirs["figures_primary"],
                        all_stats_rows=all_stats_rows,
                        figure_index=figure_index,
                        cfg=cfg,
                    )

                write_summaries(
                    gen_panel,
                    primary,
                    adirs["summaries"] / "Primary",
                    prefix="Primary: ",
                )

                # --- Supplementary seed-level figures ---
                print("  Rendering SUPPLEMENTARY Figures S1–S3 (seed-level)...")
                supp = generate_supplementary_figures(
                    seed_panel, adirs["figures_supplementary"], cfg
                )
                for name, res in supp.items():
                    self._record_figure(
                        name=name,
                        res=res,
                        util_metric=util_metric,
                        priv_key=priv_key,
                        level="supplementary",
                        src_dir=adirs["figures_supplementary"],
                        flat_dir=dirs["figures_supplementary"],
                        all_stats_rows=all_stats_rows,
                        figure_index=figure_index,
                        cfg=cfg,
                    )

                write_summaries(
                    seed_panel,
                    supp,
                    adirs["summaries"] / "Supplementary",
                    prefix="Supplementary: ",
                )

                # Per-analysis stats
                stats_df = pd.DataFrame(
                    [
                        {**result_to_row(n, *_axis_names(n), r), "Level": "primary"}
                        for n, r in primary.items()
                    ]
                    + [
                        {**result_to_row(n, *_axis_names(n), r), "Level": "supplementary"}
                        for n, r in supp.items()
                    ]
                )
                stats_df.to_csv(adirs["statistics"] / "correlation_stats.csv", index=False)

        stats_all = pd.DataFrame(all_stats_rows)
        if not stats_all.empty:
            stats_all.to_csv(dirs["statistics"] / "all_correlation_stats.csv", index=False)
            stats_all.to_excel(dirs["tables"] / "all_correlation_stats.xlsx", index=False)
            if "Level" in stats_all.columns:
                stats_all[stats_all["Level"] == "primary"].to_csv(
                    dirs["statistics"] / "primary_correlation_stats.csv", index=False
                )
                stats_all[stats_all["Level"] == "supplementary"].to_csv(
                    dirs["statistics"] / "supplementary_correlation_stats.csv", index=False
                )

        if all_seed_panels:
            pd.concat(all_seed_panels, ignore_index=True).to_csv(
                dirs["processed"] / "all_seed_panels.csv", index=False
            )
        if all_gen_panels:
            pd.concat(all_gen_panels, ignore_index=True).to_csv(
                dirs["processed"] / "all_generator_means.csv", index=False
            )

        self._write_findings(dirs["root"], figure_index)
        self._write_readme(dirs["root"])

        mirror = cfg.two_datasets_mirror
        if mirror.exists():
            shutil.rmtree(mirror)
        shutil.copytree(cfg.output_root, mirror)
        print(f"\nMirrored results → {mirror}")

        elapsed = time.time() - t0
        summary = {
            "n_primary_figures": sum(1 for f in figure_index if f["level"] == "primary"),
            "n_supplementary_figures": sum(1 for f in figure_index if f["level"] == "supplementary"),
            "output": str(cfg.output_root),
            "mirror": str(mirror),
            "elapsed_sec": round(elapsed, 1),
            "figures": figure_index,
        }
        (dirs["root"] / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.results = summary
        print(f"\nDone in {elapsed:.1f}s → {cfg.output_root}")
        return summary

    def _record_figure(
        self,
        *,
        name: str,
        res,
        util_metric: str,
        priv_key: str,
        level: str,
        src_dir: Path,
        flat_dir: Path,
        all_stats_rows: list[dict],
        figure_index: list[dict],
        cfg: CorrelationTradeoffConfig,
    ) -> None:
        for fmt in cfg.figure_formats:
            src = src_dir / f"{name}.{fmt}"
            if src.is_file():
                shutil.copy2(src, flat_dir / f"{util_metric}_{priv_key}_{name}.{fmt}")

        x_name, y_name = _axis_names(name)
        row = result_to_row(f"{util_metric}/{priv_key}/{name}", x_name, y_name, res)
        row["Level"] = level
        all_stats_rows.append(row)
        figure_index.append(
            {
                "utility_metric": util_metric,
                "privacy": priv_key,
                "level": level,
                "figure": name,
                "pearson_r": res.pearson_r,
                "spearman_rho": res.spearman_rho,
                "r2": res.r2,
                "pearson_p": res.pearson_p,
                "interpretation": res.interpretation,
                "n": res.n,
                "path": str(src_dir / name),
            }
        )

    def _write_findings(self, root: Path, figure_index: list[dict]) -> None:
        lines = [
            "# Correlation Trade-off Analysis — Findings",
            "",
            "**Primary figures:** one point per generator (mean across Cancer & Mushroom, "
            "classifiers, and seeds).",
            "",
            "**Supplementary figures:** seed-level points showing variability "
            "(Cancer = circle, Mushroom = square).",
            "",
        ]
        if not figure_index:
            lines.append("No figures were generated.")
        else:
            for level in ("primary", "supplementary"):
                lines.append(f"## {level.capitalize()}")
                lines.append("")
                subset = [r for r in figure_index if r["level"] == level]
                by_key: dict[str, list[dict]] = {}
                for row in subset:
                    key = f"{row['utility_metric']} / {row['privacy']}"
                    by_key.setdefault(key, []).append(row)
                for key, rows in by_key.items():
                    lines.append(f"### {key}")
                    lines.append("")
                    for row in rows:
                        lines.append(
                            f"- **{row['figure']}**: Pearson r = {row['pearson_r']:.3f}, "
                            f"Spearman ρ = {row['spearman_rho']:.3f}, "
                            f"R² = {row['r2']:.3f}, p = {row['pearson_p']:.4g} — "
                            f"{row['interpretation']}"
                        )
                    lines.append("")
        (root / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_readme(self, root: Path) -> None:
        text = """# Correlation Trade-off Analysis

Publication-quality trade-off figures relating **Utility**, **Fidelity**, and **Privacy**
for Cancer and Mushroom across all eight generators.

## Primary figures (use these in the paper)

**One scatter point per generator** (8 points).

Each point is the mean of the metric across:
- both datasets (Cancer, Mushroom)
- all classifiers (utility)
- all 10 random seeds (utility reconstructed from Mean±SD)

| Axis | Metric |
|------|--------|
| Utility | Mean TSTR Accuracy (and F1 variant) |
| Fidelity | Mean SDMetrics Quality Score |
| Privacy A | Mean MIA AUC (lower = more private) |
| Privacy B | Mean NNDR (higher = more private) |

Figures 1–3: Fidelity–Utility, Privacy–Utility, Fidelity–Privacy.  
Each point is labelled with the generator name and metric values.  
Statistics: Pearson *r*, Spearman *ρ*, R², *p*, OLS fit, 95% CI.

## Supplementary figures

Seed-level scatters (Generator × Seed × Dataset) with Cancer = circle and
Mushroom = square, overall + per-dataset dashed fits — for variability / robustness.

## Layout

```
Correlation_Tradeoff_Analysis/
  Figures/Primary/              # flat copies of primary figs
  Figures/Supplementary/        # flat copies of seed-level figs
  Utility_Accuracy/Analysis_A_MIA/
    Primary/Figures/
    Supplementary/Figures/
    Statistics/
    Summaries/
  … (F1 × MIA/NNDR likewise)
  Processed_Data/
  FINDINGS.md
```

Mirrored to `Results/Two_Datasets_Assessment/Correlation_Tradeoff_Analysis/`.

## Run

```bash
python run_correlation_tradeoff_analysis.py
```
"""
        (root / "README.md").write_text(text, encoding="utf-8")


def run_correlation_tradeoff_analysis() -> dict[str, Any]:
    return CorrelationTradeoffAssessment().run()
