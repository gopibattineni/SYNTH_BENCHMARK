"""Orchestrator for Wisconsin Breast Cancer & Mushroom comparative case study."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analysis.benchmarking import seed_stability_analysis
from analysis.config import PipelineConfig
from analysis.fidelity_analysis import analyze_fidelity
from analysis.privacy_analysis import analyze_privacy
from analysis.scores import build_processed_scores
from analysis.utility_analysis import analyze_utility
from analysis.two_datasets.config import DATASETS, DISPLAY_NAMES, TwoDatasetConfig
from analysis.two_datasets.data import load_two_dataset_master, save_dataset_processed
from analysis.two_datasets.figures import generate_comparison_figures
from analysis.two_datasets.tables import export_comparative_tables, export_dataset_tables, build_generator_summary_table

log = logging.getLogger(__name__)


@dataclass
class TwoDatasetAssessment:
    config: TwoDatasetConfig = field(default_factory=TwoDatasetConfig)
    results: dict[str, Any] = field(default_factory=dict)

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        tdc = self.config
        pipe = PipelineConfig(
            repo_root=tdc.repo_root,
            utility_weight=tdc.utility_weight,
            privacy_weight=tdc.privacy_weight,
            fidelity_weight=tdc.fidelity_weight,
        )

        print("=" * 60)
        print("Two-Dataset Comparative Case Study: Cancer & Mushroom")
        print("=" * 60)

        print("\n[Step 1] Loading and merging Cancer + Mushroom results...")
        master = load_two_dataset_master(tdc)
        log.info(
            "Rows — utility: %s, fidelity: %s, privacy: %s",
            len(master["utility_long"]),
            len(master["fidelity_long"]),
            len(master["privacy_long"]),
        )

        all_processed: dict[str, dict] = {}
        comparative_rows: list[pd.DataFrame] = []

        for ds_key, dataset_id in DATASETS.items():
            print(f"\n[Step 2] Scoring {DISPLAY_NAMES[ds_key]} ({dataset_id})...")
            dirs = tdc.dataset_dirs(ds_key)

            util_long = master["utility_long"][master["utility_long"]["Dataset"] == dataset_id].copy()
            fid_long = master["fidelity_long"][master["fidelity_long"]["Dataset"] == dataset_id].copy()
            priv_long = master["privacy_long"][master["privacy_long"]["Dataset"] == dataset_id].copy()
            priv_detail = master["privacy_detail"][master["privacy_detail"]["Dataset"] == dataset_id].copy()
            unified = master["unified"][master["unified"]["Dataset"] == dataset_id].copy()
            unified.to_csv(dirs["processed"] / "master_unified.csv", index=False)

            processed = build_processed_scores(util_long, fid_long, priv_long, pipe)
            ranking = processed.get("overall_scores", pd.DataFrame())
            if not ranking.empty:
                ranking = (
                    ranking.groupby("Generator", dropna=False)
                    .agg(
                        UtilityScore=("UtilityScore", "mean"),
                        FidelityScore=("FidelityScore", "mean"),
                        PrivacyScore=("PrivacyScore", "mean"),
                        OverallScore=("OverallScore", "mean"),
                    )
                    .reset_index()
                    .sort_values("OverallScore", ascending=False)
                )
                util_s = processed.get("utility_scores", pd.DataFrame())
                if not util_s.empty and "UtilityStd" in util_s.columns:
                    ranking["UtilityStd"] = ranking["Generator"].map(
                        util_s.groupby("Generator")["UtilityStd"].mean()
                    )
                fid_s = processed.get("fidelity_scores", pd.DataFrame())
                if not fid_s.empty and "FidelityStd" in fid_s.columns:
                    ranking["FidelityStd"] = ranking["Generator"].map(
                        fid_s.groupby("Generator")["FidelityStd"].mean()
                    )
                priv_s = processed.get("privacy_scores", pd.DataFrame())
                if not priv_s.empty and "PrivacyStd" in priv_s.columns:
                    ranking["PrivacyStd"] = ranking["Generator"].map(
                        priv_s.groupby("Generator")["PrivacyStd"].mean()
                    )
                ranking["Rank"] = range(1, len(ranking) + 1)
                processed["generator_ranking"] = ranking

            save_dataset_processed(processed, dirs)

            # Keep per-dataset analysis tables (no per-dataset figure spam)
            utility = analyze_utility(util_long, pipe)
            analyze_fidelity(fid_long, pipe)
            analyze_privacy(priv_long, priv_detail, pipe)
            seed = seed_stability_analysis(util_long, pipe)

            tables = {
                "generator_summary": build_generator_summary_table(
                    processed.get("generator_ranking", pd.DataFrame())
                ),
                "overall_scores": processed.get("overall_scores", pd.DataFrame()),
                "utility_scores": processed.get("utility_scores", pd.DataFrame()),
                "fidelity_scores": processed.get("fidelity_scores", pd.DataFrame()),
                "privacy_scores": processed.get("privacy_scores", pd.DataFrame()),
                "seed_stability": seed.get("seed_stability_ranks", pd.DataFrame()),
            }
            export_dataset_tables(tables, dirs["tables"], dirs["tables"])

            all_processed[ds_key] = processed
            comparative_rows.append(
                processed.get("generator_ranking", pd.DataFrame()).assign(Dataset=ds_key)
            )
            print(f"  Scores + tables → {dirs['root']}")

        print("\n[Step 3] Comparative Figures 1–8 + paper tables...")
        comp_dirs = tdc.comparison_dirs()
        cancer_rank = all_processed.get("Cancer", {}).get("generator_ranking", pd.DataFrame())
        mushroom_rank = all_processed.get("Mushroom", {}).get("generator_ranking", pd.DataFrame())

        comp_figs, extras = generate_comparison_figures(
            all_processed,
            comp_dirs,
            tdc,
            util_long=master["utility_long"],
            dataset_ids=DATASETS,
        )

        # Persist statistics used by Figure 8 / stability
        for name in ("friedman_ranks", "cd_groups", "stability_stats"):
            df = extras.get(name, pd.DataFrame())
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(comp_dirs["statistics"] / f"{name}.csv", index=False)

        export_comparative_tables(
            cancer_rank,
            mushroom_rank,
            {k: v for k, v in extras.items() if k != "combined_scores"},
            comp_dirs["tables"],
        )

        summary_dir = tdc.summary_dir()
        summary = {
            "runtime_seconds": round(time.time() - t0, 2),
            "datasets": DISPLAY_NAMES,
            "figures": comp_figs,
            "n_figures": len(comp_figs),
            "weights": {
                "utility": tdc.utility_weight,
                "fidelity": tdc.fidelity_weight,
                "privacy": tdc.privacy_weight,
            },
            "cancer_top_generator": (
                cancer_rank.iloc[0]["Generator"] if not cancer_rank.empty else None
            ),
            "mushroom_top_generator": (
                mushroom_rank.iloc[0]["Generator"] if not mushroom_rank.empty else None
            ),
            "output": {
                "figures": str(comp_dirs["figures"]),
                "tables": str(comp_dirs["tables"]),
                "statistics": str(comp_dirs["statistics"]),
            },
        }
        (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if comparative_rows:
            pd.concat(comparative_rows, ignore_index=True).to_csv(
                summary_dir / "combined_rankings.csv", index=False
            )

        md_lines = [
            "# Two-Dataset Comparative Case Study",
            "",
            f"**Runtime:** {summary['runtime_seconds']}s",
            "",
            "Publication figures compare **all 8 generators across Cancer and Mushroom** "
            "in the same plot (no separate per-dataset figure sets).",
            "",
            "## Top Generators",
            f"- **Cancer:** {summary['cancer_top_generator']}",
            f"- **Mushroom:** {summary['mushroom_top_generator']}",
            "",
            "## Figures (Comparative)",
            *[f"- `{name}`" for name in comp_figs],
            "",
            "## Output",
            f"- Figures: `{comp_dirs['figures']}`",
            f"- Tables: `{comp_dirs['tables']}`",
            f"- Statistics: `{comp_dirs['statistics']}`",
            f"- Per-dataset scores: `{tdc.output_root / 'Cancer'}` , `{tdc.output_root / 'Mushroom'}`",
        ]
        (summary_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

        print(f"\nComplete in {summary['runtime_seconds']}s")
        print(f"  Figures ({len(comp_figs)}): {comp_dirs['figures']}")
        print(f"  Tables: {comp_dirs['tables']}")
        print("=" * 60)

        self.results = {
            "master": master,
            "processed": all_processed,
            "figures": comp_figs,
            "summary": summary,
        }
        return self.results


def run_two_dataset_assessment(
    utility_weight: float = 0.40,
    privacy_weight: float = 0.30,
    fidelity_weight: float = 0.30,
) -> dict[str, Any]:
    config = TwoDatasetConfig(
        utility_weight=utility_weight,
        privacy_weight=privacy_weight,
        fidelity_weight=fidelity_weight,
    )
    return TwoDatasetAssessment(config=config).run()
