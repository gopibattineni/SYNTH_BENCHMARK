"""Orchestrator for Wisconsin Breast Cancer & Mushroom case study."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analysis.benchmarking import friedman_average_ranks, nemenyi_significance_groups, seed_stability_analysis
from analysis.config import PipelineConfig
from analysis.fidelity_analysis import analyze_fidelity
from analysis.privacy_analysis import analyze_privacy
from analysis.scores import build_processed_scores
from analysis.statistics import run_generator_comparison_tests
from analysis.utility_analysis import analyze_utility
from analysis.two_datasets.config import DATASETS, DISPLAY_NAMES, TwoDatasetConfig
from analysis.two_datasets.data import load_two_dataset_master, save_dataset_processed
from analysis.two_datasets.figures import generate_comparison_figures, generate_dataset_figures
from analysis.two_datasets.tables import (
    build_comparative_table,
    build_generator_summary_table,
    export_dataset_tables,
)

log = logging.getLogger(__name__)


def _run_dataset_statistics(clf_stats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Friedman/Nemenyi using classifiers as blocks (single-dataset design)."""
    if clf_stats.empty:
        return {"friedman_ranks": pd.DataFrame(), "cd_groups": pd.DataFrame(), "tests": {}}

    tstr = clf_stats[
        (clf_stats["EvaluationType"] == "TSTR") & (clf_stats["Metric"] == "Accuracy")
    ].copy()
    if tstr.empty:
        return {"friedman_ranks": pd.DataFrame(), "cd_groups": pd.DataFrame(), "tests": {}}

    agg = tstr.groupby(["Classifier", "Generator"], dropna=False)["Mean"].mean().reset_index()
    ranks = friedman_average_ranks(agg, block_col="Classifier", treatment_col="Generator")
    cd_groups = nemenyi_significance_groups(ranks) if not ranks.empty else pd.DataFrame()

    tests = run_generator_comparison_tests(
        tstr,
        value_col="Mean",
        block_col="Classifier",
        treatment_col="Generator",
        metric_col="Metric",
    )
    return {"friedman_ranks": ranks, "cd_groups": cd_groups, "tests": tests}


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
        print("Two-Dataset Case Study: Cancer & Mushroom")
        print("=" * 60)

        print("\n[Step 1–2] Loading and merging Cancer + Mushroom results...")
        master = load_two_dataset_master(tdc)
        log.info(
            "Rows — utility: %s, fidelity: %s, privacy: %s",
            len(master["utility_long"]),
            len(master["fidelity_long"]),
            len(master["privacy_long"]),
        )

        all_processed: dict[str, dict] = {}
        all_figures: dict[str, list[str]] = {}
        comparative_rows: list[pd.DataFrame] = []

        for ds_key, dataset_id in DATASETS.items():
            print(f"\n[Steps 3–6] Processing {DISPLAY_NAMES[ds_key]} ({dataset_id})...")
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
                    std_map = util_s.groupby("Generator")["UtilityStd"].mean()
                    ranking["UtilityStd"] = ranking["Generator"].map(std_map)
                fid_s = processed.get("fidelity_scores", pd.DataFrame())
                if not fid_s.empty and "FidelityStd" in fid_s.columns:
                    ranking["FidelityStd"] = ranking["Generator"].map(fid_s.groupby("Generator")["FidelityStd"].mean())
                priv_s = processed.get("privacy_scores", pd.DataFrame())
                if not priv_s.empty and "PrivacyStd" in priv_s.columns:
                    ranking["PrivacyStd"] = ranking["Generator"].map(priv_s.groupby("Generator")["PrivacyStd"].mean())
                ranking["Rank"] = range(1, len(ranking) + 1)
                processed["generator_ranking"] = ranking

            save_dataset_processed(processed, dirs)

            utility = analyze_utility(util_long, pipe)
            fidelity = analyze_fidelity(fid_long, pipe)
            privacy = analyze_privacy(priv_long, priv_detail, pipe)
            stats = _run_dataset_statistics(utility.get("classification_stats", pd.DataFrame()))

            for name, df in stats.get("tests", {}).items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_csv(dirs["statistics"] / f"{name}.csv", index=False)
            if not stats["friedman_ranks"].empty:
                stats["friedman_ranks"].to_csv(dirs["statistics"] / "friedman_ranks.csv", index=False)
            if not stats["cd_groups"].empty:
                stats["cd_groups"].to_csv(dirs["statistics"] / "cd_groups.csv", index=False)

            seed = seed_stability_analysis(util_long, pipe)

            tables = {
                "generator_summary": build_generator_summary_table(processed.get("generator_ranking", pd.DataFrame())),
                "overall_scores": processed.get("overall_scores", pd.DataFrame()),
                "utility_scores": processed.get("utility_scores", pd.DataFrame()),
                "fidelity_scores": processed.get("fidelity_scores", pd.DataFrame()),
                "privacy_scores": processed.get("privacy_scores", pd.DataFrame()),
                "omnibus_tests": stats.get("tests", {}).get("omnibus", pd.DataFrame()),
                "pairwise_wilcoxon": stats.get("tests", {}).get("pairwise_wilcoxon", pd.DataFrame()),
                "effect_sizes": stats.get("tests", {}).get("effect_sizes", pd.DataFrame()),
                "seed_stability": seed.get("seed_stability_ranks", pd.DataFrame()),
            }
            export_dataset_tables(tables, dirs["tables"], dirs["tables"])

            ds_master = {
                "utility_long": util_long,
                "fidelity_long": fid_long,
                "privacy_long": priv_long,
            }
            figs = generate_dataset_figures(
                ds_key, processed, utility, ds_master, stats, dirs, tdc
            )
            all_figures[ds_key] = figs
            all_processed[ds_key] = processed
            comparative_rows.append(processed.get("generator_ranking", pd.DataFrame()).assign(Dataset=ds_key))

            print(f"  Figures: {len(figs)} groups → {dirs['figures']}")

        print("\n[Comparison] Cross-dataset analysis...")
        comp_dirs = tdc.comparison_dirs()
        cancer_rank = all_processed.get("Cancer", {}).get("generator_ranking", pd.DataFrame())
        mushroom_rank = all_processed.get("Mushroom", {}).get("generator_ranking", pd.DataFrame())
        comp_table = build_comparative_table(cancer_rank, mushroom_rank)
        if not comp_table.empty:
            comp_table.to_csv(comp_dirs["tables"] / "cancer_vs_mushroom.csv", index=False)
            comp_table.to_excel(comp_dirs["tables"] / "cancer_vs_mushroom.xlsx", index=False)
            try:
                comp_table.to_latex(comp_dirs["tables"] / "cancer_vs_mushroom.tex", index=False)
            except Exception:
                pass

        comp_figs = generate_comparison_figures(all_processed, comp_dirs, tdc)
        all_figures["Comparison"] = comp_figs

        summary_dir = tdc.summary_dir()
        summary = {
            "runtime_seconds": round(time.time() - t0, 2),
            "datasets": DISPLAY_NAMES,
            "figures": {k: len(v) for k, v in all_figures.items()},
            "weights": {
                "utility": tdc.utility_weight,
                "fidelity": tdc.fidelity_weight,
                "privacy": tdc.privacy_weight,
            },
            "cancer_top_generator": cancer_rank.iloc[0]["Generator"] if not cancer_rank.empty else None,
            "mushroom_top_generator": mushroom_rank.iloc[0]["Generator"] if not mushroom_rank.empty else None,
        }
        (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if comparative_rows:
            pd.concat(comparative_rows, ignore_index=True).to_csv(
                summary_dir / "combined_rankings.csv", index=False
            )

        md_lines = [
            "# Two-Dataset Case Study Summary",
            "",
            f"**Runtime:** {summary['runtime_seconds']}s",
            "",
            "## Top Generators",
            f"- **Cancer:** {summary['cancer_top_generator']}",
            f"- **Mushroom:** {summary['mushroom_top_generator']}",
            "",
            "## Output",
            f"- Cancer: `{tdc.output_root / 'Cancer'}`",
            f"- Mushroom: `{tdc.output_root / 'Mushroom'}`",
            f"- Comparison: `{tdc.output_root / 'Comparison'}`",
        ]
        (summary_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

        print(f"\nComplete in {summary['runtime_seconds']}s")
        print(f"  Output: {tdc.output_root}")
        print("=" * 60)

        self.results = {
            "master": master,
            "processed": all_processed,
            "figures": all_figures,
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
