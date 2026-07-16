"""End-to-end orchestration of the publication analysis pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.benchmarking import run_benchmarking
from analysis.config import PipelineConfig
from analysis.correlation import hierarchical_cluster_order, metric_correlation_matrix
from analysis.data_loader import build_unified_master, load_master_data, save_master_data
from analysis.feature_fidelity import load_feature_fidelity, load_quality_from_notebooks
from analysis.fidelity_analysis import analyze_fidelity
from analysis.figures import generate_all_figures, generate_benchmark_figures
from analysis.leakage_analysis import analyze_leakage
from analysis.notebook_metrics import load_notebook_metrics_long
from analysis.publication_figures import generate_publication_figures
from analysis.privacy_analysis import analyze_privacy
from analysis.rankings import compile_rankings
from analysis.scores import build_processed_scores
from analysis.similarity import analyze_similarity
from analysis.statistics import run_generator_comparison_tests
from analysis.tables import build_comparison_table, export_tables
from analysis.tradeoff import analyze_tradeoffs
from analysis.utility_analysis import analyze_utility

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class AnalysisPipeline:
    config: PipelineConfig = field(default_factory=PipelineConfig)
    results: dict[str, Any] = field(default_factory=dict)
    build_dashboard: bool = False

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        dirs = self.config.subdirs()
        print("=" * 60)
        print("SYNTH Benchmark — Journal Paper Analysis Pipeline")
        print("=" * 60)

        # Step 1: Merge all Excel files
        print("\n[Step 1/6] Discovering and merging Excel files...")
        master = load_master_data(self.config)

        # Enrich fidelity/privacy from generator notebooks (no hardcoded filenames)
        nb_fid, nb_priv = load_notebook_metrics_long(self.config)
        notebook_quality = load_quality_from_notebooks(self.config)
        for name, frame in [
            ("fidelity", nb_fid),
            ("fidelity_q", notebook_quality),
            ("privacy", nb_priv),
        ]:
            if frame.empty:
                continue
            if name.startswith("fidelity"):
                master.fidelity_long = pd.concat(
                    [master.fidelity_long, frame], ignore_index=True
                ).drop_duplicates(subset=["Dataset", "Generator", "Metric"], keep="last")
            else:
                combined = pd.concat(
                    [master.privacy_long, frame], ignore_index=True
                )
                # Excel Hungarian summaries are authoritative for Mean_Distance;
                # notebooks fill gaps and still override other privacy metrics.
                mean_d = combined[combined["Metric"] == "Mean_Distance"].drop_duplicates(
                    subset=["Dataset", "Generator", "Metric"], keep="first"
                )
                other = combined[combined["Metric"] != "Mean_Distance"].drop_duplicates(
                    subset=["Dataset", "Generator", "Metric"], keep="last"
                )
                master.privacy_long = pd.concat([mean_d, other], ignore_index=True)

        save_master_data(master, dirs["master"])
        unified = build_unified_master(master)
        log.info(
            "Merged %d Excel files → utility=%s fidelity=%s privacy=%s unified=%s",
            len(master.file_inventory),
            f"{len(master.utility_long):,}",
            f"{len(master.fidelity_long):,}",
            f"{len(master.privacy_long):,}",
            f"{len(unified):,}",
        )

        # Steps 2–5: Category scores and overall ranking
        print("\n[Steps 2–5/6] Utility, Fidelity, Privacy & Overall scores...")
        processed = build_processed_scores(
            master.utility_long, master.fidelity_long, master.privacy_long, self.config
        )
        proc_dir = dirs["processed"]
        for name, df in processed.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(proc_dir / f"{name}.csv", index=False)
                df.to_excel(proc_dir / f"{name}.xlsx", index=False)

        # Domain-specific analysis + statistics
        print("\n[Step 6/6] Statistical analysis, figures, and tables...")
        utility = analyze_utility(master.utility_long, self.config)
        fidelity = analyze_fidelity(master.fidelity_long, self.config)
        privacy = analyze_privacy(master.privacy_long, master.privacy_detail, self.config)

        util_for_stats = utility.get("classification_stats", pd.DataFrame())
        if util_for_stats.empty:
            util_for_stats = utility.get("regression_stats", pd.DataFrame())
        if not util_for_stats.empty:
            util_for_stats = util_for_stats[util_for_stats["EvaluationType"] == "TSTR"].copy()
        statistics = run_generator_comparison_tests(
            util_for_stats,
            value_col="Mean",
            block_col="Dataset",
            treatment_col="Generator",
            metric_col="Metric",
        )

        rankings = compile_rankings(utility, privacy, fidelity, self.config)
        tradeoff = analyze_tradeoffs(utility, privacy, fidelity)

        combined_for_corr = processed.get("overall_scores", pd.DataFrame())
        if not combined_for_corr.empty:
            corr_input = combined_for_corr.rename(
                columns={
                    "UtilityScore": "Utility",
                    "FidelityScore": "Fidelity",
                    "PrivacyScore": "Privacy",
                    "OverallScore": "Overall",
                }
            )
            corr_input = corr_input.melt(
                id_vars=["Dataset", "Generator", "LeakageLevel"],
                value_vars=["Utility", "Fidelity", "Privacy", "Overall"],
                var_name="Metric",
                value_name="Mean",
            )
            correlation = metric_correlation_matrix(corr_input, index_cols=["Dataset", "Generator"])
        else:
            correlation = {"pearson": pd.DataFrame(), "spearman": pd.DataFrame(), "kendall": pd.DataFrame()}

        cluster_order = hierarchical_cluster_order(correlation.get("spearman", pd.DataFrame()))
        leakage = analyze_leakage(master.utility_long, self.config)
        similarity = analyze_similarity(master.privacy_detail)
        feature_fidelity = load_feature_fidelity(self.config)
        benchmarking = run_benchmarking(
            utility, privacy, fidelity, master.utility_long, statistics, self.config
        )

        tables = {
            "utility_classification": build_comparison_table(utility.get("classification_stats", pd.DataFrame())),
            "utility_regression": build_comparison_table(utility.get("regression_stats", pd.DataFrame())),
            "fidelity_scores": build_comparison_table(fidelity.get("fidelity_stats", pd.DataFrame())),
            "privacy_scores": build_comparison_table(privacy.get("privacy_stats", pd.DataFrame())),
            "overall_scores": processed.get("overall_scores", pd.DataFrame()),
            "generator_ranking": processed.get("generator_ranking", pd.DataFrame()),
            "weighted_rankings": rankings.get("weighted_overall", pd.DataFrame()),
            "omnibus_tests": statistics.get("omnibus", pd.DataFrame()),
            "composite_scores": benchmarking.get("composite_overall", pd.DataFrame()),
            "domain_correlations": benchmarking.get("domain_correlations", pd.DataFrame()),
            "seed_stability": benchmarking.get("seed_stability_ranks", pd.DataFrame()),
        }
        export_tables(tables, dirs["tables"], dirs["latex"])

        self.results = {
            "master": master,
            "unified": unified,
            "processed": processed,
            "utility": utility,
            "fidelity": fidelity,
            "privacy": privacy,
            "statistics": statistics,
            "rankings": rankings,
            "tradeoff": tradeoff,
            "correlation": correlation,
            "correlation_cluster_order": cluster_order,
            "leakage": leakage,
            "similarity": similarity,
            "benchmarking": benchmarking,
            "feature_fidelity": feature_fidelity,
            "tables": tables,
        }

        print("  Generating publication figures (Figures 1–14 + supplementary)...")
        figure_list: list[str] = []
        figure_list.extend(generate_publication_figures(self.results, dirs, self.config))
        figure_list.extend(generate_all_figures(self.results, dirs, self.config))
        figure_list.extend(generate_benchmark_figures(self.results, dirs, self.config))
        log.info("Generated %d figure groups", len(figure_list))

        # Supplementary exports
        summary = {
            "runtime_seconds": round(time.time() - t0, 2),
            "n_excel_files": len(master.file_inventory),
            "n_unified_rows": len(unified),
            "figures_generated": figure_list,
            "weights": {
                "utility": self.config.utility_weight,
                "privacy": self.config.privacy_weight,
                "fidelity": self.config.fidelity_weight,
            },
        }
        (dirs["supplementary"] / "pipeline_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        for name, data in [
            ("master_unified", unified),
            ("utility_classification_stats", utility.get("classification_stats")),
            ("utility_regression_stats", utility.get("regression_stats")),
            ("fidelity_stats", fidelity.get("fidelity_stats")),
            ("privacy_stats", privacy.get("privacy_stats")),
            ("pairwise_wilcoxon", statistics.get("pairwise_wilcoxon")),
            ("effect_sizes", statistics.get("effect_sizes")),
            ("overall_scores", processed.get("overall_scores")),
            ("generator_ranking", processed.get("generator_ranking")),
            ("domain_correlations", benchmarking.get("domain_correlations")),
            ("feature_fidelity_long", feature_fidelity),
        ]:
            if isinstance(data, pd.DataFrame) and not data.empty:
                data.to_csv(dirs["supplementary"] / f"{name}.csv", index=False)

        print(f"\nPipeline complete in {summary['runtime_seconds']}s")
        print(f"  Master data:    {dirs['master']}")
        print(f"  Processed data: {dirs['processed']}")
        print(f"  Figures:        {self.config.output_dir / 'Figures'}")
        print(f"  Tables:         {dirs['tables']}")
        print("=" * 60)

        if self.build_dashboard:
            try:
                from dashboard.build_pages import build_pages
                build_pages()
                print("  Dashboard built in docs/")
            except Exception as exc:
                log.warning("Dashboard build skipped: %s", exc)

        return self.results


def run_pipeline(
    output_dir: str | Path | None = None,
    utility_weight: float = 0.40,
    privacy_weight: float = 0.30,
    fidelity_weight: float = 0.30,
    build_dashboard: bool = False,
) -> dict[str, Any]:
    config = PipelineConfig(
        utility_weight=utility_weight,
        privacy_weight=privacy_weight,
        fidelity_weight=fidelity_weight,
    )
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    return AnalysisPipeline(config=config, build_dashboard=build_dashboard).run()
