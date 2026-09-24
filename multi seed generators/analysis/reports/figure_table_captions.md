# Manuscript Captions — 10-Seed Evaluation

## General phrasing

Results are reported as mean ± SD across **10 independent generator-training seeds** using the same leakage-safe train/test partition (split seed = 42).

The 10 generator-training seeds were executed in three computational batches: three seeds in Batch 1 (42, 123, 2024), three seeds in Batch 2 (68, 91, 2025), and four seeds in Batch 3 (55, 155, 255, 355). Batches are execution groups only; primary results aggregate all 10 seed-level observations.

## Workflow / experimental design

**Figure (workflow).** Leakage-safe pipeline: raw data → train/test split first → imputer fit on REAL-TRAIN only → eight generators trained on REAL-TRAIN with 10 independent generator-training seeds (Batches 1–3) → fidelity, utility, and privacy evaluation on REAL-TEST → Mean ± SD across the 10 seeds. For OASIS (Alzheimer’s), participant-level splitting ensures no participant ID overlap between train and test.

**Figure (experimental design).** Relationship among 15 datasets, 8 generators, and 10 seeds grouped into three execution batches (3 + 3 + 4). Batches are computational provenance groups, not separate scientific experiments. Total generator runs = 1,200.

## Critical-difference diagrams

Critical-difference diagram of generator rankings by utility gap (TRTR − TSTR). Within each dataset, generators were ranked by the **10-seed mean** gap (lower is better). Friedman omnibus and Nemenyi post-hoc tests (α = 0.05) were applied across datasets; generators joined by a red bar are not significantly different.

## Rank tables / robustness

Average / median / SD of per-dataset ranks (from 10-seed mean gaps). Lower average rank indicates smaller utility gaps. Robustness charts show mean rank ± SD across datasets.

## Seed-stability figures

Each distribution (or error bar) represents results from **10 independent generator-training seeds** for the same dataset × generator configuration (not 10 datasets or 10 participants).

## Trade-off figures

Points show the 10-seed mean for each generator. Where shown, error bars are the sample SD (ddof = 1) across the 10 seeds.

## Heatmaps / dataset-level panels

Cell values (or bar heights) are 10-seed means for the indicated metric. Dataset-level mean±SD panels compare the eight generators within one dataset.

## Supplementary complete-seed table

Per-metric values for all 10 seeds plus mean and sample SD, enabling full reproducibility of the aggregated Mean ± SD results.

## Main figure pack

Curated copies live in `figures/main/` (see `MANIFEST.md`). Supplementary curated copies live in `figures/supplementary/`.
