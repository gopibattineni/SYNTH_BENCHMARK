# Multi-seed generators — Batch 1

Isolated re-run of the **same SYNTH five-stage benchmark** (`fig1`), with:

1. **Leakage-safe Stage 1** — train/test split **before** imputation; imputer fitted on **Real-Train only**
2. **3 generator-training seeds** — `[42, 123, 2024]` → Mean ± SD

**Original SYNTH notebooks / results are not modified.**

See:

* `reports/workflow.md` — five-stage pipeline diagram + rules  
* `reports/leakage_audit.md` — per-dataset leakage decisions  
* `reports/experiment_summary.md` — compact design card  

---

## Manuscript reproducibility statement

> The train/test split is performed before imputation. Imputation parameters are fitted exclusively on Real-Train and then applied to Real-Test, preventing information leakage from the held-out test data.

> For OASIS, participants are assigned entirely to either Real-Train or Real-Test so that no participant appears in both partitions.

> The same fixed split is reused for generator seeds 42, 123, and 2024 so that reported Mean ± SD reflects generator stochasticity, not split variability.

---

## Design card

| Item | Value |
|---|---|
| Datasets | 15 |
| Generators | 8 |
| Seeds | 42, 123, 2024 |
| Runs | **360** |
| Split seed | 42 |
| Aggregation | mean ± SD (`ddof=1`) |

---

## How to run

```bash
cd "SYNTH/multi seed generators/1st batch"

python run_experiment.py --validate-only
python run_experiment.py --smoke      # Cancer × CTGAN × 3 seeds
python run_experiment.py --all        # resumable 360
python run_experiment.py --aggregate-only
```

Completed jobs are skipped automatically (`results/raw/seed_*/.../metrics.csv`). Use `--force` to overwrite.

---

## Folder map

```text
multi seed generators/1st batch/
├── classification/          ← 9 classification datasets (216 runs)
│   ├── results/raw|aggregated|logs
│   ├── figures/
│   ├── reports/
│   └── cache/splits/
├── regression/              ← 6 regression datasets (144 runs)
│   ├── results/raw|aggregated|logs
│   ├── figures/
│   ├── reports/
│   └── cache/splits/
├── config/                  shared
├── src/                     shared Stage 1–5 modules
├── notebooks/
├── reports/                 workflow + leakage audit
└── run_experiment.py
```

**Why Cancer earlier?** Only used as a *smoke test* (small/fast). Full run covers all 15 datasets.

## Reused (read-only)

* `hive/datasets.json`, `docs/data/meta.json`
* SDV, `CTAB-GAN-Plus/`, `Diffusion GANs/diffusion_generators.py`
* Existing metric naming from `Results/MasterData/*_long.csv`
