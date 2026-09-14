"""Core multi-seed experiment runner with resume + failure logging.

Results are stored under:
  classification/results/...   for classification datasets
  regression/results/...       for regression datasets
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config_loader import dataset_by_id, generator_names, load_datasets, load_seeds
from .datasets import load_and_split
from .generators import generate_synthetic
from .metrics.fidelity import compute_fidelity, fidelity_to_rows
from .metrics.privacy import compute_privacy, privacy_to_rows
from .metrics.utility import compute_utility, utility_to_rows
from .paths import ensure_task_dirs, task_paths


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_for(dataset_id: str) -> str:
    return dataset_by_id(dataset_id)["task"]


def _paths_for(dataset_id: str) -> dict[str, Path]:
    return ensure_task_dirs(_task_for(dataset_id))


def _run_dir(seed: int, dataset_id: str, generator: str) -> Path:
    raw = _paths_for(dataset_id)["raw"]
    d = raw / f"seed_{seed}" / dataset_id / generator
    d.mkdir(parents=True, exist_ok=True)
    return d


def _result_path(seed: int, dataset_id: str, generator: str) -> Path:
    return _run_dir(seed, dataset_id, generator) / "metrics.csv"


def _meta_path(seed: int, dataset_id: str, generator: str) -> Path:
    return _run_dir(seed, dataset_id, generator) / "run_meta.json"


def is_complete(seed: int, dataset_id: str, generator: str) -> bool:
    p = _result_path(seed, dataset_id, generator)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
        return len(df) > 0 and "metric_value" in df.columns
    except Exception:
        return False


def append_log(dataset_id: str, row: dict[str, Any]) -> None:
    logs = _paths_for(dataset_id)["logs"]
    log_path = logs / "experiment_log.csv"
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def run_one(
    dataset_id: str,
    generator: str,
    seed: int,
    *,
    force: bool = False,
    utility_classifier_seeds: Optional[list[int]] = None,
) -> str:
    """Run one dataset × generator × seed. Returns SUCCESS | FAILED | SKIPPED."""
    seeds_cfg = load_seeds()
    split_seed = int(seeds_cfg["split_seed"])
    test_size = float(seeds_cfg["test_size"])
    utility_classifier_seeds = utility_classifier_seeds or list(range(42, 52))

    if not force and is_complete(seed, dataset_id, generator):
        append_log(
            dataset_id,
            {
                "dataset": dataset_id,
                "generator": generator,
                "seed": seed,
                "status": "SKIPPED",
                "start_time": _utc_now(),
                "end_time": _utc_now(),
                "error": "",
            },
        )
        return "SKIPPED"

    start = _utc_now()
    t0 = time.perf_counter()
    try:
        cfg = dataset_by_id(dataset_id)
        task = cfg["task"]
        paths = ensure_task_dirs(task)

        split = load_and_split(cfg, test_size=test_size, split_seed=split_seed, use_cache=True)
        train, test, cfg = split["train"], split["test"], split["cfg"]
        target = cfg["target"]
        n_samples = int(cfg.get("n_samples", 1000))

        assert id(train) != id(test)
        assert len(train) > 0 and len(test) > 0

        t_train0 = time.perf_counter()
        synth = generate_synthetic(generator, train, target, n_samples, seed, task)
        training_time = time.perf_counter() - t_train0
        generation_time = 0.0

        t_eval0 = time.perf_counter()
        fid = compute_fidelity(train, synth)
        util = compute_utility(train, test, synth, target, task, utility_classifier_seeds)
        priv = compute_privacy(train, synth)
        evaluation_time = time.perf_counter() - t_eval0
        total_time = time.perf_counter() - t0

        base_meta = {
            "dataset": cfg["name"],
            "dataset_id": cfg["id"],
            "generator": generator,
            "seed": seed,
            "split_seed": split_seed,
            "n_synthetic_samples": n_samples,
            "n_train": len(train),
            "n_test": len(test),
            "task": task,
            "training_time_seconds": training_time,
            "generation_time_seconds": generation_time,
            "evaluation_time_seconds": evaluation_time,
            "total_time_seconds": total_time,
        }
        rows = []
        rows += fidelity_to_rows(fid, base_meta)
        rows += utility_to_rows(util, base_meta)
        rows += privacy_to_rows(priv, base_meta)
        for k in [
            "training_time_seconds",
            "generation_time_seconds",
            "evaluation_time_seconds",
            "total_time_seconds",
        ]:
            rows.append(
                {
                    **base_meta,
                    "metric_category": "Compute",
                    "metric_name": k,
                    "metric_value": base_meta[k],
                }
            )

        out_csv = _result_path(seed, dataset_id, generator)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        with _meta_path(seed, dataset_id, generator).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **base_meta,
                    "split_meta": split["meta"],
                    "status": "SUCCESS",
                    "timestamp": _utc_now(),
                },
                f,
                indent=2,
                default=str,
            )

        # Task-level long raw file
        global_raw = paths["raw"] / "multi_seed_raw_results.csv"
        rdf = pd.DataFrame(rows)
        if global_raw.exists():
            old = pd.read_csv(global_raw)
            mask = (
                ~(
                    (old["dataset_id"] == dataset_id)
                    & (old["generator"] == generator)
                    & (old["seed"] == seed)
                )
                if "dataset_id" in old.columns
                else pd.Series([True] * len(old))
            )
            old = old[mask]
            pd.concat([old, rdf], ignore_index=True).to_csv(global_raw, index=False)
        else:
            rdf.to_csv(global_raw, index=False)

        append_log(
            dataset_id,
            {
                "dataset": dataset_id,
                "generator": generator,
                "seed": seed,
                "status": "SUCCESS",
                "start_time": start,
                "end_time": _utc_now(),
                "error": "",
            },
        )
        return "SUCCESS"
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        fail_path = _run_dir(seed, dataset_id, generator) / "error.txt"
        fail_path.write_text(err, encoding="utf-8")
        append_log(
            dataset_id,
            {
                "dataset": dataset_id,
                "generator": generator,
                "seed": seed,
                "status": "FAILED",
                "start_time": start,
                "end_time": _utc_now(),
                "error": str(e)[:2000],
            },
        )
        return "FAILED"


def iter_jobs(
    datasets: Optional[list[str]] = None,
    generators: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    task: Optional[str] = None,
) -> list[tuple[str, str, int]]:
    seeds_cfg = load_seeds()
    all_seeds = seeds or list(seeds_cfg["generator_seeds"])
    all_ds_cfg = load_datasets()
    if task:
        t = task.lower()
        all_ds_cfg = [d for d in all_ds_cfg if d["task"].lower().startswith(t[:5])]
    all_ds = datasets or [d["id"] for d in all_ds_cfg]
    all_gen = generators or generator_names()
    jobs = []
    for ds in all_ds:
        for gen in all_gen:
            for seed in all_seeds:
                jobs.append((ds, gen, int(seed)))
    return jobs


def run_experiment(
    datasets: Optional[list[str]] = None,
    generators: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    force: bool = False,
    task: Optional[str] = None,
) -> pd.DataFrame:
    # Ensure both task trees exist
    ensure_task_dirs("classification")
    ensure_task_dirs("regression")
    jobs = iter_jobs(datasets, generators, seeds, task=task)
    rows = []
    for ds, gen, seed in jobs:
        print(f">>> {ds} | {gen} | seed={seed}")
        status = run_one(ds, gen, seed, force=force)
        print(f"    -> {status}")
        rows.append({"dataset": ds, "generator": gen, "seed": seed, "status": status})
    return pd.DataFrame(rows)
