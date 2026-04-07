"""
Multi-seed dataset generator for RLHF training data.

The corruption engine (engine.py) runs once with a fixed seed (42) — giving
you exactly ONE dirty version of each dataset. That's not enough to train a
model; it would just memorise that one example.

This script re-runs the same corruption logic N times, each time with a
different random seed, so each seed breaks DIFFERENT rows. You end up with N
distinct dirty variants per task — enough variety to actually learn from.

Output layout (inside data/{task_id}/):
    dirty/
        seed_0.csv        ← dirty CSV for seed 0
        seed_1.csv        ← dirty CSV for seed 1
        ...
    mappings/
        seed_0.json       ← which cells were broken and how, for seed 0
        seed_1.json       ← for seed 1
        ...

Usage:
    python tools/generator.py                        # 20 seeds, all 6 tasks
    python tools/generator.py --n-seeds 5            # 5 seeds, all tasks
    python tools/generator.py --tasks titanic_easy   # one task only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import the engine ────────────────────────────────────────────────────────
# We reuse the engine's corruption functions and task configs directly.
# No duplication — one source of truth.
sys.path.insert(0, str(Path(__file__).parent))
from corruption.engine import (
    TASK_CONFIGS,
    apply_corruptions,
    build_error_map,
    build_severity_map,
    load_titanic,
    load_wine,
)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")

# Map base_dataset name → loader function
DATASET_LOADERS = {
    "titanic": load_titanic,
    "wine_quality": load_wine,
}


# ── Core: generate one seed for one task ────────────────────────────────────

def generate_seed(
    task_config: dict,
    clean_df: pd.DataFrame,
    seed: int,
    task_dir: Path,
) -> dict:
    """
    Run the corruption pipeline once with a specific seed.

    This is the heart of the generator. It:
    1. Re-seeds the random number generator to `seed`
       → so the SAME corruption functions pick DIFFERENT rows each time
    2. Runs all corruption steps defined in the task config
       → records every cell that was changed in `error_log`
    3. Saves the dirty CSV and the mutation map to disk

    Returns a summary dict (used later by dataset_builder).

    Why re-seeding works:
        numpy's RNG decides WHICH rows to corrupt using random numbers.
        With seed=42 it picks row [27, 51, 68...].
        With seed=7  it picks row [3,  99, 201...].
        Same corruption type, completely different rows. That's variety.
    """
    # Step 1: Re-seed both numpy and Python's random module
    # We do both because some corruption functions use each
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Patch the engine's module-level RNG with our seed-specific one.
    # The corruption functions reference `RNG` from the engine module,
    # so we swap it out temporarily.
    import corruption.engine as _engine
    _engine.RNG = rng

    # Step 2: Run all corruption steps, collecting every broken cell
    error_log: list[dict] = []
    dirty_df = apply_corruptions(
        clean_df.copy(),          # always start from clean data
        task_config["corruptions"],
        error_log=error_log,      # corruption fns write into this list
    )

    # Step 3: Build the structured error map from the flat log
    # error_map groups by (row, col) and keeps highest severity if a cell
    # was corrupted more than once
    error_map = build_error_map(error_log)
    severity_map = build_severity_map(error_map)

    # Step 4: Save the dirty CSV
    dirty_dir = task_dir / "dirty"
    dirty_dir.mkdir(parents=True, exist_ok=True)
    dirty_path = dirty_dir / f"seed_{seed}.csv"
    dirty_df.to_csv(dirty_path, index=False)

    # Step 5: Save the mutation map (error_map + metadata)
    mappings_dir = task_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "task_id": task_config["task_id"],
        "seed": seed,
        "clean_path": str(task_dir / "clean.csv"),
        "dirty_path": str(dirty_path),
        "error_map": error_map,
        "severity_map": severity_map,
    }
    mapping_path = mappings_dir / f"seed_{seed}.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2, default=str)

    return {
        "task_id": task_config["task_id"],
        "seed": seed,
        "dirty_path": str(dirty_path),
        "mapping_path": str(mapping_path),

        "total_cell_errors": len(error_map["cell_errors"]),
        "total_spurious_rows": len(error_map["spurious_rows"]),
        "total_severity": severity_map["total_severity"],
    }


# ── Generate all seeds for all (or selected) tasks ──────────────────────────

def generate(n_seeds: int = 20, task_ids: list[str] | None = None) -> list[dict]:
    """
    Run the generator for all tasks × all seeds.

    Args:
        n_seeds:  How many seed variants to create per task (default 20).
        task_ids: Filter to specific tasks. None = all 6 tasks.

    Returns:
        List of summary dicts — one per (task, seed) combination.
        This becomes the manifest used by dataset_builder.
    """
    print(f"Loading clean datasets...")
    clean_datasets = {name: loader().reset_index(drop=True)
                      for name, loader in DATASET_LOADERS.items()}
    print(f"  titanic: {clean_datasets['titanic'].shape}")
    print(f"  wine_quality: {clean_datasets['wine_quality'].shape}")

    # Filter task configs if specific tasks were requested
    configs = TASK_CONFIGS
    if task_ids:
        configs = [c for c in TASK_CONFIGS if c["task_id"] in task_ids]
        if not configs:
            print(f"ERROR: No matching tasks found for {task_ids}")
            print(f"Available: {[c['task_id'] for c in TASK_CONFIGS]}")
            return []

    all_summaries: list[dict] = []

    for config in configs:
        task_id = config["task_id"]
        task_dir = DATA_DIR / task_id

        # Ensure clean.csv exists (engine creates it; we just verify)
        clean_csv = task_dir / "clean.csv"
        if not clean_csv.exists():
            print(f"  WARNING: {clean_csv} missing — run engine.py first")
            continue

        clean_df = clean_datasets[config["base_dataset"]]
        print(f"\n[{task_id}] Generating {n_seeds} seeds...")

        for seed in range(n_seeds):
            summary = generate_seed(config, clean_df, seed, task_dir)
            all_summaries.append(summary)
            print(
                f"  seed {seed:>2}: "
                f"{summary['total_cell_errors']} cell errors, "
                f"{summary['total_spurious_rows']} spurious rows, "
                f"severity={summary['total_severity']:.1f}"
            )

    # Save a manifest so dataset_builder knows what exists
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nManifest saved → {manifest_path}")
    print(f"Total episodes generated: {len(all_summaries)}")

    return all_summaries


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed dirty data generator")
    parser.add_argument(
        "--n-seeds", type=int, default=20,
        help="Number of seed variants per task (default: 20)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Specific task IDs to generate (default: all tasks)"
    )
    args = parser.parse_args()
    generate(n_seeds=args.n_seeds, task_ids=args.tasks)


if __name__ == "__main__":
    main()
