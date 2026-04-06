"""
Dataset Builder — assembles all per-seed mappings into a single Parquet file
ready for DPO / PPO RLHF training.

WHY THIS EXISTS
---------------
After running the generator you have 120 mapping files spread across 6 task
folders. No trainer can consume that directly. This script:

  1. Walks every data/{task_id}/mappings/seed_N.json
  2. Calls the fix synthesizer → chosen_code + rejected_code
  3. Builds a prompt string for each episode (what the LLM sees)
  4. Grades dirty vs clean to get initial_reward
  5. Grades after chosen/rejected code to get final rewards
  6. Packs everything into one DataFrame → data/dataset.parquet

One row in the parquet = one complete DPO training example.

HOW THE PROMPT IS BUILT
-----------------------
The prompt matches what the live environment shows the agent in inference.py.
It includes the task description + a data summary of the dirty CSV.
This ensures the trained model's knowledge transfers directly to live episodes.

HOW REWARDS ARE COMPUTED
------------------------
We import the grader from server/grader.py directly. For each episode:
  initial_reward    → grade(dirty_df)        before any fixes
  reward_if_chosen  → grade(result after chosen_code runs)
  reward_if_rejected→ grade(result after rejected_code runs)

The reward gap (reward_if_chosen - reward_if_rejected) shows how much better
the correct fix is. A large gap = strong DPO training signal.

Usage:
    python tools/dataset_builder.py
    python tools/dataset_builder.py --data-dir data --out data/dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
# We need to import from two sibling directories:
#   tools/fix_synthesizer.py  (our module)
#   server/grader.py          (existing grader)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))

from fix_synthesizer import synthesize_chosen, synthesize_rejected

try:
    from grader import grade
except ImportError:
    grade = None  # grader unavailable → rewards will be None


# ── Prompt builder ────────────────────────────────────────────────────────────


def _data_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Concise text summary of the DataFrame — shape, dtypes, nulls, sample rows.
    This is exactly what the live environment shows the agent, so training
    on this format means the model is ready for real episodes.
    """
    buf = StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write("Columns and types:\n")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_info = f" ({null_count} nulls)" if null_count > 0 else ""
        buf.write(f"  {col}: {df[col].dtype}{null_info}\n")
    buf.write(f"\nSample rows (first {max_rows}):\n")
    buf.write(df.head(max_rows).to_string(index=False))
    return buf.getvalue()


def build_prompt(task_description: str, dirty_df: pd.DataFrame) -> str:
    """
    Build the prompt string that an LLM agent would receive at the start
    of a live episode.

    Format:
        Task: <description>

        Data summary:
        <shape, dtypes, nulls, sample rows>

    This is intentionally simple — no constraint lists (the new engine doesn't
    store constraints in task configs). The model learns from data signals alone.
    """
    return (
        f"Task: {task_description}\n\n"
        f"Data summary:\n{_data_summary(dirty_df)}"
    )


# ── Code executor ─────────────────────────────────────────────────────────────


def _run_code(code: str, dirty_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Safely execute synthesized fix code on a copy of dirty_df.

    Uses exec() in an isolated local namespace — this is acceptable for an
    offline data generation tool (not a production server). The sandbox.py
    in the server uses subprocess + AST checks for production safety.

    Returns the resulting df, or None if the code raised an exception.
    """
    local_ns: dict[str, Any] = {"df": dirty_df.copy(), "pd": pd, "np": None}
    try:
        import numpy as np
        local_ns["np"] = np
        exec(code, {}, local_ns)  # noqa: S102
        result = local_ns.get("df")
        if isinstance(result, pd.DataFrame):
            return result
    except Exception:
        pass
    return None


# ── Reward computation ────────────────────────────────────────────────────────


def _compute_reward(
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame | None,
    error_map: dict[str, Any],
    steps: int,
    min_steps: int,
    max_steps: int,
) -> float | None:
    """
    Call the grader for a single (result, error_map) pair.
    Returns a float in [0.0, 1.0] or None if grader is unavailable.
    """
    if grade is None or result_df is None:
        return None
    try:
        _, reward = grade(
            clean_df=clean_df,
            result_df=result_df,
            error_map=error_map,
            transform_steps=steps,
            min_transform_steps=min_steps,
            max_transform_steps=max_steps,
        )
        return round(reward, 4)
    except Exception:
        return None


# ── Single episode builder ────────────────────────────────────────────────────


def build_episode(
    mapping_path: Path,
    task_configs: dict[str, dict],
) -> dict[str, Any] | None:
    """
    Build one row of the training dataset from a single seed mapping file.

    Steps:
      1. Load the mapping (task_id, seed, error_map, severity_map)
      2. Load clean + dirty CSVs
      3. Synthesize chosen and rejected code
      4. Build the prompt
      5. Compute initial_reward, reward_if_chosen, reward_if_rejected
      6. Return a flat dict → becomes one DataFrame row

    Returns None if any required file is missing.
    """
    # 1. Load mapping
    try:
        mapping = json.load(open(mapping_path))
    except Exception as e:
        print(f"  SKIP {mapping_path}: {e}")
        return None

    task_id: str = mapping["task_id"]
    seed: int = mapping["seed"]
    error_map: dict = mapping["error_map"]
    severity_map: dict = mapping["severity_map"]

    # 2. Load CSVs
    clean_path = Path(mapping["clean_path"])
    dirty_path = Path(mapping["dirty_path"])

    if not clean_path.exists() or not dirty_path.exists():
        print(f"  SKIP {task_id}/seed_{seed}: CSV files missing")
        return None

    clean_df = pd.read_csv(clean_path)
    dirty_df = pd.read_csv(dirty_path)

    # 3. Synthesize fix code from the mutation map
    chosen_code = synthesize_chosen(error_map, clean_df)
    rejected_code = synthesize_rejected(error_map, clean_df)

    # 4. Build the prompt (what the LLM sees during training)
    task_cfg = task_configs.get(task_id, {})
    description = task_cfg.get("description", f"Clean the {task_id} dataset.")
    prompt = build_prompt(description, dirty_df)

    # 5. Compute rewards — grade dirty (initial) and after each fix attempt
    min_steps = task_cfg.get("min_transform_steps", 1)
    max_steps = task_cfg.get("max_transform_steps", 10)

    initial_reward = _compute_reward(
        clean_df, dirty_df, error_map,
        steps=0, min_steps=min_steps, max_steps=max_steps,
    )

    chosen_result_df = _run_code(chosen_code, dirty_df)
    reward_if_chosen = _compute_reward(
        clean_df, chosen_result_df, error_map,
        steps=min_steps, min_steps=min_steps, max_steps=max_steps,
    )

    rejected_result_df = _run_code(rejected_code, dirty_df)
    reward_if_rejected = _compute_reward(
        clean_df, rejected_result_df, error_map,
        steps=min_steps, min_steps=min_steps, max_steps=max_steps,
    )

    # 6. Pack into a flat dict
    return {
        "episode_id": f"{task_id}_seed{seed}_{uuid.uuid4().hex[:6]}",
        "task_id": task_id,
        "seed": seed,
        "clean_path": str(clean_path),
        "dirty_path": str(dirty_path),
        "mapping_path": str(mapping_path),
        # Training columns — what DPOTrainer reads
        "prompt": prompt,
        "chosen": chosen_code,
        "rejected": rejected_code,
        # Reward columns — what PPO reward model uses
        "initial_reward": initial_reward,
        "reward_if_chosen": reward_if_chosen,
        "reward_if_rejected": reward_if_rejected,
        "reward_gap": (
            round(reward_if_chosen - reward_if_rejected, 4)
            if reward_if_chosen is not None and reward_if_rejected is not None
            else None
        ),
        # Metadata — for filtering and analysis
        "total_cell_errors": len(error_map.get("cell_errors", {})),
        "total_spurious_rows": len(error_map.get("spurious_rows", {})),
        "total_severity": severity_map.get("total_severity", 0.0),
        "corruptions_present": json.dumps(
            list(severity_map.get("by_corruption", {}).keys())
        ),
    }


# ── Main assembler ────────────────────────────────────────────────────────────


def build_dataset(
    data_dir: Path = Path("data"),
    out_path: Path = Path("data/dataset.parquet"),
) -> pd.DataFrame:
    """
    Walk all data/{task_id}/mappings/seed_*.json files, build one episode
    per file, and save the assembled DataFrame as a Parquet file.

    The Parquet file is the final deliverable — plug it straight into
    HuggingFace DPOTrainer or PPOTrainer.
    """
    # Load all task configs once (to get descriptions + step budgets)
    task_configs: dict[str, dict] = {}
    tasks_dir = ROOT / "tasks"
    if tasks_dir.exists():
        for p in tasks_dir.glob("*.json"):
            try:
                cfg = json.load(open(p))
                task_configs[cfg["task_id"]] = cfg
            except Exception:
                pass

    # Find all mapping files
    mapping_files = sorted(data_dir.glob("*/mappings/seed_*.json"))
    if not mapping_files:
        print(f"No mapping files found under {data_dir}/")
        print("Run: python tools/generator.py --n-seeds 20")
        return pd.DataFrame()

    print(f"Found {len(mapping_files)} mapping files across "
          f"{len(set(p.parent.parent.name for p in mapping_files))} tasks\n")

    rows: list[dict] = []
    for mp in mapping_files:
        task_id = mp.parent.parent.name
        seed = mp.stem.replace("seed_", "")
        print(f"  [{task_id}] seed {seed:>2} ...", end=" ")
        row = build_episode(mp, task_configs)
        if row:
            rows.append(row)
            reward_gap = row.get("reward_gap")
            gap_str = f"gap={reward_gap:+.3f}" if reward_gap is not None else "gap=n/a"
            print(f"errors={row['total_cell_errors']:>3}  {gap_str}")
        else:
            print("SKIPPED")

    if not rows:
        print("No episodes built — check your data/ directory.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Save as Parquet — best format for ML pipelines + HuggingFace datasets
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n{'─'*50}")
    print(f"Dataset saved → {out_path}")
    print(f"Total rows   : {len(df)}")
    print(f"Tasks        : {df['task_id'].nunique()} ({sorted(df['task_id'].unique())})")
    print(f"Seeds        : {df['seed'].nunique()} per task")
    if "reward_gap" in df.columns and df["reward_gap"].notna().any():
        print(f"Avg reward gap (chosen - rejected): {df['reward_gap'].mean():+.4f}")
        print(f"  chosen   avg: {df['reward_if_chosen'].mean():.4f}")
        print(f"  rejected avg: {df['reward_if_rejected'].mean():.4f}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Assemble RLHF training dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/dataset.parquet"))
    args = parser.parse_args()
    build_dataset(data_dir=args.data_dir, out_path=args.out)


if __name__ == "__main__":
    main()
