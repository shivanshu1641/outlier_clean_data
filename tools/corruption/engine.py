"""
Synthetic corruption engine — standalone tool for generating dirty datasets.

NOT part of the OpenEnv deployment. Run once to produce data/ artifacts.

Each task gets 4 artifacts:
  clean.csv       — ground truth
  dirty.csv       — corrupted version
  error_map.json  — cell/row errors with severity and clean value
  severity_map.json — total severity and per-corruption breakdown

Usage:
    python tools/corruption/engine.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

# Severity per corruption type
CORRUPTION_SEVERITY = {
    "inject_nulls": 2.0,       # high
    "type_mangle": 3.0,        # critical
    "duplicate_rows": 2.0,     # high
    "whitespace_noise": 1.0,   # medium
    "format_inconsistency": 1.0,  # medium
    "outlier_injection": 3.0,  # critical
}


# ── Corruption Functions ─────────────────────────────────────────────────────


def inject_nulls(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Randomly set cells to NaN."""
    df = df.copy()
    severity = CORRUPTION_SEVERITY["inject_nulls"]
    for col in columns:
        if col not in df.columns:
            continue
        mask = RNG.random(len(df)) < fraction
        indices = df.index[mask]
        # accepted_fill: "any" = any non-null value, "mean" = only mean,
        # "mode" = only mode, "median" = only median, "exact" = only clean value
        accepted_fill = kwargs.get("accepted_fill", "any")
        if error_log is not None:
            for idx in indices:
                val = df.at[idx, col]
                if not pd.isna(val):  # only log if cell wasn't already null
                    error_log.append({
                        "row": int(idx),
                        "col": col,
                        "type": "cell",
                        "corruption": "null_injected",
                        "clean_value": val,
                        "severity": severity,
                        "accepted_fill": accepted_fill,
                    })
        df.loc[indices, col] = np.nan
    return df


def type_mangle(
    df: pd.DataFrame,
    columns: list[str],
    error_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Mix string garbage into numeric columns."""
    df = df.copy()
    severity = CORRUPTION_SEVERITY["type_mangle"]
    garbage = ["N/A", "unknown", "##", "n/a", "-", "???", "null", "NA"]
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(object)
        n_corrupt = max(1, int(len(df) * 0.05))
        idxs = RNG.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = df.at[idx, col]
            garb = garbage[int(RNG.integers(0, len(garbage)))]
            if error_log is not None:
                error_log.append({
                    "row": int(idx),
                    "col": col,
                    "type": "cell",
                    "corruption": "type_mangled",
                    "clean_value": None if pd.isna(clean_val) else clean_val,
                    "severity": severity,
                })
            df.at[idx, col] = garb
    return df


def duplicate_rows(
    df: pd.DataFrame,
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Insert exact duplicate rows."""
    n_dupes = max(1, int(len(df) * fraction))
    severity = CORRUPTION_SEVERITY["duplicate_rows"]
    dupe_pos = RNG.choice(len(df), size=n_dupes, replace=True)
    dupes = df.iloc[dupe_pos].copy()
    result = pd.concat([df, dupes], ignore_index=True)
    if error_log is not None:
        # Log the new spurious row indices (appended at end)
        start_idx = len(df)
        for i in range(n_dupes):
            error_log.append({
                "row": start_idx + i,
                "col": None,
                "type": "spurious_row",
                "corruption": "duplicate_rows",
                "clean_value": None,
                "severity": severity,
            })
    return result


def format_inconsistency(
    df: pd.DataFrame,
    columns: list[str],
    error_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Inject inconsistent string formats (e.g., casing)."""
    df = df.copy()
    severity = CORRUPTION_SEVERITY["format_inconsistency"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        n_corrupt = max(1, int(len(df) * 0.15))
        idxs = RNG.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = df.at[idx, col]
            choice = int(RNG.integers(0, 3))
            new_val = val.upper() if choice == 0 else (val.lower() if choice == 1 else val.title())
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "row": int(idx),
                        "col": col,
                        "type": "cell",
                        "corruption": "format_inconsistency",
                        "clean_value": None if pd.isna(clean_val) else clean_val,
                        "severity": severity,
                    })
    return df


def whitespace_noise(
    df: pd.DataFrame,
    columns: list[str],
    error_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Add leading/trailing/double spaces."""
    df = df.copy()
    severity = CORRUPTION_SEVERITY["whitespace_noise"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        n_corrupt = max(1, int(len(df) * 0.1))
        idxs = RNG.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = df.at[idx, col]
            choice = int(RNG.integers(0, 3))
            if choice == 0:
                new_val = "  " + val
            elif choice == 1:
                new_val = val + "  "
            else:
                new_val = val.replace(" ", "  ", 1) if " " in val else "  " + val
            if error_log is not None:
                error_log.append({
                    "row": int(idx),
                    "col": col,
                    "type": "cell",
                    "corruption": "whitespace_noise",
                    "clean_value": None if pd.isna(clean_val) else clean_val,
                    "severity": severity,
                })
            df.at[idx, col] = new_val
    return df


def outlier_injection(
    df: pd.DataFrame,
    columns: list[str],
    error_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Inject extreme outlier values into numeric columns."""
    df = df.copy()
    severity = CORRUPTION_SEVERITY["outlier_injection"]
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        col_mean = numeric.mean()
        col_std = numeric.std()
        n_outliers = max(1, int(len(df) * 0.02))
        idxs = RNG.choice(len(df), size=n_outliers, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = df.at[idx, col]
            multiplier = float(RNG.uniform(5, 20))
            new_val = col_mean + col_std * multiplier if RNG.random() > 0.5 else col_mean - col_std * multiplier
            if error_log is not None:
                error_log.append({
                    "row": int(idx),
                    "col": col,
                    "type": "cell",
                    "corruption": "outlier_injected",
                    "clean_value": None if pd.isna(clean_val) else (float(clean_val) if isinstance(clean_val, (int, float)) else None),
                    "severity": severity,
                })
            df.at[idx, col] = new_val
    return df


# ── Corruption Pipeline ──────────────────────────────────────────────────────


DISPATCH = {
    "inject_nulls": inject_nulls,
    "type_mangle": type_mangle,
    "duplicate_rows": duplicate_rows,
    "format_inconsistency": format_inconsistency,
    "whitespace_noise": whitespace_noise,
    "outlier_injection": outlier_injection,
}


def apply_corruptions(
    df: pd.DataFrame,
    config: list[dict[str, Any]],
    error_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Apply a sequence of corruption operations, optionally tracking errors."""
    for step in config:
        fn_name = step["function"]
        kwargs = {k: v for k, v in step.items() if k != "function"}
        fn = DISPATCH[fn_name]
        df = fn(df, error_log=error_log, **kwargs)
    return df


def build_error_map(error_log: list[dict]) -> dict[str, Any]:
    """Convert the flat error log into the nested error_map schema."""
    cell_errors: dict[str, dict] = {}
    spurious_rows: dict[str, dict] = {}

    for entry in error_log:
        if entry["type"] == "cell":
            key = f"{entry['row']},{entry['col']}"
            # If same cell corrupted multiple times, keep highest severity
            if key not in cell_errors or entry["severity"] > cell_errors[key]["severity"]:
                error_entry = {
                    "severity": entry["severity"],
                    "clean_value": entry["clean_value"],
                    "corruption": entry["corruption"],
                }
                if "accepted_fill" in entry:
                    error_entry["accepted_fill"] = entry["accepted_fill"]
                cell_errors[key] = error_entry
        elif entry["type"] == "spurious_row":
            row_str = str(entry["row"])
            spurious_rows[row_str] = {"severity": entry["severity"]}

    return {
        "cell_errors": cell_errors,
        "spurious_rows": spurious_rows,
        "missing_rows": {},  # reserved for future use
    }


def build_severity_map(error_map: dict[str, Any]) -> dict[str, Any]:
    """Compute severity totals from error_map."""
    cell_errors = error_map.get("cell_errors", {})
    spurious_rows = error_map.get("spurious_rows", {})
    missing_rows = error_map.get("missing_rows", {})

    by_corruption: dict[str, float] = {}
    for info in cell_errors.values():
        c = info["corruption"]
        by_corruption[c] = by_corruption.get(c, 0.0) + info["severity"]
    for info in spurious_rows.values():
        by_corruption["duplicate_rows"] = by_corruption.get("duplicate_rows", 0.0) + info["severity"]
    for info in missing_rows.values():
        by_corruption["missing_rows"] = by_corruption.get("missing_rows", 0.0) + info["severity"]

    total = sum(
        list(by_corruption.values())
    )

    return {
        "total_severity": total,
        "total_cell_errors": len(cell_errors),
        "total_spurious_rows": len(spurious_rows),
        "total_missing_rows": len(missing_rows),
        "by_corruption": by_corruption,
    }


# ── Dataset Loaders ──────────────────────────────────────────────────────────


def load_titanic() -> pd.DataFrame:
    """Load Titanic dataset from bundled CSV or download."""
    clean_path = Path("data/clean/titanic.csv")
    if clean_path.exists():
        return pd.read_csv(clean_path)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    return df


def load_wine() -> pd.DataFrame:
    """Load Wine Quality dataset from bundled CSV or download."""
    clean_path = Path("data/clean/wine_quality.csv")
    if clean_path.exists():
        return pd.read_csv(clean_path)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    return df


# ── Task Corruption Configs ──────────────────────────────────────────────────
# Each dataset gets easy/medium/hard variants.
# Difficulty is controlled by corruption fractions and which corruption types are applied.

TASK_CONFIGS: list[dict[str, Any]] = [
    # ── Titanic ──────────────────────────────────────────────────────────────
    {
        "task_id": "titanic_easy",
        "description": (
            "Clean the Titanic passenger dataset (easy). "
            "Fix a small number of null values in Age and Embarked, "
            "and remove whitespace from Name."
        ),
        "base_dataset": "titanic",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["Age"], "fraction": 0.08},
            {"function": "whitespace_noise", "columns": ["Name"]},
        ],
        "min_transform_steps": 1,
        "max_transform_steps": 6,
    },
    {
        "task_id": "titanic_medium",
        "description": (
            "Clean the Titanic passenger dataset (medium). "
            "Fix nulls in Age and Embarked, correct type errors in Fare, "
            "and remove whitespace from Name and Ticket."
        ),
        "base_dataset": "titanic",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["Age", "Embarked"], "fraction": 0.15},
            {"function": "type_mangle", "columns": ["Fare"]},
            {"function": "whitespace_noise", "columns": ["Name", "Ticket"]},
        ],
        "min_transform_steps": 3,
        "max_transform_steps": 10,
    },
    {
        "task_id": "titanic_hard",
        "description": (
            "Clean the Titanic passenger dataset (hard). "
            "Fix heavy null injection across multiple columns, type errors, "
            "outliers in Age and Fare, format inconsistencies in Embarked, "
            "duplicate rows, and whitespace noise."
        ),
        "base_dataset": "titanic",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["Age", "Embarked", "Fare"], "fraction": 0.30},
            {"function": "type_mangle", "columns": ["Fare", "Age"]},
            {"function": "duplicate_rows", "fraction": 0.08},
            {"function": "format_inconsistency", "columns": ["Embarked", "Name"]},
            {"function": "whitespace_noise", "columns": ["Name", "Ticket"]},
            {"function": "outlier_injection", "columns": ["Age", "Fare"]},
        ],
        "min_transform_steps": 6,
        "max_transform_steps": 15,
    },
    # ── Wine ─────────────────────────────────────────────────────────────────
    {
        "task_id": "wine_easy",
        "description": (
            "Clean the Wine Quality dataset (easy). "
            "Fix null values in pH and alcohol."
        ),
        "base_dataset": "wine_quality",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["pH", "alcohol"], "fraction": 0.08},
        ],
        "min_transform_steps": 1,
        "max_transform_steps": 6,
    },
    {
        "task_id": "wine_medium",
        "description": (
            "Clean the Wine Quality dataset (medium). "
            "Remove duplicates, fix null values in pH, alcohol, and residual sugar, "
            "correct type errors in acidity columns, and fix outliers in pH."
        ),
        "base_dataset": "wine_quality",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["pH", "alcohol", "residual sugar"], "fraction": 0.12},
            {"function": "type_mangle", "columns": ["fixed acidity", "volatile acidity"]},
            {"function": "duplicate_rows", "fraction": 0.05},
            {"function": "outlier_injection", "columns": ["pH"]},
        ],
        "min_transform_steps": 3,
        "max_transform_steps": 10,
    },
    {
        "task_id": "wine_hard",
        "description": (
            "Clean the Wine Quality dataset (hard). "
            "Heavy nulls across multiple columns, type errors, duplicates, "
            "outliers in pH and alcohol, and whitespace noise in quality."
        ),
        "base_dataset": "wine_quality",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["pH", "alcohol", "residual sugar", "fixed acidity"], "fraction": 0.30},
            {"function": "type_mangle", "columns": ["fixed acidity", "volatile acidity", "citric acid"]},
            {"function": "duplicate_rows", "fraction": 0.10},
            {"function": "whitespace_noise", "columns": ["quality"]},
            {"function": "outlier_injection", "columns": ["pH", "alcohol"]},
        ],
        "min_transform_steps": 6,
        "max_transform_steps": 15,
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────


def generate_task(config: dict[str, Any], clean_df: pd.DataFrame, data_dir: Path) -> None:
    """Generate all 4 artifacts for a single task."""
    task_id = config["task_id"]
    task_dir = data_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save clean data
    clean_path = task_dir / "clean.csv"
    clean_df.to_csv(clean_path, index=False)

    # Apply corruptions, tracking errors
    error_log: list[dict] = []
    dirty_df = apply_corruptions(clean_df.copy(), config["corruptions"], error_log=error_log)

    # Save dirty data
    dirty_path = task_dir / "dirty.csv"
    dirty_df.to_csv(dirty_path, index=False)

    # Build and save error map
    error_map = build_error_map(error_log)
    with open(task_dir / "error_map.json", "w") as f:
        json.dump(error_map, f, indent=2, default=str)

    # Build and save severity map
    severity_map = build_severity_map(error_map)
    with open(task_dir / "severity_map.json", "w") as f:
        json.dump(severity_map, f, indent=2)

    # Save task config (without internal corruptions list)
    task_config = {
        "task_id": task_id,
        "description": config["description"],
        "base_dataset": config["base_dataset"],
        "min_transform_steps": config["min_transform_steps"],
        "max_transform_steps": config["max_transform_steps"],
        "explore_cost_per_step": config.get("explore_cost_per_step", 0.01),
        "explore_timeout_cost": config.get("explore_timeout_cost", 0.03),
        "clean_data_path": str(clean_path),
        "dirty_data_path": str(dirty_path),
        "error_map_path": str(task_dir / "error_map.json"),
        "severity_map_path": str(task_dir / "severity_map.json"),
    }

    tasks_dir = Path("tasks")
    tasks_dir.mkdir(exist_ok=True)
    with open(tasks_dir / f"task_{task_id}.json", "w") as f:
        json.dump(task_config, f, indent=2)

    print(f"  [{task_id}] clean={clean_df.shape}, dirty={dirty_df.shape}, "
          f"errors={len(error_map['cell_errors'])} cells + {len(error_map['spurious_rows'])} spurious rows, "
          f"total_severity={severity_map['total_severity']:.1f}")


def generate_all():
    """Generate all task datasets and configs."""
    data_dir = Path("data")

    print("Loading clean datasets...")
    titanic = load_titanic()
    wine = load_wine()
    print(f"  Titanic: {titanic.shape}, Wine: {wine.shape}")

    loaders = {"titanic": titanic, "wine_quality": wine}

    print("\nGenerating tasks...")
    for config in TASK_CONFIGS:
        base = config["base_dataset"]
        clean_df = loaders[base].reset_index(drop=True)
        generate_task(config, clean_df, data_dir)

    print(f"\nDone! Generated {len(TASK_CONFIGS)} tasks.")


if __name__ == "__main__":
    generate_all()
