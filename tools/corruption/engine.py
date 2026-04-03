"""
Synthetic corruption engine — standalone tool for generating dirty datasets.

NOT part of the OpenEnv deployment. Run once to produce data/dirty/ files.

Usage:
    python tools/corruption/engine.py
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)


# ── Corruption Functions ─────────────────────────────────────────────────────


def inject_nulls(df: pd.DataFrame, columns: list[str], fraction: float = 0.1) -> pd.DataFrame:
    """Randomly set cells to NaN."""
    df = df.copy()
    for col in columns:
        mask = RNG.random(len(df)) < fraction
        df.loc[mask, col] = np.nan
    return df


def type_mangle(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mix string garbage into numeric columns."""
    df = df.copy()
    garbage = ["N/A", "unknown", "##", "n/a", "-", "???", "null", "NA"]
    for col in columns:
        df[col] = df[col].astype(object)
        n_corrupt = max(1, int(len(df) * 0.05))
        indices = RNG.choice(len(df), size=n_corrupt, replace=False)
        for idx in indices:
            df.at[df.index[idx], col] = RNG.choice(garbage)
    return df


def duplicate_rows(df: pd.DataFrame, fraction: float = 0.05) -> pd.DataFrame:
    """Insert exact duplicate rows."""
    n_dupes = max(1, int(len(df) * fraction))
    dupe_indices = RNG.choice(len(df), size=n_dupes, replace=True)
    dupes = df.iloc[dupe_indices].copy()
    return pd.concat([df, dupes], ignore_index=True)


def format_inconsistency(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Inject inconsistent string formats (e.g., casing, abbreviations)."""
    df = df.copy()
    for col in columns:
        if df[col].dtype == object:
            n_corrupt = max(1, int(len(df) * 0.15))
            indices = RNG.choice(len(df), size=n_corrupt, replace=False)
            for idx in indices:
                val = str(df.iloc[idx, df.columns.get_loc(col)])
                choice = RNG.integers(0, 3)
                if choice == 0:
                    df.iloc[idx, df.columns.get_loc(col)] = val.upper()
                elif choice == 1:
                    df.iloc[idx, df.columns.get_loc(col)] = val.lower()
                else:
                    df.iloc[idx, df.columns.get_loc(col)] = val.title()
    return df


def whitespace_noise(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add leading/trailing/double spaces."""
    df = df.copy()
    for col in columns:
        if df[col].dtype == object:
            n_corrupt = max(1, int(len(df) * 0.1))
            indices = RNG.choice(len(df), size=n_corrupt, replace=False)
            for idx in indices:
                val = str(df.iloc[idx, df.columns.get_loc(col)])
                choice = RNG.integers(0, 3)
                if choice == 0:
                    val = "  " + val
                elif choice == 1:
                    val = val + "  "
                else:
                    val = val.replace(" ", "  ", 1) if " " in val else "  " + val
                df.iloc[idx, df.columns.get_loc(col)] = val
    return df


def outlier_injection(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Inject extreme outlier values into numeric columns."""
    df = df.copy()
    for col in columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        col_mean = numeric.mean()
        col_std = numeric.std()
        n_outliers = max(1, int(len(df) * 0.02))
        indices = RNG.choice(len(df), size=n_outliers, replace=False)
        for idx in indices:
            if RNG.random() > 0.5:
                df.iloc[idx, df.columns.get_loc(col)] = col_mean + col_std * RNG.uniform(5, 20)
            else:
                df.iloc[idx, df.columns.get_loc(col)] = col_mean - col_std * RNG.uniform(5, 20)
    return df


# ── Corruption Pipeline ──────────────────────────────────────────────────────


def apply_corruptions(df: pd.DataFrame, config: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply a sequence of corruption operations to a DataFrame."""
    dispatch = {
        "inject_nulls": inject_nulls,
        "type_mangle": type_mangle,
        "duplicate_rows": duplicate_rows,
        "format_inconsistency": format_inconsistency,
        "whitespace_noise": whitespace_noise,
        "outlier_injection": outlier_injection,
    }
    for step in config:
        fn_name = step["function"]
        kwargs = {k: v for k, v in step.items() if k != "function"}
        fn = dispatch[fn_name]
        df = fn(df, **kwargs)
    return df


# ── Dataset Loaders ──────────────────────────────────────────────────────────


def load_titanic() -> pd.DataFrame:
    """Load Titanic dataset from bundled CSV or download."""
    clean_path = Path("data/clean/titanic.csv")
    if clean_path.exists():
        return pd.read_csv(clean_path)
    # Download from a reliable source
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


EASY_CONFIG = {
    "task_id": "easy_titanic",
    "description": (
        "Clean the Titanic passenger dataset. Fix null values in Age and Embarked, "
        "correct type errors in Fare, and remove whitespace from Name and Ticket."
    ),
    "base_dataset": "titanic",
    "corruptions": [
        {"function": "inject_nulls", "columns": ["Age", "Embarked"], "fraction": 0.15},
        {"function": "type_mangle", "columns": ["Fare"]},
        {"function": "whitespace_noise", "columns": ["Name", "Ticket"]},
    ],
    "constraints": [
        {"id": "c1", "type": "no_nulls", "column": "Age", "severity": "high", "description": "No null values in Age"},
        {"id": "c2", "type": "no_nulls", "column": "Embarked", "severity": "high", "description": "No null values in Embarked"},
        {"id": "c3", "type": "dtype", "column": "Fare", "dtype": "float64", "severity": "critical", "description": "Fare must be float64"},
        {"id": "c4", "type": "no_whitespace", "column": "Name", "severity": "medium", "description": "No leading/trailing whitespace in Name"},
        {"id": "c5", "type": "no_whitespace", "column": "Ticket", "severity": "medium", "description": "No leading/trailing whitespace in Ticket"},
    ],
    "min_transform_steps": 2,
    "max_transform_steps": 8,
}

MEDIUM_CONFIG = {
    "task_id": "medium_wine",
    "description": (
        "Clean the Wine Quality dataset. Remove duplicates, fix null values, "
        "correct type errors, normalize column formats, and ensure all values "
        "are within valid scientific ranges."
    ),
    "base_dataset": "wine_quality",
    "corruptions": [
        {"function": "inject_nulls", "columns": ["pH", "alcohol", "residual sugar"], "fraction": 0.1},
        {"function": "type_mangle", "columns": ["fixed acidity", "volatile acidity"]},
        {"function": "duplicate_rows", "fraction": 0.05},
        {"function": "whitespace_noise", "columns": ["quality"]},
        {"function": "outlier_injection", "columns": ["pH"]},
    ],
    "constraints": [
        {"id": "c1", "type": "no_nulls", "column": "pH", "severity": "critical", "description": "No null values in pH"},
        {"id": "c2", "type": "no_nulls", "column": "alcohol", "severity": "critical", "description": "No null values in alcohol"},
        {"id": "c3", "type": "no_nulls", "column": "residual sugar", "severity": "high", "description": "No null values in residual sugar"},
        {"id": "c4", "type": "dtype", "column": "fixed acidity", "dtype": "float64", "severity": "critical", "description": "fixed acidity must be float64"},
        {"id": "c5", "type": "dtype", "column": "volatile acidity", "dtype": "float64", "severity": "critical", "description": "volatile acidity must be float64"},
        {"id": "c6", "type": "no_duplicates", "key": None, "severity": "high", "description": "No exact duplicate rows"},
        {"id": "c7", "type": "value_range", "column": "pH", "min": 2.0, "max": 5.0, "severity": "critical", "description": "pH must be between 2.0 and 5.0"},
        {"id": "c8", "type": "value_range", "column": "alcohol", "min": 5.0, "max": 20.0, "severity": "high", "description": "alcohol must be between 5.0 and 20.0"},
        {"id": "c9", "type": "dtype", "column": "quality", "dtype": "int64", "severity": "medium", "description": "quality must be integer"},
        {"id": "c10", "type": "row_count_range", "min": 1590, "max": 1600, "severity": "high", "description": "Row count should be ~1599 (no duplicates added)"},
    ],
    "min_transform_steps": 4,
    "max_transform_steps": 12,
}

HARD_CONFIG = {
    "task_id": "hard_combined",
    "description": (
        "Clean a merged Titanic + Wine Quality dataset. This combined dataset has "
        "complex cross-column dependencies, mixed types, duplicates, outliers, "
        "format inconsistencies, and encoding issues. You must handle all corruption "
        "types and ensure cross-column constraints are satisfied."
    ),
    "base_dataset": "combined",
    "corruptions": [
        {"function": "inject_nulls", "columns": ["Age", "pH", "alcohol", "Embarked"], "fraction": 0.12},
        {"function": "type_mangle", "columns": ["Fare", "fixed acidity", "volatile acidity"]},
        {"function": "duplicate_rows", "fraction": 0.08},
        {"function": "format_inconsistency", "columns": ["Name", "Embarked"]},
        {"function": "whitespace_noise", "columns": ["Name", "Ticket"]},
        {"function": "outlier_injection", "columns": ["Age", "pH", "Fare"]},
    ],
    "constraints": [
        {"id": "c1", "type": "no_nulls", "column": "Age", "severity": "critical", "description": "No null values in Age"},
        {"id": "c2", "type": "no_nulls", "column": "pH", "severity": "critical", "description": "No null values in pH"},
        {"id": "c3", "type": "no_nulls", "column": "alcohol", "severity": "critical", "description": "No null values in alcohol"},
        {"id": "c4", "type": "no_nulls", "column": "Embarked", "severity": "high", "description": "No null values in Embarked"},
        {"id": "c5", "type": "dtype", "column": "Fare", "dtype": "float64", "severity": "critical", "description": "Fare must be float64"},
        {"id": "c6", "type": "dtype", "column": "fixed acidity", "dtype": "float64", "severity": "critical", "description": "fixed acidity must be float64"},
        {"id": "c7", "type": "dtype", "column": "volatile acidity", "dtype": "float64", "severity": "critical", "description": "volatile acidity must be float64"},
        {"id": "c8", "type": "no_duplicates", "key": None, "severity": "critical", "description": "No exact duplicate rows"},
        {"id": "c9", "type": "value_range", "column": "Age", "min": 0, "max": 120, "severity": "high", "description": "Age must be between 0 and 120"},
        {"id": "c10", "type": "value_range", "column": "pH", "min": 2.0, "max": 5.0, "severity": "critical", "description": "pH must be between 2.0 and 5.0"},
        {"id": "c11", "type": "value_range", "column": "Fare", "min": 0, "max": 600, "severity": "high", "description": "Fare must be between 0 and 600"},
        {"id": "c12", "type": "no_whitespace", "column": "Name", "severity": "medium", "description": "No leading/trailing whitespace in Name"},
        {"id": "c13", "type": "no_whitespace", "column": "Ticket", "severity": "medium", "description": "No leading/trailing whitespace in Ticket"},
        {"id": "c14", "type": "consistent_case", "column": "Embarked", "severity": "high", "description": "Embarked values must be consistently cased"},
        {"id": "c15", "type": "unique_values", "column": "Embarked", "allowed": ["S", "C", "Q"], "severity": "high", "description": "Embarked must be one of S, C, Q"},
        {"id": "c16", "type": "value_range", "column": "alcohol", "min": 5.0, "max": 20.0, "severity": "high", "description": "alcohol must be between 5.0 and 20.0"},
        {"id": "c17", "type": "dtype", "column": "quality", "dtype": "int64", "severity": "medium", "description": "quality must be integer"},
        {"id": "c18", "type": "row_count_range", "min": 800, "max": 920, "severity": "high", "description": "Row count should be ~891 (no duplicates added)"},
    ],
    "min_transform_steps": 7,
    "max_transform_steps": 15,
}


# ── Main ─────────────────────────────────────────────────────────────────────


def create_combined_dataset(titanic: pd.DataFrame, wine: pd.DataFrame) -> pd.DataFrame:
    """Create a combined dataset by sampling from both and joining on index."""
    # Take all titanic rows, sample matching wine rows
    wine_sampled = wine.sample(n=len(titanic), replace=True, random_state=SEED).reset_index(drop=True)
    titanic_reset = titanic.reset_index(drop=True)
    combined = pd.concat([titanic_reset, wine_sampled], axis=1)
    # Remove duplicate column names if any
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def generate_all():
    """Generate all dirty datasets and task configs."""
    data_dir = Path("data")
    tasks_dir = Path("tasks")
    tasks_dir.mkdir(exist_ok=True)

    # Load clean datasets
    print("Loading clean datasets...")
    titanic = load_titanic()
    wine = load_wine()
    combined = create_combined_dataset(titanic, wine)

    # Save combined clean
    combined.to_csv(data_dir / "clean" / "combined.csv", index=False)

    configs = [
        ("easy", EASY_CONFIG, titanic),
        ("medium", MEDIUM_CONFIG, wine),
        ("hard", HARD_CONFIG, combined),
    ]

    for difficulty, config, clean_df in configs:
        print(f"\nGenerating {difficulty} task: {config['task_id']}...")
        print(f"  Clean shape: {clean_df.shape}")

        # Apply corruptions
        dirty_df = apply_corruptions(clean_df.copy(), config["corruptions"])
        print(f"  Dirty shape: {dirty_df.shape}")

        # Save dirty data
        dirty_path = data_dir / "dirty" / f"{config['task_id']}.csv"
        dirty_path.parent.mkdir(parents=True, exist_ok=True)
        dirty_df.to_csv(dirty_path, index=False)
        print(f"  Saved: {dirty_path}")

        # Save task config
        task_path = tasks_dir / f"task_{difficulty}.json"
        task_config = {k: v for k, v in config.items() if k != "corruptions"}
        task_config["dirty_data_path"] = str(dirty_path)
        task_config["clean_data_path"] = str(data_dir / "clean" / f"{config['base_dataset']}.csv")
        with open(task_path, "w") as f:
            json.dump(task_config, f, indent=2)
        print(f"  Task config: {task_path}")

    print("\nDone! Generated 3 tasks.")


if __name__ == "__main__":
    generate_all()
