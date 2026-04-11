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

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Severity per corruption type
CORRUPTION_SEVERITY = {
    "inject_nulls": 2.0,
    "type_mangle": 3.0,
    "duplicate_rows": 2.0,
    "whitespace_noise": 1.0,
    "format_inconsistency": 1.0,
    "outlier_injection": 3.0,
    "drop_rows": 2.5,
    "decimal_shift": 3.0,
    "value_swap": 2.5,
    "typo_injection": 1.5,
}


def _make_rng(task_id: str) -> tuple[np.random.Generator, random.Random]:
    """Create per-task RNG from a deterministic seed derived from task_id."""
    seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed), random.Random(seed)


def _get_clean_val(clean_df: pd.DataFrame | None, df: pd.DataFrame, idx: int, col: str) -> Any:
    """Get clean value from original clean_df if available and in bounds, else from current df."""
    if clean_df is not None and idx < len(clean_df) and col in clean_df.columns:
        return clean_df.at[idx, col]
    return df.at[idx, col]


# ── Corruption Functions ─────────────────────────────────────────────────────


def inject_nulls(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Randomly set cells to NaN."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["inject_nulls"]
    accepted_fill = kwargs.get("accepted_fill", "any")
    for col in columns:
        if col not in df.columns:
            continue
        mask = rng.random(len(df)) < fraction
        indices = df.index[mask]
        if error_log is not None:
            for idx in indices:
                val = _get_clean_val(clean_df, df, idx, col)
                if not pd.isna(val):
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
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Mix string garbage into numeric columns."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["type_mangle"]
    garbage = ["N/A", "unknown", "##", "n/a", "-", "???", "null", "NA"]
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(object)
        n_corrupt = max(1, int(len(df) * fraction))
        idxs = rng.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            garb = garbage[int(rng.integers(0, len(garbage)))]
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
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Insert exact duplicate rows."""
    rng = rng or np.random.default_rng(42)
    n_dupes = max(1, int(len(df) * fraction))
    severity = CORRUPTION_SEVERITY["duplicate_rows"]
    dupe_pos = rng.choice(len(df), size=n_dupes, replace=True)
    dupes = df.iloc[dupe_pos].copy()
    result = pd.concat([df, dupes], ignore_index=True)
    if error_log is not None:
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
    fraction: float = 0.15,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject inconsistent string formats (e.g., casing)."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["format_inconsistency"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        n_corrupt = max(1, int(len(df) * fraction))
        idxs = rng.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            choice = int(rng.integers(0, 3))
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
    fraction: float = 0.1,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Add leading/trailing/double spaces."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["whitespace_noise"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        n_corrupt = max(1, int(len(df) * fraction))
        idxs = rng.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            clean_val = _get_clean_val(clean_df, df, idx, col)
            choice = int(rng.integers(0, 3))
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
    fraction: float = 0.02,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject extreme outlier values into numeric columns."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["outlier_injection"]
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        col_mean = numeric.mean()
        col_std = numeric.std()
        n_outliers = max(1, int(len(df) * fraction))
        idxs = rng.choice(len(df), size=n_outliers, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            multiplier = float(rng.uniform(5, 20))
            new_val = col_mean + col_std * multiplier if rng.random() > 0.5 else col_mean - col_std * multiplier
            if error_log is not None:
                error_log.append({
                    "row": int(idx),
                    "col": col,
                    "type": "cell",
                    "corruption": "outlier_injected",
                    "clean_value": None if pd.isna(clean_val) else (float(clean_val) if isinstance(clean_val, (int, float, np.integer, np.floating)) else None),
                    "severity": severity,
                })
            df.at[idx, col] = new_val
    return df


def drop_rows(
    df: pd.DataFrame,
    fraction: float = 0.03,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Randomly remove rows from the DataFrame."""
    rng = rng or np.random.default_rng(42)
    n_drop = max(1, int(len(df) * fraction))
    severity = CORRUPTION_SEVERITY["drop_rows"]
    drop_positions = rng.choice(len(df), size=n_drop, replace=False)
    drop_indices = df.index[drop_positions]
    if error_log is not None:
        for idx in drop_indices:
            error_log.append({
                "row": int(idx),
                "col": None,
                "type": "missing_row",
                "corruption": "drop_rows",
                "clean_value": None,
                "severity": severity,
            })
    return df.drop(index=drop_indices).reset_index(drop=True)


def decimal_shift(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.03,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Shift decimal point by multiplying/dividing by 10 or 100."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["decimal_shift"]
    shifts = [0.01, 0.1, 10, 100]
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        n_corrupt = max(1, int(len(df) * fraction))
        idxs = rng.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            clean_val = _get_clean_val(clean_df, df, idx, col)
            try:
                fval = float(df.at[idx, col])
            except (TypeError, ValueError):
                continue
            shift = shifts[int(rng.integers(0, len(shifts)))]
            df.at[idx, col] = fval * shift
            if error_log is not None:
                error_log.append({
                    "row": int(idx),
                    "col": col,
                    "type": "cell",
                    "corruption": "decimal_shift",
                    "clean_value": None if pd.isna(clean_val) else float(clean_val),
                    "severity": severity,
                })
    return df


def value_swap(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.03,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Swap values between pairs of columns for some rows."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    severity = CORRUPTION_SEVERITY["value_swap"]
    if len(columns) < 2:
        return df
    for i in range(0, len(columns) - 1, 2):
        col_a, col_b = columns[i], columns[i + 1]
        if col_a not in df.columns or col_b not in df.columns:
            continue
        n_corrupt = max(1, int(len(df) * fraction))
        idxs = rng.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            val_a, val_b = df.at[idx, col_a], df.at[idx, col_b]
            clean_a = _get_clean_val(clean_df, df, idx, col_a)
            clean_b = _get_clean_val(clean_df, df, idx, col_b)
            df.at[idx, col_a], df.at[idx, col_b] = val_b, val_a
            if error_log is not None:
                error_log.append({
                    "row": int(idx),
                    "col": col_a,
                    "type": "cell",
                    "corruption": "value_swap",
                    "clean_value": None if pd.isna(clean_a) else clean_a,
                    "severity": severity,
                })
                error_log.append({
                    "row": int(idx),
                    "col": col_b,
                    "type": "cell",
                    "corruption": "value_swap",
                    "clean_value": None if pd.isna(clean_b) else clean_b,
                    "severity": severity,
                })
    return df


def typo_injection(
    df: pd.DataFrame,
    columns: list[str],
    fraction: float = 0.05,
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject character-level typos: transpose, delete, substitute."""
    df = df.copy()
    rng = rng or np.random.default_rng(42)
    py_rng = py_rng or random.Random(42)
    severity = CORRUPTION_SEVERITY["typo_injection"]
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        n_corrupt = max(1, int(len(df) * fraction))
        idxs = rng.choice(len(df), size=n_corrupt, replace=False)
        for pos in idxs:
            idx = df.index[pos]
            val = str(df.at[idx, col])
            if len(val) < 3:
                continue
            clean_val = _get_clean_val(clean_df, df, idx, col)
            op = int(rng.integers(0, 3))
            chars = list(val)
            if op == 0 and len(chars) >= 2:
                # transpose two adjacent characters
                p = py_rng.randint(0, len(chars) - 2)
                chars[p], chars[p + 1] = chars[p + 1], chars[p]
            elif op == 1:
                # delete a character
                p = py_rng.randint(0, len(chars) - 1)
                chars.pop(p)
            else:
                # substitute a character
                p = py_rng.randint(0, len(chars) - 1)
                chars[p] = py_rng.choice("abcdefghijklmnopqrstuvwxyz")
            new_val = "".join(chars)
            if new_val != val:
                df.at[idx, col] = new_val
                if error_log is not None:
                    error_log.append({
                        "row": int(idx),
                        "col": col,
                        "type": "cell",
                        "corruption": "typo_injection",
                        "clean_value": None if pd.isna(clean_val) else clean_val,
                        "severity": severity,
                    })
    return df


# ── Corruption Pipeline ──────────────────────────────────────────────────────


DISPATCH = {
    "inject_nulls": inject_nulls,
    "type_mangle": type_mangle,
    "duplicate_rows": duplicate_rows,
    "format_inconsistency": format_inconsistency,
    "whitespace_noise": whitespace_noise,
    "outlier_injection": outlier_injection,
    "drop_rows": drop_rows,
    "decimal_shift": decimal_shift,
    "value_swap": value_swap,
    "typo_injection": typo_injection,
}


def apply_corruptions(
    df: pd.DataFrame,
    config: list[dict[str, Any]],
    error_log: list[dict] | None = None,
    clean_df: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    py_rng: random.Random | None = None,
) -> pd.DataFrame:
    """Apply a sequence of corruption operations, optionally tracking errors."""
    for step in config:
        fn_name = step["function"]
        kwargs = {k: v for k, v in step.items() if k != "function"}
        fn = DISPATCH[fn_name]
        kwargs["clean_df"] = clean_df
        kwargs["rng"] = rng
        if fn_name == "typo_injection":
            kwargs["py_rng"] = py_rng
        df = fn(df, error_log=error_log, **kwargs)
    return df


def build_error_map(error_log: list[dict]) -> dict[str, Any]:
    """Convert the flat error log into the nested error_map schema."""
    cell_errors: dict[str, dict] = {}
    spurious_rows: dict[str, dict] = {}
    missing_rows: dict[str, dict] = {}

    # Collect spurious row indices first so we can filter cell errors on them
    spurious_row_indices: set[int] = set()
    for entry in error_log:
        if entry["type"] == "spurious_row":
            spurious_row_indices.add(entry["row"])
        elif entry["type"] == "missing_row":
            row_str = str(entry["row"])
            missing_rows[row_str] = {"severity": entry["severity"]}

    for entry in error_log:
        if entry["type"] == "cell":
            # Skip cell errors on spurious rows — those rows should be removed entirely
            if entry["row"] in spurious_row_indices:
                continue
            key = f"{entry['row']},{entry['col']}"
            # If same cell corrupted multiple times, keep highest severity
            # but prefer the entry with a non-None clean_value
            if key not in cell_errors or entry["severity"] > cell_errors[key]["severity"]:
                error_entry = {
                    "severity": entry["severity"],
                    "clean_value": entry["clean_value"],
                    "corruption": entry["corruption"],
                }
                if "accepted_fill" in entry:
                    error_entry["accepted_fill"] = entry["accepted_fill"]
                cell_errors[key] = error_entry
            elif (entry["severity"] == cell_errors[key]["severity"]
                  and cell_errors[key]["clean_value"] is None
                  and entry["clean_value"] is not None):
                # Same severity but existing has no clean_value — prefer this one
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
        "missing_rows": missing_rows,
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
        by_corruption["drop_rows"] = by_corruption.get("drop_rows", 0.0) + info["severity"]

    total = sum(by_corruption.values())

    return {
        "total_severity": total,
        "total_cell_errors": len(cell_errors),
        "total_spurious_rows": len(spurious_rows),
        "total_missing_rows": len(missing_rows),
        "by_corruption": by_corruption,
    }


# ── Validation ──────────────────────────────────────────────────────────────


def validate_artifacts(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    error_map: dict[str, Any],
) -> list[str]:
    """Validate that error_map is consistent with clean/dirty data. Returns list of warnings."""
    warnings: list[str] = []
    cell_errors = error_map.get("cell_errors", {})

    for key, info in cell_errors.items():
        row_str, col = key.rsplit(",", 1)
        try:
            row_idx = int(row_str)
        except ValueError:
            warnings.append(f"Invalid row index in error_map: {key}")
            continue

        # Check clean_value matches actual clean data
        if row_idx < len(clean_df) and col in clean_df.columns:
            actual_clean = clean_df.at[row_idx, col]
            logged_clean = info["clean_value"]
            if logged_clean is not None:
                try:
                    if pd.isna(actual_clean) and pd.isna(logged_clean):
                        pass  # both NaN, ok
                    elif pd.isna(actual_clean) or pd.isna(logged_clean):
                        warnings.append(f"clean_value mismatch at {key}: logged={logged_clean}, actual={actual_clean}")
                    else:
                        fa, fb = float(actual_clean), float(logged_clean)
                        if abs(fa - fb) > 1e-6:
                            warnings.append(f"clean_value mismatch at {key}: logged={logged_clean}, actual={actual_clean}")
                except (TypeError, ValueError):
                    if str(actual_clean) != str(logged_clean):
                        warnings.append(f"clean_value mismatch at {key}: logged={logged_clean}, actual={actual_clean}")

        # Check dirty data actually differs from clean
        if row_idx < len(dirty_df) and col in dirty_df.columns and row_idx < len(clean_df):
            dirty_val = dirty_df.at[row_idx, col]
            clean_val = clean_df.at[row_idx, col]
            try:
                both_nan = pd.isna(dirty_val) and pd.isna(clean_val)
            except (TypeError, ValueError):
                both_nan = False
            if both_nan:
                warnings.append(f"Phantom error at {key}: dirty and clean are both NaN")
            elif not pd.isna(dirty_val) and not pd.isna(clean_val):
                try:
                    if abs(float(dirty_val) - float(clean_val)) < 1e-6:
                        warnings.append(f"Phantom error at {key}: dirty==clean ({dirty_val})")
                except (TypeError, ValueError):
                    if str(dirty_val) == str(clean_val):
                        warnings.append(f"Phantom error at {key}: dirty==clean ({dirty_val})")

    return warnings


# ── Dataset Loaders ──────────────────────────────────────────────────────────


def load_titanic() -> pd.DataFrame:
    """Load Titanic dataset from bundled CSV or download.

    The raw Titanic CSV has NaN in Age, Cabin, and Embarked.
    We fill these so the clean data has zero NaN — all NaN in dirty data
    are then guaranteed to be from corruption, making fillna always safe.
    """
    clean_path = Path("data/clean/titanic.csv")
    if clean_path.exists():
        return pd.read_csv(clean_path)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Fill original NaN so clean data is fully populated
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
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
            {"function": "whitespace_noise", "columns": ["Name"], "fraction": 0.08},
        ],
        "min_transform_steps": 1,
        "max_transform_steps": 6,
    },
    {
        "task_id": "titanic_medium",
        "description": (
            "Clean the Titanic passenger dataset (medium). "
            "Fix nulls in Age and Embarked, correct type errors in Fare, "
            "remove whitespace from Name and Ticket, and fix typos in Name."
        ),
        "base_dataset": "titanic",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["Age", "Embarked"], "fraction": 0.15},
            {"function": "type_mangle", "columns": ["Fare"], "fraction": 0.05},
            {"function": "whitespace_noise", "columns": ["Name", "Ticket"], "fraction": 0.10},
            {"function": "typo_injection", "columns": ["Name"], "fraction": 0.03},
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
            "duplicate rows, decimal shifts in Fare, typos in Name, and whitespace noise."
        ),
        "base_dataset": "titanic",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["Age", "Embarked", "Fare"], "fraction": 0.30},
            {"function": "type_mangle", "columns": ["Fare", "Age"], "fraction": 0.05},
            {"function": "duplicate_rows", "fraction": 0.08},
            {"function": "format_inconsistency", "columns": ["Embarked", "Name"], "fraction": 0.15},
            {"function": "whitespace_noise", "columns": ["Name", "Ticket"], "fraction": 0.10},
            {"function": "outlier_injection", "columns": ["Age", "Fare"], "fraction": 0.02},
            {"function": "decimal_shift", "columns": ["Fare"], "fraction": 0.03},
            {"function": "typo_injection", "columns": ["Name"], "fraction": 0.05},
        ],
        "min_transform_steps": 6,
        "max_transform_steps": 18,
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
            "correct type errors in acidity columns, fix outliers in pH, "
            "and correct decimal shifts in alcohol."
        ),
        "base_dataset": "wine_quality",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["pH", "alcohol", "residual sugar"], "fraction": 0.12},
            {"function": "type_mangle", "columns": ["fixed acidity", "volatile acidity"], "fraction": 0.05},
            {"function": "duplicate_rows", "fraction": 0.05},
            {"function": "outlier_injection", "columns": ["pH"], "fraction": 0.02},
            {"function": "decimal_shift", "columns": ["alcohol"], "fraction": 0.02},
        ],
        "min_transform_steps": 3,
        "max_transform_steps": 12,
    },
    {
        "task_id": "wine_hard",
        "description": (
            "Clean the Wine Quality dataset (hard). "
            "Heavy nulls across multiple columns, type errors, duplicates, "
            "outliers in pH and alcohol, decimal shifts, value swaps between "
            "acidity columns, and whitespace noise in quality."
        ),
        "base_dataset": "wine_quality",
        "corruptions": [
            {"function": "inject_nulls", "columns": ["pH", "alcohol", "residual sugar", "fixed acidity"], "fraction": 0.30},
            {"function": "type_mangle", "columns": ["fixed acidity", "volatile acidity", "citric acid"], "fraction": 0.05},
            {"function": "duplicate_rows", "fraction": 0.10},
            {"function": "whitespace_noise", "columns": ["quality"], "fraction": 0.10},
            {"function": "outlier_injection", "columns": ["pH", "alcohol"], "fraction": 0.02},
            {"function": "decimal_shift", "columns": ["residual sugar", "citric acid"], "fraction": 0.03},
            {"function": "value_swap", "columns": ["fixed acidity", "volatile acidity"], "fraction": 0.03},
        ],
        "min_transform_steps": 6,
        "max_transform_steps": 18,
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────


def generate_task(config: dict[str, Any], clean_df: pd.DataFrame, data_dir: Path) -> None:
    """Generate all 4 artifacts for a single task."""
    task_id = config["task_id"]
    task_dir = data_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Per-task deterministic RNG
    rng, py_rng = _make_rng(task_id)

    # Save clean data
    clean_path = task_dir / "clean.csv"
    clean_df.to_csv(clean_path, index=False)

    # Apply corruptions, tracking errors — pass original clean_df for correct clean_value capture
    error_log: list[dict] = []
    dirty_df = apply_corruptions(
        clean_df.copy(),
        config["corruptions"],
        error_log=error_log,
        clean_df=clean_df,
        rng=rng,
        py_rng=py_rng,
    )

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

    # Validate artifacts
    warnings = validate_artifacts(clean_df, dirty_df, error_map)
    if warnings:
        print(f"  [{task_id}] ⚠ {len(warnings)} validation warnings:")
        for w in warnings[:5]:
            print(f"    - {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")

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
          f"errors={len(error_map['cell_errors'])} cells + {len(error_map['spurious_rows'])} spurious + {len(error_map['missing_rows'])} missing rows, "
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
