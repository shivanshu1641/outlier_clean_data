"""Load benchmark results, episode logs, and catalog data for the UI."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from server.corruption.categories import CATEGORY_FORMAT_MAP, CATEGORY_CORRUPTION_MAP

_PROJECT_ROOT = Path(__file__).parent.parent


# ── Category assignment ────────────────────────────────────────────────────────

def _infer_category(difficulty: str, fmt: str) -> str:
    """Map a task's (difficulty, format) to its primary benchmark category.

    Heuristic based on CATEGORY_FORMAT_MAP:
    - hard + non-csv format → CP (compound)
    - non-csv format → FP (format parsing)
    - csv easy → pick from VR/MD randomly? No — use a stable assignment.

    For csv-only tasks we can't distinguish VR/MD/SR/SV from format alone,
    so we rotate through them based on a hash for stable assignment.
    In practice, the benchmark runner should tag category explicitly.
    """
    # Non-csv formats are primarily format parsing challenges
    if fmt not in ("csv", "tsv"):
        if difficulty == "hard":
            return "CP"
        return "FP"

    # TSV → could be SR (structural repair uses csv/tsv)
    if fmt == "tsv":
        return "SR"

    # CSV tasks: distribute across VR, MD, SR, SV based on difficulty
    # This is a rough heuristic — real benchmark_runner should tag explicitly
    if difficulty == "easy":
        return "VR"  # easy csv = value repair focus
    if difficulty == "medium":
        return "MD"  # medium csv = missing data + value repair
    return "SR"  # hard csv = structural


def _parse_task_id(task_id: str) -> dict:
    """Parse 'titanic_easy_csv' → {dataset, difficulty, format}."""
    parts = task_id.rsplit("_", 2)
    if len(parts) == 3:
        return {"dataset": parts[0], "difficulty": parts[1], "format": parts[2]}
    # Try 2-part split for legacy IDs
    parts = task_id.rsplit("_", 1)
    if len(parts) == 2:
        return {"dataset": parts[0], "difficulty": parts[1], "format": "csv"}
    return {"dataset": task_id, "difficulty": "unknown", "format": "csv"}


# ── Results loading ────────────────────────────────────────────────────────────

def load_results(benchmark_dir: str = "outputs/benchmark") -> pd.DataFrame:
    """Load benchmark results. Tries benchmark_runner's results.jsonl first,
    falls back to inference.py's results.csv in outputs/logs/."""

    # Resolve relative paths against project root (not CWD)
    if not os.path.isabs(benchmark_dir):
        benchmark_dir = str(_PROJECT_ROOT / benchmark_dir)

    # Primary: benchmark_runner output (has category field)
    jsonl_path = os.path.join(benchmark_dir, "results.jsonl")
    if os.path.exists(jsonl_path):
        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    # Flatten scores dict
                    scores = row.pop("scores", {})
                    for k, v in scores.items():
                        row[f"score_{k}"] = v
                    records.append(row)
        if records:
            df = pd.DataFrame(records)
            # Ensure standard columns exist
            if "model" not in df.columns and "model_name" in df.columns:
                df["model"] = df["model_name"]
            # Synthesize task_id if missing
            if "task_id" not in df.columns and "dataset_id" in df.columns:
                df["task_id"] = df["dataset_id"] + "_" + df.get("category", "") + "_" + df.get("difficulty", "")
            # Alias dataset_id → dataset for consistency
            if "dataset" not in df.columns and "dataset_id" in df.columns:
                df["dataset"] = df["dataset_id"]
            return df

    # Fallback: inference.py's results.csv (no category — infer it)
    # Only use fallback when using the default benchmark_dir
    csv_path = str(_PROJECT_ROOT / "outputs" / "logs" / "results.csv")
    if not os.path.exists(os.path.join(benchmark_dir, "results.jsonl")):
        # benchmark_dir was explicitly set and has no results — don't fall back
        if benchmark_dir != str(_PROJECT_ROOT / "outputs" / "benchmark"):
            return pd.DataFrame()
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    # Parse task_id into components
    parsed = df["task_id"].apply(_parse_task_id).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Assign category
    df["category"] = df.apply(
        lambda r: _infer_category(r["difficulty"], r["format"]), axis=1
    )

    return df


def load_best_per_model_task(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the best reward per (model, task_id) pair."""
    if df.empty:
        return df
    subset = [c for c in ["model", "task_id"] if c in df.columns]
    if not subset:
        return df
    return df.sort_values("reward", ascending=False).drop_duplicates(
        subset=subset, keep="first"
    ).reset_index(drop=True)


# ── Episode logs ───────────────────────────────────────────────────────────────

def load_episode_log(path: str) -> list[dict]:
    """Load a JSONL episode log as list of event dicts."""
    if not os.path.exists(path):
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def list_episode_files(log_dir: str = "outputs/benchmark/episodes") -> list[dict]:
    """List available episode JSONL files with metadata from their first event."""
    if not os.path.isabs(log_dir):
        log_dir = str(_PROJECT_ROOT / log_dir)
    if not os.path.isdir(log_dir):
        return []
    episodes = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(log_dir, fname)
        file_size = os.path.getsize(path)
        # Read first line for metadata
        try:
            with open(path) as f:
                first_line = f.readline().strip()
            if not first_line:
                continue
            meta = json.loads(first_line)
            # Read last line for final results
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            last = json.loads(lines[-1]) if lines else {}
        except (json.JSONDecodeError, IndexError):
            continue  # truly corrupt file
        status = "failed" if file_size < 500 else "completed"
        episodes.append({
            "file": fname,
            "path": path,
            "task_id": meta.get("task_id", fname),
            "model": meta.get("model", "unknown"),
            "final_reward": last.get("final_reward", last.get("reward", 0.0)),
            "steps": last.get("total_steps", 0),
            "status": status,
        })
    return episodes


# ── Catalog ────────────────────────────────────────────────────────────────────

def load_catalog(catalog_path: str = "datasets/catalog.json") -> dict:
    """Load the dataset catalog."""
    if not os.path.isabs(catalog_path):
        catalog_path = str(_PROJECT_ROOT / catalog_path)
    if not os.path.exists(catalog_path):
        return {}
    with open(catalog_path) as f:
        return json.load(f)


# ── Backward-compat aliases & helpers ─────────────────────────────────────────

load_benchmark_summary = load_results


def get_available_models(df: pd.DataFrame) -> list[str]:
    """Return unique model names from a results DataFrame."""
    if "model" not in df.columns:
        return []
    return df["model"].unique().tolist()


def get_available_datasets(df: pd.DataFrame) -> list[str]:
    """Return unique dataset IDs from a results DataFrame."""
    if "dataset_id" not in df.columns:
        return []
    return df["dataset_id"].unique().tolist()
