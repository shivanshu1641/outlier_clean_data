"""Load benchmark results, episode logs, and catalog data for the UI."""
from __future__ import annotations

import json
import os

import pandas as pd


def load_benchmark_summary(output_dir: str = "outputs/benchmark") -> pd.DataFrame:
    """Load all benchmark results from JSONL into a DataFrame."""
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                scores = row.pop("scores", {})
                for k, v in scores.items():
                    row[f"score_{k}"] = v
                records.append(row)
    return pd.DataFrame(records) if records else pd.DataFrame()


def load_episode_log(episode_path: str) -> list[dict]:
    """Load an episode JSONL log as a list of step dicts."""
    if not os.path.exists(episode_path):
        return []
    steps = []
    with open(episode_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def load_catalog(catalog_path: str = "datasets/catalog.json") -> dict:
    """Load the dataset catalog."""
    if not os.path.exists(catalog_path):
        return {}
    with open(catalog_path) as f:
        return json.load(f)


def get_available_models(df: pd.DataFrame) -> list[str]:
    if df.empty or "model" not in df.columns:
        return []
    return sorted(df["model"].unique().tolist())


def get_available_datasets(df: pd.DataFrame) -> list[str]:
    if df.empty or "dataset_id" not in df.columns:
        return []
    return sorted(df["dataset_id"].unique().tolist())
