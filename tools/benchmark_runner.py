"""Benchmark runner: config loading and task matrix generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Project root relative to this file
_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class BenchmarkTask:
    dataset_id: str
    category: str
    difficulty: str
    model_name: str
    model_api_base: str
    model_api_key_env: str
    seed: int


def load_config(
    config_path: str | Path,
    models: Optional[List[Dict[str, Any]]] = None,
    categories: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load YAML config and apply optional overrides."""
    config_path = Path(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    if models is not None:
        config["models"] = models
    if categories is not None:
        config["categories"] = categories
    if difficulties is not None:
        config["difficulties"] = difficulties
    if datasets is not None:
        config["datasets"] = datasets

    return config


def _discover_datasets(project_root: Path = _PROJECT_ROOT) -> List[str]:
    """Find dataset IDs that have clean CSVs AND exist in datasets/catalog.json."""
    catalog_path = project_root / "datasets" / "catalog.json"
    clean_dir = project_root / "data" / "clean"

    if not catalog_path.exists() or not clean_dir.exists():
        return []

    with catalog_path.open() as f:
        catalog: Dict[str, Any] = json.load(f)

    discovered = []
    for dataset_id, meta in catalog.items():
        filename = meta.get("filename", f"{dataset_id}.csv")
        if (clean_dir / filename).exists():
            discovered.append(dataset_id)

    return sorted(discovered)


def generate_task_matrix(config: Dict[str, Any]) -> List[BenchmarkTask]:
    """Generate the full list of BenchmarkTask objects from config."""
    datasets = config.get("datasets") or []
    if not datasets:
        datasets = _discover_datasets()

    models = config.get("models", [])
    categories = config.get("categories", [])
    difficulties = config.get("difficulties", [])
    seeds_per_combo = config.get("seeds_per_combo", 1)
    base_seed = config.get("base_seed", 42)

    tasks: List[BenchmarkTask] = []
    for dataset_id in datasets:
        for category in categories:
            for difficulty in difficulties:
                for model in models:
                    for i in range(seeds_per_combo):
                        seed = base_seed + i
                        tasks.append(
                            BenchmarkTask(
                                dataset_id=dataset_id,
                                category=category,
                                difficulty=difficulty,
                                model_name=model["name"],
                                model_api_base=model["api_base"],
                                model_api_key_env=model["api_key_env"],
                                seed=seed,
                            )
                        )

    return tasks
