"""Benchmark runner: config loading and task matrix generation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
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


@dataclass
class BenchmarkResult:
    dataset_id: str
    category: str
    difficulty: str
    model: str
    seed: int
    reward: float
    scores: Dict[str, Any]
    steps: int
    episode_log_path: str
    elapsed_s: float


_SUMMARY_BASE_FIELDS = [
    "dataset_id",
    "category",
    "difficulty",
    "model",
    "seed",
    "reward",
    "steps",
    "episode_log_path",
    "elapsed_s",
]


def _safe_task_key(task_key: str) -> str:
    """Sanitize task keys for safe filesystem use."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_key)


def _summary_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Build a stable CSV schema from all seen rows."""
    score_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in _SUMMARY_BASE_FIELDS
        }
    )
    return [*_SUMMARY_BASE_FIELDS, *score_fields]


def save_result(result: BenchmarkResult, output_dir: str | Path) -> None:
    """Append result to JSONL file and update summary CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Append to JSONL
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")

    # Rewrite summary CSV with a stable union schema across all rows.
    csv_path = output_dir / "summary.csv"
    row = {
        k: v for k, v in asdict(result).items() if k != "scores"
    }
    for score_key, score_val in result.scores.items():
        row[f"score_{score_key}"] = score_val

    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    rows.append(row)

    fieldnames = _summary_fieldnames(rows)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in rows:
            writer.writerow(existing_row)


def load_results_summary(output_dir: str | Path) -> List[Dict[str, Any]]:
    """Load all results from JSONL file. Returns empty list if file doesn't exist."""
    jsonl_path = Path(output_dir) / "results.jsonl"
    if not jsonl_path.exists():
        return []
    results = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


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


logger = logging.getLogger("benchmark")


async def run_benchmark_task(
    task: BenchmarkTask,
    env_url: str = "http://localhost:7860",
    max_steps: int = 50,
    min_call_interval: float = 2.5,
    config: dict | None = None,
) -> BenchmarkResult:
    """Run a single benchmark task by delegating to inference.run_task.

    This avoids duplicating the inference loop — all improvements
    (diagnostic phase, templates, escalation ladder, dynamic temperature,
    code sanitization) are inherited automatically.
    """
    sys.path.insert(0, str(_PROJECT_ROOT))
    import inference as inf_mod

    # Configure the inference module for this benchmark task
    inf_mod.API_BASE_URL = task.model_api_base
    inf_mod.API_KEY = os.environ.get(task.model_api_key_env, "")
    inf_mod.MODEL_NAME = task.model_name
    inf_mod.llm_client = None  # force re-init with new settings
    inf_mod.MIN_CALL_INTERVAL = min_call_interval
    inf_mod.ENV_URL = env_url

    task_key = f"{task.dataset_id}_{task.category}_{task.difficulty}_{task.model_name}_{task.seed}"
    logger.info("Starting benchmark task: %s", task_key)

    # Point inference JSONL logs to the benchmark episodes dir so we capture
    # the full step-by-step log per task (not just a summary).
    episode_dir = Path(
        config.get("output_dir", "outputs/benchmark") if config else "outputs/benchmark"
    ) / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    safe_key = _safe_task_key(task_key)
    episode_path = episode_dir / f"{safe_key}.jsonl"

    # Redirect inference module's JSONL log to our episode file
    old_jsonl = inf_mod._jsonl_file
    inf_mod._jsonl_file = open(episode_path, "w", buffering=1)

    t0 = time.time()

    try:
        # Delegate to the canonical inference loop — gets all improvements for free
        reward = await inf_mod.run_task(
            task.dataset_id, task.difficulty, fmt="csv",
        )
    except Exception:
        logger.exception("Task %s failed", task_key)
        inf_mod._jsonl_file.close()
        inf_mod._jsonl_file = old_jsonl
        # Clean up episode file on error
        if episode_path.exists():
            episode_path.unlink()
        return None
    finally:
        if not inf_mod._jsonl_file.closed:
            inf_mod._jsonl_file.close()
        inf_mod._jsonl_file = old_jsonl

    elapsed = time.time() - t0

    return BenchmarkResult(
        dataset_id=task.dataset_id, category=task.category,
        difficulty=task.difficulty, model=task.model_name,
        seed=task.seed, reward=reward, scores={},
        steps=0,  # step count is in the inference JSONL log
        episode_log_path=str(episode_path),
        elapsed_s=round(elapsed, 2),
    )


def _load_completed_keys(output_dir: str | Path) -> set[str]:
    """Load (dataset_id, category, difficulty, model, seed) keys already in results."""
    existing = load_results_summary(output_dir)
    keys = set()
    for row in existing:
        key = (
            row.get("dataset_id", ""),
            row.get("category", ""),
            row.get("difficulty", ""),
            row.get("model", ""),
            str(row.get("seed", "")),
        )
        keys.add(key)
    return keys


async def run_benchmark(config: dict) -> list[BenchmarkResult]:
    """Run the full benchmark matrix sequentially, skipping already-completed tasks."""
    tasks = generate_task_matrix(config)
    output_dir = config.get("output_dir", "outputs/benchmark")
    env_url = config.get("env_url", "http://localhost:7860")
    max_steps = config.get("max_steps", 50)
    min_call_interval = config.get("min_call_interval", 2.5)

    completed = _load_completed_keys(output_dir)

    logger.info("Benchmark matrix: %d tasks (%d already completed)", len(tasks), len(completed))
    results = []
    skipped = 0
    for i, task in enumerate(tasks):
        key = (task.dataset_id, task.category, task.difficulty, task.model_name, str(task.seed))
        if key in completed:
            skipped += 1
            logger.info("SKIP %d/%d: %s/%s/%s/%s/seed=%d (already exists)",
                         i + 1, len(tasks), task.dataset_id, task.category,
                         task.difficulty, task.model_name, task.seed)
            continue

        logger.info("Task %d/%d: %s/%s/%s/%s/seed=%d",
                     i + 1, len(tasks), task.dataset_id, task.category,
                     task.difficulty, task.model_name, task.seed)
        result = await run_benchmark_task(task, env_url=env_url, max_steps=max_steps, min_call_interval=min_call_interval, config=config)
        if result is None:
            logger.warning("  FAILED — not saving to results")
            continue
        save_result(result, output_dir=output_dir)
        results.append(result)
        logger.info("  reward=%.4f steps=%d elapsed=%.1fs", result.reward, result.steps, result.elapsed_s)
    logger.info("Benchmark complete: %d ran, %d skipped", len(results), skipped)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run data cleaning benchmark")
    parser.add_argument("--config", default="tools/benchmark_config.yaml")
    parser.add_argument("--models", nargs="*", help="Filter to these model names (must exist in config)")
    parser.add_argument("--model-name", help="Run a single model by name (creates entry if not in config)")
    parser.add_argument("--api-base", default="http://localhost:8080/v1", help="API base URL for --model-name")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var for API key")
    parser.add_argument("--categories", nargs="*", help="Filter to these categories")
    parser.add_argument("--difficulties", nargs="*", help="Filter to these difficulties")
    parser.add_argument("--datasets", nargs="*", help="Filter to these dataset IDs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")

    config = load_config(args.config, categories=args.categories, difficulties=args.difficulties, datasets=args.datasets)

    # Single model from CLI (creates entry on the fly)
    if args.model_name:
        config["models"] = [{
            "name": args.model_name,
            "api_base": args.api_base,
            "api_key_env": args.api_key_env,
        }]
    elif args.models:
        config["models"] = [m for m in config.get("models", []) if m["name"] in args.models]

    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
