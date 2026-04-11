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
) -> BenchmarkResult:
    """Run a single benchmark task and return the result."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from client import DataCleaningClient
    from inference import (
        build_system_prompt, build_user_prompt, get_agent_action, action_from_dict,
    )
    from models import DoneAction

    import inference as inf_mod
    inf_mod.API_BASE_URL = task.model_api_base
    inf_mod.API_KEY = os.environ.get(task.model_api_key_env, "")
    inf_mod.MODEL_NAME = task.model_name
    inf_mod.llm_client = None
    inf_mod.MIN_CALL_INTERVAL = min_call_interval

    task_key = f"{task.dataset_id}_{task.category}_{task.difficulty}_{task.model_name}_{task.seed}"
    logger.info("Starting benchmark task: %s", task_key)

    t0 = time.time()
    current_reward = 0.0
    step_num = 0
    action_history: list[dict] = []

    env_client = DataCleaningClient(base_url=env_url)

    try:
        async with env_client as env:
            step_result = await env.reset(
                task_id=task.dataset_id, difficulty=task.difficulty,
                category=task.category, seed=task.seed,
            )
            obs = step_result.observation
            current_reward = step_result.reward or 0.0

            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(obs, current_reward, action_history=action_history)},
            ]

            while step_num < max_steps:
                step_num += 1
                action_dict, latency, usage = get_agent_action(messages)
                action = action_from_dict(action_dict)
                action_type = action_dict.get("type", "done")

                step_result = await env.step(action)
                obs = step_result.observation
                current_reward = step_result.reward if step_result.reward is not None else current_reward

                constraint_status = obs.constraint_status or {}
                fixed = sum(1 for v in constraint_status.values() if v)
                total = len(constraint_status)

                action_content = action_dict.get("query") or action_dict.get("code") or ""
                action_history.append({
                    "step": step_num, "type": action_type,
                    "summary": action_content[:100],
                    "reward_after": current_reward, "errors_fixed": fixed,
                })

                messages.append({"role": "assistant", "content": json.dumps(action_dict)})
                messages.append({"role": "user", "content": build_user_prompt(obs, current_reward, action_history=action_history)})

                if len(messages) > 20:
                    messages = [messages[0]] + messages[-19:]

                if step_result.done:
                    break
                if total > 0 and fixed == total:
                    done_result = await env.step(DoneAction())
                    current_reward = (
                        done_result.reward
                        if done_result.reward is not None
                        else current_reward
                    )
                    break

                recent_fixed = [h["errors_fixed"] for h in action_history if h["type"] == "transform"]
                if len(recent_fixed) >= 3 and len(set(recent_fixed[-3:])) == 1:
                    done_result = await env.step(DoneAction())
                    current_reward = (
                        done_result.reward
                        if done_result.reward is not None
                        else current_reward
                    )
                    break

    except Exception:
        logger.exception("Task %s failed", task_key)
        raise

    elapsed = time.time() - t0

    episode_dir = os.path.join("outputs", "episodes")
    os.makedirs(episode_dir, exist_ok=True)
    episode_path = os.path.join(episode_dir, f"{_safe_task_key(task_key)}.jsonl")
    with open(episode_path, "w") as f:
        for h in action_history:
            f.write(json.dumps(h) + "\n")

    return BenchmarkResult(
        dataset_id=task.dataset_id, category=task.category,
        difficulty=task.difficulty, model=task.model_name,
        seed=task.seed, reward=current_reward, scores={},
        steps=step_num, episode_log_path=episode_path,
        elapsed_s=round(elapsed, 2),
    )


async def run_benchmark(config: dict) -> list[BenchmarkResult]:
    """Run the full benchmark matrix sequentially."""
    tasks = generate_task_matrix(config)
    output_dir = config.get("output_dir", "outputs/benchmark")
    env_url = config.get("env_url", "http://localhost:7860")
    max_steps = config.get("max_steps", 50)
    min_call_interval = config.get("min_call_interval", 2.5)

    logger.info("Benchmark matrix: %d tasks", len(tasks))
    results = []
    for i, task in enumerate(tasks):
        logger.info("Task %d/%d: %s/%s/%s/%s/seed=%d",
                     i + 1, len(tasks), task.dataset_id, task.category,
                     task.difficulty, task.model_name, task.seed)
        result = await run_benchmark_task(task, env_url=env_url, max_steps=max_steps, min_call_interval=min_call_interval)
        save_result(result, output_dir=output_dir)
        results.append(result)
        logger.info("  reward=%.4f steps=%d elapsed=%.1fs", result.reward, result.steps, result.elapsed_s)
    logger.info("Benchmark complete: %d tasks", len(results))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run data cleaning benchmark")
    parser.add_argument("--config", default="tools/benchmark_config.yaml")
    parser.add_argument("--models", nargs="*", help="Filter to these model names")
    parser.add_argument("--categories", nargs="*", help="Filter to these categories")
    parser.add_argument("--difficulties", nargs="*", help="Filter to these difficulties")
    parser.add_argument("--datasets", nargs="*", help="Filter to these dataset IDs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")

    # CLI --models filter works by name matching
    models_filter = None
    if args.models:
        models_filter = args.models

    config = load_config(args.config, categories=args.categories, difficulties=args.difficulties, datasets=args.datasets)

    # Filter models by name if --models provided
    if models_filter:
        config["models"] = [m for m in config.get("models", []) if m["name"] in models_filter]

    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
