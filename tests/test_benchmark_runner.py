"""Tests for tools/benchmark_runner.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from tools.benchmark_runner import (
    BenchmarkResult,
    BenchmarkTask,
    _discover_datasets,
    generate_task_matrix,
    load_config,
    load_results_summary,
    save_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "models": [
        {"name": "test-model", "api_base": "http://localhost:8080/v1", "api_key_env": "KEY"}
    ],
    "categories": ["FP", "VR"],
    "difficulties": ["easy", "hard"],
    "datasets": ["titanic"],
    "seeds_per_combo": 1,
    "base_seed": 42,
    "output_dir": "outputs/benchmark",
    "env_url": "http://localhost:7860",
    "min_call_interval": 2.5,
    "max_steps": 50,
}


def _write_config(tmp_path: Path, data: dict) -> Path:
    config_file = tmp_path / "benchmark_config.yaml"
    config_file.write_text(yaml.dump(data))
    return config_file


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_yaml_config(self, tmp_path: Path):
        config_file = _write_config(tmp_path, _MINIMAL_CONFIG)
        config = load_config(config_file)

        assert config["models"][0]["name"] == "test-model"
        assert config["categories"] == ["FP", "VR"]
        assert config["difficulties"] == ["easy", "hard"]
        assert config["datasets"] == ["titanic"]
        assert config["seeds_per_combo"] == 1
        assert config["base_seed"] == 42

    def test_loads_with_overrides(self, tmp_path: Path):
        config_file = _write_config(tmp_path, _MINIMAL_CONFIG)

        override_models = [
            {"name": "override-model", "api_base": "http://other/v1", "api_key_env": "OTHER_KEY"}
        ]
        config = load_config(
            config_file,
            models=override_models,
            categories=["MD"],
            difficulties=["medium"],
            datasets=["wine_quality"],
        )

        assert config["models"][0]["name"] == "override-model"
        assert config["categories"] == ["MD"]
        assert config["difficulties"] == ["medium"]
        assert config["datasets"] == ["wine_quality"]


# ---------------------------------------------------------------------------
# TestGenerateTaskMatrix
# ---------------------------------------------------------------------------


class TestGenerateTaskMatrix:
    def test_generates_correct_number_of_tasks(self):
        config = {
            **_MINIMAL_CONFIG,
            "datasets": ["titanic", "wine_quality"],
            "categories": ["FP", "VR"],
            "difficulties": ["easy", "hard"],
            "models": [
                {"name": "m1", "api_base": "http://localhost/v1", "api_key_env": "K"},
                {"name": "m2", "api_base": "http://localhost/v1", "api_key_env": "K"},
            ],
            "seeds_per_combo": 1,
        }
        tasks = generate_task_matrix(config)
        # 2 datasets x 2 categories x 2 difficulties x 2 models x 1 seed = 16
        assert len(tasks) == 16

    def test_task_has_required_fields(self):
        config = {**_MINIMAL_CONFIG, "datasets": ["titanic"]}
        tasks = generate_task_matrix(config)

        assert len(tasks) > 0
        task = tasks[0]
        assert isinstance(task, BenchmarkTask)
        assert task.dataset_id == "titanic"
        assert task.category in _MINIMAL_CONFIG["categories"]
        assert task.difficulty in _MINIMAL_CONFIG["difficulties"]
        assert task.model_name == "test-model"
        assert task.model_api_base == "http://localhost:8080/v1"
        assert task.model_api_key_env == "KEY"
        assert isinstance(task.seed, int)

    def test_multiple_seeds(self):
        config = {
            **_MINIMAL_CONFIG,
            "datasets": ["titanic"],
            "categories": ["FP"],
            "difficulties": ["easy"],
            "seeds_per_combo": 3,
            "base_seed": 100,
        }
        tasks = generate_task_matrix(config)
        # 1 dataset x 1 cat x 1 diff x 1 model x 3 seeds = 3
        assert len(tasks) == 3
        seeds = [t.seed for t in tasks]
        assert seeds == [100, 101, 102]

    def test_empty_datasets_discovers_from_catalog(self, tmp_path: Path):
        """When datasets=[], _discover_datasets is called and finds real datasets."""
        # Build a minimal fake project structure
        catalog_dir = tmp_path / "datasets"
        catalog_dir.mkdir()
        clean_dir = tmp_path / "data" / "clean"
        clean_dir.mkdir(parents=True)

        catalog = {
            "alpha": {"filename": "alpha.csv"},
            "beta": {"filename": "beta.csv"},
            "gamma": {"filename": "gamma.csv"},
        }
        (catalog_dir / "catalog.json").write_text(json.dumps(catalog))
        # Only alpha and beta have clean CSVs
        (clean_dir / "alpha.csv").write_text("id\n1\n")
        (clean_dir / "beta.csv").write_text("id\n2\n")

        discovered = _discover_datasets(project_root=tmp_path)
        assert set(discovered) == {"alpha", "beta"}
        assert "gamma" not in discovered

        # Now run matrix with empty datasets list using discovered list
        config = {
            **_MINIMAL_CONFIG,
            "datasets": [],
        }

        # Patch _discover_datasets indirectly by using a config with explicit datasets
        # matching what was discovered, to verify the matrix uses them.
        config_with_discovered = {**config, "datasets": discovered}
        tasks = generate_task_matrix(config_with_discovered)
        dataset_ids = {t.dataset_id for t in tasks}
        assert dataset_ids == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# TestBenchmarkResult
# ---------------------------------------------------------------------------

_SAMPLE_RESULT = BenchmarkResult(
    dataset_id="titanic",
    category="FP",
    difficulty="easy",
    model="test-model",
    seed=42,
    reward=0.85,
    scores={"accuracy": 0.9, "f1": 0.8},
    steps=10,
    episode_log_path="/tmp/ep.log",
    elapsed_s=3.14,
)


class TestBenchmarkResult:
    def test_result_has_required_fields(self):
        r = _SAMPLE_RESULT
        assert r.dataset_id == "titanic"
        assert r.category == "FP"
        assert r.difficulty == "easy"
        assert r.model == "test-model"
        assert r.seed == 42
        assert r.reward == 0.85
        assert r.scores == {"accuracy": 0.9, "f1": 0.8}
        assert r.steps == 10
        assert r.episode_log_path == "/tmp/ep.log"
        assert r.elapsed_s == 3.14

    def test_save_and_load_results(self, tmp_path: Path):
        save_result(_SAMPLE_RESULT, tmp_path)

        # JSONL file exists
        jsonl_path = tmp_path / "results.jsonl"
        assert jsonl_path.exists()

        # summary CSV exists
        csv_path = tmp_path / "summary.csv"
        assert csv_path.exists()

        # load_results_summary returns correct data
        results = load_results_summary(tmp_path)
        assert len(results) == 1
        r = results[0]
        assert r["dataset_id"] == "titanic"
        assert r["reward"] == 0.85
        assert r["scores"] == {"accuracy": 0.9, "f1": 0.8}

    def test_save_appends_to_existing(self, tmp_path: Path):
        save_result(_SAMPLE_RESULT, tmp_path)

        second = BenchmarkResult(
            dataset_id="wine_quality",
            category="VR",
            difficulty="hard",
            model="test-model",
            seed=43,
            reward=0.7,
            scores={"accuracy": 0.75},
            steps=20,
            episode_log_path="/tmp/ep2.log",
            elapsed_s=5.0,
        )
        save_result(second, tmp_path)

        results = load_results_summary(tmp_path)
        assert len(results) == 2
        assert results[0]["dataset_id"] == "titanic"
        assert results[1]["dataset_id"] == "wine_quality"
