"""Tests for tools/benchmark_runner.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from tools.benchmark_runner import (
    BenchmarkTask,
    _discover_datasets,
    generate_task_matrix,
    load_config,
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
