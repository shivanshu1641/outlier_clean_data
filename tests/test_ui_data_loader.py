import pytest
import json
from ui.data_loader import (
    load_benchmark_summary, load_episode_log, load_catalog,
    get_available_models, get_available_datasets,
)


@pytest.fixture
def benchmark_dir(tmp_path):
    results = [
        {"dataset_id": "titanic", "category": "VR", "difficulty": "medium",
         "model": "qwen3-8b", "seed": 42, "reward": 0.82,
         "scores": {"schema": 1.0, "row": 0.95, "cell": 0.78, "distribution": 0.90, "semantic": 0.85},
         "steps": 8, "episode_log_path": "", "elapsed_s": 45.2},
        {"dataset_id": "iris", "category": "MD", "difficulty": "easy",
         "model": "gemma-2-9b", "seed": 42, "reward": 0.91,
         "scores": {"schema": 1.0, "row": 1.0, "cell": 0.88, "distribution": 0.95, "semantic": 0.90},
         "steps": 5, "episode_log_path": "", "elapsed_s": 30.1},
        {"dataset_id": "titanic", "category": "VR", "difficulty": "medium",
         "model": "gemma-2-9b", "seed": 42, "reward": 0.75,
         "scores": {"schema": 1.0, "row": 0.90, "cell": 0.65, "distribution": 0.85, "semantic": 0.80},
         "steps": 12, "episode_log_path": "", "elapsed_s": 60.5},
    ]
    jsonl_path = tmp_path / "results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path)


@pytest.fixture
def episode_file(tmp_path):
    steps = [
        {"step": 1, "type": "explore", "summary": "df.head()", "reward_after": 0.1, "errors_fixed": 0},
        {"step": 2, "type": "transform", "summary": "df['Age'].fillna(28)", "reward_after": 0.5, "errors_fixed": 3},
        {"step": 3, "type": "done", "summary": "", "reward_after": 0.5, "errors_fixed": 3},
    ]
    ep_path = tmp_path / "episode.jsonl"
    with open(ep_path, "w") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")
    return str(ep_path)


class TestLoadBenchmarkSummary:
    def test_loads_as_dataframe(self, benchmark_dir):
        df = load_benchmark_summary(benchmark_dir)
        assert len(df) == 3
        assert "reward" in df.columns
        assert "model" in df.columns

    def test_returns_empty_for_missing_dir(self, tmp_path):
        df = load_benchmark_summary(str(tmp_path / "nonexistent"))
        assert len(df) == 0


class TestLoadEpisodeLog:
    def test_loads_steps(self, episode_file):
        steps = load_episode_log(episode_file)
        assert len(steps) == 3
        assert steps[0]["type"] == "explore"
        assert steps[1]["reward_after"] == 0.5

    def test_returns_empty_for_missing(self):
        steps = load_episode_log("/nonexistent/path.jsonl")
        assert steps == []


class TestHelpers:
    def test_get_available_models(self, benchmark_dir):
        df = load_benchmark_summary(benchmark_dir)
        models = get_available_models(df)
        assert set(models) == {"qwen3-8b", "gemma-2-9b"}

    def test_get_available_datasets(self, benchmark_dir):
        df = load_benchmark_summary(benchmark_dir)
        datasets = get_available_datasets(df)
        assert set(datasets) == {"titanic", "iris"}


class TestLoadCatalog:
    def test_loads_catalog_dict(self):
        catalog = load_catalog()
        assert isinstance(catalog, dict)
        assert "titanic" in catalog
