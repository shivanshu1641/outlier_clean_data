# Runbook: Run Benchmark

Systematically evaluates one or more LLM agents across datasets, categories, and difficulties.

## Prerequisites

1. Server running: `uvicorn server.app:app --port 7860 --ws-ping-interval 60 --ws-ping-timeout 120`
2. Datasets downloaded: `python tools/download_datasets.py`
3. Config set in `tools/benchmark_config.yaml` (edit models/categories/difficulties as needed)

## Command

```bash
source .venv/bin/activate

# Full run using config file
python tools/benchmark_runner.py

# Filter to specific models
python tools/benchmark_runner.py --models qwen3 gemma3:4b

# Filter to categories
python tools/benchmark_runner.py --categories FP VR MD

# Filter to difficulties
python tools/benchmark_runner.py --difficulties easy medium

# Filter to specific datasets
python tools/benchmark_runner.py --datasets titanic iris wine_quality

# Combine filters
python tools/benchmark_runner.py --models qwen3 --categories FP --difficulties easy medium

# Custom config file
python tools/benchmark_runner.py --config path/to/my_config.yaml
```

## Config file: `tools/benchmark_config.yaml`

```yaml
models:
  - name: qwen3           # display name
    api_base: "http://localhost:11434/v1"
    api_key_env: "BENCHMARK_API_KEY"   # env var holding the key (empty string is fine)

categories: [FP, VR, MD, SR, SV, CP]  # all 6, or subset
difficulties: [easy, medium, hard]
datasets: []               # empty = all datasets in catalog

seeds_per_combo: 1         # episodes per (dataset × category × difficulty × model)
base_seed: 42
max_steps: 50              # max actions per episode
min_call_interval: 2.5     # seconds between LLM calls (rate limiting)

output_dir: "outputs/benchmark"
env_url: "http://localhost:7860"
```

### Categories

| ID | Name | Description |
|----|------|-------------|
| FP | Format & Parsing | Malformed delimiters, encoding, whitespace |
| VR | Value Range | Out-of-range values, unit inconsistencies |
| MD | Missing Data | Nulls, empty strings, missing rows |
| SR | Schema & References | Wrong dtypes, referential violations |
| SV | Semantic Violations | Business rule violations, cross-column errors |
| CP | Comprehensive | All corruption types combined |

## Output

Results written to `outputs/benchmark/`:
- `results_<timestamp>.jsonl` — one JSON line per episode
- `summary_<timestamp>.csv` — pivot table (model × category × difficulty)

Load results:
```python
from server.benchmark import BenchmarkResult
results = BenchmarkResult.load("outputs/benchmark/results_<timestamp>.jsonl")
```

## Tips

- For a quick smoke-test, use `--categories FP --difficulties easy --datasets iris`
- Use `seeds_per_combo: 3` in config for statistically meaningful averages
- For local Ollama models, set `api_base: "http://localhost:11434/v1"` and `api_key_env` to any env var (can be empty)
- Results from multiple runs accumulate in the output dir; the Leaderboard UI tab reads all of them
