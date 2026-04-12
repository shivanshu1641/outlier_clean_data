---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
default_tab: custom
---

# Data Cleaning Benchmark

Data cleaning is unglamorous but unavoidable. Before any model trains, report runs, or pipeline fires, someone has to fix the data. Nulls get injected, values get mangled, formats break, schemas drift. The question we wanted to answer: **how well can AI agents actually do this?**

This is a standardized benchmark where AI agents are given corrupted tabular datasets and must clean them by writing Python/pandas code inside a sandboxed environment. No pre-built solutions. No hints beyond what a real data engineer would see. Each episode is generated fresh; the corruption pipeline runs at reset time against a clean dataset, producing a unique dirty file every run.

Built for the [Meta PyTorch OpenEnv Hackathon x Scaler School of Technology](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

## How an Episode Works

1. **Reset**: the server corrupts a clean dataset and hands the agent the dirty file plus an error summary and initial reward
2. **Explore**: the agent inspects the data with read-only pandas queries (`df.describe()`, `df.isnull().sum()`, etc.)
3. **Transform**: the agent submits Python/pandas code; it runs in a sandboxed subprocess, graded immediately against ground truth, new reward returned
4. **Undo**: if a transform made things worse, the agent rolls back to the last checkpoint
5. **Validate**: the agent can request a structured error breakdown (2 uses per episode)
6. **Done**: if score < 1.0 on the first done, the agent gets one last look at remaining errors before the episode closes

Reward is on **[0.0 to 1.0]**. Reaching 1.0 requires both perfect accuracy (every corrupted cell fixed, no collateral damage) and efficiency (no excess transforms or exploration).

## Benchmark Categories

| Code   | Category            | What it tests                                                                         |
| ------ | ------------------- | ------------------------------------------------------------------------------------- |
| **VR** | Value Repair        | Corrupted cell values: typos, type mangling, decimal shifts, outliers, homoglyphs     |
| **FP** | Format Parsing      | File-level damage: JSON nesting errors, Excel encoding issues, delimiter noise        |
| **MD** | Missing Data        | Injected nulls across numeric and categorical columns                                 |
| **SR** | Structural Repair   | Duplicate rows, column shifts, repeated headers, schema drift                         |
| **SV** | Semantic Validation | Business rule violations: impossible values, unit inconsistencies, cross-column logic |
| **CP** | Compound            | 7+ corruption types simultaneously, non-CSV format, overlapping errors                |

## Difficulty Levels

| Level  | Corruption types | Typical error count | Notes                                                 |
| ------ | ---------------- | ------------------- | ----------------------------------------------------- |
| Easy   | 1 focused type   | 33-98 errors        | Single corruption family                              |
| Medium | 3-4 types        | 136-447 errors      | Mixed types, light format noise                       |
| Hard   | 7-10 types       | 1,500-3,000 errors  | Row-level ops, heavy format noise, wide column spread |

## Scoring

The grader compares the agent's cleaned file against the ground truth using a pre-generated error map. Five components combine into a final reward:

| Component      | Weight | What it measures                                                                  |
| -------------- | ------ | --------------------------------------------------------------------------------- |
| Cell accuracy  | 50%    | Per-cell correctness: full credit for fixes, 1.5x penalty for wrong values        |
| Row integrity  | 15-20% | Duplicate removal, missing row recovery (content-based matching, not index)       |
| Schema         | 15%    | Column names, count, structural compatibility                                     |
| Distribution   | 10%    | Statistical similarity (mean, std, quantiles)                                     |
| Semantic rules | 10%    | Auto-inferred business rules: range, enum, regex, dtype, uniqueness, cross-column |

An **efficiency factor** (floor 0.5) scales the base score down for excessive explores, undos, or transforms. An agent that cleans everything in 8 steps scores more than one that takes 80.

## Benchmark Results

57 runs across 6 models, seed 42, April 2026. Scores are average reward [0.0 to 1.0].

| Model                       | Params | Avg  |   Easy   | Medium |   Hard   |
| --------------------------- | ------ | :--: | :------: | :----: | :------: |
| Nemotron-Nano-8B            | 8B     | 0.31 |   0.15   |  0.33  | **0.46** |
| gemma-4-E2B-it Q4_K_M       | 2B     | 0.31 | **0.43** |  0.14  |   0.06   |
| Qwen3-8B Q4_K_M             | 8B     | 0.26 |   0.42   |  0.16  |   0.25   |
| gpt-4.1-mini                | API    | 0.23 |   0.32   |  0.16  |   0.22   |
| Qwen3.5-0.8B Q4_K_XL        | 0.8B   | 0.14 |   0.14   |  N/A   |   N/A    |
| DeepSeek-R1-Distill-Qwen-7B | 7B     | 0.08 |   0.00   |  0.10  |   N/A    |

**Per-category breakdown:**

| Model            |    FP    |    VR    |  MD  |  SR  |    SV    |  CP  |
| ---------------- | :------: | :------: | :--: | :--: | :------: | :--: |
| Nemotron-Nano-8B |   0.16   |   0.00   | N/A  | 0.23 | **0.49** | N/A  |
| gemma-4-E2B      |   0.24   | **0.65** | 0.21 | 0.19 |   0.66   | 0.20 |
| Qwen3-8B         |   0.04   |   N/A    | 0.29 | 0.45 |   N/A    | 0.20 |
| gpt-4.1-mini     |   0.30   |   N/A    | 0.05 | 0.31 | **0.85** | N/A  |
| Qwen3.5-0.8B     | **0.55** |   N/A    | N/A  | 0.00 |   0.00   | 0.10 |
| DeepSeek-R1 7B   |   0.07   |   N/A    | 0.17 | 0.00 |   N/A    | N/A  |

N/A means the model has not been evaluated on that category yet. Full interactive breakdowns are in the dashboard.

**Key observations:**

- Nemotron-Nano-8B leads overall and is the only model that improves on harder tasks (0.46 on hard vs 0.15 on easy)
- gemma-4-E2B punches above its weight at 2B params, leading on Value Repair (0.65) and matching Nemotron on average
- gpt-4.1-mini dominates Semantic Validation (0.85) but underperforms on structural and missing data tasks
- DeepSeek-R1 struggles significantly, scoring 0.00 on easy tasks despite its reasoning-focused training

## Quick Start

**Run the environment server:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Run the baseline agent:**

```bash
python inference.py titanic easy csv   # single task
python inference.py                    # full 18-task eval suite
```

**Run the full benchmark:**

```bash
python -m tools.benchmark_runner --config tools/benchmark_config.yaml
```

**Launch the dashboard:**

```bash
python -m ui.app --port 7861
```

**Or with Docker:**

```bash
docker build -t data-cleaning-bench .
docker run -p 7860:7860 data-cleaning-bench
```

## Project Structure

```
inference.py                 # Baseline agent: explores, transforms, validates, done
client.py                    # WebSocket client for the environment
models.py                    # Pydantic types: actions, observations, rewards

server/
  app.py                     # FastAPI server: /reset, /step, WebSocket
  environment.py             # Episode lifecycle: generative reset, action dispatch
  sandbox.py                 # AST safety scanning, sandboxed code execution, checkpoints
  grader.py                  # Multi-level reward: cell accuracy, row matching, semantic rules
  corruption/                # 22 value corruptions, ~40 format corruptions, difficulty profiles
  rules/                     # 7 semantic rule types, auto-inferred from clean data

tools/
  benchmark_runner.py        # CLI orchestrator: task matrix, runs, JSONL + CSV output
  benchmark_config.yaml      # Model list, categories, difficulties, seeds

ui/
  app.py                     # Gradio dashboard entry point (HF Spaces compatible)
  leaderboard.py             # Model rankings, bar chart, category heatmap
  explorer.py                # Step-by-step episode replay viewer
  catalog_view.py            # Dataset browser with semantic rule inspection
  data_loader.py             # Loads benchmark results and episode logs

datasets/
  catalog.json               # 25 datasets with auto-inferred semantic rules
```

## Environment Variables

| Variable                      | Default                     | Purpose                                                          |
| ----------------------------- | --------------------------- | ---------------------------------------------------------------- |
| `API_BASE_URL`                | `https://api.openai.com/v1` | LLM endpoint, works with OpenAI, Ollama, llama.cpp, HF Inference |
| `MODEL_NAME`                  | `gpt-4o`                    | Model identifier                                                 |
| `HF_TOKEN` / `OPENAI_API_KEY` | (none)                      | API key                                                          |
| `ENV_URL`                     | `http://localhost:7860`     | Environment server URL                                           |
| `MIN_CALL_INTERVAL`           | `2.5`                       | Seconds between LLM calls (set to 0 for local models)            |
| `LOG_DIR`                     | `outputs/logs`              | JSONL episode log directory                                      |
