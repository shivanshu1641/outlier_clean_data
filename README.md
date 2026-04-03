# Data Cleaning OpenEnv Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to clean dirty CSV data by writing Python/pandas code, graded on severity-weighted constraint satisfaction and step efficiency.

Built for the [Meta PyTorch OpenEnv Hackathon x Scaler School of Technology](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt

# Start the environment server
uvicorn server.app:app --port 8000

# Run the baseline agent (in another terminal)
API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=qwen3 ENV_URL=http://localhost:8000 python inference.py
```

## How It Works

1. **Reset**: Agent receives a dirty dataset + list of constraints to satisfy
2. **Explore**: Agent inspects the data (pandas queries, no modification, 10/cycle budget)
3. **Transform**: Agent submits Python/pandas code executed in a sandbox
4. **Grade**: Constraints are checked, severity-weighted score computed
5. **Done**: Agent signals completion or hits max transform steps

## Tasks

| Task | Dataset | Constraints | Difficulty |
|------|---------|-------------|------------|
| `easy_titanic` | Titanic (891 rows) | 5 | Easy — nulls, types, whitespace |
| `medium_wine` | Wine Quality (1599 rows) | 10 | Medium — + duplicates, outliers, ranges |
| `hard_combined` | Titanic + Wine (891 rows) | 18 | Hard — + cross-column, format, casing |

## Action Space

| Action | Description |
|--------|-------------|
| `ExploreAction(query="df.isnull().sum()")` | Read-only data inspection |
| `TransformAction(code="df['Age'] = df['Age'].fillna(0)")` | Execute pandas code in sandbox |
| `DoneAction()` | End the episode |

## Scoring

```
score = Σ(severity × solved) / Σ(severity)    # constraint satisfaction
reward = score × efficiency_factor              # penalizes excess steps
```

Severity levels: `critical` (3×), `high` (2×), `medium` (1×)

## Documentation

- [technical.md](technical.md) — Architecture, models, sandbox, grading details
- [deployment.md](deployment.md) — Local dev, Docker, HF Spaces deployment guide

## Project Structure

```
├── models.py              # Typed Pydantic models (Action, Observation, State)
├── client.py              # EnvClient subclass (WebSocket)
├── inference.py           # Baseline LLM agent
├── server/
│   ├── environment.py     # Core reset/step/state loop
│   ├── sandbox.py         # Sandboxed code execution
│   ├── grader.py          # Constraint checkers + scoring
│   └── app.py             # FastAPI application
├── tasks/                 # Task configs (easy/medium/hard)
├── data/                  # Clean + dirty datasets
└── tools/corruption/      # Standalone dirty data generator
```
