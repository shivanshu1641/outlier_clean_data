# Runbook: Generate Task Artifacts

## When to run
- First-time setup
- After changing corruption configs in `tools/corruption/engine.py`
- After adding new tasks or datasets

## Command
```bash
source .venv/bin/activate
python tools/corruption/engine.py
```

## What it produces
For each of the 6 tasks, under `data/<task_id>/`:
```
data/titanic_easy/
├── clean.csv         — ground truth (original dataset)
├── dirty.csv         — corrupted version (agent input)
├── error_map.json    — per-cell/row errors with severity + clean value
└── severity_map.json — total severity and breakdown by corruption type

tasks/task_<task_id>.json  — task config loaded by the server
```

## Output format
```
[titanic_easy] clean=(891, 12), dirty=(891, 12), errors=120 cells + 0 spurious rows, total_severity=180.0
[titanic_medium] ...
...
Done! Generated 6 tasks.
```

## Notes
- Seed is fixed (42) — regenerating produces identical artifacts
- Clean datasets are downloaded on first run and cached in `data/clean/`
- Existing `data/<task_id>/` directories are overwritten
- The `tasks/` directory is wiped and rebuilt
