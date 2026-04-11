# OpenEnv Data Cleaning — v2 Extensions Design

**Date:** 2026-04-11
**Status:** Draft
**Author:** Shivanshu + Claude

## Overview

Four workstreams extending the generative data cleaning environment. Two are core (semantic constraints, category system) and two are extensions (benchmark runner, HF Spaces UI with embedded episode visualizer). The core feeds the extensions: rules and categories make the benchmark meaningful, and the benchmark produces the data the UI displays.

**Build order:** Semantic Constraints → Category System → Benchmark Runner → HF Spaces UI (includes episode visualizer)

---

## 1. Semantic Constraints (Core)

### Problem

The current `business_rule_violation` corruption type is one of 22 corruption types with hardcoded logic. There's no per-dataset rule catalog, no auto-inference, and the grader doesn't have a dedicated dimension for semantic validity. The agent doesn't know what rules the data should satisfy.

### Design

#### 1.1 Rule Types

Seven rule types, all JSON-serializable:

| Type | Schema | Example |
|------|--------|---------|
| `range` | `{type, column, min, max}` | `age ∈ [0, 120]` |
| `regex` | `{type, column, pattern}` | `email matches ^[\w.]+@[\w.]+\.\w+$` |
| `enum` | `{type, column, values}` | `sex ∈ {male, female, other}` |
| `dtype` | `{type, column, expected_dtype}` | `age must be integer` |
| `not_null` | `{type, column}` | `passenger_id cannot be null` |
| `unique` | `{type, column}` | `passenger_id has no duplicates` |
| `cross_column` | `{type, columns, condition, hint}` | `if embarked='S' then port='Southampton'` |

All rules carry an optional `source` field: `"inferred"` or `"manual"`. Manual rules override inferred ones for the same column+type.

#### 1.2 Auto-Inference (`server/rules/inferrer.py`)

Runs over clean data at dataset prep time. Three layers:

1. **Domain heuristics** — Recognizes common column names and applies known domain rules:
   - `age`, `years` → range [0, 120]
   - `email` → standard email regex
   - `zip`, `zipcode`, `pincode` → digit-format regex
   - `phone` → phone format regex
   - `salary`, `price`, `income` → range [0, ∞)
   - `state` → US state enum (when dataset domain suggests US data)
   - `country` → ISO-3166 enum
   - Column names matching `*_id` → unique + not_null

2. **Statistical inference** — For columns not matched by heuristics:
   - Numeric columns: range = [min - 0.2*(max-min), max + 0.2*(max-min)], clamped to 0 if all values ≥ 0
   - Categorical columns (≤30 unique values): enum from observed values
   - String columns with consistent pattern: regex inferred from common format
   - Columns with 0% nulls in clean data: not_null rule

3. **Cross-column hints** — Limited auto-inference:
   - Functional dependencies detected via value co-occurrence (e.g., city → state)
   - Date ordering (start_date < end_date) detected from column name pairs
   - These are surfaced as hints with natural-language descriptions since they're harder to express as rigid rules

Output: list of rule dicts written to `catalog.json` under each dataset's `rules` field.

#### 1.3 Manual Overrides

Dataset entries in `catalog.json` can include a `rules_override` field with manually authored rules. The enrichment script merges: manual rules take precedence over inferred rules for the same column+type combination.

#### 1.4 Catalog Schema Addition

Rules are **auto-generated** by running `catalog_enricher.py` across all clean datasets. The enricher writes the `rules` field automatically. Manual overrides are optional — use `rules_override` only when auto-inference isn't precise enough for a specific dataset.

```json
{
  "id": "titanic",
  "rules": [
    {"type": "range", "column": "age", "min": 0, "max": 120, "source": "heuristic"},
    {"type": "range", "column": "fare", "min": 0, "max": 1000, "source": "statistical"},
    {"type": "enum", "column": "sex", "values": ["male", "female"], "source": "statistical"},
    {"type": "not_null", "column": "passengerid", "source": "statistical"},
    {"type": "unique", "column": "passengerid", "source": "heuristic"},
    {"type": "cross_column", "columns": ["embarked", "port"], "condition": "functional_dependency", "hint": "Embarkation code must match port name (S=Southampton, C=Cherbourg, Q=Queenstown)", "source": "cross_column_inference"}
  ],
  "rules_override": [
    {"type": "range", "column": "fare", "min": 0, "max": 600, "source": "manual"}
  ]
}
```

The `source` field tracks provenance: `"heuristic"` (domain name match), `"statistical"` (data-driven), `"cross_column_inference"` (co-occurrence), or `"manual"` (human override). At merge time, manual rules replace any inferred rule for the same column+type.

#### 1.5 Corruption Integration

At `reset()`, the `CorruptionPipeline`:
1. Loads rules for the selected dataset from catalog
2. When `business_rule_violation` is selected (or for SV category tasks), it reads the rules and deliberately creates violations:
   - Range rules → inject out-of-range values
   - Regex rules → inject malformed strings
   - Enum rules → inject invalid category values
   - Not-null rules → inject nulls in required columns
   - Unique rules → inject duplicate IDs
   - Cross-column rules → break the relationship
3. Violations are tracked in `error_map` with a new `rule_violation` field per cell, referencing which rule was broken

#### 1.6 Grading Integration

New 5th scoring dimension in `grader.py`:

- **Weight rebalance:** schema 15%, row 15%, cell 50%, distribution 10%, semantic 10%
- `semantic_score = 1.0 - (violations_in_result / total_rules)`
- Only counts rules that were relevant (i.e., the column exists in the result)
- A rule violation in the agent's output that was NOT in the dirty data (collateral damage) counts double
- If no rules exist for a dataset, `semantic_score` defaults to 1.0 (no penalty)

#### 1.7 Observation

The agent receives the rules in the observation so it knows what to enforce:

```python
class DataCleaningObservation:
    # ... existing fields ...
    semantic_rules: list[dict]  # rules for this dataset
```

Cross-column rules include their `hint` field so the agent understands the relationship in natural language.

#### 1.8 New Files

```
server/rules/
├── __init__.py
├── types.py          # Rule dataclasses, serialization
├── inferrer.py       # Auto-inference engine (3 layers)
├── validator.py      # Validate df against ruleset, return violations
└── catalog_enricher.py  # CLI script to enrich catalog.json
```

---

## 2. Category System (Core)

### Problem

The current environment generates random corruption combinations per difficulty. For benchmarking, we need to produce tasks that target specific skill categories and ensure balanced coverage.

### Design

#### 2.1 Category Definitions

Six categories, each mapping to a subset of the 22 corruption types:

| Category | Code | Corruption Types | Description |
|----------|------|-----------------|-------------|
| Format Parsing | `FP` | format corruptions (10 formats), encoding_noise, header_in_data | Can the agent read the mess? File arrives as JSON/Excel/XML with format-level issues. |
| Value Repair | `VR` | type_mangle, decimal_shift, value_swap, typo_injection, unicode_homoglyph, html_entity_leak, leading_zero_strip | Can the agent fix corrupted cell values? |
| Missing Data | `MD` | inject_nulls, drop_rows | Can the agent fill gaps intelligently? |
| Structural Repair | `SR` | duplicate_rows, column_shift, schema_drift, header_in_data | Can the agent repair the shape? |
| Semantic Validation | `SV` | business_rule_violation, unit_inconsistency, outlier_injection + semantic rules | Does the agent understand the data? |
| Compound | `CP` | 7+ corruption types, non-CSV format, mixed categories | Can the agent handle everything at once? |

Some corruptions appear in multiple categories (e.g., `header_in_data` in FP and SR). This is intentional — the category constrains the primary focus, not exclusive membership.

#### 2.2 Generator Integration

New file: `server/corruption/categories.py`

```python
CATEGORY_CORRUPTION_MAP = {
    "FP": ["encoding_noise", "header_in_data"],  # + format corruptions
    "VR": ["type_mangle", "decimal_shift", "value_swap", "typo_injection", "unicode_homoglyph", "html_entity_leak", "leading_zero_strip"],
    "MD": ["inject_nulls", "drop_rows"],
    "SR": ["duplicate_rows", "column_shift", "schema_drift", "header_in_data"],
    "SV": ["business_rule_violation", "unit_inconsistency", "outlier_injection"],
    "CP": None,  # any 7+ types
}

CATEGORY_FORMAT_MAP = {
    "FP": ["json", "jsonl", "excel", "xml", "html_table", "fixed_width", "sql_dump", "yaml"],
    "VR": ["csv"],  # isolate value issues from format issues
    "MD": ["csv"],
    "SR": ["csv", "tsv"],
    "SV": ["csv"],
    "CP": ["json", "excel", "xml", "html_table"],  # non-CSV
}
```

#### 2.3 Environment API

`reset()` gains an optional `category` parameter:

```python
def reset(self, task_id=None, difficulty=None, category=None, seed=None):
    # If category specified, constrain corruption selection
    # If category is CP, select 7+ types from all categories
    # If category is FP, force non-CSV format + format-specific corruptions
```

When `category` is None (default), behavior is unchanged — random selection per difficulty profile.

#### 2.4 Difficulty × Category Matrix

Each category supports all three difficulty levels. Difficulty controls:
- Number of corruptions within the category's pool
- Corruption fraction (% of cells affected)
- Hint granularity

For FP: difficulty also controls format complexity (easy=json/tsv, medium=excel/fixed_width, hard=xml/html_table/sql_dump).

#### 2.5 New Files

```
server/corruption/categories.py  # Category definitions, corruption/format maps
```

Modifications to: `profiles.py` (category-aware selection), `pipeline.py` (accept category param), `environment.py` (pass category through reset).

---

## 3. Benchmark Runner (Extension)

### Problem

The current eval suite has 15 hardcoded tasks across 8 datasets. We need comprehensive coverage across 118+ datasets, 6 categories, and multiple open-source models.

### Design

#### 3.1 Benchmark Matrix

- **Datasets:** All 118+ from catalog.json (those with downloaded clean CSVs)
- **Categories:** 6 (FP, VR, MD, SR, SV, CP)
- **Difficulties:** 3 (easy, medium, hard)
- **Models:** Open-source models that fit in 24GB VRAM:
  - Qwen3-8B, Qwen3-1.7B
  - Gemma-2-9B, Gemma-2-2B
  - Llama-3.1-8B, Llama-3.2-3B
  - DeepSeek-V2-Lite, DeepSeek-Coder-7B
  - (exact list TBD based on availability and inference speed)

#### 3.2 Task Generation

A benchmark task is defined by `(dataset_id, category, difficulty, seed)`. The benchmark runner:

1. Iterates over the matrix (or a configured subset)
2. For each task, calls `reset(dataset_id, difficulty, category, seed)`
3. Runs the inference agent
4. Collects: reward, per-dimension scores, episode log (all steps), metadata

#### 3.3 Output Format

Results saved as JSONL, one line per task:

```json
{
  "dataset_id": "titanic",
  "category": "VR",
  "difficulty": "medium",
  "model": "qwen3-8b",
  "seed": 42,
  "reward": 0.82,
  "scores": {"schema": 1.0, "row": 0.95, "cell": 0.78, "distribution": 0.90, "semantic": 0.85},
  "steps": 8,
  "episode_log": "outputs/episodes/titanic_VR_medium_qwen3-8b_42.jsonl"
}
```

Episode logs (for the UI explorer) capture each step: action type, code/query, stdout, observation changes, score delta.

#### 3.4 Runner Script

`tools/benchmark_runner.py` — CLI tool that:
- Accepts filters: `--models`, `--categories`, `--difficulties`, `--datasets`
- Supports `--parallel N` for concurrent episodes (multiple model servers)
- Outputs to `outputs/benchmark/` with per-run directories
- Produces a summary CSV for the UI to consume

#### 3.5 New Files

```
tools/benchmark_runner.py      # CLI benchmark orchestrator
tools/benchmark_config.yaml    # Default benchmark matrix config
outputs/benchmark/             # Results directory
```

---

## 4. HF Spaces UI (Extension)

### Problem

No visual interface exists. Benchmark results and episode replays need a self-contained, pre-loaded UI that runs on HF Spaces without API credentials.

### Design

#### 4.1 Technology

Gradio app — already the standard for HF Spaces, fits the existing Dockerfile pattern. All data pre-loaded from `outputs/benchmark/`.

#### 4.2 Tabs

**Tab 1: Benchmark Leaderboard**
- Model × Category score matrix (like the screenshot)
- Filterable by difficulty, dataset, category
- Bar charts per category showing model comparison
- Overall ranking with aggregate scores
- Data source: static CSV/JSON from benchmark runner output

**Tab 2: Episode Explorer**
- Select: model, dataset, category, difficulty
- Step-by-step replay of the agent's journey:
  - Step N: action type + code executed
  - Data diff (before/after for that step)
  - Score progression (running chart)
  - Agent's reasoning/observation at each step
- Pre-loaded from episode JSONL logs
- Clear label: "These are pre-computed results"

**Tab 3: Dataset Catalog**
- Browse all 118+ datasets
- View metadata: domain, rows, cols, dtypes
- View inferred + manual semantic rules for each dataset
- View available categories and difficulty levels

#### 4.3 Data Flow

```
outputs/benchmark/summary.csv  →  Leaderboard tab
outputs/episodes/*.jsonl       →  Explorer tab
datasets/catalog.json          →  Catalog tab (with rules)
```

#### 4.4 New Files

```
ui/
├── app.py              # Gradio app entry point
├── leaderboard.py      # Benchmark leaderboard tab
├── explorer.py         # Episode replay tab
├── catalog_view.py     # Dataset catalog tab
└── data_loader.py      # Load pre-computed results
```

Dockerfile updated to include `ui/` and `outputs/benchmark/` data.

---

## 5. Cross-Cutting Concerns

### 5.1 Backward Compatibility

- `reset()` without `category` param behaves exactly as before
- Grading without rules defaults `semantic_score` to 1.0
- Existing LEGACY_TASK_MAP still works
- Existing 15-task eval suite still works

### 5.2 Reproducibility

- Semantic rule violations use the same seeded RNG as other corruptions
- Benchmark tasks identified by `(dataset_id, category, difficulty, seed)` tuple
- Same seed → same corruptions → same rules violated

### 5.3 Testing

Each component gets unit tests:
- `tests/test_rules.py` — rule types, inference, validation
- `tests/test_categories.py` — category mapping, constrained generation
- `tests/test_grader_semantic.py` — semantic scoring dimension
- `tests/test_benchmark_runner.py` — runner output format, filtering

---

## Summary

| Component | Type | New Files | Modifications |
|-----------|------|-----------|---------------|
| Semantic Constraints | Core | `server/rules/{__init__,types,inferrer,validator,catalog_enricher}.py` | `grader.py`, `corruption/pipeline.py`, `corruption/value_corruptions.py`, `models.py`, `catalog.json` |
| Category System | Core | `server/corruption/categories.py` | `profiles.py`, `pipeline.py`, `environment.py` |
| Benchmark Runner | Extension | `tools/benchmark_runner.py`, `tools/benchmark_config.yaml` | `inference.py` (minor) |
| HF Spaces UI | Extension | `ui/{app,leaderboard,explorer,catalog_view,data_loader}.py` | `Dockerfile` |
