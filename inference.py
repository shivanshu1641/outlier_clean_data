"""
Baseline inference script for the Data Cleaning OpenEnv environment.

Uses any OpenAI-compatible API endpoint. Configure via environment variables:
  API_BASE_URL  — defaults to https://api.openai.com/v1 if unset
  OPENAI_API_KEY / HF_TOKEN — API token (HF_TOKEN takes precedence)
  MODEL_NAME    — e.g. gemma-4-E2B-it, qwen3, gpt-4o
  ENV_URL       — http://localhost:7860
  LOG_LEVEL     — INFO (default) or DEBUG for full LLM I/O
  LOG_DIR       — directory for JSONL log files (default: outputs/logs)

Usage:
    python inference.py                      # runs all tasks
    python inference.py titanic easy         # run one task (default fmt=csv)
    python inference.py titanic easy json    # run one task with explicit format
    python inference.py titanic/easy/json    # equivalent slash form
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from client import DataCleaningClient
from models import (
    DoneAction,
    ExploreAction,
    TransformAction,
    UndoAction,
    ValidateAction,
)

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("inference")

LOG_DIR = Path(os.environ.get("LOG_DIR", "outputs/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
_jsonl_file = open(LOG_DIR / f"run_{_RUN_ID}.jsonl", "w", buffering=1)


def _jlog(event: str, **fields):
    """Append a JSON line to the run log file."""
    entry = {"event": event, "timestamp": time.time(), **fields}
    _jsonl_file.write(json.dumps(entry) + "\n")


# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Each entry: (dataset_id, difficulty, format)
# Formats are pinned per task for deterministic, reproducible episodes.
# easy: csv + tsv | medium: csv + json + jsonl | hard: csv + json + xml
EVAL_TASKS: list[tuple[str, str, str]] = [
    ("titanic", "easy", "csv"),
    ("titanic", "medium", "csv"),
    ("titanic", "hard", "csv"),
    ("iris", "easy", "csv"),
    ("iris", "medium", "csv"),
    ("iris", "medium", "jsonl"),
    ("boston_housing", "medium", "csv"),
    ("boston_housing", "hard", "csv"),
    ("boston_housing", "hard", "json"),
    ("diabetes", "medium", "csv"),
    ("diabetes", "hard", "csv"),
    ("diabetes", "hard", "json"),
    ("wine_quality", "easy", "csv"),
    ("wine_quality", "medium", "csv"),
    ("wine_quality", "hard", "csv"),
    ("breast_cancer", "easy", "csv"),
    ("breast_cancer", "medium", "csv"),
    ("breast_cancer", "medium", "jsonl"),
]

TASKS = EVAL_TASKS

# Max agent steps per difficulty — hard tasks get more time
MAX_STEPS_BY_DIFFICULTY = {"easy": 30, "medium": 60, "hard": 100}

# Minimum seconds between LLM calls to avoid rate limits (Groq free: 30 req/min)
MIN_CALL_INTERVAL = float(os.environ.get("MIN_CALL_INTERVAL", "2.5"))

# ── Corruption-type transform templates (C) ───────────────────────────────────

TRANSFORM_TEMPLATES: dict[str, str] = {
    "whitespace_noise": (
        "df['{col}'] = df['{col}'].str.replace(r'\\s+', ' ', regex=True).str.strip()"
    ),
    "type_mangle": (
        "# Find only the values that actually fail numeric conversion:\n"
        "bad = df['{col}'][pd.to_numeric(df['{col}'], errors='coerce').isna() & df['{col}'].notna()]\n"
        "df['{col}'] = df['{col}'].replace(bad.unique().tolist(), float('nan'))\n"
        "df['{col}'] = pd.to_numeric(df['{col}'])\n"
        "df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
    ),
    "inject_nulls": (
        "df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
    ),
    "null_injected": (
        "# Numeric column:\n"
        "df['{col}'] = df['{col}'].fillna(df['{col}'].median())\n"
        "# Categorical column:\n"
        "# df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])"
    ),
    "outlier_injection": (
        "Q1 = df['{col}'].quantile(0.25)\n"
        "Q3 = df['{col}'].quantile(0.75)\n"
        "IQR = Q3 - Q1\n"
        "med = df['{col}'].median()\n"
        "df.loc[df['{col}'] > Q3 + 1.5 * IQR, '{col}'] = med\n"
        "df.loc[df['{col}'] < Q1 - 1.5 * IQR, '{col}'] = med"
    ),
    "decimal_shift": (
        "med = df['{col}'].median()\n"
        "shifted_up = df['{col}'] > med * 5\n"
        "shifted_dn = df['{col}'] < med / 5\n"
        "df.loc[shifted_up, '{col}'] = df.loc[shifted_up, '{col}'] / 10\n"
        "df.loc[shifted_dn, '{col}'] = df.loc[shifted_dn, '{col}'] * 10"
    ),
    "duplicate_rows": "df = df.drop_duplicates()",
    "category_misspell": (
        "# Check value_counts first, map misspellings to canonical form:\n"
        "# df['{col}'] = df['{col}'].str.lower().str.strip()\n"
        "# df['{col}'] = df['{col}'].replace({'misspelling': 'canonical'})"
    ),
    "typo_injection": (
        "# Check value_counts first, map typos to canonical form:\n"
        "# df['{col}'] = df['{col}'].replace({'typo': 'correct_value'})"
    ),
    "format_inconsistency": (
        "df['{col}'] = df['{col}'].str.lower().str.strip()"
    ),
    "value_swap": (
        "# Inspect df[['col_a', 'col_b']].head(20) to confirm swap pattern.\n"
        "# Then swap back: mask = <condition>\n"
        "# df.loc[mask, ['{col}', 'other_col']] = df.loc[mask, ['other_col', '{col}']].values"
    ),
}
_last_call_time = 0.0

# ── OpenAI Client ─────────────────────────────────────────────────────────────

llm_client: OpenAI | None = None


def get_api_base_url() -> str:
    return API_BASE_URL


def get_api_key() -> str:
    return API_KEY


def get_model_name() -> str:
    return MODEL_NAME


def get_llm_client() -> OpenAI:
    global llm_client
    if llm_client is None:
        api_base_url = get_api_base_url()
        api_key = get_api_key()
        llm_client = OpenAI(base_url=api_base_url, api_key=api_key)
    return llm_client


# ── Structured stdout logging (machine-readable) ──────────────────────────────


ENV_NAME = "data_cleaning_env"


def log_start(task_id: str, category: str = "", seed: int = 0):
    model_name = get_model_name()
    api_base_url = get_api_base_url()
    print(f"[START] task={task_id} env={ENV_NAME} model={model_name}", flush=True)
    _jlog("task_start", task_id=task_id, model=model_name, api_base=api_base_url,
          category=category, seed=seed)


def log_step(
    step_num: int,
    action_type: str,
    reward: float,
    done: bool,
    action_content: str = "",
    error: str | None = None,
    latency: float = 0.0,
    usage: dict | None = None,
    errors_fixed: int = 0,
    errors_total: int = 0,
):
    action_summary = (
        f"{action_type}()"
        if action_type == "done"
        else f"{action_type}('{action_content[:80]}')"
    )
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step_num} action={action_summary} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )
    _jlog(
        "step",
        step=step_num,
        action_type=action_type,
        action_content=action_content[:500] if action_content else "",
        reward=reward,
        errors_fixed=errors_fixed,
        errors_total=errors_total,
        llm_latency_s=round(latency, 3),
        **(usage or {}),
    )


def log_end(
    task_id: str,
    final_reward: float,
    total_steps: int,
    elapsed: float,
    rewards: list[float],
    category: str = "",
    seed: int = 0,
):
    score = max(0.001, min(final_reward, 0.999))
    success = str(score >= 0.5).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success} steps={total_steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )
    _jlog(
        "task_end",
        task_id=task_id,
        final_reward=final_reward,
        total_steps=total_steps,
        elapsed_s=round(elapsed, 2),
        category=category,
        seed=seed,
    )


# ── Agent Logic ───────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
You are a data cleaning agent. You receive a dirty CSV dataset and information
about what errors exist. Your goal is to fix all errors using as few transform
steps as possible.

You have five action types:
1. **explore**: Inspect the data without modifying it. Your query should be a valid pandas expression using `df` (a pandas DataFrame). Examples: `df.describe()`, `df['Age'].isnull().sum()`, `df.head(10)`
2. **transform**: Submit Python/pandas code that operates on `df` to clean it. Do NOT include import statements, print(), or comments — pandas (pd), numpy (np), re, datetime, string, math, json, csv, io, collections, itertools, functools are already pre-imported. The variable `df` is pre-loaded. Keep code concise.
3. **done**: Signal you're finished cleaning.
4. **validate**: Get detailed error breakdown (budget: 2 per episode). Use {"type": "validate"} when stuck.
5. **undo**: Restore to a previous checkpoint. ONLY use when you see "REGRESSION" in the feedback.

Respond with EXACTLY one JSON object (no markdown, no explanation):
- For explore: {"type": "explore", "query": "<pandas expression>"}
- For transform: {"type": "transform", "code": "<python code>"}
- For done: {"type": "done"}
- For undo: {"type": "undo", "step": 0}
- For validate: {"type": "validate"}

Strategy:
- DIAGNOSTIC FIRST: Before any transform, explore to understand the scope. Run `df.isnull().sum()` and check value counts on columns mentioned in the error summary.
- On medium/hard tasks, do at least one targeted explore per error type before your first transform
- Tackle the largest error group first (highest count), then work down
- Fix ONE column per transform — short, focused code. Do NOT try to fix multiple columns at once.
- Do NOT write comments in your code — they waste tokens. Code only.
- Check error status after each transform
- If your reward or errors-fixed DROPS after a transform, immediately undo to restore your best state, then try a different approach
- Submit 'done' when all errors are fixed or you can't improve further
- Be efficient: fewer transform steps = higher reward
- Warning: changing a cell to the wrong value is penalized MORE than leaving it dirty
- NEVER drop rows with dropna() unless you are specifically removing duplicate rows
- Only fix columns mentioned in the error summary — do not touch other columns
- For whitespace errors: use str.replace(r'\\s+', ' ', regex=True).str.strip() not just str.strip()

Fix templates (copy-paste, replace 'col' with actual column name):
- inject_nulls: df['col'] = df['col'].fillna(df['col'].median())
- whitespace_noise: df['col'] = df['col'].astype(str).str.replace(r'\\s+', ' ', regex=True).str.strip()
- duplicate_rows: df = df.drop_duplicates()
- decimal_shift: med = df['col'].median(); df.loc[df['col'] > med * 8, 'col'] /= 10; df.loc[(df['col'] > 0) & (df['col'] < med / 8), 'col'] *= 10
- outlier_injection: q1, q3 = df['col'].quantile([0.25, 0.75]); iqr = q3 - q1; mask = (df['col'] < q1 - 3*iqr) | (df['col'] > q3 + 3*iqr); df.loc[mask, 'col'] = df['col'].median()
- type_mangle: bad = df['col'][pd.to_numeric(df['col'], errors='coerce').isna() & df['col'].notna()]; df['col'] = df['col'].replace(bad.unique().tolist(), float('nan')); df['col'] = pd.to_numeric(df['col']); df['col'] = df['col'].fillna(df['col'].median())
- category_misspell: valid = df['col'].value_counts().head(10).index.tolist(); df['col'] = df['col'].apply(lambda x: min(valid, key=lambda v: sum(a!=b for a,b in zip(str(x).lower(),v.lower()))) if str(x) not in valid else x)
- leading_zero_strip: df['col'] = df['col'].astype(str).str.zfill(5)
- value_swap: explore first to identify swapped pairs, then swap back targeted rows

Important rules:
- Do NOT repeat the same explore query — check your action history below
- If your last transform fixed 0 errors, your approach is WRONG. Do NOT repeat it. Explore the affected columns to understand what the dirty values actually look like, then try a completely different fix.
- If your last transform didn't improve the score, try a COMPLETELY different approach
- Your action history is shown in every observation — use it to avoid repeating mistakes
- If you're stuck, submit 'done' rather than repeating the same failing transform
- The clean data has ZERO NaN — every NaN in the dirty data is from corruption. fillna() is always safe: `df['col'] = df['col'].fillna(df['col'].median())` for numeric, `df['col'] = df['col'].fillna(df['col'].mode()[0])` for categorical.
- fillna(value, inplace=True) on a column may silently no-op in pandas 2.x. Use assignment: df['col'] = df['col'].fillna(value)
- For type_mangle: find ONLY the values that fail numeric conversion, then replace those specifically: `bad = df['col'][pd.to_numeric(df['col'], errors='coerce').isna() & df['col'].notna()]; df['col'] = df['col'].replace(bad.unique().tolist(), float('nan')); df['col'] = pd.to_numeric(df['col']); df['col'] = df['col'].fillna(df['col'].median())`. This preserves valid values. Do NOT use blanket to_numeric(errors='coerce') on the whole column — it destroys valid cells.
- NEVER invent or re-sequence identifier columns (e.g. PassengerId, id, index, key). Inspect the exact dirty tokens with explore first; do not overwrite IDs with sequential integers or any fabricated values.
- A single cell can have MULTIPLE corruptions (e.g. type_mangle + inject_nulls on the same column). One transform may only fix one layer — check errors after each step and apply follow-up transforms for remaining issues.
"""


def build_system_prompt() -> str:
    return BASE_SYSTEM_PROMPT


def _extract_remaining_error_targets(
    constraints: list[str] | None,
) -> dict[str, list[str]]:
    """Parse compact `corruption in: col1, col2` lines from the observation."""
    targets: dict[str, list[str]] = {}
    for desc in constraints or []:
        for raw_line in str(desc).splitlines():
            line = raw_line.strip()
            # Handle row-level corruptions (no columns): "duplicate_rows: 59 extra rows"
            if "duplicate_rows" in line and "extra rows" in line:
                targets["duplicate_rows"] = []
                continue
            if " in: " not in line:
                continue
            left, right = line.split(" in: ", 1)
            # left is like "type_mangle (108 errors)" — strip the "(N errors)" part
            ctype = left.split("(")[0].strip().split()[-1].rstrip(":")
            cols = [c.strip() for c in right.split(",") if c.strip()]
            if cols:
                targets[ctype] = cols
    return targets


def _suggest_explore_queries(
    obs, action_history: list[dict] | None = None
) -> list[str]:
    """Generate a few concrete explore queries from the current error summary."""
    targets = _extract_remaining_error_targets(obs.constraints or [])
    suggestions: list[str] = []

    def add(query: str) -> None:
        if query not in suggestions:
            suggestions.append(query)

    for ctype, cols in targets.items():
        shown = cols[:3]
        cols_expr = "[" + ", ".join(repr(c) for c in shown) + "]"

        if ctype in {"inject_nulls", "null_injected"}:
            add(f"df[{cols_expr}].isna().sum()")
            if shown:
                add(f"df[{repr(shown[0])}].value_counts(dropna=False).head(10)")
        elif ctype == "type_mangle":
            add(f"df[{cols_expr}].head(15)")
            if shown:
                add(
                    f"pd.to_numeric(df[{repr(shown[0])}], errors='coerce').isna().sum()"
                )
        elif ctype in {"outlier_injection", "decimal_shift"}:
            add(f"df[{cols_expr}].describe(include='all')")
            if shown:
                add(f"df[{repr(shown[0])}].sort_values().tail(10)")
        elif ctype in {
            "whitespace_noise",
            "format_inconsistency",
            "category_misspell",
            "typo_injection",
        }:
            add(f"df[{cols_expr}].head(15)")
            if shown:
                add(f"df[{repr(shown[0])}].value_counts(dropna=False).head(20)")
        else:
            add(f"df[{cols_expr}].head(15)")

    if not suggestions:
        add("df.head(10)")
        add("df.info()")

    recent_explores = {
        h["summary"] for h in (action_history or []) if h.get("type") == "explore"
    }
    filtered = [q for q in suggestions if q[:100] not in recent_explores]
    return (filtered or suggestions)[:5]


def _explore_manual(obs) -> list[str]:
    """Return concise exploration heuristics for the current error mix."""
    targets = _extract_remaining_error_targets(obs.constraints or [])
    bullets: list[str] = []

    if "type_mangle" in targets:
        bullets.append(
            "For type_mangle: find ONLY values that fail numeric conversion, then replace those: "
            "`bad = df['col'][pd.to_numeric(df['col'], errors='coerce').isna() & df['col'].notna()]; "
            "df['col'] = df['col'].replace(bad.unique().tolist(), float('nan')); "
            "df['col'] = pd.to_numeric(df['col']); "
            "df['col'] = df['col'].fillna(df['col'].median())`. "
            "This preserves valid values. Do NOT use blanket to_numeric(errors='coerce') on the whole column."
        )
    if "inject_nulls" in targets or "null_injected" in targets:
        bullets.append(
            "For inject_nulls: all NaN are from corruption so fillna is safe. "
            "Numeric: `df['col'] = df['col'].fillna(df['col'].median())` "
            "Categorical: `df['col'] = df['col'].fillna(df['col'].mode()[0])`"
        )
    if "outlier_injection" in targets or "decimal_shift" in targets:
        bullets.append(
            "For outlier_injection / decimal_shift: explore `df['col'].describe()` and "
            "`df['col'].sort_values().tail(10)` to find shifted values. "
            "Fix with a clip or IQR replacement — e.g. "
            "`df.loc[df['col'] > upper, 'col'] = median_val`. "
            "Casting dtype does NOT fix outlier values."
        )
    if "duplicate_rows" in targets:
        bullets.append(
            "For duplicate_rows: fix with `df = df.drop_duplicates()` — "
            "this is the ONE safe use of drop; confirm duplicates exist first with `df.duplicated().sum()`."
        )
    if any(
        t in targets
        for t in (
            "whitespace_noise",
            "format_inconsistency",
            "category_misspell",
            "typo_injection",
        )
    ):
        bullets.append(
            "For string corruption, inspect examples or value counts before normalizing so you target the right columns."
        )

    bullets.append(
        "Do not repeat the same explore query. After 1-2 targeted explores, either transform or validate."
    )
    return bullets[:5]


def _build_template_hints(obs) -> str:
    """Build concrete code templates from the current error summary (change C)."""
    targets = _extract_remaining_error_targets(obs.constraints or [])
    if not targets:
        return ""
    lines: list[str] = []
    # Only show top 2 corruption types to keep prompt short
    shown = 0
    for ctype, cols in targets.items():
        if shown >= 2:
            break
        template = TRANSFORM_TEMPLATES.get(ctype)
        if template is None:
            continue
        if cols:
            lines.append(f"### {ctype}: affects {', '.join(cols[:4])}")
        else:
            lines.append(f"### {ctype}")
        if "{col}" in template and cols:
            # Only show template for first column
            lines.append(template.replace("{col}", cols[0]))
        else:
            lines.append(template)
        lines.append("")
        shown += 1
    return "\n".join(lines).strip()


def _consecutive_explore_count(action_history: list[dict]) -> int:
    count = 0
    for h in reversed(action_history):
        if h["type"] != "explore":
            break
        count += 1
    return count


def _same_query_streak(action_history: list[dict]) -> int:
    explores = [h for h in action_history if h["type"] == "explore"]
    if not explores:
        return 0
    last = explores[-1]["summary"]
    streak = 0
    for h in reversed(explores):
        if h["summary"] == last:
            streak += 1
        else:
            break
    return streak


def build_user_prompt(
    obs,
    result_reward: float | None = None,
    action_history: list[dict] | None = None,
    warnings: list[str] | None = None,
    diagnostic_text: str | None = None,
    template_hints: str | None = None,
) -> str:
    """Build a compact, decision-oriented prompt from the current observation."""
    constraint_status = obs.constraint_status or {}
    fixed = sum(1 for v in constraint_status.values() if v)
    total = len(constraint_status)
    reward = result_reward if result_reward is not None else (obs.reward or 0.0)

    parts = [
        f"Task: {obs.task_description}",
        f"File format: {obs.file_format or 'csv'}",
        "",
        "Score:",
        f"  Reward: {reward:.4f}",
        "",
        "Progress:",
        f"  Errors fixed: {fixed}/{total}",
    ]

    step_info = obs.step_info
    if step_info:
        parts.extend(
            [
                f"  Explore budget: {step_info.explore_steps_used}/{step_info.explore_budget}",
                f"  Transform steps: {step_info.transform_steps_used}/{step_info.max_transform_steps}",
                f"  Validate budget: {step_info.validate_uses}/{step_info.validate_budget}",
            ]
        )

    # ── CRITICAL: Surface execution errors and wrong-value warnings at the TOP ──
    # Small models only reliably read the first ~200 tokens of feedback.
    # Execution failures and wrong-value regressions must be unmissable.
    transform_result = obs.transform_result or ""
    if "Execution: failed" in transform_result:
        parts.extend([
            "",
            "*** YOUR LAST TRANSFORM FAILED TO EXECUTE ***",
            "The code crashed. Do NOT repeat it. Try fixing a DIFFERENT corruption type first and come back to this one later.",
            transform_result,
            "Do NOT use import statements — pandas (pd), numpy (np), re, math, datetime, string, json, csv are already available.",
            "Write ONLY the transform code that operates on df.",
        ])
    elif "Wrong-value delta: +" in transform_result and "Wrong-value delta: +0" not in transform_result:
        parts.extend([
            "",
            "*** WARNING: YOUR LAST TRANSFORM INTRODUCED WRONG VALUES ***",
            "Wrong values are penalized 1.5x — worse than leaving cells dirty.",
            "Undo immediately or target a DIFFERENT corruption type.",
            transform_result,
        ])

    # Detect when all original errors have been attempted (0 unfixed remaining)
    # — only wrong_value errors remain, model should stop or undo
    all_attempted = False
    error_text = str(obs.constraints[0]) if obs.constraints else ""
    if "0 errors still unfixed" in error_text or ("wrong value" in error_text.lower() and "still unfixed" not in error_text):
        # Check: are there ONLY wrong_value lines (no "Still need fixing")?
        if "Still need fixing" not in error_text:
            all_attempted = True
            parts.extend([
                "",
                "*** ALL ORIGINAL ERRORS HAVE BEEN ATTEMPTED ***",
                "Only wrong_value cells remain — these are cells YOUR transforms broke.",
                "You cannot fix them with more transforms. Submit {\"type\": \"done\"} now.",
            ])
            # Suppress explore/template hints — nothing productive to do
            template_hints = None
    elif transform_result:
        parts.extend(["", "Last action outcome:", transform_result])

    if warnings:
        parts.extend([""])
        for w in warnings:
            parts.append(f"WARNING: {w}")

    if obs.constraints:
        parts.extend(["", "Remaining errors:"])
        for desc in obs.constraints:
            for line in str(desc).splitlines():
                parts.append(f"  {line}")

    # Show prioritized fix plan with counts on every step (not just hard tasks)
    if total - fixed > 0 and obs.constraints:
        import re as _re_fix

        _PRIORITY = {
            "duplicate_rows": 0,
            "drop_rows": 0,
            "header_in_data": 0,
            "column_shift": 1,
            "value_swap": 1,
            "inject_nulls": 2,
            "whitespace_noise": 2,
            "type_mangle": 3,
            "format_inconsistency": 3,
            "outlier_injection": 4,
            "decimal_shift": 4,
            "category_misspell": 5,
            "typo_injection": 5,
        }
        # Parse using the same logic as _extract_remaining_error_targets, plus counts
        fix_plan_items: list[tuple[str, str, str]] = []
        for desc in obs.constraints or []:
            for raw_line in str(desc).splitlines():
                line = raw_line.strip()
                if "duplicate_rows" in line and "extra rows" in line:
                    count_m = _re_fix.search(r"(\d+)\s*extra", line)
                    fix_plan_items.append(("duplicate_rows", "", count_m.group(1) if count_m else ""))
                    continue
                if " in: " not in line:
                    continue
                left, right = line.split(" in: ", 1)
                # left is like "type_mangle (108 errors)"
                ctype = left.split("(")[0].strip().split()[-1].rstrip(":")
                count_m = _re_fix.search(r"\((\d+)\s*errors?\)", left)
                count = count_m.group(1) if count_m else ""
                cols = right.strip()
                fix_plan_items.append((ctype, cols, count))

        if fix_plan_items:
            sorted_types = sorted(fix_plan_items, key=lambda t: _PRIORITY.get(t[0], 6))
            parts.extend(["", "FIX PLAN (work top to bottom, one at a time):"])
            for i, (ctype, cols, count) in enumerate(sorted_types, 1):
                count_str = f" ({count} errors)" if count else ""
                col_str = f" in {cols}" if cols else ""
                parts.append(f"  {i}. {ctype}{col_str}{count_str}")

    if obs.diagnosis:
        parts.extend(["", "Diagnosis:", obs.diagnosis])

    explore_suggestions = _suggest_explore_queries(obs, action_history=action_history)
    if explore_suggestions:
        parts.extend(["", "Suggested explore queries:"])
        parts.extend(f"  {q}" for q in explore_suggestions)

    manual = _explore_manual(obs)
    if manual:
        parts.extend(["", "Explore guide:"])
        parts.extend(f"  - {line}" for line in manual)

    if obs.explore_result:
        parts.extend(["", "Last explore result:", obs.explore_result])
    if obs.validate_result:
        parts.extend(["", "Last validate result:", obs.validate_result])
        parts.append(
            "\nACTION REQUIRED: The validate breakdown above shows exactly which cells are wrong "
            "and what their expected values are. Use this to write a TARGETED transform — "
            "do NOT repeat your previous approach. Fix the specific cells/columns listed above."
        )

    parts.extend(["", "Data summary:", obs.data_summary])

    if diagnostic_text:
        parts.extend(["", "## Diagnostic Results (pre-computed — no explore budget used)", diagnostic_text])

    if template_hints:
        parts.extend(["", "## Fix Templates (adapt to actual data — do NOT copy blindly):", template_hints])

    if action_history:
        parts.extend(["", "Recent action history:"])
        for h in action_history[-10:]:
            summary = h["summary"][:80]
            parts.append(
                f'  Step {h["step"]}: {h["type"]} "{summary}" -> reward={h["reward_after"]:.4f}'
            )
        recent_transforms = [h for h in action_history if h["type"] == "transform"][-3:]
        if len(recent_transforms) >= 2:
            parts.extend(["", "## ALREADY TRIED — do NOT repeat these exact transforms:"])
            for h in recent_transforms:
                code = h["summary"].replace("\n", " | ")[:250]
                result_tag = "FAILED" if h.get("exec_failed") else f"fixed {h.get('errors_fixed', 0)}"
                parts.append(f'  Step {h["step"]} ({result_tag}): {code}')

    return "\n".join(parts)


def get_agent_action(
    messages: list[dict], temperature: float = 0.1
) -> tuple[dict, float, dict | None]:
    """Query the LLM for the next action. Returns (action_dict, latency_s, usage).

    Retries on rate limit errors with exponential backoff.
    temperature is dynamic (higher when model is stuck, lower after progress).
    """
    logger.debug(
        "LLM request — %d messages, last role=%s", len(messages), messages[-1]["role"]
    )
    logger.debug("Full messages:\n%s", json.dumps(messages, indent=2)[:3000])
    _jlog("llm_request", num_messages=len(messages),
          last_user_msg=messages[-1]["content"][:2000] if messages[-1]["role"] == "user" else "")

    global _last_call_time
    # Pace calls to avoid rate limits
    elapsed_since_last = time.time() - _last_call_time
    if elapsed_since_last < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed_since_last)

    max_retries = 5
    backoff = 5.0  # seconds

    for attempt in range(max_retries):
        t0 = time.time()
        try:
            response = get_llm_client().chat.completions.create(
                model=get_model_name(),
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
            latency = time.time() - t0
            _last_call_time = time.time()
            break
        except Exception as e:
            latency = time.time() - t0
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "too many" in err_str:
                wait = backoff * (2**attempt)
                logger.warning(
                    "Rate limit hit — waiting %.0fs before retry %d/%d | error: %s",
                    wait,
                    attempt + 1,
                    max_retries,
                    str(e)[:300],
                )
                _jlog("rate_limit_retry", attempt=attempt + 1, wait_s=wait)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            elif "context" in err_str or "exceed" in err_str or "400" in err_str:
                logger.warning(
                    "Context size exceeded or bad request — auto-submitting done | error: %s",
                    str(e)[:300],
                )
                _jlog("context_exceeded", error=str(e)[:200])
                return {"type": "done"}, latency, None
            else:
                raise

    content = response.choices[0].message.content.strip()
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    logger.debug("LLM response (%.2fs):\n%s", latency, content[:2000])
    _jlog(
        "llm_response",
        latency_s=round(latency, 3),
        **(usage or {}),
        response_preview=content[:500],
    )

    # Strip markdown code blocks if present
    if "```" in content:
        lines = content.split("\n")
        json_lines, in_block = [], False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        content = "\n".join(json_lines)

    try:
        action = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract the first complete JSON object using a decoder
        decoder = json.JSONDecoder()
        start = content.find("{")
        if start >= 0:
            try:
                action, _ = decoder.raw_decode(content, start)
            except json.JSONDecodeError:
                # Truncated JSON — try to salvage transform code
                import re as _salvage_re
                code_match = _salvage_re.search(
                    r'"code"\s*:\s*"((?:[^"\\]|\\.)*)',
                    content,
                )
                if code_match and '"type"' in content and '"transform"' in content:
                    action = {"type": "transform", "code": code_match.group(1)}
                    action["code"] = action["code"].encode().decode("unicode_escape")
                    logger.warning("Salvaged truncated transform (%d chars)", len(action["code"]))
                else:
                    action = {"type": "done"}
        else:
            action = {"type": "done"}

    return action, latency, usage


def _sanitize_transform_code(code: str) -> str:
    """Fix common small-model code issues before sandbox execution.

    1. Strip redundant import lines — pd, np, re, etc. are pre-loaded in the
       worker namespace. Models (esp. 9B+) prepend `import pandas as pd` which
       crashes because the sandbox blocks __import__.
    2. Fix spurious indentation from models that generate code as if inside a
       function body (first line at col 0, rest uniformly indented).
    """
    import re as _re
    import textwrap

    # ── Phase 1: Strip redundant imports ─────────────────────────────────────
    # These modules are already available in the worker namespace.
    _WORKER_MODULES = {
        "pandas", "pd", "numpy", "np", "re", "datetime", "string",
        "math", "json", "csv", "io", "collections", "itertools",
        "functools", "openpyxl", "yaml", "lxml",
        # Also strip these — not in worker but import crashes are confusing.
        # The code after the import will fail with NameError instead,
        # giving the model a clearer signal to use pd/np.
        "scipy", "sklearn", "statsmodels", "seaborn", "matplotlib",
    }
    cleaned_lines: list[str] = []
    for line in code.split("\n"):
        stripped = line.strip()
        # `import X`, `import X as Y`, `import X, Y`
        if stripped.startswith("import "):
            modules = stripped[7:].split(",")
            all_redundant = all(
                m.strip().split()[0].split(".")[0] in _WORKER_MODULES
                for m in modules if m.strip()
            )
            if all_redundant:
                continue  # drop this line entirely
        # `from X import ...`
        elif stripped.startswith("from "):
            match = _re.match(r"from\s+(\S+)", stripped)
            if match and match.group(1).split(".")[0] in _WORKER_MODULES:
                continue
        cleaned_lines.append(line)
    code = "\n".join(cleaned_lines)

    # ── Phase 1a: Strip comments to save tokens ──────────────────────────────
    # Only strip line-level comments (lines starting with optional whitespace + #)
    # to avoid breaking '#' inside strings like df['col'].replace(['##', ...])
    code = _re.sub(r"^\s*#[^\n]*$", "", code, flags=_re.MULTILINE)

    # ── Phase 1b: Strip df = pd.read_csv(...) reloads ───────────────────────
    # Small models re-load the file even though df is pre-loaded. The file
    # doesn't exist in sandbox, so the line either errors or silently resets df.
    code = _re.sub(
        r"^\s*df\s*=\s*pd\.read_(csv|json|excel|table|fwf)\(.*\)\s*$",
        "",
        code,
        flags=_re.MULTILINE,
    )

    # ── Phase 2: Fix spurious indentation ────────────────────────────────────
    lines = code.split("\n")
    if len(lines) > 1:
        first_nonempty = next((l for l in lines if l.strip()), None)
        if first_nonempty is not None:
            first_indent = len(first_nonempty) - len(first_nonempty.lstrip())

            if first_indent > 0:
                # First line itself is indented — textwrap.dedent handles it
                code = textwrap.dedent(code)
            else:
                # First line at col 0 — check if subsequent lines have uniform extra indent
                rest_non_empty = [l for l in lines[1:] if l.strip()]
                if rest_non_empty:
                    rest_indents = [len(l) - len(l.lstrip()) for l in rest_non_empty]
                    min_rest = min(rest_indents)

                    if min_rest > 0 and not first_nonempty.rstrip().endswith(":"):
                        new_lines = [lines[0]]
                        for line in lines[1:]:
                            if not line.strip():
                                new_lines.append(line)
                            elif len(line) - len(line.lstrip()) >= min_rest:
                                new_lines.append(line[min_rest:])
                            else:
                                new_lines.append(line)
                        code = "\n".join(new_lines)

    # ── Phase 2b: Warn if too many columns modified (don't block, just log) ──
    col_assignments = _re.findall(r"df\['([^']+)'\]\s*=", code)
    unique_cols = set(col_assignments)
    if len(unique_cols) > 3:
        logger.warning(
            "Transform touches %d columns (%s) — risk of wrong-value blowup",
            len(unique_cols), ", ".join(sorted(unique_cols)[:5]),
        )

    # ── Phase 3: Auto-complete truncated type_mangle ───────────────────────
    # Models sometimes generate 3 of 4 lines of the safe type_mangle pattern,
    # ending at pd.to_numeric() without the fillna(median) step.  The NaN
    # values left behind count as wrong_value cells.
    _mangle_pat = _re.search(
        r"df\['([^']+)'\]\s*=\s*pd\.to_numeric\(df\['([^']+)'\]",
        code,
    )
    if _mangle_pat and "fillna" not in code.split("to_numeric")[-1]:
        col = _mangle_pat.group(1)
        code = code.rstrip().rstrip(";")
        code += f"\ndf['{col}'] = df['{col}'].fillna(df['{col}'].median())"
        logger.warning("Auto-completed truncated type_mangle for '%s'", col)

    return code


def action_from_dict(d: dict):
    """Convert a dict to the correct Action type."""
    t = d.get("type", "done")
    if t == "explore":
        return ExploreAction(query=d.get("query", "df.head()"))
    elif t == "transform":
        code = _sanitize_transform_code(d.get("code", ""))
        d["code"] = code  # update dict so logs capture sanitized version
        return TransformAction(code=code)
    elif t == "undo":
        return UndoAction(step=int(d.get("step", 0)))
    elif t == "validate":
        return ValidateAction()
    else:
        return DoneAction()


# ── Diagnostic phase (A + B) ──────────────────────────────────────────────────

# Max programmatic explores in the diagnostic phase — keeps LLM explore budget usable.
DIAG_MAX_EXPLORES = 4


async def run_diagnostic_phase(env, obs, *, skip_validate: bool = False) -> tuple[str, str, object]:
    """Run pre-LLM diagnostics: targeted explore battery + optional validate.

    Returns (explore_text, validate_text, updated_obs).
    Explore results are collected from the env without spending the LLM's
    conversational explore budget more than DIAG_MAX_EXPLORES steps.
    """
    targets = _extract_remaining_error_targets(obs.constraints or [])

    explore_blocks: list[str] = []
    n_explores = 0

    async def _explore(query: str) -> None:
        nonlocal n_explores, obs
        if n_explores >= DIAG_MAX_EXPLORES:
            return
        try:
            result = await env.step(ExploreAction(query=query))
            obs = result.observation
            text = (obs.explore_result or "").strip()
            if text:
                explore_blocks.append(f">>> {query}\n{text}")
            n_explores += 1
        except Exception as exc:
            logger.warning("Diagnostic explore '%s' failed: %s", query, exc)

    # 1. Global null overview — always useful
    await _explore("df.isnull().sum()")

    # 2. Column-specific probes keyed by corruption type (highest-signal first)
    probed_cols: set[str] = set()
    priority_order = [
        "type_mangle",
        "whitespace_noise",
        "category_misspell",
        "typo_injection",
        "format_inconsistency",
        "inject_nulls",
        "null_injected",
        "outlier_injection",
        "decimal_shift",
        "value_swap",
        "duplicate_rows",
    ]
    for ctype in priority_order:
        if n_explores >= DIAG_MAX_EXPLORES:
            break
        cols = targets.get(ctype, [])
        for col in cols[:2]:
            if col in probed_cols or n_explores >= DIAG_MAX_EXPLORES:
                continue
            probed_cols.add(col)
            if ctype in (
                "whitespace_noise",
                "type_mangle",
                "format_inconsistency",
                "category_misspell",
                "typo_injection",
            ):
                await _explore(f"df[{repr(col)}].value_counts(dropna=False).head(15)")
            elif ctype in ("inject_nulls", "null_injected"):
                await _explore(f"df[{repr(col)}].isnull().sum()")
            elif ctype in ("outlier_injection", "decimal_shift"):
                await _explore(f"df[{repr(col)}].describe()")
            else:
                await _explore(f"df[{repr(col)}].head(10)")

    # 3. Validate — gives per-cell expected vs actual breakdown
    # Skip for easy tasks to preserve validate budget (costs 0.01 penalty each)
    validate_text = ""
    if not skip_validate:
        try:
            vresult = await env.step(ValidateAction())
            obs = vresult.observation
            validate_text = (obs.validate_result or "").strip()
        except Exception as exc:
            logger.warning("Diagnostic validate failed: %s", exc)

    explore_text = "\n\n".join(explore_blocks)
    _jlog(
        "diagnostic_phase",
        n_explores=n_explores,
        has_validate=bool(validate_text),
    )
    logger.info(
        "Diagnostic phase complete — %d explores, validate=%s",
        n_explores,
        bool(validate_text),
    )
    return explore_text, validate_text, obs


# ── Run Task ──────────────────────────────────────────────────────────────────


async def run_task(
    dataset_id: str,
    difficulty: str,
    fmt: str | None = "csv",
    category: str | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
) -> float:
    """Run the agent on a single task via WebSocket. Returns final reward."""
    task_id = f"{dataset_id}_{difficulty}_{fmt or 'auto'}"
    log_start(task_id, category=category or "", seed=seed or 0)
    task_start = time.time()
    current_reward = 0.0
    step_num = 0
    step_rewards: list[float] = []
    logger.info(
        "Starting task: %s | model: %s | endpoint: %s",
        task_id,
        get_model_name(),
        get_api_base_url(),
    )

    env_client = DataCleaningClient(base_url=ENV_URL)

    async with env_client as env:
        reset_kwargs = dict(task_id=dataset_id, difficulty=difficulty)
        if fmt is not None:
            reset_kwargs["format"] = fmt
        if category is not None:
            reset_kwargs["category"] = category
        if seed is not None:
            reset_kwargs["seed"] = seed
        step_result = await env.reset(**reset_kwargs)
        obs = step_result.observation
        current_reward = step_result.reward or 0.0

        constraint_status = obs.constraint_status or {}
        fixed = sum(1 for v in constraint_status.values() if v)
        total = len(constraint_status)
        logger.info(
            "Reset complete — %d/%d errors fixed, reward=%.4f",
            fixed,
            total,
            current_reward,
        )

        # ── Diagnostic phase: auto-explore + validate before LLM's first turn (A+B) ──
        # Skip validate for easy tasks — saves budget (0.01 penalty) and they're small enough
        diag_explore_text, diag_validate_text, obs = await run_diagnostic_phase(
            env, obs, skip_validate=(difficulty == "easy"),
        )
        # Re-read constraint status after diagnostic validate may have updated obs
        constraint_status = obs.constraint_status or {}
        fixed = sum(1 for v in constraint_status.values() if v)
        total = len(constraint_status)

        # Build corruption-type templates from the error summary (C)
        template_hints = _build_template_hints(obs)

        # ── Auto-submit safe transforms for known corruption types ───────────
        # Deterministic first-pass: submit template code for unambiguous types
        # before the LLM starts. This handles types small models struggle with.
        auto_action_history: list[dict] = []
        auto_step = 0
        targets = _extract_remaining_error_targets(obs.constraints)
        _SAFE_TEMPLATES: dict[str, callable] = {
            "duplicate_rows": lambda _col: "df = df.drop_duplicates()",
            "inject_nulls": lambda col: (
                f"df['{col}'] = df['{col}'].fillna("
                f"df['{col}'].median() if pd.api.types.is_numeric_dtype(df['{col}']) "
                f"else df['{col}'].mode().iloc[0])"
            ),
            "whitespace_noise": lambda col: (
                f"df['{col}'] = df['{col}'].astype(str)"
                f".str.replace(r'\\s+', ' ', regex=True).str.strip()"
            ),
        }
        for ctype, cols in targets.items():
            if ctype not in _SAFE_TEMPLATES:
                continue
            if ctype == "duplicate_rows":
                code = _SAFE_TEMPLATES[ctype]("")
                auto_step += 1
                try:
                    result = await env.step(TransformAction(code=code))
                    obs = result.observation
                    current_reward = result.reward if result.reward is not None else current_reward
                    constraint_status = obs.constraint_status or {}
                    new_fixed = sum(1 for v in constraint_status.values() if v)
                    if new_fixed >= fixed:
                        logger.info(
                            "Auto-transform %s: %d/%d fixed, reward=%.4f",
                            ctype, new_fixed, total, current_reward,
                        )
                        auto_action_history.append({
                            "step": auto_step, "type": "transform",
                            "summary": f"auto: {code[:60]}",
                            "reward_after": current_reward, "errors_fixed": new_fixed,
                        })
                        fixed = new_fixed
                    else:
                        logger.warning("Auto-transform %s regressed %d->%d, skipping", ctype, fixed, new_fixed)
                except Exception as exc:
                    logger.warning("Auto-transform %s failed: %s", ctype, exc)
            else:
                for col in cols[:2]:
                    code = _SAFE_TEMPLATES[ctype](col)
                    auto_step += 1
                    try:
                        result = await env.step(TransformAction(code=code))
                        obs = result.observation
                        current_reward = result.reward if result.reward is not None else current_reward
                        constraint_status = obs.constraint_status or {}
                        new_fixed = sum(1 for v in constraint_status.values() if v)
                        if new_fixed >= fixed:
                            logger.info(
                                "Auto-transform %s(%s): %d/%d fixed, reward=%.4f",
                                ctype, col, new_fixed, total, current_reward,
                            )
                            auto_action_history.append({
                                "step": auto_step, "type": "transform",
                                "summary": f"auto: {code[:60]}",
                                "reward_after": current_reward, "errors_fixed": new_fixed,
                            })
                            fixed = new_fixed
                        else:
                            logger.warning("Auto-transform %s(%s) regressed %d->%d, skipping", ctype, col, fixed, new_fixed)
                    except Exception as exc:
                        logger.warning("Auto-transform %s(%s) failed: %s", ctype, col, exc)

        # Refresh template hints after auto-transforms
        if auto_action_history:
            template_hints = _build_template_hints(obs)

        action_history: list[dict] = auto_action_history
        step_rewards: list[float] = []
        best_reward: float = current_reward
        best_fixed: int = fixed
        # Undo target: if auto-transforms improved state, checkpoint at that step
        # so auto-undo doesn't revert past auto-transform gains
        best_transform_step: int = auto_step if auto_action_history else 0

        # Escalation ladder state (D)
        stale_count: int = 0   # consecutive non-improving transforms
        current_temp: float = 0.1  # dynamic temperature (F)
        consecutive_blocks: int = 0  # consecutive blocked actions (validate/undo/dup)
        total_auto_undos: int = 0   # total auto-undos (escalation + regression)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    obs,
                    current_reward,
                    action_history=action_history,
                    diagnostic_text=diag_explore_text or None,
                    template_hints=template_hints or None,
                ),
            },
        ]

        step_num = 0
        max_steps = max_steps or MAX_STEPS_BY_DIFFICULTY.get(difficulty, 60)

        while step_num < max_steps:
            step_num += 1

            action_dict, latency, usage = get_agent_action(messages, temperature=current_temp)
            action = action_from_dict(action_dict)
            action_type = action_dict.get("type", "done")

            # ── Block model-initiated undos that would destroy progress ────────
            # Replace last user message instead of appending — avoids inflating context.
            if action_type == "undo" and best_fixed > 0:
                logger.warning("BLOCKED model undo (best_fixed=%d)", best_fixed)
                _jlog("blocked_undo", step=step_num, best_fixed=best_fixed)
                step_num -= 1
                messages[-1] = {
                    "role": "user",
                    "content": build_user_prompt(
                        obs, current_reward, action_history=action_history,
                        warnings=[
                            f"BLOCKED: You tried to undo, but you have {best_fixed}/{total} errors fixed. "
                            f"Focus on fixing the remaining {total - best_fixed} errors with a transform "
                            f"targeting a different corruption type."
                        ],
                    ),
                }
                continue

            # ── Block validate loops — model stuck calling validate repeatedly ────
            if action_type == "validate":
                recent_validates = sum(
                    1 for h in action_history[-4:] if h["type"] == "validate"
                )
                validate_budget_left = bool(
                    obs.step_info
                    and obs.step_info.validate_uses < obs.step_info.validate_budget
                )
                if recent_validates >= 2 or not validate_budget_left:
                    consecutive_blocks += 1
                    if consecutive_blocks >= 5:
                        logger.warning("Model hopelessly stuck (%d blocks) — auto-done", consecutive_blocks)
                        break
                    logger.warning(
                        "BLOCKED validate loop (recent_validates=%d, budget_left=%s, blocks=%d)",
                        recent_validates, validate_budget_left, consecutive_blocks,
                    )
                    _jlog("blocked_validate_loop", step=step_num, recent_validates=recent_validates)
                    step_num -= 1
                    messages[-1] = {
                        "role": "user",
                        "content": build_user_prompt(
                            obs, current_reward, action_history=action_history,
                            warnings=[
                                "BLOCKED: You cannot call validate again right now. "
                                "You MUST submit a transform. "
                                "Use the error summary above to pick the most common corruption type and fix it."
                            ],
                            template_hints=_build_template_hints(obs),
                        ),
                    }
                    continue

            # ── Block premature done — model calls done before fixing anything ────
            if action_type == "done" and fixed == 0 and total > 0:
                consecutive_blocks += 1
                if consecutive_blocks >= 3:
                    logger.warning("Model done with 0 fixed and stuck — forcing exit")
                    break
                logger.warning("BLOCKED premature done — 0/%d errors fixed", total)
                _jlog("blocked_premature_done", step=step_num, fixed=fixed, total=total)
                step_num -= 1
                messages[-1] = {
                    "role": "user",
                    "content": build_user_prompt(
                        obs, current_reward, action_history=action_history,
                        warnings=[
                            f"BLOCKED: You called done but have fixed 0/{total} errors. "
                            "You MUST submit at least one transform before finishing. "
                            "Use the error summary above to identify the corruption type and fix it."
                        ],
                        template_hints=_build_template_hints(obs),
                    ),
                }
                continue

            # ── Block duplicate transforms — models repeat exact same code ────────
            # Replace last user message instead of appending — avoids inflating context.
            if action_type == "transform":
                new_code = action_dict.get("code", "").strip()
                past_codes = [
                    h["summary"].strip()
                    for h in action_history
                    if h["type"] == "transform"
                ]
                if any(new_code[:150] == old[:150] for old in past_codes if old):
                    consecutive_blocks += 1
                    if consecutive_blocks >= 5:
                        logger.warning("Model hopelessly stuck (%d blocks) — auto-done", consecutive_blocks)
                        break
                    logger.warning("BLOCKED duplicate transform — same code as a previous step (blocks=%d)", consecutive_blocks)
                    _jlog("blocked_duplicate_transform", step=step_num, consecutive_blocks=consecutive_blocks)
                    step_num -= 1
                    # Build a directive hint: which corruption types haven't been successfully fixed yet?
                    remaining = _extract_remaining_error_targets(obs.constraints or [])
                    # Figure out which types the model already tried (from action_history)
                    tried_codes = [h["summary"] for h in action_history if h["type"] == "transform"]
                    tried_cols = set()
                    for c in tried_codes:
                        for col_match in __import__("re").findall(r"df\['([^']+)'\]", c):
                            tried_cols.add(col_match)
                    # Find untried corruption types (types with columns not yet attempted)
                    untried = []
                    for ctype, cols in remaining.items():
                        untried_cols = [c for c in cols if c not in tried_cols] if cols else []
                        if untried_cols or (not cols and ctype not in str(tried_codes)):
                            untried.append((ctype, untried_cols))

                    block_warning = (
                        "BLOCKED: You submitted the EXACT SAME transform code as a previous step. "
                        "You MUST target a DIFFERENT corruption type or different columns."
                    )
                    if untried:
                        next_type, next_cols = untried[0]
                        if next_cols:
                            block_warning += f"\n>>> TRY THIS: Fix '{next_type}' in column '{next_cols[0]}' next."
                        else:
                            block_warning += f"\n>>> TRY THIS: Fix '{next_type}' next."
                    elif remaining:
                        # All types tried but some still unfixed — suggest re-approach
                        first_type = next(iter(remaining))
                        block_warning += f"\n>>> Your previous attempt at '{first_type}' didn't fully work. Try a different approach for it."

                    messages[-1] = {
                        "role": "user",
                        "content": build_user_prompt(
                            obs, current_reward, action_history=action_history,
                            warnings=[block_warning],
                            template_hints=_build_template_hints(obs),
                        ),
                    }
                    continue

            # Log action content at INFO
            if action_type == "explore":
                logger.info(
                    "Step %d | explore | query: %s | latency=%.2fs%s | temp=%.1f",
                    step_num,
                    action_dict.get("query", "")[:200],
                    latency,
                    f" | tokens={usage['total_tokens']}" if usage else "",
                    current_temp,
                )
            elif action_type == "transform":
                code_preview = action_dict.get("code", "").replace("\n", " ")[:200]
                logger.info(
                    "Step %d | transform | code: %s | latency=%.2fs%s | temp=%.1f",
                    step_num,
                    code_preview,
                    latency,
                    f" | tokens={usage['total_tokens']}" if usage else "",
                    current_temp,
                )
            else:
                logger.info("Step %d | %s | latency=%.2fs", step_num, action_type, latency)

            if action_type in ("transform", "done", "undo"):
                consecutive_blocks = 0  # real progress action — reset block counter
            step_result = await env.step(action)
            obs = step_result.observation
            current_reward = (
                step_result.reward if step_result.reward is not None else current_reward
            )

            constraint_status = obs.constraint_status or {}
            fixed = sum(1 for v in constraint_status.values() if v)
            total = len(constraint_status)

            logger.info(
                "Step %d result | reward=%.4f | %d/%d errors fixed | stale=%d",
                step_num,
                current_reward,
                fixed,
                total,
                stale_count,
            )
            logger.debug(
                "Observation data_summary:\n%s",
                obs.data_summary[:1000] if obs.data_summary else "",
            )
            if obs.explore_result:
                logger.debug("Explore result:\n%s", obs.explore_result[:1000])
            if obs.transform_result:
                logger.debug("Transform result:\n%s", obs.transform_result[:500])

            action_content = action_dict.get("query") or action_dict.get("code") or ""
            step_rewards.append(current_reward)
            log_step(
                step_num,
                action_type,
                current_reward,
                done=step_result.done,
                action_content=action_content,
                latency=latency,
                usage=usage,
                errors_fixed=fixed,
                errors_total=total,
            )

            # Detect execution failures from transform_result
            transform_exec_failed = (
                action_type == "transform"
                and "Execution: failed" in (obs.transform_result or "")
            )

            # Track action history
            action_history.append(
                {
                    "step": step_num,
                    "type": action_type,
                    "summary": (
                        action_dict.get("query") or action_dict.get("code", "")
                    )[:300],
                    "reward_after": current_reward,
                    "errors_fixed": fixed,
                    "exec_failed": transform_exec_failed,
                }
            )

            # Per-step warnings — initialized here so auto-undo can append before
            # the later warning-generation block adds more.
            warnings: list[str] = []

            # ── Major regression: auto-undo immediately ────────────────────────
            # MUST check BEFORE updating best state, otherwise reward drop is masked
            did_regression_undo = False
            if action_type == "transform" and not transform_exec_failed and best_fixed > 0:
                # Case 1: errors-fixed count dropped significantly
                fixed_regression = fixed < best_fixed and (best_fixed - fixed) / best_fixed >= 0.25
                # Case 2: reward dropped significantly — wrong values introduced
                # (even if fixed count increased, reward drop means wrong values)
                wrong_value_regression = (
                    best_reward > 0
                    and current_reward < best_reward * 0.7  # 30%+ reward drop
                )
                if (fixed_regression or wrong_value_regression) and total_auto_undos < 4:
                    reason = "wrong values" if wrong_value_regression else "errors-fixed drop"
                    logger.warning(
                        "Auto-undo (%s): reward %.4f→%.4f, fixed %d→%d — reverting to checkpoint %d",
                        reason,
                        best_reward,
                        current_reward,
                        best_fixed,
                        fixed,
                        best_transform_step,
                    )
                    undo_result = await env.step(UndoAction(step=best_transform_step))
                    obs = undo_result.observation
                    current_reward = (
                        undo_result.reward
                        if undo_result.reward is not None
                        else current_reward
                    )
                    step_rewards.append(current_reward)
                    constraint_status = obs.constraint_status or {}
                    fixed = sum(1 for v in constraint_status.values() if v)
                    total = len(constraint_status)
                    log_step(
                        step_num,
                        "undo",
                        current_reward,
                        done=False,
                        errors_fixed=fixed,
                        errors_total=total,
                    )
                    action_history.append(
                        {
                            "step": step_num,
                            "type": "undo",
                            "summary": f"auto-undo ({reason}) to checkpoint {best_transform_step}",
                            "reward_after": current_reward,
                            "errors_fixed": fixed,
                        }
                    )
                    stale_count = max(0, stale_count - 1)  # partial credit for undo
                    did_regression_undo = True
                    total_auto_undos += 1
                    # Sync best_fixed to the post-undo state (data matches checkpoint).
                    # Keep best_reward as-is — the pre-regression peak is the true
                    # best.  Resetting to post-undo value (which includes undo penalty)
                    # would permanently lower the bar and mask future regressions.
                    best_fixed = fixed
                    # Build explicit post-undo warning so model avoids the same mistake
                    failed_code = action_dict.get("code", "")
                    failed_cols = __import__("re").findall(r"df\['([^']+)'\]", failed_code)
                    undo_warning = (
                        f"AUTO-UNDO: Your last transform caused a {reason} "
                        f"(reward {best_reward:.4f}→{current_reward:.4f}). "
                        f"Reverted to checkpoint {best_transform_step} "
                        f"({fixed}/{total} fixed)."
                    )
                    if failed_cols:
                        undo_warning += (
                            f" Avoid column(s) {', '.join(set(failed_cols))} "
                            f"with that approach — target a DIFFERENT corruption type or column."
                        )
                    else:
                        undo_warning += (
                            " Target a DIFFERENT corruption type or column next."
                        )
                    warnings.append(undo_warning)

            # ── Update best state + stale_count after transforms ──────────────
            # (placed AFTER regression check so reward drop isn't masked)
            if action_type == "transform" and not did_regression_undo:
                if transform_exec_failed:
                    stale_count += 1
                    current_temp = min(0.1 + stale_count * 0.2, 0.7)
                    _jlog("transform_exec_failed", step=step_num, stale_count=stale_count,
                          error=(obs.transform_result or "")[:200])
                    logger.warning("Step %d transform FAILED to execute — stale=%d", step_num, stale_count)
                elif fixed > best_fixed:
                    # Always update checkpoint when more errors fixed (for undo targeting)
                    best_fixed = fixed
                    best_transform_step = sum(
                        1
                        for h in action_history
                        if h["type"] == "transform" and not h.get("exec_failed")
                    )
                    if current_reward >= best_reward:
                        # Both improved — clear stale, update reward checkpoint
                        stale_count = 0
                        best_reward = current_reward
                        current_temp = 0.1
                    else:
                        # Fixed more but reward dropped — wrong values introduced
                        stale_count += 1
                        current_temp = min(0.1 + stale_count * 0.2, 0.7)
                elif current_reward > best_reward:
                    # Reward improved without fixing more — still good progress
                    stale_count = 0
                    best_reward = current_reward
                    current_temp = 0.1
                else:
                    stale_count += 1
                    current_temp = min(0.1 + stale_count * 0.2, 0.7)
                    _jlog("stale_transform", step=step_num, stale_count=stale_count,
                          fixed=fixed, best_fixed=best_fixed, temp=current_temp)

            # ── Generate per-step warnings ────────────────────────────────────
            if action_type == "explore":
                query = action_dict.get("query", "")
                recent_explores = [
                    h["summary"] for h in action_history[:-1] if h["type"] == "explore"
                ][-3:]
                if query[:100] in recent_explores:
                    warnings.append(
                        "You already ran this exact explore query. Try a different query or submit a transform."
                    )
                same_query_streak = _same_query_streak(action_history)
                consecutive_explores = _consecutive_explore_count(action_history)
                if same_query_streak >= 2:
                    warnings.append(
                        "You repeated the same explore query multiple times. Use validate or transform instead of asking again."
                    )
                if consecutive_explores >= 3:
                    warnings.append(
                        f"BLOCKED: You have explored {consecutive_explores} times in a row without transforming. "
                        f"You MUST submit a transform on your next action. Target the highest-count corruption type in the FIX PLAN."
                    )
            if action_type == "transform":
                recent_transform_fixed = [
                    h["errors_fixed"]
                    for h in action_history
                    if h["type"] == "transform"
                ]
                if len(recent_transform_fixed) >= 2:
                    prev = recent_transform_fixed[-2]
                    curr = recent_transform_fixed[-1]
                    if curr < prev:
                        warnings.append(
                            f"REGRESSION: Your last transform REDUCED errors fixed from {prev}/{total} to {curr}/{total}. "
                            f"The system will auto-undo if needed. Focus on a DIFFERENT corruption type next."
                        )
                    elif curr == prev:
                        if curr == 0:
                            warnings.append(
                                f"Your last {stale_count} transforms fixed ZERO errors. "
                                f"Your approach is fundamentally wrong. Inspect the ACTUAL dirty values "
                                f"(e.g. df['col'].unique()[:20]) — do not guess. Then try a completely different fix."
                            )
                        else:
                            warnings.append(
                                f"Your last {stale_count} transforms fixed the same number of errors ({fixed}/{total}). "
                                f"Try a fundamentally different approach. You must target a DIFFERENT corruption type."
                            )

            # ── Escalation ladder (D): stale transform handling ────────────────
            should_auto_done = False

            # Explore loop → validate or done (kept from previous logic)
            if action_type == "explore":
                same_query_streak = _same_query_streak(action_history)
                consecutive_explores = _consecutive_explore_count(action_history)
                validate_available = bool(
                    obs.step_info
                    and obs.step_info.validate_uses < obs.step_info.validate_budget
                )
                budget_exhausted = (
                    "budget exhausted" in (obs.explore_result or "").lower()
                )

                if validate_available and (
                    same_query_streak >= 2
                    or consecutive_explores >= 4
                    or budget_exhausted
                ):
                    logger.warning(
                        "Explore loop — auto-validate (same_query=%d consec=%d)",
                        same_query_streak,
                        consecutive_explores,
                    )
                    vresult = await env.step(ValidateAction())
                    obs = vresult.observation
                    current_reward = vresult.reward if vresult.reward is not None else current_reward
                    step_rewards.append(current_reward)
                    constraint_status = obs.constraint_status or {}
                    fixed = sum(1 for v in constraint_status.values() if v)
                    total = len(constraint_status)
                    log_step(step_num + 1, "validate", current_reward,
                             done=vresult.done, errors_fixed=fixed, errors_total=total)
                    action_history.append({"step": step_num + 1, "type": "validate",
                                           "summary": "", "reward_after": current_reward,
                                           "errors_fixed": fixed})
                elif same_query_streak >= 3 or consecutive_explores >= 6:
                    logger.warning("Explore loop persists — auto-done")
                    should_auto_done = True

            # Transform stale → escalation ladder (only if no regression-undo already fired)
            if action_type == "transform" and not did_regression_undo:
                validate_available = bool(
                    obs.step_info
                    and obs.step_info.validate_uses < obs.step_info.validate_budget
                )

                if stale_count >= 1 and validate_available:
                    # Tier 2: auto-validate to give the model exact cell-level info
                    logger.warning(
                        "Escalation tier 2: %d stale transforms — auto-validate", stale_count
                    )
                    step_num += 1
                    vresult = await env.step(ValidateAction())
                    obs = vresult.observation
                    current_reward = vresult.reward if vresult.reward is not None else current_reward
                    step_rewards.append(current_reward)
                    constraint_status = obs.constraint_status or {}
                    fixed = sum(1 for v in constraint_status.values() if v)
                    total = len(constraint_status)
                    log_step(step_num, "validate", current_reward,
                             done=vresult.done, errors_fixed=fixed, errors_total=total)
                    action_history.append({"step": step_num, "type": "validate",
                                           "summary": "escalation-auto", "reward_after": current_reward,
                                           "errors_fixed": fixed})
                    _jlog("escalation_validate", step=step_num, stale_count=stale_count)

                elif stale_count == 3 and best_fixed > 0 and total_auto_undos >= 2:
                    # Too many undos — each costs reward, stop thrashing
                    logger.warning(
                        "Escalation: %d auto-undos already — auto-done instead of undo #%d",
                        total_auto_undos, total_auto_undos + 1,
                    )
                    should_auto_done = True

                elif stale_count == 3 and best_fixed > 0:
                    # Tier 3: auto-undo to best known good state
                    logger.warning(
                        "Escalation tier 3: %d stale transforms — auto-undo to checkpoint %d",
                        stale_count,
                        best_transform_step,
                    )
                    step_num += 1
                    undo_result = await env.step(UndoAction(step=best_transform_step))
                    obs = undo_result.observation
                    current_reward = (
                        undo_result.reward if undo_result.reward is not None else current_reward
                    )
                    step_rewards.append(current_reward)
                    constraint_status = obs.constraint_status or {}
                    fixed = sum(1 for v in constraint_status.values() if v)
                    total = len(constraint_status)
                    log_step(step_num, "undo", current_reward,
                             done=False, errors_fixed=fixed, errors_total=total)
                    action_history.append({"step": step_num, "type": "undo",
                                           "summary": f"escalation-undo to checkpoint {best_transform_step}",
                                           "reward_after": current_reward, "errors_fixed": fixed})
                    total_auto_undos += 1
                    # Sync best_fixed to the post-undo state (data matches checkpoint).
                    # Keep best_reward as-is — the pre-undo peak is the true best.
                    best_fixed = fixed
                    _jlog("escalation_undo", step=step_num, stale_count=stale_count,
                          best_transform_step=best_transform_step,
                          total_auto_undos=total_auto_undos)
                    # Reset stale_count so the agent gets a real attempt after the undo
                    stale_count = 0
                    current_temp = 0.3  # slightly elevated — still needs diversity
                    warnings.append(
                        f"ESCALATION: Auto-undone to your best checkpoint ({best_fixed}/{total} fixed). "
                        f"The remaining errors need a DIFFERENT approach — look at the Diagnostic Results "
                        f"above and target a corruption type you haven't fixed yet."
                    )

                elif stale_count >= 3 and best_fixed == 0:
                    # No checkpoint to undo to — nothing ever worked, give up
                    logger.warning(
                        "Escalation tier 3 (no checkpoint): %d stale transforms, best_fixed=0 — auto-done",
                        stale_count,
                    )
                    should_auto_done = True

                elif stale_count >= 4:
                    # Tier 4: give up
                    logger.warning(
                        "Escalation tier 4: %d stale transforms — auto-done", stale_count
                    )
                    should_auto_done = True

            if should_auto_done:
                _jlog("auto_done_stuck", step=step_num, reward=current_reward,
                      stale_count=stale_count)
                done_result = await env.step(DoneAction())
                current_reward = (
                    done_result.reward if done_result.reward is not None else current_reward
                )
                step_rewards.append(current_reward)
                log_step(
                    step_num + 1,
                    "done",
                    current_reward,
                    done=True,
                    errors_fixed=fixed,
                    errors_total=total,
                )
                break

            # Re-include template hints when model is stuck — hints drop out
            # after the first turn due to message trimming, leaving the model
            # without guidance on the correct fix pattern for each corruption type.
            rehint = _build_template_hints(obs) if stale_count > 0 else None

            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            messages.append(
                {
                    "role": "user",
                    "content": build_user_prompt(
                        obs,
                        current_reward,
                        action_history=action_history,
                        warnings=warnings,
                        template_hints=rehint,
                    ),
                }
            )

            # Keep message history bounded — retain system + last N exchanges.
            MAX_HISTORY_MESSAGES = 16 if rehint else 20
            if len(messages) > MAX_HISTORY_MESSAGES:
                messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1) :]

            # Adaptive context guard — estimate tokens (~3.5 chars/token) and trim
            # until we're safely under the context limit (leave 2K for completion).
            CTX_LIMIT = 32768  # match llama-server -c setting
            COMPLETION_RESERVE = 2048
            MAX_PROMPT_TOKENS = CTX_LIMIT - COMPLETION_RESERVE
            while len(messages) > 3:  # keep at least system + 1 exchange
                est_tokens = sum(len(m["content"]) for m in messages) // 3
                if est_tokens <= MAX_PROMPT_TOKENS:
                    break
                # Drop oldest exchange (assistant + user pair after system)
                messages = [messages[0]] + messages[3:]
                logger.debug("Trimmed messages to %d (est ~%d tokens)", len(messages), est_tokens)

            if step_result.done:
                break

            # ── Soft-done handling: env returns done=false on first done() ────
            # Small models don't understand "soft done" — they repeat transforms
            # and get stuck. Auto-finalize by sending done() again.
            if action_type == "done" and not step_result.done:
                logger.info("Soft-done received (reward=%.4f, %d/%d fixed) — auto-finalizing", current_reward, fixed, total)
                final_result = await env.step(DoneAction())
                current_reward = final_result.reward if final_result.reward is not None else current_reward
                step_rewards.append(current_reward)
                log_step(step_num + 1, "done", current_reward, done=True, errors_fixed=fixed, errors_total=total)
                _jlog("auto_finalize_done", step=step_num + 1, reward=current_reward)
                break

            if total > 0 and fixed == total:
                done_result = await env.step(DoneAction())
                current_reward = (
                    done_result.reward
                    if done_result.reward is not None
                    else current_reward
                )
                step_rewards.append(current_reward)
                log_step(
                    step_num + 1,
                    "done",
                    current_reward,
                    done=True,
                    errors_fixed=fixed,
                    errors_total=total,
                )
                _jlog("auto_done", step=step_num + 1, reward=current_reward)
                logger.info("All errors fixed — submitted done action")
                break

    elapsed = time.time() - task_start
    log_end(task_id, current_reward, step_num, elapsed, step_rewards,
            category=category or "", seed=seed or 0)
    logger.info(
        "Task %s complete | reward=%.4f | steps=%d | elapsed=%.1fs",
        task_id,
        current_reward,
        step_num,
        elapsed,
    )
    return current_reward


# ── Main ──────────────────────────────────────────────────────────────────────


def _parse_cli_tasks(args: list[str]) -> list[tuple[str, str]]:
    """Parse CLI args as 'dataset_id difficulty' pairs or 'dataset_id/difficulty'."""
    tasks = []
    i = 0
    while i < len(args):
        arg = args[i]
        parts = arg.split("/")
        if len(parts) == 3:
            tasks.append((parts[0], parts[1], parts[2]))
            i += 1
        elif len(parts) == 2:
            tasks.append((parts[0], parts[1], "csv"))
            i += 1
        elif i + 1 < len(args) and args[i + 1] in ("easy", "medium", "hard"):
            fmt = (
                args[i + 2]
                if i + 2 < len(args) and args[i + 2] not in ("easy", "medium", "hard")
                else "csv"
            )
            step = 3 if fmt != "csv" else 2
            tasks.append((arg, args[i + 1], fmt))
            i += step
        else:
            raise SystemExit(
                f"Cannot parse task: {arg!r}.\n"
                f"Examples: titanic easy  |  titanic/easy  |  titanic/easy/json"
            )
    return tasks


async def amain():
    tasks = _parse_cli_tasks(sys.argv[1:]) if len(sys.argv) > 1 else TASKS
    results: dict[str, float] = {}

    for dataset_id, difficulty, fmt in tasks:
        task_id = f"{dataset_id}_{difficulty}_{fmt}"
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Running task: {task_id}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        task_start = time.time()
        try:
            reward = await run_task(dataset_id, difficulty, fmt)
            results[task_id] = reward
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            elapsed = time.time() - task_start
            log_end(task_id, 0.0, 0, elapsed, [])
            results[task_id] = 0.0

    print(f"\n{'='*60}", file=sys.stderr)
    print("Summary:", file=sys.stderr)
    for tid, reward in results.items():
        print(f"  {tid}: {reward}", file=sys.stderr)
    avg = sum(results.values()) / len(results) if results else 0
    print(f"  Average: {avg:.4f}", file=sys.stderr)

    # Append results to CSV for cross-run comparison
    import csv as csv_mod

    csv_path = LOG_DIR / "results.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv_mod.writer(f)
        if write_header:
            writer.writerow(["run_id", "model", "task_id", "reward", "timestamp"])
        run_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for tid, reward in results.items():
            writer.writerow([_RUN_ID, get_model_name(), tid, round(reward, 4), run_ts])


def main():
    try:
        asyncio.run(amain())
    finally:
        _jsonl_file.close()


if __name__ == "__main__":
    main()
