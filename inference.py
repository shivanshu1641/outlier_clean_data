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
from models import DoneAction, ExploreAction, TransformAction, UndoAction, ValidateAction

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
    ("titanic",        "easy",   "csv"),
    ("titanic",        "easy",   "tsv"),
    ("titanic",        "medium", "csv"),
    ("titanic",        "medium", "json"),
    ("titanic",        "hard",   "csv"),
    ("titanic",        "hard",   "json"),
    ("titanic",        "hard",   "xml"),
    ("iris",           "easy",   "csv"),
    ("iris",           "medium", "csv"),
    ("iris",           "medium", "jsonl"),
    ("boston_housing", "medium", "csv"),
    ("boston_housing", "medium", "json"),
    ("boston_housing", "hard",   "csv"),
    ("boston_housing", "hard",   "xml"),
    ("diabetes",       "medium", "csv"),
    ("diabetes",       "medium", "jsonl"),
    ("diabetes",       "hard",   "csv"),
    ("diabetes",       "hard",   "json"),
    ("wine_quality",   "easy",   "csv"),
    ("wine_quality",   "medium", "csv"),
    ("wine_quality",   "medium", "json"),
    ("wine_quality",   "hard",   "csv"),
    ("wine_quality",   "hard",   "xml"),
    ("breast_cancer",  "easy",   "csv"),
    ("breast_cancer",  "medium", "csv"),
    ("breast_cancer",  "medium", "jsonl"),
]

TASKS = EVAL_TASKS

# Max agent steps per difficulty — hard tasks get more time
MAX_STEPS_BY_DIFFICULTY = {"easy": 30, "medium": 60, "hard": 100}

# Minimum seconds between LLM calls to avoid rate limits (Groq free: 30 req/min)
MIN_CALL_INTERVAL = float(os.environ.get("MIN_CALL_INTERVAL", "2.5"))
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


def log_start(task_id: str):
    model_name = get_model_name()
    api_base_url = get_api_base_url()
    print(f"[START] task={task_id} env={ENV_NAME} model={model_name}", flush=True)
    _jlog("task_start", task_id=task_id, model=model_name, api_base=api_base_url)


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
    )


# ── Agent Logic ───────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
You are a data cleaning agent. You receive a dirty CSV dataset and information
about what errors exist. Your goal is to fix all errors using as few transform
steps as possible.

You have five action types:
1. **explore**: Inspect the data without modifying it. Your query should be a valid pandas expression using `df` (a pandas DataFrame). Examples: `df.describe()`, `df['Age'].isnull().sum()`, `df.head(10)`
2. **transform**: Submit Python/pandas code that operates on `df` to clean it. Available imports: pandas (pd), numpy (np), re, datetime, string, math, json, csv. The variable `df` is pre-loaded.
3. **done**: Signal you're finished cleaning.
4. **undo**: Restore to a previous checkpoint. Use {"type": "undo", "step": N} where N=0 means the original dirty state. USE THIS IMMEDIATELY when your reward or errors-fixed count drops after a transform — go back to your best state and try a different approach.
5. **validate**: Get detailed error breakdown (budget: 2 per episode). Use {"type": "validate"} when stuck.

Respond with EXACTLY one JSON object (no markdown, no explanation):
- For explore: {"type": "explore", "query": "<pandas expression>"}
- For transform: {"type": "transform", "code": "<python code>"}
- For done: {"type": "done"}
- For undo: {"type": "undo", "step": 0}
- For validate: {"type": "validate"}

Strategy:
- DIAGNOSTIC FIRST: Before any transform, explore to understand the scope. Run `df.isnull().sum()` and check value counts on columns mentioned in the error summary so you know which error type affects how many cells.
- On medium/hard tasks, do at least one targeted explore per error type before your first transform unless the prompt already shows an explicit transform execution error you are fixing
- Tackle the largest error group first (highest count), then work down
- Then apply targeted transforms to fix all errors
- Check error status after each transform
- If your reward or errors-fixed DROPS after a transform, immediately undo to restore your best state, then try a different approach
- Submit 'done' when all errors are fixed or you can't improve further
- Be efficient: fewer transform steps = higher reward
- Warning: changing a cell to the wrong value is penalized MORE than leaving it dirty
- NEVER drop rows with dropna() unless you are specifically removing duplicate rows
- Only fix columns mentioned in the error summary — do not touch other columns
- For whitespace errors: use str.replace(r'\\s+', ' ', regex=True).str.strip() not just str.strip()

Important rules:
- Do NOT repeat the same explore query — check your action history below
- If your last transform fixed 0 errors, your approach is WRONG. Do NOT repeat it. Explore the affected columns to understand what the dirty values actually look like, then try a completely different fix.
- If your last transform didn't improve the score, try a COMPLETELY different approach
- Your action history is shown in every observation — use it to avoid repeating mistakes
- If you're stuck, submit 'done' rather than repeating the same failing transform
- NEVER invent or re-sequence identifier columns (e.g. PassengerId, id, index, key). Inspect the exact dirty tokens with explore first; do not overwrite IDs with sequential integers or any fabricated values.
- pd.to_numeric(col, errors='coerce') turns bad strings into NaN, which is NOT the same as fixing them. You must also fill those NaN values correctly afterward.
- fillna(value, inplace=True) on a column may silently no-op in pandas 2.x. Use assignment: df['col'] = df['col'].fillna(value)
- A single cell can have MULTIPLE corruptions (e.g. type_mangle + inject_nulls on the same column). One transform may only fix one layer — check errors after each step and apply follow-up transforms for remaining issues.
"""


def build_system_prompt() -> str:
    return BASE_SYSTEM_PROMPT


def _extract_remaining_error_targets(constraints: list[str] | None) -> dict[str, list[str]]:
    """Parse compact `corruption in: col1, col2` lines from the observation."""
    targets: dict[str, list[str]] = {}
    for desc in constraints or []:
        for raw_line in str(desc).splitlines():
            line = raw_line.strip()
            if " in: " not in line:
                continue
            left, right = line.split(" in: ", 1)
            ctype = left.split()[-1].rstrip(":")
            cols = [c.strip() for c in right.split(",") if c.strip()]
            if cols:
                targets[ctype] = cols
    return targets


def _suggest_explore_queries(obs, action_history: list[dict] | None = None) -> list[str]:
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
                add(f"pd.to_numeric(df[{repr(shown[0])}], errors='coerce').isna().sum()")
        elif ctype in {"outlier_injection", "decimal_shift"}:
            add(f"df[{cols_expr}].describe(include='all')")
            if shown:
                add(f"df[{repr(shown[0])}].sort_values().tail(10)")
        elif ctype in {"whitespace_noise", "format_inconsistency", "category_misspell", "typo_injection"}:
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
            "For type_mangle: explore with `df['col'].unique()[:20]` to see the bad tokens, "
            "then fix with `df['col'] = pd.to_numeric(df['col'], errors='coerce')`. "
            "Do NOT use to_numeric if values are already parsed as NaN — confirm bad tokens exist first."
        )
    if "inject_nulls" in targets or "null_injected" in targets:
        bullets.append(
            "For inject_nulls: explore `df['col'].isnull().sum()` per affected column, "
            "then fill with the column's mode or median: "
            "`df['col'] = df['col'].fillna(df['col'].median())` for numeric, "
            "`df['col'] = df['col'].fillna(df['col'].mode()[0])` for categorical."
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
    if any(t in targets for t in ("whitespace_noise", "format_inconsistency", "category_misspell", "typo_injection")):
        bullets.append(
            "For string corruption, inspect examples or value counts before normalizing so you target the right columns."
        )

    bullets.append("Do not repeat the same explore query. After 1-2 targeted explores, either transform or validate.")
    return bullets[:5]


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

    if obs.constraints:
        parts.extend(["", "Remaining errors:"])
        for desc in obs.constraints:
            for line in str(desc).splitlines():
                parts.append(f"  {line}")

    # On hard tasks (many errors), add a prioritized fix plan so the model doesn't flail
    if total - fixed > 200 and obs.constraints:
        error_text = str(obs.constraints[0]) if obs.constraints else ""
        # Parse corruption types and their columns from the error summary
        import re as _re
        type_lines = _re.findall(r"(\w+) in: ([^\n]+)", error_text)
        if type_lines:
            # Priority order: structural first, then value-level
            _PRIORITY = {
                "duplicate_rows": 0, "drop_rows": 0, "header_in_data": 0,
                "column_shift": 1, "value_swap": 1,
                "inject_nulls": 2, "whitespace_noise": 2,
                "type_mangle": 3, "format_inconsistency": 3,
                "outlier_injection": 4, "decimal_shift": 4,
                "category_misspell": 5, "typo_injection": 5,
            }
            sorted_types = sorted(type_lines, key=lambda t: _PRIORITY.get(t[0], 6))
            parts.extend(["", "FIX PRIORITY (work in this order):"])
            for i, (ctype, cols) in enumerate(sorted_types, 1):
                parts.append(f"  {i}. {ctype} in {cols}")

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

    if obs.transform_result:
        parts.extend(["", "Last action outcome:", obs.transform_result])
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

    if action_history:
        parts.extend(["", "Recent action history:"])
        for h in action_history[-10:]:
            summary = h["summary"][:80]
            parts.append(
                f'  Step {h["step"]}: {h["type"]} "{summary}" -> reward={h["reward_after"]:.4f}'
            )

    if warnings:
        parts.extend([""])
        for w in warnings:
            parts.append(f"WARNING: {w}")

    return "\n".join(parts)


def get_agent_action(messages: list[dict]) -> tuple[dict, float, dict | None]:
    """Query the LLM for the next action. Returns (action_dict, latency_s, usage).

    Retries on rate limit errors with exponential backoff.
    """
    logger.debug(
        "LLM request — %d messages, last role=%s", len(messages), messages[-1]["role"]
    )
    logger.debug("Full messages:\n%s", json.dumps(messages, indent=2)[:3000])
    _jlog("llm_request", num_messages=len(messages))

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
                temperature=0.1,
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
                action = {"type": "done"}
        else:
            action = {"type": "done"}

    return action, latency, usage


def action_from_dict(d: dict):
    """Convert a dict to the correct Action type."""
    t = d.get("type", "done")
    if t == "explore":
        return ExploreAction(query=d.get("query", "df.head()"))
    elif t == "transform":
        return TransformAction(code=d.get("code", ""))
    elif t == "undo":
        return UndoAction(step=int(d.get("step", 0)))
    elif t == "validate":
        return ValidateAction()
    else:
        return DoneAction()


# ── Run Task ──────────────────────────────────────────────────────────────────


async def run_task(dataset_id: str, difficulty: str, fmt: str = "csv") -> float:
    """Run the agent on a single task via WebSocket. Returns final reward."""
    task_id = f"{dataset_id}_{difficulty}_{fmt}"
    log_start(task_id)
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
        step_result = await env.reset(task_id=dataset_id, difficulty=difficulty, format=fmt)
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

        action_history: list[dict] = []
        step_rewards: list[float] = []
        best_reward: float = current_reward
        best_fixed: int = fixed
        best_transform_step: int = 0  # checkpoint step for undo (0 = original)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    obs, current_reward, action_history=action_history
                ),
            },
        ]

        step_num = 0
        max_steps = MAX_STEPS_BY_DIFFICULTY.get(difficulty, 60)

        while step_num < max_steps:
            step_num += 1

            action_dict, latency, usage = get_agent_action(messages)
            action = action_from_dict(action_dict)
            action_type = action_dict.get("type", "done")

            # Hard enforce: must explore before first transform
            has_any_explore = any(h["type"] == "explore" for h in action_history)
            if action_type == "transform" and not has_any_explore:
                logger.warning("Model tried to transform before any explore — auto-injecting diagnostic explore")
                action_dict = {"type": "explore", "query": "df.isnull().sum()"}
                action = action_from_dict(action_dict)
                action_type = "explore"

            # Log action content at INFO
            if action_type == "explore":
                logger.info(
                    "Step %d | explore | query: %s | latency=%.2fs%s",
                    step_num,
                    action_dict.get("query", "")[:200],
                    latency,
                    f" | tokens={usage['total_tokens']}" if usage else "",
                )
            elif action_type == "transform":
                code_preview = action_dict.get("code", "").replace("\n", " ")[:200]
                logger.info(
                    "Step %d | transform | code: %s | latency=%.2fs%s",
                    step_num,
                    code_preview,
                    latency,
                    f" | tokens={usage['total_tokens']}" if usage else "",
                )
            else:
                logger.info("Step %d | done | latency=%.2fs", step_num, latency)

            step_result = await env.step(action)
            obs = step_result.observation
            current_reward = (
                step_result.reward if step_result.reward is not None else current_reward
            )

            constraint_status = obs.constraint_status or {}
            fixed = sum(1 for v in constraint_status.values() if v)
            total = len(constraint_status)

            logger.info(
                "Step %d result | reward=%.4f | %d/%d errors fixed",
                step_num,
                current_reward,
                fixed,
                total,
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

            # Track action history
            action_history.append(
                {
                    "step": step_num,
                    "type": action_type,
                    "summary": (
                        action_dict.get("query") or action_dict.get("code", "")
                    )[:100],
                    "reward_after": current_reward,
                    "errors_fixed": fixed,
                }
            )

            # Track best state and auto-undo on significant regression
            if action_type == "transform" and fixed > best_fixed:
                best_reward = current_reward
                best_fixed = fixed
                best_transform_step = sum(1 for h in action_history if h["type"] == "transform")

            if action_type == "transform" and fixed < best_fixed and best_fixed > 0:
                regression_pct = (best_fixed - fixed) / best_fixed
                if regression_pct >= 0.25:
                    logger.warning(
                        "Major regression: %d/%d fixed (was %d/%d, %.0f%% drop) — auto-undoing to step %d",
                        fixed, total, best_fixed, total, regression_pct * 100, best_transform_step,
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
                        step_num, "undo", current_reward,
                        done=False, errors_fixed=fixed, errors_total=total,
                    )
                    action_history.append(
                        {
                            "step": step_num,
                            "type": "undo",
                            "summary": f"auto-undo to step {best_transform_step}",
                            "reward_after": current_reward,
                            "errors_fixed": fixed,
                        }
                    )

            # Generate warnings for repeated/stale actions
            warnings: list[str] = []
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
                if consecutive_explores >= 4:
                    warnings.append(
                        f"You have explored {consecutive_explores} times in a row without transforming. Stop exploring and either validate or transform."
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
                            f"Your previous state was better. Use undo to go back: {{\"type\": \"undo\", \"step\": 0}} restores the original, "
                            f"or use a higher step number to restore a later checkpoint. Do NOT keep transforming without undoing first."
                        )
                    elif curr == prev:
                        stale_count = 1
                        for f in reversed(recent_transform_fixed[:-1]):
                            if f == curr:
                                stale_count += 1
                            else:
                                break
                        if curr == 0:
                            warnings.append(
                                f"Your last {stale_count + 1} transforms fixed ZERO errors. "
                                f"Your approach is fundamentally wrong. Use explore to inspect the actual dirty values "
                                f"(e.g. df['col'].value_counts().head(20)) before trying another transform."
                            )
                        else:
                            warnings.append(
                                f"Your last {stale_count + 1} transforms fixed the same number of errors ({fixed}/{total}). "
                                f"Try a fundamentally different approach."
                            )

            # Auto-validate / auto-done circuit breakers
            recent_transform_fixed = [
                h["errors_fixed"] for h in action_history if h["type"] == "transform"
            ]
            should_auto_validate = False
            should_auto_done = False

            if action_type == "explore":
                same_query_streak = _same_query_streak(action_history)
                consecutive_explores = _consecutive_explore_count(action_history)
                validate_available = bool(
                    obs.step_info and obs.step_info.validate_uses < obs.step_info.validate_budget
                )
                budget_exhausted = "budget exhausted" in (obs.explore_result or "").lower()

                if validate_available and (
                    same_query_streak >= 2 or consecutive_explores >= 5 or budget_exhausted
                ):
                    logger.warning(
                        "Repeated explore loop detected (same_query_streak=%d consecutive_explores=%d budget_exhausted=%s) — auto-submitting validate",
                        same_query_streak,
                        consecutive_explores,
                        budget_exhausted,
                    )
                    should_auto_validate = True
                elif same_query_streak >= 3 or consecutive_explores >= 8:
                    logger.warning(
                        "Explore loop persists without progress (same_query_streak=%d consecutive_explores=%d) — auto-submitting done",
                        same_query_streak,
                        consecutive_explores,
                    )
                    should_auto_done = True

            if should_auto_validate:
                validate_result = await env.step(ValidateAction())
                obs = validate_result.observation
                current_reward = (
                    validate_result.reward
                    if validate_result.reward is not None
                    else current_reward
                )
                step_rewards.append(current_reward)
                constraint_status = obs.constraint_status or {}
                fixed = sum(1 for v in constraint_status.values() if v)
                total = len(constraint_status)
                log_step(
                    step_num + 1,
                    "validate",
                    current_reward,
                    done=validate_result.done,
                    errors_fixed=fixed,
                    errors_total=total,
                )
                action_history.append(
                    {
                        "step": step_num + 1,
                        "type": "validate",
                        "summary": "",
                        "reward_after": current_reward,
                        "errors_fixed": fixed,
                    }
                )

            # Stale: 2 consecutive transforms (since last validate) with same errors_fixed → auto-validate
            # Only count transforms after the most recent validate to avoid burning both validates quickly
            transforms_since_validate = []
            for h in reversed(action_history):
                if h["type"] == "validate":
                    break
                if h["type"] == "transform":
                    transforms_since_validate.append(h["errors_fixed"])
            transforms_since_validate.reverse()

            last2 = transforms_since_validate[-2:] if len(transforms_since_validate) >= 2 else []
            stale_validate_fired = False
            if not should_auto_validate and len(last2) == 2 and len(set(last2)) == 1:
                validate_available = bool(
                    obs.step_info and obs.step_info.validate_uses < obs.step_info.validate_budget
                )
                if validate_available:
                    logger.warning(
                        "No improvement in last 2 transforms (%d/%d fixed) — auto-submitting validate before escalating",
                        last2[-1],
                        total,
                    )
                    validate_result = await env.step(ValidateAction())
                    obs = validate_result.observation
                    current_reward = (
                        validate_result.reward
                        if validate_result.reward is not None
                        else current_reward
                    )
                    step_rewards.append(current_reward)
                    constraint_status = obs.constraint_status or {}
                    fixed = sum(1 for v in constraint_status.values() if v)
                    total = len(constraint_status)
                    log_step(
                        step_num + 1,
                        "validate",
                        current_reward,
                        done=validate_result.done,
                        errors_fixed=fixed,
                        errors_total=total,
                    )
                    action_history.append(
                        {
                            "step": step_num + 1,
                            "type": "validate",
                            "summary": "",
                            "reward_after": current_reward,
                            "errors_fixed": fixed,
                        }
                    )
                    stale_validate_fired = True

            # Stale: 3 consecutive transforms with same errors_fixed → auto-done
            # Skip if a stale-validate just fired this iteration to avoid validate+done in same step
            last3 = recent_transform_fixed[-3:]
            if not stale_validate_fired and len(last3) == 3 and len(set(last3)) == 1:
                logger.warning(
                    "No improvement in last 3 transforms (%d/%d fixed) — auto-submitting done",
                    last3[-1],
                    total,
                )
                should_auto_done = True

            # Regression: 2 consecutive transforms that both reduced errors_fixed
            if not should_auto_done and len(recent_transform_fixed) >= 3:
                if (
                    recent_transform_fixed[-1]
                    < recent_transform_fixed[-2]
                    < recent_transform_fixed[-3]
                ):
                    logger.warning(
                        "2 consecutive regressions (%d -> %d -> %d/%d) — auto-submitting done",
                        recent_transform_fixed[-3],
                        recent_transform_fixed[-2],
                        recent_transform_fixed[-1],
                        total,
                    )
                    should_auto_done = True

            if should_auto_done:
                _jlog("auto_done_stuck", step=step_num, reward=current_reward)
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
                break

            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            messages.append(
                {
                    "role": "user",
                    "content": build_user_prompt(
                        obs,
                        current_reward,
                        action_history=action_history,
                        warnings=warnings,
                    ),
                }
            )

            # Keep message history bounded — retain system + last N exchanges
            MAX_HISTORY_MESSAGES = 20  # system + 10 exchanges (assistant+user pairs)
            if len(messages) > MAX_HISTORY_MESSAGES:
                messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

            if step_result.done:
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
    log_end(task_id, current_reward, step_num, elapsed, step_rewards)
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
            fmt = args[i + 2] if i + 2 < len(args) and args[i + 2] not in ("easy", "medium", "hard") else "csv"
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
