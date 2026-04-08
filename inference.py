"""
Baseline inference script for the Data Cleaning OpenEnv environment.

Uses any OpenAI-compatible API endpoint. Configure via .env:
  API_BASE_URL  — defaults to the active hosted endpoint if unset
  API_KEY       — required via `API_KEY` or `HF_TOKEN`
  MODEL_NAME    — e.g. qwen3, nvidia/nemotron-super-49b-v1, gpt-4o
  ENV_URL       — http://localhost:8000
  LOCAL_IMAGE_NAME / IMAGE_NAME — optional docker image name for local env startup
  LOG_LEVEL     — INFO (default) or DEBUG for full LLM I/O
  LOG_DIR       — directory for JSONL log files (default: outputs/logs)

Usage:
    python inference.py                      # runs all tasks
    python inference.py titanic_easy         # run specific task(s)
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
from models import DoneAction, ExploreAction, TransformAction

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

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_BASE_URL") or os.getenv("ENV_URL") or "http://localhost:8000"
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

TASK_IDS = [
    "titanic_easy", "titanic_medium", "titanic_hard",
    "wine_easy", "wine_medium", "wine_hard",
]

# ── Few-shot examples from dataset.parquet ────────────────────────────────────

def _load_few_shot_examples() -> dict[str, str]:
    parquet_path = Path("data/dataset.parquet")
    if not parquet_path.exists():
        return {}
    try:
        import pandas as _pd
        df = _pd.read_parquet(parquet_path)
        best = (
            df.sort_values("reward_gap", ascending=False)
              .drop_duplicates(subset="task_id", keep="first")
              .set_index("task_id")["chosen"]
              .to_dict()
        )
        logger.info("Loaded few-shot examples for %d tasks from dataset.parquet", len(best))
        return best
    except Exception as e:
        logger.warning("Could not load dataset.parquet for few-shot examples: %s", e)
        return {}

FEW_SHOT_EXAMPLES: dict[str, str] = _load_few_shot_examples()

# Minimum seconds between LLM calls to avoid rate limits (Groq free: 30 req/min)
MIN_CALL_INTERVAL = float(os.environ.get("MIN_CALL_INTERVAL", "2.5"))
_last_call_time = 0.0

# ── OpenAI Client ─────────────────────────────────────────────────────────────

llm_client: OpenAI | None = None


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL


def get_api_key() -> str:
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    if api_key is None:
        raise RuntimeError(
            "Missing required environment variable: API_KEY or HF_TOKEN. "
            "Set one in your environment or .env before running inference.py."
        )
    return api_key


def get_model_name() -> str:
    return os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME


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


def log_step(step_num: int, action_type: str, reward: float, done: bool,
             action_content: str = "", error: str | None = None,
             latency: float = 0.0, usage: dict | None = None,
             errors_fixed: int = 0, errors_total: int = 0):
    action_summary = f"{action_type}()" if action_type == "done" else f"{action_type}('{action_content[:80]}')"
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step_num} action={action_summary} reward={reward:.2f} done={done_val} error={error_val}", flush=True)
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


def log_end(task_id: str, final_reward: float, total_steps: int, elapsed: float, rewards: list[float]):
    score = max(1e-6, min(final_reward, 1 - 1e-6))
    success = str(score >= 0.5).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success} steps={total_steps} score={score:.3f} rewards={rewards_str}", flush=True)
    _jlog("task_end", task_id=task_id, final_reward=final_reward,
          total_steps=total_steps, elapsed_s=round(elapsed, 2))


# ── Agent Logic ───────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
You are a data cleaning agent. You receive a dirty CSV dataset and information
about what errors exist. Your goal is to fix all errors using as few transform
steps as possible.

You have three action types:
1. **explore**: Inspect the data without modifying it. Your query should be a valid pandas expression using `df` (a pandas DataFrame). Examples: `df.describe()`, `df['Age'].isnull().sum()`, `df.head(10)`
2. **transform**: Submit Python/pandas code that operates on `df` to clean it. Available imports: pandas (pd), numpy (np), re, datetime, string, math, json, csv. The variable `df` is pre-loaded.
3. **done**: Signal you're finished cleaning.

Respond with EXACTLY one JSON object (no markdown, no explanation):
- For explore: {"type": "explore", "query": "<pandas expression>"}
- For transform: {"type": "transform", "code": "<python code>"}
- For done: {"type": "done"}

Strategy:
- First explore to understand the data issues (you have 10 explores per transform cycle)
- Then apply targeted transforms to fix all errors
- Check error status after each transform
- Submit 'done' when all errors are fixed or you can't improve further
- Be efficient: fewer transform steps = higher reward
- Warning: changing a cell to the wrong value is penalized MORE than leaving it dirty
- NEVER drop rows with dropna() unless you are specifically removing duplicate rows
- Only fix columns mentioned in the error summary — do not touch other columns
- For whitespace errors: use str.replace(r'\\s+', ' ', regex=True).str.strip() not just str.strip()

Important rules:
- Do NOT repeat the same explore query — check your action history below
- If your last transform didn't improve the score, try a COMPLETELY different approach
- Your action history is shown in every observation — use it to avoid repeating mistakes
- If you're stuck, submit 'done' rather than repeating the same failing transform
"""


def build_system_prompt(task_id: str) -> str:
    if os.environ.get("NO_FEW_SHOT"):
        return BASE_SYSTEM_PROMPT
    example = FEW_SHOT_EXAMPLES.get(task_id)
    if not example:
        return BASE_SYSTEM_PROMPT
    few_shot_section = (
        "\n\nHere is a verified correct solution for a similar version of this "
        "task (different rows affected, same corruption types). Use this as a "
        "template — adapt column names and values to the actual data you see:\n"
        "```python\n" + example + "\n```\n"
        "Study this pattern. Apply the same logic to the current dirty data."
    )
    return BASE_SYSTEM_PROMPT + few_shot_section


def build_user_prompt(
    obs,
    result_reward: float | None = None,
    action_history: list[dict] | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Build the user message from the current observation."""
    parts = [
        f"Task: {obs.task_description}",
        "",
        "Current status:",
    ]
    for desc in (obs.constraints or []):
        parts.append(f"  {desc}")

    constraint_status = obs.constraint_status or {}
    fixed = sum(1 for v in constraint_status.values() if v)
    total = len(constraint_status)
    reward = result_reward if result_reward is not None else (obs.reward or 0.0)
    parts.extend([
        "",
        f"Errors fixed: {fixed}/{total}",
        f"Current reward: {reward}",
        "",
        "Data summary:",
        obs.data_summary,
    ])

    if obs.explore_result:
        parts.extend(["", "Last explore result:", obs.explore_result])
    if obs.transform_result:
        parts.extend(["", "Last transform result:", obs.transform_result])

    step_info = obs.step_info
    if step_info:
        parts.extend([
            "",
            f"Explore budget: {step_info.explore_steps_used}/{step_info.explore_budget}",
            f"Transform steps: {step_info.transform_steps_used}/{step_info.max_transform_steps}",
        ])

    # Action history — survives message truncation
    if action_history:
        parts.extend(["", "Action history:"])
        for h in action_history:
            summary = h["summary"][:80]
            parts.append(f'  Step {h["step"]}: {h["type"]} "{summary}" -> reward={h["reward_after"]:.4f}')

    # Warnings about repeated/stale actions
    if warnings:
        parts.extend([""])
        for w in warnings:
            parts.append(f"WARNING: {w}")

    return "\n".join(parts)


def get_agent_action(messages: list[dict]) -> tuple[dict, float, dict | None]:
    """Query the LLM for the next action. Returns (action_dict, latency_s, usage).

    Retries on rate limit errors with exponential backoff.
    """
    logger.debug("LLM request — %d messages, last role=%s", len(messages), messages[-1]["role"])
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
                wait = backoff * (2 ** attempt)
                logger.warning("Rate limit hit — waiting %.0fs before retry %d/%d | error: %s", wait, attempt + 1, max_retries, str(e)[:300])
                _jlog("rate_limit_retry", attempt=attempt + 1, wait_s=wait)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            elif "context" in err_str or "exceed" in err_str or "400" in err_str:
                logger.warning("Context size exceeded or bad request — auto-submitting done | error: %s", str(e)[:300])
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
    _jlog("llm_response", latency_s=round(latency, 3), **(usage or {}), response_preview=content[:500])

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
    else:
        return DoneAction()


# ── Run Task ──────────────────────────────────────────────────────────────────


async def run_task(task_id: str) -> float:
    """Run the agent on a single task via WebSocket. Returns final reward."""
    log_start(task_id)
    task_start = time.time()
    current_reward = 0.0
    step_num = 0
    step_rewards: list[float] = []
    logger.info("Starting task: %s | model: %s | endpoint: %s", task_id, get_model_name(), get_api_base_url())

    if IMAGE_NAME:
        env_client = DataCleaningClient.from_docker_image(IMAGE_NAME)
    else:
        env_client = DataCleaningClient(base_url=ENV_URL)

    async with env_client as env:
        step_result = await env.reset(task_id=task_id)
        obs = step_result.observation
        current_reward = step_result.reward or 0.0

        constraint_status = obs.constraint_status or {}
        fixed = sum(1 for v in constraint_status.values() if v)
        total = len(constraint_status)
        logger.info("Reset complete — %d/%d errors fixed, reward=%.4f", fixed, total, current_reward)

        action_history: list[dict] = []
        step_rewards: list[float] = []

        messages = [
            {"role": "system", "content": build_system_prompt(task_id)},
            {"role": "user", "content": build_user_prompt(obs, current_reward, action_history=action_history)},
        ]

        step_num = 0
        max_steps = 50

        while step_num < max_steps:
            step_num += 1

            action_dict, latency, usage = get_agent_action(messages)
            action = action_from_dict(action_dict)
            action_type = action_dict.get("type", "done")

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
            current_reward = step_result.reward if step_result.reward is not None else current_reward

            constraint_status = obs.constraint_status or {}
            fixed = sum(1 for v in constraint_status.values() if v)
            total = len(constraint_status)

            logger.info(
                "Step %d result | reward=%.4f | %d/%d errors fixed",
                step_num, current_reward, fixed, total,
            )
            logger.debug("Observation data_summary:\n%s", obs.data_summary[:1000] if obs.data_summary else "")
            if obs.explore_result:
                logger.debug("Explore result:\n%s", obs.explore_result[:1000])
            if obs.transform_result:
                logger.debug("Transform result:\n%s", obs.transform_result[:500])

            action_content = action_dict.get("query") or action_dict.get("code") or ""
            step_rewards.append(current_reward)
            log_step(step_num, action_type, current_reward, done=step_result.done,
                     action_content=action_content, latency=latency, usage=usage,
                     errors_fixed=fixed, errors_total=total)

            # Track action history
            action_history.append({
                "step": step_num,
                "type": action_type,
                "summary": (action_dict.get("query") or action_dict.get("code", ""))[:100],
                "reward_after": current_reward,
                "errors_fixed": fixed,
            })

            # Generate warnings for repeated/stale actions
            warnings: list[str] = []
            if action_type == "explore":
                query = action_dict.get("query", "")
                recent_explores = [h["summary"] for h in action_history[:-1] if h["type"] == "explore"][-3:]
                if query[:100] in recent_explores:
                    warnings.append("You already ran this exact explore query. Try a different query or submit a transform.")
            if action_type == "transform":
                recent_transform_fixed = [h["errors_fixed"] for h in action_history if h["type"] == "transform"]
                if len(recent_transform_fixed) >= 2:
                    prev = recent_transform_fixed[-2]
                    curr = recent_transform_fixed[-1]
                    if curr < prev:
                        warnings.append(
                            f"REGRESSION: Your last transform REDUCED errors fixed from {prev}/{total} to {curr}/{total}. "
                            f"Your previous state was better. Submit 'done' now to lock in your best score, "
                            f"or try a completely different approach."
                        )
                    elif curr == prev:
                        stale_count = 1
                        for f in reversed(recent_transform_fixed[:-1]):
                            if f == curr:
                                stale_count += 1
                            else:
                                break
                        warnings.append(
                            f"Your last {stale_count + 1} transforms fixed the same number of errors ({fixed}/{total}). "
                            f"Try a fundamentally different approach."
                        )

            # Auto-done circuit breakers
            recent_transform_fixed = [h["errors_fixed"] for h in action_history if h["type"] == "transform"]
            should_auto_done = False

            # Stale: 3 consecutive transforms with same errors_fixed
            last3 = recent_transform_fixed[-3:]
            if len(last3) == 3 and len(set(last3)) == 1:
                logger.warning("No improvement in last 3 transforms (%d/%d fixed) — auto-submitting done",
                               last3[-1], total)
                should_auto_done = True

            # Regression: 2 consecutive transforms that both reduced errors_fixed
            if not should_auto_done and len(recent_transform_fixed) >= 3:
                if recent_transform_fixed[-1] < recent_transform_fixed[-2] < recent_transform_fixed[-3]:
                    logger.warning("2 consecutive regressions (%d -> %d -> %d/%d) — auto-submitting done",
                                   recent_transform_fixed[-3], recent_transform_fixed[-2],
                                   recent_transform_fixed[-1], total)
                    should_auto_done = True

            if should_auto_done:
                _jlog("auto_done_stuck", step=step_num, reward=current_reward)
                done_result = await env.step(DoneAction())
                current_reward = done_result.reward if done_result.reward is not None else current_reward
                step_rewards.append(current_reward)
                log_step(step_num + 1, "done", current_reward, done=True,
                         errors_fixed=fixed, errors_total=total)
                break

            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            messages.append({"role": "user", "content": build_user_prompt(
                obs, current_reward, action_history=action_history, warnings=warnings,
            )})

            # Keep message history bounded — retain system + last N exchanges
            MAX_HISTORY_MESSAGES = 20  # system + 10 exchanges (assistant+user pairs)
            if len(messages) > MAX_HISTORY_MESSAGES:
                messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

            if step_result.done:
                break

            if total > 0 and fixed == total:
                done_result = await env.step(DoneAction())
                current_reward = done_result.reward if done_result.reward is not None else current_reward
                step_rewards.append(current_reward)
                log_step(step_num + 1, "done", current_reward, done=True,
                         errors_fixed=fixed, errors_total=total)
                _jlog("auto_done", step=step_num + 1, reward=current_reward)
                logger.info("All errors fixed — submitted done action")
                break

    elapsed = time.time() - task_start
    log_end(task_id, current_reward, step_num, elapsed, step_rewards)
    logger.info(
        "Task %s complete | reward=%.4f | steps=%d | elapsed=%.1fs",
        task_id, current_reward, step_num, elapsed,
    )
    return current_reward


# ── Main ──────────────────────────────────────────────────────────────────────


async def amain():
    task_ids = sys.argv[1:] if len(sys.argv) > 1 else TASK_IDS
    results = {}

    for task_id in task_ids:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Running task: {task_id}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        task_start = time.time()
        try:
            reward = await run_task(task_id)
            results[task_id] = reward
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Ensure [END] is emitted even on crash
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
