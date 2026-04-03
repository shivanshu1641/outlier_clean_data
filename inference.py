"""
Baseline inference script for the Data Cleaning OpenEnv environment.

Uses any OpenAI-compatible API endpoint. Configure via .env:
  API_BASE_URL  — e.g. http://localhost:11434/v1 or https://integrate.api.nvidia.com/v1
  API_KEY       — leave empty for local endpoints that don't require auth
  MODEL_NAME    — e.g. qwen3, nvidia/nemotron-super-49b-v1, gpt-4o
  ENV_URL       — http://localhost:8000
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

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

TASK_IDS = [
    "titanic_easy", "titanic_medium", "titanic_hard",
    "wine_easy", "wine_medium", "wine_hard",
]

# Minimum seconds between LLM calls to avoid rate limits (Groq free: 30 req/min)
MIN_CALL_INTERVAL = float(os.environ.get("MIN_CALL_INTERVAL", "2.5"))
_last_call_time = 0.0

# ── OpenAI Client ─────────────────────────────────────────────────────────────

llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY or "not-needed",
    max_retries=0,  # we handle retries ourselves with pacing
)

# ── Structured stdout logging (machine-readable) ──────────────────────────────


def log_start(task_id: str):
    payload = {
        "type": "[START]",
        "task_id": task_id,
        "model": MODEL_NAME,
        "timestamp": time.time(),
    }
    print(json.dumps(payload))
    _jlog("task_start", task_id=task_id, model=MODEL_NAME, api_base=API_BASE_URL)


def log_step(step_num: int, action_type: str, reward: float, errors_fixed: int, errors_total: int,
             action_content: str = "", latency: float = 0.0, usage: dict | None = None):
    payload = {
        "type": "[STEP]",
        "step": step_num,
        "action_type": action_type,
        "reward": reward,
        "errors_fixed": errors_fixed,
        "errors_total": errors_total,
        "timestamp": time.time(),
    }
    print(json.dumps(payload))
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


def log_end(task_id: str, final_reward: float, total_steps: int, elapsed: float):
    payload = {
        "type": "[END]",
        "task_id": task_id,
        "final_reward": final_reward,
        "total_steps": total_steps,
        "elapsed_s": round(elapsed, 2),
        "timestamp": time.time(),
    }
    print(json.dumps(payload))
    _jlog("task_end", task_id=task_id, final_reward=final_reward,
          total_steps=total_steps, elapsed_s=round(elapsed, 2))


# ── Agent Logic ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
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
"""


def build_user_prompt(obs, result_reward: float | None = None) -> str:
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
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
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
    logger.info("Starting task: %s | model: %s | endpoint: %s", task_id, MODEL_NAME, API_BASE_URL)

    async with DataCleaningClient(base_url=ENV_URL) as env:
        step_result = await env.reset(task_id=task_id)
        obs = step_result.observation
        current_reward = step_result.reward or 0.0

        constraint_status = obs.constraint_status or {}
        fixed = sum(1 for v in constraint_status.values() if v)
        total = len(constraint_status)
        logger.info("Reset complete — %d/%d errors fixed, reward=%.4f", fixed, total, current_reward)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs, current_reward)},
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
            log_step(step_num, action_type, current_reward, fixed, total,
                     action_content=action_content, latency=latency, usage=usage)

            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            messages.append({"role": "user", "content": build_user_prompt(obs, current_reward)})

            # Keep message history bounded — retain system + last N exchanges
            MAX_HISTORY_MESSAGES = 20  # system + 10 exchanges (assistant+user pairs)
            if len(messages) > MAX_HISTORY_MESSAGES:
                messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

            if step_result.done:
                break

            if total > 0 and fixed == total:
                done_result = await env.step(DoneAction())
                current_reward = done_result.reward if done_result.reward is not None else current_reward
                log_step(step_num + 1, "done", current_reward, fixed, total)
                _jlog("auto_done", step=step_num + 1, reward=current_reward)
                logger.info("All errors fixed — submitted done action")
                break

    elapsed = time.time() - task_start
    log_end(task_id, current_reward, step_num, elapsed)
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
        try:
            reward = await run_task(task_id)
            results[task_id] = reward
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            results[task_id] = 0.0

    print(f"\n{'='*60}", file=sys.stderr)
    print("Summary:", file=sys.stderr)
    for tid, reward in results.items():
        print(f"  {tid}: {reward}", file=sys.stderr)
    avg = sum(results.values()) / len(results) if results else 0
    print(f"  Average: {avg:.4f}", file=sys.stderr)


def main():
    try:
        asyncio.run(amain())
    finally:
        _jsonl_file.close()


if __name__ == "__main__":
    main()
