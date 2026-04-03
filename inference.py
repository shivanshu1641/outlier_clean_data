"""
Baseline inference script for the Data Cleaning OpenEnv environment.

Uses the OpenAI Client for LLM calls as required by the hackathon.
Implements the [START]/[STEP]/[END] logging format.

Usage:
    python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from client import DataCleaningClient
from models import DoneAction, ExploreAction, TransformAction

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

TASK_IDS = ["easy_titanic", "medium_wine", "hard_combined"]

# ── OpenAI Client ────────────────────────────────────────────────────────────

llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=NVIDIA_API_KEY or "not-needed",
)

# ── Structured Logging ───────────────────────────────────────────────────────


def log_start(task_id: str):
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "model": MODEL_NAME,
        "timestamp": time.time(),
    }))


def log_step(step_num: int, action_type: str, reward: float, constraints_satisfied: int, constraints_total: int):
    print(json.dumps({
        "type": "[STEP]",
        "step": step_num,
        "action_type": action_type,
        "reward": reward,
        "constraints_satisfied": constraints_satisfied,
        "constraints_total": constraints_total,
        "timestamp": time.time(),
    }))


def log_end(task_id: str, final_reward: float, total_steps: int):
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "final_reward": final_reward,
        "total_steps": total_steps,
        "timestamp": time.time(),
    }))


# ── Agent Logic ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a data cleaning agent. You receive a dirty CSV dataset and a list of constraints.
Your goal is to clean the data so all constraints are satisfied, using as few transform steps as possible.

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
- Then apply targeted transforms to fix constraint violations
- Check constraint status after each transform
- Submit 'done' when all constraints are satisfied or you can't improve further
- Be efficient: fewer transform steps = higher reward
"""


def build_user_prompt(obs, result_reward: float | None = None) -> str:
    """Build the user message from the current observation."""
    parts = [
        f"Task: {obs.task_description}",
        "",
        "Constraints (severity: critical=3x, high=2x, medium=1x weight):",
    ]
    constraint_status = obs.constraint_status or {}
    for i, desc in enumerate(obs.constraints):
        cid = f"c{i+1}"
        status = "PASS" if constraint_status.get(cid, False) else "FAIL"
        parts.append(f"  [{status}] {desc}")

    reward = result_reward if result_reward is not None else (obs.reward or 0.0)
    parts.extend([
        "",
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


def get_agent_action(messages: list[dict]) -> dict:
    """Query the LLM for the next action."""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
    )
    content = response.choices[0].message.content.strip()

    # Handle markdown code blocks
    if "```" in content:
        lines = content.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        content = "\n".join(json_lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        return {"type": "done"}


def action_from_dict(d: dict):
    """Convert a dict to the correct Action type."""
    t = d.get("type", "done")
    if t == "explore":
        return ExploreAction(query=d.get("query", "df.head()"))
    elif t == "transform":
        return TransformAction(code=d.get("code", ""))
    else:
        return DoneAction()


# ── Run Task ─────────────────────────────────────────────────────────────────


async def run_task(task_id: str) -> float:
    """Run the agent on a single task via WebSocket. Returns final reward."""
    log_start(task_id)

    async with DataCleaningClient(base_url=ENV_URL) as env:
        step_result = await env.reset(task_id=task_id)
        obs = step_result.observation
        current_reward = step_result.reward or 0.0

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs, current_reward)},
        ]

        step_num = 0
        max_steps = 50  # safety limit

        while step_num < max_steps:
            step_num += 1

            action_dict = get_agent_action(messages)
            action = action_from_dict(action_dict)
            action_type = action_dict.get("type", "done")

            step_result = await env.step(action)
            obs = step_result.observation
            current_reward = step_result.reward if step_result.reward is not None else current_reward

            constraint_status = obs.constraint_status or {}
            satisfied = sum(1 for v in constraint_status.values() if v)
            total = len(constraint_status)

            log_step(step_num, action_type, current_reward, satisfied, total)

            # Add to conversation
            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            messages.append({"role": "user", "content": build_user_prompt(obs, current_reward)})

            # Check termination
            if step_result.done:
                break

            # Auto-done if all constraints pass
            if total > 0 and satisfied == total:
                done_result = await env.step(DoneAction())
                current_reward = done_result.reward if done_result.reward is not None else current_reward
                log_step(step_num + 1, "done", current_reward, satisfied, total)
                break

    log_end(task_id, current_reward, step_num)
    return current_reward


# ── Main ─────────────────────────────────────────────────────────────────────


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
            print(f"Final reward: {reward}", file=sys.stderr)
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
    asyncio.run(amain())


if __name__ == "__main__":
    main()
