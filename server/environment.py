"""
Core Data Cleaning Environment — implements OpenEnv reset/step/state API.
"""

from __future__ import annotations

import json
import os
import uuid
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

try:
    from ..models import (
        DataCleaningObservation,
        DataCleaningState,
        DoneAction,
        ExploreAction,
        StepInfo,
        TransformAction,
    )
except ImportError:
    from models import (
        DataCleaningObservation,
        DataCleaningState,
        DoneAction,
        ExploreAction,
        StepInfo,
        TransformAction,
    )

from openenv.core import Environment

try:
    from .grader import grade, summarize_errors
    from .sandbox import ExecutionResult, create_sandbox, execute_explore, execute_transform, terminate_worker
except ImportError:
    from grader import grade, summarize_errors
    from sandbox import ExecutionResult, create_sandbox, execute_explore, execute_transform, terminate_worker


# Type alias for the discriminated union
ActionType = Union[ExploreAction, TransformAction, DoneAction]

TASKS_DIR = os.environ.get("TASKS_DIR", "tasks")
DATA_DIR = os.environ.get("DATA_DIR", "data")
SANDBOX_BASE = os.environ.get("SANDBOX_BASE", "outputs/sandbox")


def _load_task(task_id: str) -> dict[str, Any]:
    """Load a task config by searching tasks/ directory."""
    tasks_dir = Path(TASKS_DIR)
    for task_file in tasks_dir.glob("*.json"):
        with open(task_file) as f:
            config = json.load(f)
        if config.get("task_id") == task_id:
            return config
    raise ValueError(f"Task not found: {task_id}")


def _list_tasks() -> list[str]:
    """List all available task IDs."""
    tasks_dir = Path(TASKS_DIR)
    task_ids = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        with open(task_file) as f:
            config = json.load(f)
        task_ids.append(config["task_id"])
    return task_ids


def _data_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Generate a concise summary of the DataFrame for the agent."""
    buf = StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")

    buf.write("Columns and types:\n")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = f" ({null_count} nulls, {null_count/len(df)*100:.1f}%)" if null_count > 0 else ""
        buf.write(f"  {col}: {df[col].dtype}{null_pct}\n")

    buf.write(f"\nSample rows (first {max_rows}):\n")
    buf.write(df.head(max_rows).to_string(index=False))
    return buf.getvalue()


def _error_summary(error_status: dict[str, str], summary: dict[str, Any]) -> str:
    """Human-readable error fix progress for the agent."""
    total = summary["total_errors"]
    fixed = summary["fixed"]
    wrong = summary["wrong_value"]
    unfixed = summary["unfixed"]
    lines = [
        f"Error fix progress: {fixed}/{total} fixed",
    ]
    if wrong:
        lines.append(f"  {wrong} cells changed to wrong value (penalized 1.5x)")
    if unfixed:
        lines.append(f"  {unfixed} errors still unfixed")
    return "\n".join(lines)


class DataCleaningEnvironment(
    Environment[ActionType, DataCleaningObservation, DataCleaningState]
):
    """OpenEnv environment for AI-powered data cleaning."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_config: dict[str, Any] = {}
        self._error_map: dict[str, Any] = {}
        self._sandbox_dir: str = ""
        self._episode_id: str = ""
        self._clean_df: pd.DataFrame = pd.DataFrame()
        self._worker_proc = None
        self._current_df: pd.DataFrame = pd.DataFrame()
        self._explore_steps_cycle: int = 0
        self._explore_steps_total: int = 0
        self._explore_timeouts: int = 0
        self._transform_steps: int = 0
        self._current_reward: float = 0.0
        self._error_status: dict[str, str] = {}
        self._error_summary_cache: dict[str, Any] = {}
        self._done: bool = False
        self._done_count: int = 0
        self._step_count: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        """Reset the environment for a new episode."""
        task_id = kwargs.get("task_id")
        if not task_id:
            available = _list_tasks()
            task_id = available[0] if available else "titanic_easy"

        self._task_config = _load_task(task_id)
        self._episode_id = episode_id or str(uuid.uuid4())[:8]
        self._explore_steps_cycle = 0
        self._explore_steps_total = 0
        self._explore_timeouts = 0
        self._transform_steps = 0
        self._current_reward = 0.0
        self._done = False
        self._done_count = 0
        self._step_count = 0

        # Load error map
        error_map_path = self._task_config["error_map_path"]
        with open(error_map_path) as f:
            self._error_map = json.load(f)

        # Load clean data (for grading)
        self._clean_df = pd.read_csv(self._task_config["clean_data_path"])

        # Terminate any existing worker from a previous episode
        if self._worker_proc is not None:
            terminate_worker(self._worker_proc)
            self._worker_proc = None

        # Create sandbox and spawn persistent worker
        dirty_path = self._task_config["dirty_data_path"]
        self._sandbox_dir, self._worker_proc = create_sandbox(
            self._episode_id, dirty_path, base_dir=SANDBOX_BASE
        )

        # Load dirty data for initial summary + initial grade
        self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
        df = self._current_df
        self._error_status, self._current_reward = grade(
            self._clean_df,
            df,
            self._error_map,
            self._transform_steps,
            self._task_config["min_transform_steps"],
            self._task_config["max_transform_steps"],
            explore_steps=self._explore_steps_total,
            explore_timeouts=self._explore_timeouts,
            explore_cost_per_step=self._task_config.get("explore_cost_per_step", 0.01),
            explore_timeout_cost=self._task_config.get("explore_timeout_cost", 0.03),
        )
        self._error_summary_cache = summarize_errors(self._error_status, self._error_map)

        return DataCleaningObservation(
            task_id=self._task_config["task_id"],
            task_description=self._task_config["description"],
            constraints=[_error_summary(self._error_status, self._error_summary_cache)],
            data_summary=_data_summary(df),
            explore_result=None,
            transform_result=None,
            constraint_status={k: (v == "fixed") for k, v in self._error_status.items()},
            reward=self._current_reward,
            done=False,
            step_info=self._make_step_info(),
        )

    def step(
        self,
        action: ActionType,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        """Process an agent action."""
        if self._done:
            return self._make_observation(
                explore_result=None,
                transform_result="Episode is already done.",
            )

        self._step_count += 1

        if isinstance(action, ExploreAction):
            return self._handle_explore(action)
        elif isinstance(action, TransformAction):
            return self._handle_transform(action)
        elif isinstance(action, DoneAction):
            return self._handle_done()
        else:
            return self._make_observation(
                transform_result=f"Unknown action type: {type(action).__name__}",
            )

    @property
    def state(self) -> DataCleaningState:
        """Return current environment state."""
        summary = self._error_summary_cache
        satisfied = summary.get("fixed", 0)
        total = summary.get("total_errors", 0)
        return DataCleaningState(
            episode_id=self._episode_id,
            task_id=self._task_config.get("task_id", ""),
            step_count=self._step_count,
            explore_steps_total=self._explore_steps_total,
            transform_steps_total=self._transform_steps,
            current_reward=self._current_reward,
            constraints_satisfied=satisfied,
            constraints_total=total,
        )

    # ── Action Handlers ──────────────────────────────────────────────────────

    def _handle_explore(self, action: ExploreAction) -> DataCleaningObservation:
        budget = self._task_config.get("explore_budget", 10)
        if self._explore_steps_cycle >= budget:
            return self._make_observation(
                explore_result=f"Explore budget exhausted ({budget}/{budget}). Submit a transform to reset it.",
            )

        self._explore_steps_cycle += 1
        self._explore_steps_total += 1

        result = execute_explore(
            action.query,
            self._worker_proc,
            self._explore_steps_total,
        )

        if not result.success:
            self._explore_timeouts += 1

        return self._make_observation(
            explore_result=result.stdout if result.success else f"Error: {result.error}",
        )

    def _handle_transform(self, action: TransformAction) -> DataCleaningObservation:
        self._transform_steps += 1
        self._explore_steps_cycle = 0  # reset explore budget

        result = execute_transform(
            action.code,
            self._worker_proc,
            self._transform_steps,
        )

        if result.success:
            prev_df = self._current_df
            self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
            df = self._current_df

            # Detect if data actually changed
            data_changed = not prev_df.equals(df)

            self._error_status, self._current_reward = grade(
                self._clean_df,
                df,
                self._error_map,
                self._transform_steps,
                self._task_config["min_transform_steps"],
                self._task_config["max_transform_steps"],
                explore_steps=self._explore_steps_total,
                explore_timeouts=self._explore_timeouts,
                explore_cost_per_step=self._task_config.get("explore_cost_per_step", 0.01),
                explore_timeout_cost=self._task_config.get("explore_timeout_cost", 0.03),
            )
            self._error_summary_cache = summarize_errors(self._error_status, self._error_map)
            transform_msg = "Transform applied successfully."
            if not data_changed:
                transform_msg += "\nWARNING: Data was not modified by this transform. Common cause: using inplace=True on a column (e.g. df['col'].fillna(val, inplace=True)) does NOT modify df. Use df['col'] = df['col'].fillna(val) instead."
            if result.stdout:
                transform_msg += f"\nOutput: {result.stdout[:500]}"
        else:
            transform_msg = f"Transform failed: {result.error}"
            if result.stderr:
                transform_msg += f"\nStderr: {result.stderr[:500]}"

        if self._transform_steps >= self._task_config["max_transform_steps"]:
            self._done = True
            transform_msg += "\nMax transform steps reached. Episode ending."
            if self._worker_proc is not None:
                terminate_worker(self._worker_proc)
                self._worker_proc = None

        return self._make_observation(transform_result=transform_msg)

    def _handle_done(self) -> DataCleaningObservation:
        self._done_count += 1

        # Always grade current state
        self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
        df = self._current_df
        self._error_status, self._current_reward = grade(
            self._clean_df,
            df,
            self._error_map,
            self._transform_steps,
            self._task_config["min_transform_steps"],
            self._task_config["max_transform_steps"],
            explore_steps=self._explore_steps_total,
            explore_timeouts=self._explore_timeouts,
            explore_cost_per_step=self._task_config.get("explore_cost_per_step", 0.01),
            explore_timeout_cost=self._task_config.get("explore_timeout_cost", 0.03),
        )
        self._error_summary_cache = summarize_errors(self._error_status, self._error_map)

        # Soft done: first attempt with imperfect score — give agent another chance
        if self._done_count == 1 and self._current_reward < 1.0:
            unfixed = sum(1 for s in self._error_status.values() if s != "fixed")
            return self._make_observation(
                transform_result=(
                    f"Soft done — your score is {self._current_reward:.4f} with {unfixed} errors remaining. "
                    f"You can continue exploring/transforming, or submit done again to finalize."
                ),
            )

        # Hard done: perfect score, or second done attempt
        self._done = True
        if self._worker_proc is not None:
            terminate_worker(self._worker_proc)
            self._worker_proc = None
        return self._make_observation(
            transform_result="Episode complete. Final grading applied.",
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_step_info(self) -> StepInfo:
        return StepInfo(
            explore_steps_used=self._explore_steps_cycle,
            explore_budget=self._task_config.get("explore_budget", 10),
            transform_steps_used=self._transform_steps,
            max_transform_steps=self._task_config.get("max_transform_steps", 10),
            min_transform_steps=self._task_config.get("min_transform_steps", 2),
            done_count=self._done_count,
        )

    def _make_observation(
        self,
        explore_result: Optional[str] = None,
        transform_result: Optional[str] = None,
    ) -> DataCleaningObservation:
        df = self._current_df

        return DataCleaningObservation(
            task_id=self._task_config["task_id"],
            task_description=self._task_config["description"],
            constraints=[_error_summary(self._error_status, self._error_summary_cache)],
            data_summary=_data_summary(df),
            explore_result=explore_result,
            transform_result=transform_result,
            constraint_status={k: (v == "fixed") for k, v in self._error_status.items()},
            reward=self._current_reward,
            done=self._done,
            step_info=self._make_step_info(),
        )
