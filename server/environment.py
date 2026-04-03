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
    from .grader import compute_reward, grade
    from .sandbox import ExecutionResult, create_sandbox, execute_explore, execute_transform
except ImportError:
    from grader import compute_reward, grade
    from sandbox import ExecutionResult, create_sandbox, execute_explore, execute_transform


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
    # If task_id not found, try loading by difficulty prefix
    for difficulty in ["easy", "medium", "hard"]:
        path = tasks_dir / f"task_{difficulty}.json"
        if path.exists():
            with open(path) as f:
                config = json.load(f)
            if config.get("task_id") == task_id:
                return config
    raise ValueError(f"Task not found: {task_id}")


def _list_tasks() -> list[str]:
    """List all available task IDs."""
    tasks_dir = Path(TASKS_DIR)
    task_ids = []
    for task_file in tasks_dir.glob("*.json"):
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


class DataCleaningEnvironment(
    Environment[ActionType, DataCleaningObservation, DataCleaningState]
):
    """OpenEnv environment for AI-powered data cleaning."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_config: dict[str, Any] = {}
        self._constraints: list[dict[str, Any]] = []
        self._sandbox_dir: str = ""
        self._episode_id: str = ""
        self._explore_steps_cycle: int = 0  # resets each transform cycle
        self._explore_steps_total: int = 0
        self._transform_steps: int = 0
        self._current_reward: float = 0.0
        self._constraint_status: dict[str, bool] = {}
        self._done: bool = False
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
            # Default to easy if no task_id specified
            available = _list_tasks()
            task_id = available[0] if available else "easy_titanic"

        self._task_config = _load_task(task_id)
        self._constraints = self._task_config["constraints"]
        self._episode_id = episode_id or str(uuid.uuid4())[:8]
        self._explore_steps_cycle = 0
        self._explore_steps_total = 0
        self._transform_steps = 0
        self._current_reward = 0.0
        self._done = False
        self._step_count = 0

        # Create sandbox with dirty data
        dirty_path = self._task_config["dirty_data_path"]
        self._sandbox_dir = create_sandbox(
            self._episode_id, dirty_path, base_dir=SANDBOX_BASE
        )

        # Load dirty data for initial summary
        df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))

        # Initial constraint check
        self._constraint_status = grade(df, self._constraints)
        self._current_reward = compute_reward(
            self._constraint_status,
            self._constraints,
            self._transform_steps,
            self._task_config["min_transform_steps"],
            self._task_config["max_transform_steps"],
        )

        return DataCleaningObservation(
            task_id=self._task_config["task_id"],
            task_description=self._task_config["description"],
            constraints=[c["description"] for c in self._constraints],
            data_summary=_data_summary(df),
            explore_result=None,
            transform_result=None,
            constraint_status=self._constraint_status,
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
        satisfied = sum(1 for v in self._constraint_status.values() if v)
        return DataCleaningState(
            episode_id=self._episode_id,
            task_id=self._task_config.get("task_id", ""),
            step_count=self._step_count,
            explore_steps_total=self._explore_steps_total,
            transform_steps_total=self._transform_steps,
            current_reward=self._current_reward,
            constraints_satisfied=satisfied,
            constraints_total=len(self._constraints),
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
            self._sandbox_dir,
            self._explore_steps_total,
        )

        return self._make_observation(
            explore_result=result.stdout if result.success else f"Error: {result.error}",
        )

    def _handle_transform(self, action: TransformAction) -> DataCleaningObservation:
        self._transform_steps += 1
        self._explore_steps_cycle = 0  # reset explore budget

        result = execute_transform(
            action.code,
            self._sandbox_dir,
            self._transform_steps,
        )

        if result.success:
            # Re-grade
            df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
            self._constraint_status = grade(df, self._constraints)
            self._current_reward = compute_reward(
                self._constraint_status,
                self._constraints,
                self._transform_steps,
                self._task_config["min_transform_steps"],
                self._task_config["max_transform_steps"],
            )
            transform_msg = "Transform applied successfully."
            if result.stdout:
                transform_msg += f"\nOutput: {result.stdout[:500]}"
        else:
            transform_msg = f"Transform failed: {result.error}"
            if result.stderr:
                transform_msg += f"\nStderr: {result.stderr[:500]}"

        # Check if max transform steps reached
        if self._transform_steps >= self._task_config["max_transform_steps"]:
            self._done = True
            transform_msg += "\nMax transform steps reached. Episode ending."

        return self._make_observation(transform_result=transform_msg)

    def _handle_done(self) -> DataCleaningObservation:
        self._done = True
        # Final grading
        df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
        self._constraint_status = grade(df, self._constraints)
        self._current_reward = compute_reward(
            self._constraint_status,
            self._constraints,
            self._transform_steps,
            self._task_config["min_transform_steps"],
            self._task_config["max_transform_steps"],
        )
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
        )

    def _make_observation(
        self,
        explore_result: Optional[str] = None,
        transform_result: Optional[str] = None,
    ) -> DataCleaningObservation:
        # Load current data for summary
        current_csv = os.path.join(self._sandbox_dir, "current.csv")
        df = pd.read_csv(current_csv)

        return DataCleaningObservation(
            task_id=self._task_config["task_id"],
            task_description=self._task_config["description"],
            constraints=[c["description"] for c in self._constraints],
            data_summary=_data_summary(df),
            explore_result=explore_result,
            transform_result=transform_result,
            constraint_status=self._constraint_status,
            reward=self._current_reward,
            done=self._done,
            step_info=self._make_step_info(),
        )
