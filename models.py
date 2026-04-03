"""Typed Pydantic models for the Data Cleaning OpenEnv environment."""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Union

from openenv.core import Action as BaseAction
from openenv.core import Observation as BaseObservation
from openenv.core import State as BaseState
from pydantic import Field


# ── Actions ──────────────────────────────────────────────────────────────────


class ExploreAction(BaseAction):
    """Explore the current dataset without modifying it. No reward change."""

    type: Literal["explore"] = "explore"
    query: str = Field(
        ...,
        description=(
            "A natural-language or pandas-style query to inspect the data. "
            "Examples: 'df.describe()', 'df[\"Age\"].value_counts()', "
            "'df.isnull().sum()', 'df.head(10)'"
        ),
    )


class TransformAction(BaseAction):
    """Submit Python/pandas code to clean the dataset. Gets executed in a sandbox."""

    type: Literal["transform"] = "transform"
    code: str = Field(
        ...,
        description=(
            "Python code that operates on a variable `df` (a pandas DataFrame). "
            "The code should modify `df` in-place or reassign it. "
            "Available imports: pandas, numpy, re, datetime, string, math, "
            "collections, itertools, functools, json, csv."
        ),
    )


class DoneAction(BaseAction):
    """Signal that the agent is finished cleaning."""

    type: Literal["done"] = "done"


Action = Annotated[
    Union[ExploreAction, TransformAction, DoneAction],
    Field(discriminator="type"),
]


class ActionWrapper(BaseAction):
    """Wrapper that deserializes into the correct action subtype via discriminator.

    OpenEnv's create_app needs a single class with model_validate().
    This delegates to the discriminated union.
    """

    type: str = "explore"

    @classmethod
    def model_validate(cls, obj, **kwargs):
        """Route to the correct subclass based on 'type' field."""
        if isinstance(obj, dict):
            action_type = obj.get("type", "done")
            if action_type == "explore":
                return ExploreAction.model_validate(obj, **kwargs)
            elif action_type == "transform":
                return TransformAction.model_validate(obj, **kwargs)
            elif action_type == "done":
                return DoneAction.model_validate(obj, **kwargs)
        return super().model_validate(obj, **kwargs)


# ── Step Info ────────────────────────────────────────────────────────────────


class StepInfo(BaseObservation):
    """Tracks step budgets and usage within an episode."""

    done: bool = False
    reward: Optional[float] = None

    explore_steps_used: int = Field(0, description="Explores used in current transform cycle")
    explore_budget: int = Field(10, description="Max explores per transform cycle")
    transform_steps_used: int = Field(0, description="Total transform steps taken")
    max_transform_steps: int = Field(10, description="Max transform steps allowed")
    min_transform_steps: int = Field(2, description="Known minimum transforms for this task")
    done_count: int = Field(0, description="Number of times agent has submitted done")


# ── Observation ──────────────────────────────────────────────────────────────


class DataCleaningObservation(BaseObservation):
    """What the agent sees after each step."""

    task_id: str = ""
    task_description: str = ""
    constraints: List[str] = Field(default_factory=list, description="Human-readable constraint descriptions")
    data_summary: str = Field("", description="Shape, dtypes, null counts, sample rows")
    explore_result: Optional[str] = Field(None, description="Result of last explore query")
    transform_result: Optional[str] = Field(None, description="Success/error from last transform")
    constraint_status: Dict[str, bool] = Field(default_factory=dict, description="Constraint ID → satisfied")
    step_info: Optional[StepInfo] = None


# ── State ────────────────────────────────────────────────────────────────────


class DataCleaningState(BaseState):
    """Internal environment state exposed via state()."""

    task_id: str = ""
    explore_steps_total: int = 0
    transform_steps_total: int = 0
    current_reward: float = 0.0
    constraints_satisfied: int = 0
    constraints_total: int = 0
