"""EnvClient subclass for the Data Cleaning environment."""

from __future__ import annotations

from typing import Any, Dict, Union

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    DataCleaningObservation,
    DataCleaningState,
    DoneAction,
    ExploreAction,
    TransformAction,
    UndoAction,
    ValidateAction,
)

ActionType = Union[ExploreAction, TransformAction, DoneAction, UndoAction, ValidateAction]


class DataCleaningClient(
    EnvClient[ActionType, DataCleaningObservation, DataCleaningState]
):
    """Client for interacting with the Data Cleaning environment server."""

    def _step_payload(self, action: ActionType) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DataCleaningObservation]:
        obs_data = payload.get("observation", payload)
        obs = DataCleaningObservation.model_validate(obs_data)
        # OpenEnv may put reward/done at top level OR inside observation
        reward = payload.get("reward") if payload.get("reward") is not None else obs.reward
        done = payload.get("done") if payload.get("done") is not None else obs.done
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DataCleaningState:
        return DataCleaningState.model_validate(payload)
