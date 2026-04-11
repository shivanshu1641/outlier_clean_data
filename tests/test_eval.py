"""Smoke tests for the task suite configuration."""
from __future__ import annotations

import pytest

from inference import EVAL_TASKS
from models import (
    ActionWrapper, DoneAction, ExploreAction, TransformAction, UndoAction, ValidateAction
)


class TestSuiteConfig:
    def test_task_count(self):
        assert len(EVAL_TASKS) == 14, f"Expected 14 tasks, got {len(EVAL_TASKS)}"

    def test_task_format(self):
        for dataset_id, difficulty in EVAL_TASKS:
            assert isinstance(dataset_id, str) and dataset_id, "dataset_id must be non-empty string"
            assert difficulty in ("easy", "medium", "hard"), f"Invalid difficulty: {difficulty}"

    def test_action_wrapper_routes_undo(self):
        action = ActionWrapper.model_validate({"type": "undo", "step": 2})
        assert isinstance(action, UndoAction)
        assert action.step == 2

    def test_action_wrapper_routes_validate(self):
        action = ActionWrapper.model_validate({"type": "validate"})
        assert isinstance(action, ValidateAction)

    def test_action_wrapper_routes_all_types(self):
        for t, cls in [
            ("explore", ExploreAction),
            ("transform", TransformAction),
            ("done", DoneAction),
            ("undo", UndoAction),
            ("validate", ValidateAction),
        ]:
            d = {"type": t}
            if t == "explore":
                d["query"] = "df.head()"
            if t == "transform":
                d["code"] = "pass"
            if t == "undo":
                d["step"] = 0
            action = ActionWrapper.model_validate(d)
            assert isinstance(action, cls), f"Expected {cls.__name__} for type={t}"
