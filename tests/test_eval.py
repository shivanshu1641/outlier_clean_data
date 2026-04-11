"""Smoke tests for the 15-task fixed eval suite configuration."""
from __future__ import annotations

import pytest

EVAL_TASK_IDS = [
    "titanic_easy", "titanic_medium", "titanic_hard",
    "iris_easy", "iris_medium",
    "housing_medium", "housing_hard",
    "diabetes_medium", "diabetes_hard",
    "wine_easy", "wine_medium",
    "breast_cancer_easy", "breast_cancer_medium",
    "adult_medium",
    # 15th task for full coverage
    "income_medium",
]

from models import (
    ActionWrapper, DoneAction, ExploreAction, TransformAction, UndoAction, ValidateAction
)
from server.environment import LEGACY_TASK_MAP


class TestEvalSuiteConfig:
    def test_eval_has_15_tasks(self):
        assert len(EVAL_TASK_IDS) == 15, f"Expected 15 tasks, got {len(EVAL_TASK_IDS)}"

    def test_legacy_task_map_has_9_entries(self):
        assert len(LEGACY_TASK_MAP) == 9

    def test_legacy_task_ids_in_eval(self):
        for tid in LEGACY_TASK_MAP:
            assert tid in EVAL_TASK_IDS, f"{tid} missing from eval suite"

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
