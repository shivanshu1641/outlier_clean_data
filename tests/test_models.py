"""Tests for Pydantic models."""
import pytest
from models import (
    UndoAction, ValidateAction, ExploreAction, TransformAction, DoneAction,
    ActionWrapper, DataCleaningObservation, StepInfo, ErrorMap, CellError, RowError,
)


def test_undo_action_valid():
    a = UndoAction(step=3)
    assert a.type == "undo"
    assert a.step == 3


def test_undo_action_step_zero():
    a = UndoAction(step=0)
    assert a.step == 0


def test_undo_action_negative_step_rejected():
    with pytest.raises(Exception):
        UndoAction(step=-1)


def test_validate_action():
    a = ValidateAction()
    assert a.type == "validate"


def test_action_wrapper_routes_undo():
    obj = {"type": "undo", "step": 2}
    action = ActionWrapper.model_validate(obj)
    assert isinstance(action, UndoAction)
    assert action.step == 2


def test_action_wrapper_routes_validate():
    obj = {"type": "validate"}
    action = ActionWrapper.model_validate(obj)
    assert isinstance(action, ValidateAction)


def test_action_wrapper_routes_existing_types():
    assert isinstance(ActionWrapper.model_validate({"type": "explore", "query": "df.shape"}), ExploreAction)
    assert isinstance(ActionWrapper.model_validate({"type": "transform", "code": "pass"}), TransformAction)
    assert isinstance(ActionWrapper.model_validate({"type": "done"}), DoneAction)


def test_error_map_schema():
    em = ErrorMap(
        cell_errors={
            "0,Age": CellError(severity=2.0, clean_value=29.0, corruption="inject_nulls", accepted_fill="any"),
        },
        spurious_rows={
            "5": RowError(severity=2.0),
        },
        missing_rows={
            "missing_0": RowError(severity=2.5, clean_values={"Name": "Alice", "Age": 25}),
        },
    )
    assert em.cell_errors["0,Age"].severity == 2.0
    assert em.cell_errors["0,Age"].corruption == "inject_nulls"
    assert em.spurious_rows["5"].severity == 2.0
    assert em.missing_rows["missing_0"].clean_values["Name"] == "Alice"


def test_error_map_empty():
    em = ErrorMap()
    assert em.cell_errors == {}
    assert em.spurious_rows == {}
    assert em.missing_rows == {}


def test_observation_new_fields():
    obs = DataCleaningObservation(
        task_id="test",
        file_format="json",
        target_schema={"Age": "float64", "Name": "object"},
        file_preview='{"name": "Alice"}',
        diagnosis="~10 null values detected",
        validate_result="Nulls in: Age (5), Fare (3)",
    )
    assert obs.file_format == "json"
    assert obs.target_schema["Age"] == "float64"
    assert obs.diagnosis is not None


def test_observation_new_fields_optional():
    obs = DataCleaningObservation(task_id="test")
    assert obs.file_format is None
    assert obs.target_schema is None
    assert obs.file_preview is None
    assert obs.diagnosis is None
    assert obs.validate_result is None


def test_step_info_new_fields():
    info = StepInfo(undo_count=2, validate_uses=1, validate_budget=2)
    assert info.undo_count == 2
    assert info.validate_uses == 1
    assert info.validate_budget == 2


def test_step_info_defaults():
    info = StepInfo()
    assert info.undo_count == 0
    assert info.validate_uses == 0
    assert info.validate_budget == 2
