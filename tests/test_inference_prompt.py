"""Tests for inference prompt construction."""

from types import SimpleNamespace

from inference import build_user_prompt


def test_build_user_prompt_includes_validate_result():
    obs = SimpleNamespace(
        task_description="Clean sample data",
        constraints=["Error fix progress: 0/2 fixed"],
        constraint_status={"a": False, "b": False},
        reward=0.0,
        data_summary="Shape: 2 rows x 2 columns",
        explore_result=None,
        transform_result=None,
        validate_result="Cell errors (2 unfixed): ...",
        step_info=None,
    )

    prompt = build_user_prompt(obs, result_reward=0.0, action_history=[])

    assert "Last validate result:" in prompt
    assert "Cell errors (2 unfixed): ..." in prompt
