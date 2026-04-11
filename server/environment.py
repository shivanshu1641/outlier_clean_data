"""
Core Data Cleaning Environment — implements OpenEnv reset/step/state API.

Generative: datasets and corruptions are applied dynamically at reset() time
using a seeded RNG. Supports 9 file formats, 22 corruption types, undo/validate.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

try:
    from ..models import (
        DataCleaningObservation,
        DataCleaningState,
        DoneAction,
        ExploreAction,
        StepInfo,
        TransformAction,
        UndoAction,
        ValidateAction,
    )
except ImportError:
    from models import (
        DataCleaningObservation,
        DataCleaningState,
        DoneAction,
        ExploreAction,
        StepInfo,
        TransformAction,
        UndoAction,
        ValidateAction,
    )

from openenv.core import Environment

try:
    from .corruption.format_corruptions import apply_format_corruptions, convert_to_format, format_preview
    from .corruption.hints import generate_format_hints, generate_hints
    from .corruption.pipeline import CorruptionPipeline
    from .corruption.profiles import DIFFICULTY_PROFILES
    from .grader import grade, summarize_errors
    from .sandbox import ExecutionResult, create_sandbox, execute_explore, execute_transform, terminate_worker
except ImportError:
    from corruption.format_corruptions import apply_format_corruptions, convert_to_format, format_preview
    from corruption.hints import generate_format_hints, generate_hints
    from corruption.pipeline import CorruptionPipeline
    from corruption.profiles import DIFFICULTY_PROFILES
    from grader import grade, summarize_errors
    from sandbox import ExecutionResult, create_sandbox, execute_explore, execute_transform, terminate_worker

ActionType = Union[ExploreAction, TransformAction, DoneAction, UndoAction, ValidateAction]

DATA_DIR = os.environ.get("DATA_DIR", "data")
SANDBOX_BASE = os.environ.get("SANDBOX_BASE", "outputs/sandbox")
CATALOG_PATH = os.environ.get("CATALOG_PATH", str(Path(__file__).parent.parent / "datasets" / "catalog.json"))

# Defaults for profile keys not present in DIFFICULTY_PROFILES
_EXPLORE_BUDGET_DEFAULT = 10
_EXPLORE_COST_PER_STEP_DEFAULT = 0.01
_EXPLORE_TIMEOUT_COST_DEFAULT = 0.03
_VALIDATE_BUDGET_DEFAULT = 2
_UNDO_COST_DEFAULT = 0.02
_VALIDATE_COST_DEFAULT = 0.01

LEGACY_TASK_MAP: dict[str, dict] = {
    "titanic_easy":    {"dataset_id": "titanic",   "difficulty": "easy",   "format": "csv"},
    "titanic_medium":  {"dataset_id": "titanic",   "difficulty": "medium", "format": "csv"},
    "titanic_hard":    {"dataset_id": "titanic",   "difficulty": "hard",   "format": "csv"},
    "iris_easy":       {"dataset_id": "iris",      "difficulty": "easy",   "format": "csv"},
    "iris_medium":     {"dataset_id": "iris",      "difficulty": "medium", "format": "csv"},
    "housing_medium":  {"dataset_id": "housing",   "difficulty": "medium", "format": "csv"},
    "housing_hard":    {"dataset_id": "housing",   "difficulty": "hard",   "format": "csv"},
    "diabetes_medium": {"dataset_id": "diabetes",  "difficulty": "medium", "format": "csv"},
    "diabetes_hard":   {"dataset_id": "diabetes",  "difficulty": "hard",   "format": "csv"},
}


def _load_catalog() -> list[dict]:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _find_dataset(catalog: list[dict], dataset_id: str) -> Optional[dict]:
    for entry in catalog:
        if entry.get("id") == dataset_id:
            return entry
    return None


def _ensure_dataset(entry: dict) -> pd.DataFrame:
    """Load dataset from local_path, downloading if necessary."""
    local_path = entry.get("local_path", "")
    if local_path and Path(local_path).exists():
        return pd.read_csv(local_path)
    # Try to download
    try:
        import sys
        tools_dir = str(Path(__file__).parent.parent / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from download_datasets import download_one
        dest = Path(DATA_DIR)
        dest.mkdir(parents=True, exist_ok=True)
        path = download_one(entry, dest)
        if path:
            return pd.read_csv(path)
    except Exception:
        pass
    raise RuntimeError(f"Dataset not available: {entry.get('id')} — run tools/download_datasets.py first")


def _data_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    buf = StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write("Columns and types:\n")
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_info = f" ({null_count} nulls, {null_count/len(df)*100:.1f}%)" if null_count > 0 else ""
        buf.write(f"  {col}: {df[col].dtype}{null_info}\n")
    buf.write(f"\nSample rows (first {max_rows}):\n")
    buf.write(df.head(max_rows).to_string(index=False))
    return buf.getvalue()


def _target_schema(df: pd.DataFrame) -> dict[str, str]:
    return {col: str(df[col].dtype) for col in df.columns}


def _error_summary(error_status: dict[str, str], summary: dict[str, Any]) -> str:
    total = summary.get("total_errors", 0)
    fixed = summary.get("fixed", 0)
    wrong = summary.get("wrong_value", 0)
    unfixed = summary.get("unfixed", 0)
    lines = [f"Error fix progress: {fixed}/{total} fixed"]
    if wrong:
        lines.append(f"  {wrong} cells changed to wrong value (penalized 1.5x)")
    if unfixed:
        lines.append(f"  {unfixed} errors still unfixed")
    return "\n".join(lines)


def _validate_breakdown(error_status: dict[str, str], error_map: dict, clean_df: pd.DataFrame) -> str:
    """Generate detailed error breakdown for ValidateAction."""
    cell_errors = error_map.get("cell_errors", {})
    missing = error_map.get("missing_rows", {})
    lines = ["=== Validate: Detailed Error Breakdown ===\n"]

    unfixed_cells = [
        (k, v) for k, v in error_status.items()
        if v != "fixed" and not k.startswith("spurious_") and not k.startswith("missing_")
    ]
    if unfixed_cells:
        lines.append(f"Cell errors ({len(unfixed_cells)} unfixed):")
        for key, status in unfixed_cells[:20]:
            info = cell_errors.get(key, {})
            corruption = info.get("corruption", "unknown")
            clean_val = info.get("clean_value")
            lines.append(f"  [{status}] key={key!r}  corruption={corruption}  expected={clean_val!r}")
        if len(unfixed_cells) > 20:
            lines.append(f"  ... and {len(unfixed_cells)-20} more")

    unfixed_spurious = [k for k, v in error_status.items() if k.startswith("spurious_") and v != "fixed"]
    if unfixed_spurious:
        lines.append(f"\nSpurious rows ({len(unfixed_spurious)} unfixed):")
        for key in unfixed_spurious[:5]:
            row_str = key.replace("spurious_", "")
            lines.append(f"  Row index {row_str} is a duplicate — remove it")

    unfixed_missing = [k for k, v in error_status.items() if k.startswith("missing_") and v != "fixed"]
    if unfixed_missing:
        lines.append(f"\nMissing rows ({len(unfixed_missing)} unfixed):")
        for key in unfixed_missing[:5]:
            row_str = key.replace("missing_", "")
            info = missing.get(row_str, {})
            clean_vals = info.get("clean_values", {})
            lines.append(f"  Row {row_str} is missing — should contain: {clean_vals}")

    if not unfixed_cells and not unfixed_spurious and not unfixed_missing:
        lines.append("All errors fixed! Submit done.")

    return "\n".join(lines)


class DataCleaningEnvironment(
    Environment[ActionType, DataCleaningObservation, DataCleaningState]
):
    """OpenEnv environment for AI-powered data cleaning. Generative version."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._episode_id: str = ""
        self._difficulty: str = "medium"
        self._profile: dict = {}
        self._clean_df: pd.DataFrame = pd.DataFrame()
        self._dirty_df: pd.DataFrame = pd.DataFrame()
        self._dirty_content: str | bytes = ""
        self._file_format: str = "csv"
        self._error_map: dict[str, Any] = {}
        self._fmt_metadata: list[dict] = []
        self._sandbox_dir: str = ""
        self._worker_proc = None
        self._current_df: pd.DataFrame = pd.DataFrame()
        self._checkpoints: list[pd.DataFrame] = []  # index 0 = dirty_df, index N = after transform N
        self._error_status: dict[str, str] = {}
        self._error_summary_cache: dict[str, Any] = {}
        self._current_reward: float = 0.0
        self._explore_steps_cycle: int = 0
        self._explore_steps_total: int = 0
        self._explore_timeouts: int = 0
        self._transform_steps: int = 0
        self._undo_count: int = 0
        self._validate_uses: int = 0
        self._done: bool = False
        self._done_count: int = 0
        self._step_count: int = 0
        self._dataset_name: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        task_id = kwargs.get("task_id")
        difficulty = kwargs.get("difficulty", "medium")

        # Resolve dataset from task_id or legacy map
        catalog = _load_catalog()
        dataset_entry = None

        if task_id and task_id in LEGACY_TASK_MAP:
            legacy = LEGACY_TASK_MAP[task_id]
            difficulty = legacy["difficulty"]
            dataset_entry = _find_dataset(catalog, legacy["dataset_id"])
        elif task_id:
            # Try direct dataset id match
            dataset_entry = _find_dataset(catalog, task_id)

        if dataset_entry is None:
            # Pick random dataset
            rng_pick = random.Random(seed)
            dataset_entry = rng_pick.choice(catalog)

        self._difficulty = difficulty
        self._profile = DIFFICULTY_PROFILES[difficulty]
        self._episode_id = episode_id or str(uuid.uuid4())[:8]
        self._dataset_name = dataset_entry.get("name", dataset_entry.get("id", "unknown"))

        # Reset counters
        self._explore_steps_cycle = 0
        self._explore_steps_total = 0
        self._explore_timeouts = 0
        self._transform_steps = 0
        self._undo_count = 0
        self._validate_uses = 0
        self._done = False
        self._done_count = 0
        self._step_count = 0
        self._checkpoints = []

        # Load clean data
        clean_df = _ensure_dataset(dataset_entry)
        self._clean_df = clean_df.reset_index(drop=True)

        # Seeded RNGs
        np_rng = np.random.default_rng(seed if seed is not None else 42)
        py_rng = random.Random(seed if seed is not None else 42)

        # Run corruption pipeline
        pipeline = CorruptionPipeline()
        pipeline.select_format(np_rng)  # MUST be called before corrupt()
        dirty_df, error_map, _severity_map, pipeline_metadata = pipeline.corrupt(
            self._clean_df, difficulty, np_rng, py_rng
        )

        self._file_format = pipeline_metadata.get("format", "csv")
        # Normalize error_map to plain dict
        if hasattr(error_map, "model_dump"):
            self._error_map = error_map.model_dump()
        else:
            self._error_map = error_map
        self._dirty_df = dirty_df.reset_index(drop=True)

        # Convert to file format and apply format corruptions
        dirty_content = convert_to_format(dirty_df, self._file_format)
        dirty_content, fmt_meta = apply_format_corruptions(
            dirty_content, self._file_format, np_rng, py_rng, difficulty=difficulty
        )
        self._dirty_content = dirty_content
        self._fmt_metadata = fmt_meta

        # Terminate previous worker if any
        if self._worker_proc is not None:
            terminate_worker(self._worker_proc)
            self._worker_proc = None

        # Create sandbox (always uses CSV internally)
        import tempfile
        tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        dirty_df.to_csv(tmp_csv.name, index=False)
        tmp_csv.close()
        try:
            self._sandbox_dir, self._worker_proc = create_sandbox(
                self._episode_id, tmp_csv.name, base_dir=SANDBOX_BASE
            )
        finally:
            os.unlink(tmp_csv.name)

        # Load initial current_df
        self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))

        # Initial grade
        self._regrade()

        return self._make_observation()

    def step(
        self,
        action: ActionType,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        if self._done:
            return self._make_observation(transform_result="Episode is already done.")

        self._step_count += 1

        if isinstance(action, ExploreAction):
            return self._handle_explore(action)
        elif isinstance(action, TransformAction):
            return self._handle_transform(action)
        elif isinstance(action, DoneAction):
            return self._handle_done()
        elif isinstance(action, UndoAction):
            return self._handle_undo(action)
        elif isinstance(action, ValidateAction):
            return self._handle_validate()
        else:
            return self._make_observation(transform_result=f"Unknown action type: {type(action).__name__}")

    @property
    def state(self) -> DataCleaningState:
        summary = self._error_summary_cache
        return DataCleaningState(
            task_id=f"{self._dataset_name}_{self._difficulty}",
            explore_steps_total=self._explore_steps_total,
            transform_steps_total=self._transform_steps,
            current_reward=self._current_reward,
            constraints_satisfied=summary.get("fixed", 0),
            constraints_total=summary.get("total_errors", 0),
        )

    # ── Action Handlers ──────────────────────────────────────────────────────

    def _handle_explore(self, action: ExploreAction) -> DataCleaningObservation:
        budget = self._profile.get("explore_budget", _EXPLORE_BUDGET_DEFAULT)
        if self._explore_steps_cycle >= budget:
            return self._make_observation(
                explore_result=f"Explore budget exhausted ({budget}/{budget}). Submit a transform to reset it."
            )

        self._explore_steps_cycle += 1
        self._explore_steps_total += 1

        result = execute_explore(action.query, self._worker_proc, self._explore_steps_total)
        if not result.success:
            self._explore_timeouts += 1

        return self._make_observation(
            explore_result=result.stdout if result.success else f"Error: {result.error}"
        )

    def _handle_transform(self, action: TransformAction) -> DataCleaningObservation:
        self._transform_steps += 1
        self._explore_steps_cycle = 0

        result = execute_transform(action.code, self._worker_proc, self._transform_steps)

        if result.success:
            prev_df = self._current_df.copy()
            self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
            df = self._current_df

            # Save checkpoint: index 0 = dirty_df (before any transforms), index N = after transform N
            if len(self._checkpoints) == 0:
                self._checkpoints.append(self._dirty_df.copy())
            self._checkpoints.append(df.copy())

            data_changed = not prev_df.equals(df)
            self._regrade()

            msg = "Transform applied successfully."
            if not data_changed:
                msg += "\nWARNING: Data was not modified. Use df['col'] = df['col'].fillna(val) instead of inplace=True."
            if result.stdout:
                msg += f"\nOutput: {result.stdout[:500]}"
        else:
            msg = f"Transform failed: {result.error}"
            if result.stderr:
                msg += f"\nStderr: {result.stderr[:500]}"

        max_steps = self._profile.get("max_transform_steps", 10)
        if self._transform_steps >= max_steps:
            self._done = True
            msg += "\nMax transform steps reached. Episode ending."
            if self._worker_proc is not None:
                terminate_worker(self._worker_proc)
                self._worker_proc = None

        return self._make_observation(transform_result=msg)

    def _handle_done(self) -> DataCleaningObservation:
        self._done_count += 1
        self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
        self._regrade()

        if self._done_count == 1 and self._current_reward < 1.0:
            unfixed = sum(1 for s in self._error_status.values() if s != "fixed")
            return self._make_observation(
                transform_result=(
                    f"Soft done — score {self._current_reward:.4f} with {unfixed} errors remaining. "
                    f"Continue cleaning or submit done again to finalize."
                )
            )

        self._done = True
        if self._worker_proc is not None:
            terminate_worker(self._worker_proc)
            self._worker_proc = None
        return self._make_observation(transform_result="Episode complete. Final grading applied.")

    def _handle_undo(self, action: UndoAction) -> DataCleaningObservation:
        self._undo_count += 1
        step = action.step

        if not self._checkpoints:
            return self._make_observation(transform_result="Nothing to undo — no transforms applied yet.")

        # checkpoint[0] = dirty_df, checkpoint[N] = after transform N
        if step < len(self._checkpoints):
            restore_df = self._checkpoints[step].copy()
        else:
            return self._make_observation(
                transform_result=f"Invalid undo step {step}. Valid range: 0–{len(self._checkpoints)-1}."
            )

        # Write restored df to sandbox CSV
        csv_path = os.path.join(self._sandbox_dir, "current.csv")
        restore_df.to_csv(csv_path, index=False)
        self._current_df = restore_df

        # Truncate checkpoints to step (keep up to and including step)
        self._checkpoints = self._checkpoints[:step + 1]

        self._regrade()
        return self._make_observation(
            transform_result=f"Restored to state after transform step {step}."
        )

    def _handle_validate(self) -> DataCleaningObservation:
        budget = self._profile.get("validate_budget", _VALIDATE_BUDGET_DEFAULT)
        if self._validate_uses >= budget:
            return self._make_observation(
                validate_result=f"Validate budget exhausted ({budget}/{budget} used)."
            )

        self._validate_uses += 1
        self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))
        self._regrade()

        breakdown = _validate_breakdown(self._error_status, self._error_map, self._clean_df)
        return self._make_observation(validate_result=breakdown)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _regrade(self) -> None:
        profile = self._profile
        self._error_status, self._current_reward = grade(
            self._clean_df,
            self._current_df,
            self._error_map,
            self._transform_steps,
            profile.get("min_transform_steps", 2),
            profile.get("max_transform_steps", 10),
            explore_steps=self._explore_steps_total,
            explore_timeouts=self._explore_timeouts,
            explore_cost_per_step=profile.get("explore_cost_per_step", _EXPLORE_COST_PER_STEP_DEFAULT),
            explore_timeout_cost=profile.get("explore_timeout_cost", _EXPLORE_TIMEOUT_COST_DEFAULT),
            undo_count=self._undo_count,
            validate_count=self._validate_uses,
            undo_cost=profile.get("undo_cost", _UNDO_COST_DEFAULT),
            validate_cost=profile.get("validate_cost", _VALIDATE_COST_DEFAULT),
        )
        self._error_summary_cache = summarize_errors(self._error_status, self._error_map)

    def _make_step_info(self) -> StepInfo:
        profile = self._profile
        return StepInfo(
            done=self._done,
            reward=self._current_reward,
            explore_steps_used=self._explore_steps_cycle,
            explore_budget=profile.get("explore_budget", _EXPLORE_BUDGET_DEFAULT),
            transform_steps_used=self._transform_steps,
            max_transform_steps=profile.get("max_transform_steps", 10),
            min_transform_steps=profile.get("min_transform_steps", 2),
            done_count=self._done_count,
            undo_count=self._undo_count,
            validate_uses=self._validate_uses,
            validate_budget=profile.get("validate_budget", _VALIDATE_BUDGET_DEFAULT),
        )

    def _make_observation(
        self,
        explore_result: Optional[str] = None,
        transform_result: Optional[str] = None,
        validate_result: Optional[str] = None,
    ) -> DataCleaningObservation:
        df = self._current_df
        profile = self._profile
        hint_level = profile.get("hint_level", "categorical")

        # Build diagnosis from hints
        col_stats = {col: df[col].dropna().tolist() for col in df.columns if len(df[col].dropna()) > 0}
        diagnosis = generate_hints(self._error_map, hint_level, col_stats=col_stats)
        if self._fmt_metadata:
            diagnosis += "\n\n" + generate_format_hints(self._fmt_metadata, hint_level)

        return DataCleaningObservation(
            task_id=f"{self._dataset_name}_{self._difficulty}",
            task_description=(
                f"Clean the {self._dataset_name} dataset (difficulty: {self._difficulty}). "
                f"The data has been corrupted — restore it to its clean form."
            ),
            constraints=[_error_summary(self._error_status, self._error_summary_cache)],
            data_summary=_data_summary(df),
            explore_result=explore_result,
            transform_result=transform_result,
            constraint_status={k: (v == "fixed") for k, v in self._error_status.items()},
            file_format=self._file_format,
            target_schema=_target_schema(self._clean_df),
            file_preview=format_preview(self._dirty_content, self._file_format),
            diagnosis=diagnosis,
            validate_result=validate_result,
            step_info=self._make_step_info(),
        )
