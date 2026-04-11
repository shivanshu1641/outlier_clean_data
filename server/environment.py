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
    from .corruption.format_corruptions import (
        apply_format_corruptions,
        convert_to_format,
        format_preview,
    )
    from .corruption.pipeline import CorruptionPipeline
    from .corruption.profiles import DIFFICULTY_PROFILES
    from .grader import grade, summarize_errors
    from .sandbox import (
        ExecutionResult,
        create_sandbox,
        execute_explore,
        execute_transform,
        reload_worker_df,
        restore_checkpoint,
        save_checkpoint,
        terminate_worker,
    )
except ImportError:
    from corruption.format_corruptions import (
        apply_format_corruptions,
        convert_to_format,
        format_preview,
    )
    from corruption.pipeline import CorruptionPipeline
    from corruption.profiles import DIFFICULTY_PROFILES
    from grader import grade, summarize_errors
    from sandbox import (
        ExecutionResult,
        create_sandbox,
        execute_explore,
        execute_transform,
        reload_worker_df,
        restore_checkpoint,
        save_checkpoint,
        terminate_worker,
    )

ActionType = Union[
    ExploreAction, TransformAction, DoneAction, UndoAction, ValidateAction
]

DATA_DIR = os.environ.get("DATA_DIR", "data")
SANDBOX_BASE = os.environ.get("SANDBOX_BASE", "outputs/sandbox")
CATALOG_PATH = os.environ.get(
    "CATALOG_PATH", str(Path(__file__).parent.parent / "datasets" / "catalog.json")
)

# Defaults for profile keys not present in DIFFICULTY_PROFILES
_EXPLORE_BUDGET_DEFAULT = 10
_EXPLORE_COST_PER_STEP_DEFAULT = 0.01
_EXPLORE_TIMEOUT_COST_DEFAULT = 0.03
_VALIDATE_BUDGET_DEFAULT = 2
_UNDO_COST_DEFAULT = 0.02
_VALIDATE_COST_DEFAULT = 0.01


def _load_catalog() -> dict[str, dict]:
    """Load catalog.json as {dataset_name: entry_dict}."""
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _find_dataset(catalog: dict[str, dict], dataset_id: str) -> tuple[str, dict] | None:
    """Look up a dataset by exact catalog key. Returns (name, entry) or None."""
    if dataset_id in catalog:
        return dataset_id, catalog[dataset_id]
    return None


def _ensure_dataset(name: str, entry: dict) -> pd.DataFrame:
    """Load dataset CSV from data/clean/, downloading if necessary."""
    filename = entry.get("filename", f"{name}.csv")
    clean_dir = Path(__file__).parent.parent / DATA_DIR / "clean"
    csv_path = clean_dir / filename

    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Try to download on the fly
    try:
        import sys

        tools_dir = str(Path(__file__).parent.parent / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from download_datasets import download_one

        clean_dir.mkdir(parents=True, exist_ok=True)
        ok = download_one(name, entry, clean_dir)
        if ok and csv_path.exists():
            return pd.read_csv(csv_path)
    except Exception:
        pass
    raise RuntimeError(
        f"Dataset not available: {name} — run tools/download_datasets.py first"
    )


def _data_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    buf = StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write("Columns and types:\n")
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_info = (
            f" ({null_count} nulls, {null_count/len(df)*100:.1f}%)"
            if null_count > 0
            else ""
        )
        buf.write(f"  {col}: {df[col].dtype}{null_info}\n")
    buf.write(f"\nSample rows (first {max_rows}):\n")
    buf.write(df.head(max_rows).to_string(index=False))
    return buf.getvalue()


def _target_schema(df: pd.DataFrame) -> dict[str, str]:
    return {col: str(df[col].dtype) for col in df.columns}


def _error_summary(
    error_status: dict[str, str],
    summary: dict[str, Any],
    error_map: dict[str, Any] | None = None,
) -> str:
    total = summary.get("total_errors", 0)
    fixed = summary.get("fixed", 0)
    wrong = summary.get("wrong_value", 0)
    unfixed = summary.get("unfixed", 0)
    lines = [f"Error fix progress: {fixed}/{total} fixed"]
    if wrong:
        lines.append(f"  {wrong} cells changed to wrong value (penalized 1.5x)")
    if unfixed:
        lines.append(f"  {unfixed} errors still unfixed")
    # Compact corruption-type breakdown — only UNFIXED errors so the model
    # doesn't waste steps re-fixing already-fixed corruption types.
    if error_map:
        cell_errors = error_map.get("cell_errors", {})
        by_type: dict[str, dict] = {}  # ctype -> {cols: set, count: int}
        # Collect up to 2 sample dirty→clean pairs per corruption type
        type_samples: dict[str, list[str]] = {}
        for key, info in cell_errors.items():
            # Skip already-fixed errors
            if error_status.get(key) == "fixed":
                continue
            ctype = info.get("corruption", "unknown")
            parts = key.split(",", 1)
            col = parts[1] if len(parts) == 2 else "?"
            entry = by_type.setdefault(ctype, {"cols": set(), "count": 0})
            entry["cols"].add(col)
            entry["count"] += 1
            if len(type_samples.get(ctype, [])) < 2:
                dirty = info.get("dirty_value")
                clean = info.get("clean_value")
                if dirty is not None and clean is not None and dirty != clean:
                    type_samples.setdefault(ctype, []).append(
                        f"{col}: {dirty!r} → {clean!r}"
                    )
        if by_type:
            lines.append("  Remaining error types (already-fixed types are hidden):")
            for ctype, info in sorted(by_type.items(), key=lambda x: -x[1]["count"]):
                cols = info["cols"]
                count = info["count"]
                lines.append(
                    f"    {ctype} ({count} errors) in: {', '.join(sorted(cols))}"
                )
                for sample in type_samples.get(ctype, []):
                    lines.append(f"      example: {sample}")
        if error_map.get("spurious_rows"):
            lines.append(
                f"    duplicate_rows: {len(error_map['spurious_rows'])} extra rows"
            )
        if error_map.get("missing_rows"):
            lines.append(
                f"    missing_rows: {len(error_map['missing_rows'])} rows missing"
            )
    return "\n".join(lines)


def _remaining_error_breakdown(
    error_status: dict[str, str],
    error_map: dict[str, Any],
    max_types: int = 6,
    max_cols: int = 4,
) -> list[str]:
    """Return compact remaining-error lines grouped by corruption type."""
    cell_errors = error_map.get("cell_errors", {})
    by_type: dict[str, dict[str, Any]] = {}
    wrong_total = 0

    for key, status in error_status.items():
        if status == "fixed":
            continue
        if key.startswith("spurious_") or key.startswith("missing_"):
            continue

        info = cell_errors.get(key, {})
        ctype = info.get("corruption", "unknown")
        try:
            _, col = key.split(",", 1)
        except ValueError:
            col = "?"

        entry = by_type.setdefault(ctype, {"count": 0, "wrong": 0, "columns": set()})
        entry["count"] += 1
        entry["columns"].add(col)
        if status == "wrong_value":
            entry["wrong"] += 1
            wrong_total += 1

    # Split into unfixed (need fixing) vs wrong_value (model broke these)
    unfixed_lines: list[str] = []
    wrong_lines: list[str] = []
    for ctype, info in sorted(
        by_type.items(), key=lambda item: (-item[1]["count"], item[0])
    )[:max_types]:
        cols = sorted(info["columns"])
        col_text = ", ".join(cols[:max_cols])
        if len(cols) > max_cols:
            col_text += f", +{len(cols) - max_cols} more"
        unfixed_count = info["count"] - info["wrong"]
        if unfixed_count > 0:
            unfixed_lines.append(f"{ctype}: {unfixed_count} unfixed in {col_text}")
        if info["wrong"] > 0:
            wrong_lines.append(f"{ctype}: {info['wrong']} wrong_value in {col_text}")

    lines: list[str] = []
    if unfixed_lines:
        lines.append("Still need fixing:")
        lines.extend(f"  {l}" for l in unfixed_lines)
    if wrong_lines:
        lines.append("Broken by your transforms (undo or leave alone — do NOT re-transform these columns):")
        lines.extend(f"  {l}" for l in wrong_lines)

    spurious = sum(
        1 for k, v in error_status.items() if k.startswith("spurious_") and v != "fixed"
    )
    missing = sum(
        1 for k, v in error_status.items() if k.startswith("missing_") and v != "fixed"
    )
    if spurious:
        lines.append(f"duplicate_rows: {spurious} remaining")
    if missing:
        lines.append(f"missing_rows: {missing} remaining")
    if wrong_total and not any("wrong_value" in line for line in lines):
        lines.append(f"wrong_value: {wrong_total} remaining")
    return lines


def _changed_columns(
    prev_df: pd.DataFrame, curr_df: pd.DataFrame, max_cols: int = 8
) -> str:
    """Return a compact list of columns touched by the last transform."""
    changed: list[str] = []
    all_cols = list(dict.fromkeys(list(prev_df.columns) + list(curr_df.columns)))
    for col in all_cols:
        if col not in prev_df.columns or col not in curr_df.columns:
            changed.append(col)
            continue
        prev_col = prev_df[col].reset_index(drop=True)
        curr_col = curr_df[col].reset_index(drop=True)
        if len(prev_col) != len(curr_col) or not prev_col.equals(curr_col):
            changed.append(col)

    if not changed:
        return "none"
    if len(changed) <= max_cols:
        return ", ".join(changed)
    return ", ".join(changed[:max_cols]) + f", +{len(changed) - max_cols} more"


def _format_transform_feedback(
    success: bool,
    *,
    result: ExecutionResult,
    data_changed: bool = False,
    changed_columns: str = "none",
    reward_before: float | None = None,
    reward_after: float | None = None,
    summary_before: dict[str, Any] | None = None,
    summary_after: dict[str, Any] | None = None,
    remaining_lines: list[str] | None = None,
) -> str:
    """Build a compact transform outcome block for the agent prompt."""
    lines = [
        f"Execution: {'succeeded' if success else 'failed'}",
    ]

    if reward_before is not None and reward_after is not None:
        lines.append(
            f"Reward delta: {reward_after - reward_before:+.4f} ({reward_before:.4f} -> {reward_after:.4f})"
        )

    if summary_before and summary_after:
        lines.append(
            "Errors fixed delta: "
            f"{summary_after['fixed'] - summary_before['fixed']:+d} "
            f"({summary_after['fixed']}/{summary_after['total_errors']})"
        )
        lines.append(
            "Wrong-value delta: "
            f"{summary_after['wrong_value'] - summary_before['wrong_value']:+d} "
            f"({summary_after['wrong_value']} total)"
        )
        lines.append(
            "Unfixed delta: "
            f"{summary_after['unfixed'] - summary_before['unfixed']:+d} "
            f"({summary_after['unfixed']} remaining)"
        )

    lines.append(f"Data changed: {'yes' if data_changed else 'no'}")
    lines.append(f"Changed columns: {changed_columns}")

    if remaining_lines:
        lines.append("Remaining errors by type:")
        lines.extend(f"  - {line}" for line in remaining_lines)

    if success and result.stdout:
        lines.append(f"Output: {result.stdout[:500]}")
    elif not success:
        lines.append(f"Error: {result.error}")
        if result.stderr:
            lines.append(f"Stderr: {result.stderr[:500]}")

    return "\n".join(lines)


def _refresh_dirty_values_from_df(error_map: dict[str, Any], df: pd.DataFrame) -> None:
    """Align error_map dirty_value entries to the parsed DataFrame seen by the agent."""
    for key, info in error_map.get("cell_errors", {}).items():
        try:
            row_str, col = key.split(",", 1)
            row_idx = int(row_str)
            if row_idx >= len(df) or col not in df.columns:
                continue
            dv = df.at[row_idx, col]
            try:
                info["dirty_value"] = None if pd.isna(dv) else dv
            except (TypeError, ValueError):
                info["dirty_value"] = dv
        except (ValueError, IndexError, KeyError):
            continue


def _validate_breakdown(
    error_status: dict[str, str], error_map: dict, clean_df: pd.DataFrame
) -> str:
    """Generate detailed error breakdown for ValidateAction."""
    cell_errors = error_map.get("cell_errors", {})
    missing = error_map.get("missing_rows", {})
    lines = ["=== Validate: Detailed Error Breakdown ===\n"]

    unfixed_cells = [
        (k, v)
        for k, v in error_status.items()
        if v != "fixed"
        and not k.startswith("spurious_")
        and not k.startswith("missing_")
    ]
    if unfixed_cells:
        lines.append(f"Cell errors ({len(unfixed_cells)} unfixed):")
        for key, status in unfixed_cells[:20]:
            info = cell_errors.get(key, {})
            corruption = info.get("corruption", "unknown")
            clean_val = info.get("clean_value")
            lines.append(
                f"  [{status}] key={key!r}  corruption={corruption}  expected={clean_val!r}"
            )
        if len(unfixed_cells) > 20:
            lines.append(f"  ... and {len(unfixed_cells)-20} more")

    unfixed_spurious = [
        k for k, v in error_status.items() if k.startswith("spurious_") and v != "fixed"
    ]
    if unfixed_spurious:
        lines.append(f"\nSpurious rows ({len(unfixed_spurious)} unfixed):")
        for key in unfixed_spurious[:5]:
            row_str = key.replace("spurious_", "")
            lines.append(f"  Row index {row_str} is a duplicate — remove it")

    unfixed_missing = [
        k for k, v in error_status.items() if k.startswith("missing_") and v != "fixed"
    ]
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
        self._checkpoint_steps: int = (
            0  # number of filesystem checkpoints saved (0 = none yet)
        )
        self._error_status: dict[str, str] = {}
        self._error_summary_cache: dict[str, Any] = {}
        self._current_reward: float = 0.0
        self._reward_baseline: float = 0.0
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
        # Grading caches — avoid recomputing stable scores every step
        self._cached_schema_score: float | None = None
        self._cached_row_mapping: dict | None = None
        self._reward_stale: bool = True

    def __del__(self) -> None:
        """Ensure worker is cleaned up even if close() was never called."""
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Clean up sandbox worker and directory."""
        if self._worker_proc is not None:
            terminate_worker(self._worker_proc)
            self._worker_proc = None
        if self._sandbox_dir and os.path.isdir(self._sandbox_dir):
            import shutil

            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            self._sandbox_dir = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        task_id = kwargs.get("task_id")
        difficulty = kwargs.get("difficulty", "medium")
        category = kwargs.get("category")
        forced_format: str | None = kwargs.get("format")

        # Resolve dataset from task_id (dataset_id) directly
        catalog = _load_catalog()
        dataset_name: str | None = None
        dataset_entry: dict | None = None

        if task_id:
            result = _find_dataset(catalog, task_id)
            if result:
                dataset_name, dataset_entry = result

        if dataset_entry is None:
            # Pick random dataset
            rng_pick = random.Random(seed)
            names = list(catalog.keys())
            dataset_name = rng_pick.choice(names)
            dataset_entry = catalog[dataset_name]

        self._difficulty = difficulty
        self._category = category
        self._profile = DIFFICULTY_PROFILES[difficulty]
        self._episode_id = episode_id or str(uuid.uuid4())[:8]
        self._dataset_name = dataset_name

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
        self._cached_schema_score = None
        self._cached_row_mapping = None
        self._reward_stale = True
        self._checkpoint_steps = 0

        # Load clean data
        clean_df = _ensure_dataset(dataset_name, dataset_entry)
        self._clean_df = clean_df.reset_index(drop=True)

        # Seeded RNGs
        np_rng = np.random.default_rng(seed if seed is not None else 42)
        py_rng = random.Random(seed if seed is not None else 42)

        # Load semantic rules from catalog entry
        raw_rules = dataset_entry.get("rules", [])
        self._rules_dicts = raw_rules
        self._rules = []
        if raw_rules:
            try:
                from server.rules.types import rule_from_dict
            except ImportError:
                from rules.types import rule_from_dict
            self._rules = [rule_from_dict(r) for r in raw_rules]

        # Run corruption pipeline
        _seed = seed if seed is not None else 42
        pipeline = CorruptionPipeline(
            seed=_seed, difficulty=difficulty, category=category
        )
        selected = (
            pipeline.select_format()
        )  # MUST be called before corrupt() to preserve RNG
        self._file_format = forced_format if forced_format else selected
        dirty_df, error_map, _severity_map, pipeline_metadata = pipeline.corrupt(
            self._clean_df,
            rules=self._rules,
        )
        # Normalize error_map to plain dict
        if hasattr(error_map, "model_dump"):
            self._error_map = error_map.model_dump()
        else:
            self._error_map = error_map
        self._dirty_df = dirty_df.reset_index(drop=True)

        # Seed dirty_value from the generated dirty frame first. After the sandbox
        # round-trip we refresh this from current_df so grading matches what the
        # agent actually sees in pandas.
        _refresh_dirty_values_from_df(self._error_map, self._dirty_df)

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
        _refresh_dirty_values_from_df(self._error_map, self._current_df)

        # Save step-0 checkpoint = dirty state (before any agent transforms)
        save_checkpoint(self._sandbox_dir, 0)
        self._checkpoint_steps = 1  # next transform will be saved as step 1

        # Initial grade — capture baseline (dirty data already-correct components)
        self._regrade()
        self._reward_baseline = self._current_reward
        self._current_reward = 0.0

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
            return self._make_observation(
                transform_result=f"Unknown action type: {type(action).__name__}"
            )

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

        result = execute_explore(
            action.query, self._worker_proc, self._explore_steps_total
        )
        if not result.success:
            self._explore_timeouts += 1

        return self._make_observation(
            explore_result=result.stdout if result.success else f"Error: {result.error}"
        )

    def _handle_transform(self, action: TransformAction) -> DataCleaningObservation:
        self._transform_steps += 1
        self._explore_steps_cycle = 0

        prev_df = self._current_df.copy()
        prev_reward = self._current_reward
        prev_summary = dict(self._error_summary_cache)
        result = execute_transform(
            action.code, self._worker_proc, self._transform_steps
        )

        if result.success:
            self._current_df = pd.read_csv(
                os.path.join(self._sandbox_dir, "current.csv")
            )
            df = self._current_df

            # Save filesystem checkpoint after successful transform
            # step 0 = dirty state (saved at reset), step N = after transform N
            save_checkpoint(self._sandbox_dir, self._checkpoint_steps)
            self._checkpoint_steps += 1

            data_changed = not prev_df.equals(df)
            self._regrade()
            changed_columns = _changed_columns(prev_df, df)
            remaining_lines = _remaining_error_breakdown(
                self._error_status, self._error_map
            )
            msg = _format_transform_feedback(
                True,
                result=result,
                data_changed=data_changed,
                changed_columns=changed_columns,
                reward_before=prev_reward,
                reward_after=self._current_reward,
                summary_before=prev_summary,
                summary_after=self._error_summary_cache,
                remaining_lines=remaining_lines,
            )
            if not data_changed:
                msg += "\nWARNING: Data was not modified. Use df['col'] = df['col'].fillna(val) instead of inplace=True."
        else:
            remaining_lines = _remaining_error_breakdown(
                self._error_status, self._error_map
            )
            msg = _format_transform_feedback(
                False,
                result=result,
                data_changed=False,
                changed_columns="none",
                reward_before=prev_reward,
                reward_after=self._current_reward,
                summary_before=prev_summary,
                summary_after=prev_summary,
                remaining_lines=remaining_lines,
            )

        max_steps = self._profile.get("max_transform_steps", 10)
        if self._transform_steps >= max_steps:
            self._done = True
            msg += "\nMax transform steps reached. Episode ending."
            if self._worker_proc is not None:
                terminate_worker(self._worker_proc)
                self._worker_proc = None

        return self._make_observation(transform_result=msg)

    def _ensure_graded(self) -> None:
        """Regrade only if reward is stale (after a transform)."""
        if self._reward_stale:
            self._regrade()

    def _handle_done(self) -> DataCleaningObservation:
        self._done_count += 1
        self._ensure_graded()  # no-op if already graded; _current_df already up-to-date

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
        return self._make_observation(
            transform_result="Episode complete. Final grading applied."
        )

    def _handle_undo(self, action: UndoAction) -> DataCleaningObservation:
        self._undo_count += 1
        step = action.step

        if self._checkpoint_steps <= 1:
            return self._make_observation(
                transform_result="Nothing to undo — no transforms applied yet."
            )

        max_step = self._checkpoint_steps - 1
        if step > max_step:
            return self._make_observation(
                transform_result=f"Invalid undo step {step}. Valid range: 0–{max_step}."
            )

        # Restore filesystem checkpoint and re-read current.csv
        if not restore_checkpoint(self._sandbox_dir, step):
            return self._make_observation(
                transform_result=f"Checkpoint for step {step} not found."
            )
        self._current_df = pd.read_csv(os.path.join(self._sandbox_dir, "current.csv"))

        # Sync worker's in-memory df to the restored state so next explore/transform
        # runs against the correct checkpoint, not stale post-undo data
        if self._worker_proc is not None:
            reload_worker_df(self._worker_proc)

        # Truncate checkpoint count to step (discard checkpoints after restored step)
        self._checkpoint_steps = step + 1

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
        self._ensure_graded()  # no-op if already graded; _current_df already up-to-date

        breakdown = _validate_breakdown(
            self._error_status, self._error_map, self._clean_df
        )
        return self._make_observation(validate_result=breakdown)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _regrade(self) -> None:
        profile = self._profile
        cached_rm = (
            None  # Always recompute — row content may change without count change
        )
        self._error_status, self._current_reward, ss, rm = grade(
            self._clean_df,
            self._current_df,
            self._error_map,
            self._transform_steps,
            profile.get("min_transform_steps", 2),
            profile.get("max_transform_steps", 10),
            explore_steps=self._explore_steps_total,
            explore_timeouts=self._explore_timeouts,
            explore_cost_per_step=profile.get(
                "explore_cost_per_step", _EXPLORE_COST_PER_STEP_DEFAULT
            ),
            explore_timeout_cost=profile.get(
                "explore_timeout_cost", _EXPLORE_TIMEOUT_COST_DEFAULT
            ),
            undo_count=self._undo_count,
            validate_count=self._validate_uses,
            undo_cost=profile.get("undo_cost", _UNDO_COST_DEFAULT),
            validate_cost=profile.get("validate_cost", _VALIDATE_COST_DEFAULT),
            rules=getattr(self, "_rules", None),
            cached_schema_score=self._cached_schema_score,
            cached_row_mapping=cached_rm,
        )
        self._cached_schema_score = ss
        self._cached_row_mapping = rm
        # Normalize: subtract baseline so unfixed=0.0, fully fixed=1.0
        if self._reward_baseline < 1.0:
            self._current_reward = max(
                0.0,
                round(
                    (self._current_reward - self._reward_baseline)
                    / (1.0 - self._reward_baseline),
                    4,
                ),
            )
        self._error_summary_cache = summarize_errors(
            self._error_status, self._error_map
        )
        self._reward_stale = False

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

        return DataCleaningObservation(
            task_id=f"{self._dataset_name}_{self._difficulty}",
            task_description=(
                f"Clean the {self._dataset_name} dataset (difficulty: {self._difficulty}). "
                f"The data has been corrupted — restore it to its clean form."
            ),
            constraints=[
                _error_summary(
                    self._error_status, self._error_summary_cache, self._error_map
                )
            ],
            data_summary=_data_summary(df),
            explore_result=explore_result,
            transform_result=transform_result,
            constraint_status={
                k: (v == "fixed") for k, v in self._error_status.items()
            },
            file_format=self._file_format,
            target_schema=_target_schema(self._clean_df),
            file_preview=format_preview(self._dirty_content, self._file_format),
            validate_result=validate_result,
            semantic_rules=getattr(self, "_rules_dicts", []),
            step_info=self._make_step_info(),
            reward=self._current_reward,
            done=self._done,
        )
