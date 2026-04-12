"""Runtime corruption pipeline -- generates dirty data + error_map at reset() time."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

from .profiles import DIFFICULTY_PROFILES, DIFFICULTY_WEIGHTS
from .value_corruptions import (
    CORRUPTION_REGISTRY,
    CORRUPTION_SEVERITY,
    _CLEAN_INDEX_MAP_ATTR,
    _CleanIndexMap,
)


class CorruptionPipeline:
    """Orchestrator that selects corruption types and applies them to a clean DataFrame.

    Key rule: ``select_format()`` MUST be called BEFORE ``corrupt()`` to
    preserve RNG ordering (both methods draw from the same py_rng).
    """

    def __init__(
        self,
        seed: int,
        difficulty: Optional[str] = None,
        category: Optional[str] = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        if difficulty is None:
            diffs = list(DIFFICULTY_WEIGHTS.keys())
            weights = list(DIFFICULTY_WEIGHTS.values())
            difficulty = self.py_rng.choices(diffs, weights=weights, k=1)[0]

        self.difficulty = difficulty
        self.profile = DIFFICULTY_PROFILES[difficulty]
        self.category = category

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_format(self) -> str:
        """MUST be called BEFORE corrupt() to preserve RNG ordering."""
        if self.category:
            from .categories import get_formats_for_category

            pool = get_formats_for_category(self.category)
        else:
            pool = self.profile["format_pool"]
        return self.py_rng.choice(pool)

    def corrupt(
        self,
        clean_df: pd.DataFrame,
        rules: list | None = None,
    ) -> tuple[pd.DataFrame, dict, dict, dict]:
        """Apply selected corruptions to *clean_df*.

        Returns
        -------
        dirty_df : pd.DataFrame
        error_map : dict
        severity_map : dict
        pipeline_metadata : dict
        """
        df = clean_df.copy()
        row_mapping = _CleanIndexMap({idx: idx for idx in df.index})
        df.attrs[_CLEAN_INDEX_MAP_ATTR] = row_mapping
        self._rules = rules or []
        error_log: list[dict] = []

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        all_cols = df.columns.tolist()

        # Detect identifier-like columns: name ends with 'id'/'key'/'index', or
        # column is numeric with all-unique integer values (surrogate key pattern).
        _ID_SUFFIXES = ("id", "_id", "key", "index", "idx", "no", "_no")
        identifier_cols: set[str] = set()
        for col in all_cols:
            col_lower = col.lower()
            if any(col_lower == s or col_lower.endswith(s) for s in _ID_SUFFIXES):
                identifier_cols.add(col)
            elif col in numeric_cols:
                series = df[col].dropna()
                if len(series) == len(df) and series.nunique() == len(series):
                    try:
                        if (series == series.astype(int)).all():
                            identifier_cols.add(col)
                    except (ValueError, TypeError):
                        pass

        # Corruptions that should never touch identifier columns
        _RISKY_FOR_IDS = {
            "outlier_injection",
            "type_mangle",
            "decimal_shift",
            "integer_as_float",
            "leading_zero_strip",
        }

        # --- choose corruption types ---
        if self.category:
            from .categories import get_corruptions_for_category

            allowed = get_corruptions_for_category(self.category)
        else:
            allowed = self.profile.get("allowed_corruptions")  # None = all types

        if self.category == "CP":
            min_types, max_types = 7, min(10, len(CORRUPTION_REGISTRY))
        else:
            min_types, max_types = self.profile["num_corruption_types"]
        num_types = self.py_rng.randint(min_types, max_types)

        applicable: list[str] = []
        for name, meta in CORRUPTION_REGISTRY.items():
            if allowed is not None and name not in allowed:
                continue
            if meta["requires_numeric"] and not numeric_cols:
                continue
            if meta["requires_string"] and not string_cols:
                continue
            applicable.append(name)

        selected = self.py_rng.sample(applicable, min(num_types, len(applicable)))
        # Row-level ops first: they reset/shift the index, so cell-level ops
        # must record keys against the final row structure. Sort key is
        # (is_cell_op, name) — row ops first, alphabetical within each group.
        _ROW_CORRUPTIONS = {"drop_rows", "duplicate_rows", "header_in_data"}
        selected.sort(key=lambda c: (c not in _ROW_CORRUPTIONS, c))

        frac_min, frac_max = self.profile["fraction_range"]
        applied_corruptions: list[dict] = []
        spurious_rows: dict[str, dict] = {}
        missing_rows: dict[str, dict] = {}

        for corruption_name in selected:
            meta = CORRUPTION_REGISTRY[corruption_name]
            fraction = self.py_rng.uniform(frac_min, frac_max)

            max_cols = self.profile.get("max_columns")
            if meta["requires_numeric"]:
                pool = numeric_cols
            elif meta["requires_string"]:
                pool = string_cols
            else:
                pool = all_cols
            # Exclude identifier-like columns from corruptions that mangle values/types
            if corruption_name in _RISKY_FOR_IDS and identifier_cols:
                pool = [c for c in pool if c not in identifier_cols] or pool
            n_cols = max(1, len(pool) // 2)
            if max_cols is not None:
                n_cols = min(n_cols, max_cols)
            target_cols = self.py_rng.sample(pool, min(n_cols, len(pool)))

            extra_kwargs = {}
            if corruption_name == "business_rule_violation" and self._rules:
                extra_kwargs["rules"] = self._rules
            df.attrs[_CLEAN_INDEX_MAP_ATTR] = row_mapping
            extra_kwargs["row_mapping"] = row_mapping

            df = meta["fn"](
                df,
                columns=target_cols,
                fraction=fraction,
                error_log=error_log,
                clean_df=clean_df,
                rng=self.rng,
                py_rng=self.py_rng,
                **extra_kwargs,
            )
            mapped = df.attrs.get(_CLEAN_INDEX_MAP_ATTR)
            if isinstance(mapped, dict):
                row_mapping = mapped

            applied_corruptions.append(
                {
                    "type": corruption_name,
                    "fraction": fraction,
                    "columns": target_cols,
                }
            )

        # --- build error_map ---
        cell_errors: dict[str, dict] = {}
        for entry in error_log:
            key = entry["key"]
            if key.startswith("missing_"):
                row_key = key.replace("missing_", "", 1)
                missing_rows[row_key] = {
                    "severity": entry["severity"],
                    "clean_values": entry.get("clean_values", {}),
                    "corruption": entry.get("corruption", "drop_rows"),
                }
            elif key.startswith("spurious_"):
                row_key = key.replace("spurious_", "", 1)
                spurious_rows[row_key] = {
                    "severity": entry["severity"],
                    "corruption": entry.get("corruption", "duplicate_rows"),
                }
            else:
                cell_errors[key] = {
                    "severity": entry["severity"],
                    "clean_value": entry["clean_value"],
                    "corruption": entry["corruption"],
                }
                if "accepted_fill" in entry:
                    cell_errors[key]["accepted_fill"] = entry["accepted_fill"]
                if "rule_type" in entry:
                    cell_errors[key]["rule_type"] = entry["rule_type"]

        error_map = {
            "cell_errors": cell_errors,
            "spurious_rows": spurious_rows,
            "missing_rows": missing_rows,
        }

        # --- build severity_map ---
        total_severity = (
            sum(e["severity"] for e in cell_errors.values())
            + sum(e["severity"] for e in spurious_rows.values())
            + sum(e["severity"] for e in missing_rows.values())
        )
        by_type: dict[str, float] = {}
        by_column: dict[str, float] = {}
        for e in cell_errors.values():
            t = e["corruption"]
            by_type[t] = by_type.get(t, 0) + e["severity"]
        for key, e in cell_errors.items():
            try:
                _, col = key.split(",", 1)
            except ValueError:
                col = "__unknown__"
            by_column[col] = by_column.get(col, 0) + e["severity"]
        # Include row-level corruptions in the by_type breakdown
        for e in spurious_rows.values():
            t = e.get("corruption", "duplicate_rows")  # default for backward compat
            by_type[t] = by_type.get(t, 0) + e["severity"]
            by_column["__spurious_rows__"] = (
                by_column.get("__spurious_rows__", 0) + e["severity"]
            )
        for e in missing_rows.values():
            t = e.get("corruption", "drop_rows")  # default for backward compat
            by_type[t] = by_type.get(t, 0) + e["severity"]
            by_column["__missing_rows__"] = (
                by_column.get("__missing_rows__", 0) + e["severity"]
            )
        severity_map = {
            "total_severity": total_severity,
            "by_type": by_type,
            "by_column": by_column,
        }

        pipeline_metadata = {
            "difficulty": self.difficulty,
            "seed": self.seed,
            "corruptions_applied": applied_corruptions,
            "format_corruptions_applied": [],
        }

        return df, error_map, severity_map, pipeline_metadata
