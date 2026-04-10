"""Runtime corruption pipeline -- generates dirty data + error_map at reset() time."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

from .profiles import DIFFICULTY_PROFILES, DIFFICULTY_WEIGHTS
from .value_corruptions import CORRUPTION_REGISTRY, CORRUPTION_SEVERITY


class CorruptionPipeline:
    """Orchestrator that selects corruption types and applies them to a clean DataFrame.

    Key rule: ``select_format()`` MUST be called BEFORE ``corrupt()`` to
    preserve RNG ordering (both methods draw from the same py_rng).
    """

    def __init__(self, seed: int, difficulty: Optional[str] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        if difficulty is None:
            diffs = list(DIFFICULTY_WEIGHTS.keys())
            weights = list(DIFFICULTY_WEIGHTS.values())
            difficulty = self.py_rng.choices(diffs, weights=weights, k=1)[0]

        self.difficulty = difficulty
        self.profile = DIFFICULTY_PROFILES[difficulty]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_format(self) -> str:
        """MUST be called BEFORE corrupt() to preserve RNG ordering."""
        return self.py_rng.choice(self.profile["format_pool"])

    def corrupt(
        self, clean_df: pd.DataFrame
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
        error_log: list[dict] = []

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        string_cols = df.select_dtypes(include="object").columns.tolist()
        all_cols = df.columns.tolist()

        # --- choose corruption types ---
        min_types, max_types = self.profile["num_corruption_types"]
        num_types = self.py_rng.randint(min_types, max_types)

        applicable: list[str] = []
        for name, meta in CORRUPTION_REGISTRY.items():
            if meta["requires_numeric"] and not numeric_cols:
                continue
            if meta["requires_string"] and not string_cols:
                continue
            applicable.append(name)

        selected = self.py_rng.sample(
            applicable, min(num_types, len(applicable))
        )
        selected.sort()  # deterministic ordering

        frac_min, frac_max = self.profile["fraction_range"]
        applied_corruptions: list[dict] = []
        spurious_rows: dict[str, dict] = {}
        missing_rows: dict[str, dict] = {}

        for corruption_name in selected:
            meta = CORRUPTION_REGISTRY[corruption_name]
            fraction = self.py_rng.uniform(frac_min, frac_max)

            if meta["requires_numeric"]:
                target_cols = self.py_rng.sample(
                    numeric_cols, max(1, len(numeric_cols) // 2)
                )
            elif meta["requires_string"]:
                target_cols = self.py_rng.sample(
                    string_cols, max(1, len(string_cols) // 2)
                )
            else:
                target_cols = all_cols

            df = meta["fn"](
                df,
                columns=target_cols,
                fraction=fraction,
                error_log=error_log,
                clean_df=clean_df,
                rng=self.rng,
                py_rng=self.py_rng,
            )

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
                missing_rows[key] = {
                    "severity": entry["severity"],
                    "clean_values": entry.get("clean_values", {}),
                }
            elif key.startswith("spurious_"):
                row_key = key.replace("spurious_", "", 1)
                spurious_rows[row_key] = {
                    "severity": entry["severity"]
                }
            else:
                cell_errors[key] = {
                    "severity": entry["severity"],
                    "clean_value": entry["clean_value"],
                    "corruption": entry["corruption"],
                }
                if "accepted_fill" in entry:
                    cell_errors[key]["accepted_fill"] = entry["accepted_fill"]

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
        for e in cell_errors.values():
            t = e["corruption"]
            by_type[t] = by_type.get(t, 0) + e["severity"]
        severity_map = {"total_severity": total_severity, "by_type": by_type}

        pipeline_metadata = {
            "difficulty": self.difficulty,
            "seed": self.seed,
            "corruptions_applied": applied_corruptions,
            "format_corruptions_applied": [],
        }

        return df, error_map, severity_map, pipeline_metadata
