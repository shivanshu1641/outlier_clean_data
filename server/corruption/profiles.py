"""Difficulty profiles for corruption generation."""
from __future__ import annotations

DIFFICULTY_PROFILES = {
    "easy": {
        "num_corruption_types": (1, 1),
        "fraction_range": (0.03, 0.06),
        "overlap_allowed": False,
        "format_pool": ["csv"],
        "min_transform_steps": 1,
        "max_transform_steps": 6,
        "max_columns": 2,  # corrupt at most 2 columns
        "allowed_corruptions": [
            "inject_nulls", "whitespace_noise", "duplicate_rows",
        ],
    },
    "medium": {
        "num_corruption_types": (4, 6),
        "fraction_range": (0.08, 0.15),
        "overlap_allowed": True,
        "format_pool": ["csv", "json", "jsonl", "excel", "tsv", "fixed_width"],
        "min_transform_steps": 4,
        "max_transform_steps": 15,
        "allowed_corruptions": [
            "inject_nulls", "whitespace_noise", "duplicate_rows",
            "type_mangle", "format_inconsistency", "outlier_injection",
            "date_format_mix", "category_misspell", "leading_zero_strip",
            "typo_injection", "drop_rows", "header_in_data",
        ],
    },
    "hard": {
        "num_corruption_types": (7, 10),
        "fraction_range": (0.15, 0.30),
        "overlap_allowed": True,
        "format_pool": ["csv", "json", "jsonl", "excel", "tsv", "xml",
                        "fixed_width", "html_table", "sql_dump", "yaml"],
        "min_transform_steps": 8,
        "max_transform_steps": 25,
        # allowed_corruptions intentionally absent — hard mode uses all 22 types
    },
}

DIFFICULTY_WEIGHTS = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
