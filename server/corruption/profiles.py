"""Difficulty profiles for corruption generation."""
from __future__ import annotations

DIFFICULTY_PROFILES = {
    "easy": {
        "num_corruption_types": (2, 3),
        "fraction_range": (0.03, 0.08),
        "overlap_allowed": False,
        "hint_level": "strategy",
        "format_pool": ["csv", "tsv"],
        "min_transform_steps": 2,
        "max_transform_steps": 8,
    },
    "medium": {
        "num_corruption_types": (4, 6),
        "fraction_range": (0.08, 0.15),
        "overlap_allowed": True,
        "hint_level": "tactical",
        "format_pool": ["csv", "json", "jsonl", "excel", "tsv", "fixed_width"],
        "min_transform_steps": 4,
        "max_transform_steps": 15,
    },
    "hard": {
        "num_corruption_types": (7, 10),
        "fraction_range": (0.15, 0.30),
        "overlap_allowed": True,
        "hint_level": "categorical",
        "format_pool": ["csv", "json", "jsonl", "excel", "tsv", "xml",
                        "fixed_width", "html_table", "sql_dump", "yaml"],
        "min_transform_steps": 8,
        "max_transform_steps": 25,
    },
}

DIFFICULTY_WEIGHTS = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
