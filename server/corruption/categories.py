"""Benchmark category definitions.

Maps each of 6 skill categories to the corruption types and file formats
that stress-test that category. Used by CorruptionPipeline when
a category is specified to constrain what gets generated.
"""
from __future__ import annotations

CATEGORIES = ("FP", "VR", "MD", "SR", "SV", "CP")

# Corruption types per category. CP = None (any 7+ types from all categories).
CATEGORY_CORRUPTION_MAP: dict[str, list[str] | None] = {
    "FP": ["encoding_noise", "header_in_data"],  # + format-level corruptions
    "VR": [
        "type_mangle", "decimal_shift", "value_swap", "typo_injection",
        "unicode_homoglyph", "html_entity_leak", "leading_zero_strip",
    ],
    "MD": ["inject_nulls", "drop_rows"],
    "SR": ["duplicate_rows", "column_shift", "schema_drift", "header_in_data"],
    "SV": ["business_rule_violation", "unit_inconsistency", "outlier_injection"],
    "CP": None,
}

# File format pool per category.
CATEGORY_FORMAT_MAP: dict[str, list[str]] = {
    "FP": ["json", "jsonl", "excel", "xml", "html_table", "fixed_width", "sql_dump", "yaml"],
    "VR": ["csv"],
    "MD": ["csv"],
    "SR": ["csv", "tsv"],
    "SV": ["csv"],
    "CP": ["json", "excel", "xml", "html_table"],
}


def get_corruptions_for_category(category: str) -> list[str] | None:
    """Return the corruption type names for a category, or None for CP (all)."""
    if category not in CATEGORY_CORRUPTION_MAP:
        raise ValueError(f"Unknown category: {category!r}. Must be one of {CATEGORIES}")
    return CATEGORY_CORRUPTION_MAP[category]


def get_formats_for_category(category: str) -> list[str]:
    """Return the format pool for a category."""
    if category not in CATEGORY_FORMAT_MAP:
        raise ValueError(f"Unknown category: {category!r}. Must be one of {CATEGORIES}")
    return CATEGORY_FORMAT_MAP[category]
