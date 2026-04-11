"""Runtime corruption engine for dynamic task generation."""
from .categories import (
    CATEGORIES,
    CATEGORY_CORRUPTION_MAP,
    CATEGORY_FORMAT_MAP,
    get_corruptions_for_category,
    get_formats_for_category,
)
from .pipeline import CorruptionPipeline
from .profiles import DIFFICULTY_PROFILES, DIFFICULTY_WEIGHTS
from .value_corruptions import CORRUPTION_REGISTRY, CORRUPTION_SEVERITY

__all__ = [
    "CATEGORIES",
    "CATEGORY_CORRUPTION_MAP",
    "CATEGORY_FORMAT_MAP",
    "CorruptionPipeline",
    "DIFFICULTY_PROFILES",
    "DIFFICULTY_WEIGHTS",
    "CORRUPTION_REGISTRY",
    "CORRUPTION_SEVERITY",
    "get_corruptions_for_category",
    "get_formats_for_category",
]
