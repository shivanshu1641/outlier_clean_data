"""Theme and shared constants for the benchmark dashboard."""
from __future__ import annotations

CATEGORY_META = {
    "FP": {
        "name": "Format parsing",
        "desc": "Can the agent read the mess? File arrives as JSON/Excel/XML with format-level corruptions. Tests library knowledge and schema extraction.",
        "color": "#0d9488",
        "tasks_hint": "~10 tasks across 5 formats",
    },
    "VR": {
        "name": "Value repair",
        "desc": "Can the agent fix corrupted cell values? Types mangled, decimals shifted, outliers injected, homoglyphs swapped.",
        "color": "#2563eb",
        "tasks_hint": "~12 tasks, CSV format only",
    },
    "MD": {
        "name": "Missing data",
        "desc": "Can the agent fill gaps intelligently? Nulls injected, rows dropped.",
        "color": "#7c3aed",
        "tasks_hint": "~8 tasks, varies by difficulty",
    },
    "SR": {
        "name": "Structural repair",
        "desc": "Can the agent fix the shape? Duplicate rows, column shifts, schema drift, headers repeated as data.",
        "color": "#d97706",
        "tasks_hint": "~8 tasks",
    },
    "SV": {
        "name": "Semantic validation",
        "desc": "Does the agent understand the data? Business rule violations, impossible values, unit inconsistencies.",
        "color": "#059669",
        "tasks_hint": "~6 tasks",
    },
    "CP": {
        "name": "Compound",
        "desc": "Can the agent handle everything at once? 7+ corruption types, non-CSV format, overlapping errors.",
        "color": "#dc2626",
        "tasks_hint": "~6 tasks, hard only",
    },
}

MODEL_COLORS = [
    "#0d9488", "#2563eb", "#dc2626", "#d97706", "#7c3aed",
    "#059669", "#db2777", "#7c3aed", "#0891b2", "#65a30d",
]

CUSTOM_CSS = """
/* ── Interactive affordances — make dropdowns obviously clickable ── */
/* Target all input wrappers inside dropdown containers */
input[role="listbox"],
input[aria-expanded],
div[data-testid] input,
.border, .wrap, .secondary-wrap,
div:has(> input[role="listbox"]),
div:has(> select) {
    /* Don't override everything — just dropdowns */
}
/* Use elem_classes on the Gradio components instead — see Python code */

/* Teal-accented interactive elements via .interactive-dropdown class */
.interactive-dropdown {
    border: 2px solid #0d9488 !important;
    border-radius: 8px !important;
    background: #f0fdfa !important;
    padding: 2px !important;
    transition: box-shadow 0.15s !important;
}
.interactive-dropdown:hover {
    box-shadow: 0 0 0 3px #0d948822 !important;
}
.interactive-dropdown input,
.interactive-dropdown select {
    background: #f0fdfa !important;
}

/* ── Category cards ────────────────────────────────────────────── */
.category-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    font-weight: 700;
    font-size: 16px;
    color: #fff;
    margin-bottom: 14px;
}

/* ── Task badges ───────────────────────────────────────────────── */
.task-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
}
.task-badge.easy { background: #059669; color: #fff; }
.task-badge.medium { background: #d97706; color: #fff; }
.task-badge.hard { background: #dc2626; color: #fff; }

/* ── Step cards (episode explorer) ─────────────────────────────── */
.step-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 10px;
}

/* ── Category radio — pill-style buttons ───────────────────────── */
.category-radio .wrap {
    gap: 8px !important;
    flex-wrap: wrap !important;
}
.category-radio label:has(input[type="radio"]),
.category-radio .radio-item {
    border: 2px solid #e2e8f0 !important;
    border-radius: 24px !important;
    padding: 8px 20px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    background: #fff !important;
    color: #475569 !important;
}
.category-radio label:has(input[type="radio"]):hover,
.category-radio .radio-item:hover {
    border-color: #0d9488 !important;
    background: #f0fdfa !important;
    color: #0d9488 !important;
}
.category-radio label:has(input[type="radio"]:checked),
.category-radio .radio-item.selected {
    border-color: #0d9488 !important;
    background: #0d9488 !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* ── Tabs — active tab gets colored underline ──────────────────── */
.tabs > .tab-nav > button.selected {
    color: #0d9488 !important;
    border-bottom: 2px solid #0d9488 !important;
}
"""
