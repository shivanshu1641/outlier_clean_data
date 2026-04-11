# ADR 001: Diff-based grading over constraint checkers

## Status
Accepted (April 2026)

## Context
The original grader had 10 hardcoded constraint checker functions (`check_no_nulls`, `check_dtype`, `check_value_range`, etc.). Each task config had to enumerate constraints by type, and adding a new kind of data quality issue required writing a new checker.

This created two problems:
1. The grader had domain knowledge baked in — it knew about null checks, regex patterns, row counts, etc.
2. Constraint satisfaction is binary — either the whole column passes or fails — making it harder to give partial credit for partial fixes.

## Decision
Replace the constraint checker system with a generic diff engine and multi-level score:
- **Pipeline** tracks every cell mutation during corruption and returns `error_map`
- **Grader** compares result DataFrame against clean DataFrame using the error map
- Score weights: schema 15%, row 20%, cell 55%, distribution 10%
- Rows are matched by content, not only by positional index, so reordered rows can still earn credit

## Error map schema
```json
{
  "cell_errors": {
    "ROW,COL": {"severity": 2.0, "clean_value": "S", "corruption": "null_injected"}
  },
  "spurious_rows": {"42": {"severity": 2.0}},
  "missing_rows": {}
}
```

## Wrong value penalty
A cell changed to the wrong value is penalized at 1.5× severity — worse than leaving it dirty. Rationale: an agent that corrupts values further is actively harmful, not just ineffective.

## Consequences
- Adding new corruption types requires no grader changes — only engine.py changes
- Grading is granular across schema, row, cell, and distribution quality
- Wrong value detection catches agents that hallucinate values
- Content-based row matching avoids penalizing harmless row reordering
