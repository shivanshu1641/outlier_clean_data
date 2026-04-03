"""
Persistent sandbox worker process.

Spawned once per episode. Keeps pandas/numpy loaded in memory across all
transform/explore steps — eliminates per-step import cold-start overhead.

Protocol (stdin/stdout, newline-delimited JSON):
  Parent → Worker:
    First line: {"current_csv": "...", "artifacts_dir": "...", "scripts_dir": "..."}
    Each step:  {"type": "transform"|"explore", "code"|"query": "...", "step": N}
    Shutdown:   {"type": "exit"}

  Worker → Parent:
    Each step:  {"success": true|false, "stdout": "...", "stderr": "...", "error": "..."}
"""

from __future__ import annotations

import csv
import collections
import datetime
import functools
import io
import itertools
import json
import math
import os.path
import re
import string
import sys
import traceback

import numpy as np
import pandas as pd


def _fix_inplace_pattern(code: str) -> str:
    """Rewrite df['col'].method(val, inplace=True) → df['col'] = df['col'].method(val).

    Pandas 2.x Copy-on-Write makes chained inplace operations a no-op.
    This silently fixes the most common LLM mistake.
    """
    import re
    # Match: df['col'].fillna(..., inplace=True)  and similar methods
    # Captures: df['col'] as group 1, method call as group 2
    pattern = r"(df\[(['\"])([\w\s]+)\2\])\.(fillna|replace|drop_duplicates|dropna|ffill|bfill|interpolate|clip|where|mask)\((.+?),\s*inplace\s*=\s*True\s*\)"
    def _rewrite(m):
        accessor = m.group(1)       # df['col']
        method = m.group(4)         # fillna
        args = m.group(5)           # the arguments before inplace=True
        return f"{accessor} = {accessor}.{method}({args})"
    fixed = re.sub(pattern, _rewrite, code)
    # Also handle: df.dropna(inplace=True), df.drop_duplicates(inplace=True), etc.
    df_pattern = r"(df)\.(fillna|replace|drop_duplicates|dropna|ffill|bfill|interpolate|clip|rename|reset_index|sort_values|sort_index|set_index|drop)\((.+?),\s*inplace\s*=\s*True\s*\)"
    def _rewrite_df(m):
        obj = m.group(1)
        method = m.group(2)
        args = m.group(3)
        return f"{obj} = {obj}.{method}({args})"
    fixed = re.sub(df_pattern, _rewrite_df, fixed)
    # Handle case where inplace=True is the only argument: df.dropna(inplace=True)
    solo_col_pattern = r"(df\[(['\"])([\w\s]+)\2\])\.(fillna|replace|dropna|ffill|bfill|interpolate)\(inplace\s*=\s*True\s*\)"
    fixed = re.sub(solo_col_pattern, r"\1 = \1.\4()", fixed)
    solo_df_pattern = r"(df)\.(drop_duplicates|dropna|ffill|bfill|reset_index|sort_values|sort_index)\(inplace\s*=\s*True\s*\)"
    fixed = re.sub(solo_df_pattern, r"\1 = \1.\2()", fixed)
    return fixed


def _run():
    # ── Setup ────────────────────────────────────────────────────────────────
    setup_line = sys.stdin.readline()
    if not setup_line:
        sys.exit(1)
    setup = json.loads(setup_line)

    current_csv: str = setup["current_csv"]
    artifacts_dir: str = setup["artifacts_dir"]
    scripts_dir: str = setup["scripts_dir"]

    df = pd.read_csv(current_csv)

    # Namespace available to agent code
    _ns_base = {
        "pd": pd, "np": np,
        "re": re, "datetime": datetime, "string": string,
        "math": math, "collections": collections,
        "itertools": itertools, "functools": functools,
        "json": json, "csv": csv,
    }

    # Signal ready
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    # ── Command loop ─────────────────────────────────────────────────────────
    while True:
        line = sys.stdin.readline()
        if not line:
            break

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue

        cmd_type = cmd.get("type")
        step = cmd.get("step", 0)

        if cmd_type == "exit":
            break

        elif cmd_type == "transform":
            code = _fix_inplace_pattern(cmd.get("code", ""))
            script_path = os.path.join(scripts_dir, f"transform_{step:03d}.py")
            snapshot_path = os.path.join(artifacts_dir, f"transform_{step:03d}.csv")

            # Save script for replay/debug
            try:
                with open(script_path, "w") as f:
                    f.write(code)
            except Exception:
                pass

            buf = io.StringIO()
            old_stdout = sys.stdout
            success = False
            stdout_out = ""
            stderr_out = ""
            error_msg = ""

            try:
                sys.stdout = buf
                # Re-read CSV each step — CSV is the source of truth, not in-memory df.
                # This avoids pandas inplace=True pitfalls.
                df = pd.read_csv(current_csv)
                ns = {"df": df, **_ns_base}
                exec(code, ns)  # noqa: S102 — AST-checked by parent before sending
                sys.stdout = old_stdout
                df = ns["df"]
                df.to_csv(current_csv, index=False)
                df.to_csv(snapshot_path, index=False)
                stdout_out = buf.getvalue()
                success = True
            except Exception:
                sys.stdout = old_stdout
                stderr_out = traceback.format_exc()
                error_msg = traceback.format_exc().strip().splitlines()[-1]

            sys.stdout.write(json.dumps({
                "success": success,
                "stdout": stdout_out,
                "stderr": stderr_out,
                "error": error_msg,
            }) + "\n")
            sys.stdout.flush()

        elif cmd_type == "explore":
            query = cmd.get("query", "df.head()")
            artifact_path = os.path.join(artifacts_dir, f"explore_{step:03d}.txt")

            buf = io.StringIO()
            old_stdout = sys.stdout
            success = False
            stdout_out = ""
            error_msg = ""

            try:
                sys.stdout = buf
                df = pd.read_csv(current_csv)
                ns = {"df": df, **_ns_base}
                result = eval(query, ns)  # noqa: S307
                if result is not None:
                    print(result)
                sys.stdout = old_stdout
                stdout_out = buf.getvalue()
                success = True

                try:
                    with open(artifact_path, "w") as f:
                        f.write(stdout_out)
                except Exception:
                    pass
            except Exception:
                sys.stdout = old_stdout
                error_msg = traceback.format_exc().strip().splitlines()[-1]

            sys.stdout.write(json.dumps({
                "success": success,
                "stdout": stdout_out,
                "stderr": "",
                "error": error_msg,
            }) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    _run()
