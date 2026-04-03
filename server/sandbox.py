"""
Sandboxed code execution for agent transform actions.

Runs agent-submitted Python code in a subprocess with:
- AST-level import/call blocking
- Timeout (30s default)
- Memory limits via resource module
- Isolated working directory per episode
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Safety ───────────────────────────────────────────────────────────────────

ALLOWED_IMPORTS = frozenset({
    "pandas", "pd",
    "numpy", "np",
    "re",
    "datetime",
    "string",
    "math",
    "collections",
    "itertools",
    "functools",
    "json",
    "csv",
    "os.path",
})

BLOCKED_NAMES = frozenset({
    "exec", "eval", "compile", "__import__",
    "open",  # blocked in agent code — preamble handles file I/O
    "breakpoint", "input",
})

BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "importlib",
    "ctypes", "signal", "multiprocessing", "threading",
    "pickle", "shelve", "tempfile", "glob",
    "webbrowser", "code", "codeop",
})


class UnsafeCodeError(Exception):
    """Raised when agent code contains blocked patterns."""
    pass


def check_code_safety(code: str) -> None:
    """AST-scan code for blocked imports, calls, and attributes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error in agent code: {e}")

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in BLOCKED_MODULES:
                    raise UnsafeCodeError(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]
                if module_root in BLOCKED_MODULES:
                    raise UnsafeCodeError(f"Blocked import: from {node.module}")

        # Block dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked call: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked call: .{node.func.attr}()")


# ── Sandbox Directory Management ─────────────────────────────────────────────


def create_sandbox(episode_id: str, dirty_csv_path: str, base_dir: str = "outputs/sandbox") -> str:
    """Create an isolated sandbox directory for an episode."""
    sandbox_dir = os.path.abspath(os.path.join(base_dir, episode_id))
    os.makedirs(os.path.join(sandbox_dir, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(sandbox_dir, "scripts"), exist_ok=True)

    # Copy dirty data as input and current working copy
    shutil.copy2(dirty_csv_path, os.path.join(sandbox_dir, "input.csv"))
    shutil.copy2(dirty_csv_path, os.path.join(sandbox_dir, "current.csv"))

    return sandbox_dir


# ── Execution ────────────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


def execute_transform(
    code: str,
    sandbox_dir: str,
    step_idx: int,
    timeout: int = 30,
) -> ExecutionResult:
    """Execute agent transform code in a sandboxed subprocess."""
    # Safety check first
    try:
        check_code_safety(code)
    except UnsafeCodeError as e:
        return ExecutionResult(success=False, error=str(e))

    current_csv = os.path.join(sandbox_dir, "current.csv")
    snapshot_csv = os.path.join(sandbox_dir, "artifacts", f"transform_{step_idx:03d}.csv")
    script_path = os.path.join(sandbox_dir, "scripts", f"transform_{step_idx:03d}.py")

    # Build the full script with preamble + agent code + postamble
    full_script = f"""\
import resource
# Memory limit: 2GB
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except (ValueError, resource.error):
    pass  # May not be supported on all platforms

import pandas as pd
import numpy as np
import re
import datetime
import string
import math
import collections
import itertools
import functools
import json
import csv
import os.path

SANDBOX_DIR = {sandbox_dir!r}
ARTIFACTS_DIR = os.path.join(SANDBOX_DIR, "artifacts")
CURRENT_CSV = {current_csv!r}

# Load current state
df = pd.read_csv(CURRENT_CSV)

# ── Agent Code ──
{code}
# ── End Agent Code ──

# Save updated state
df.to_csv(CURRENT_CSV, index=False)
# Save snapshot
df.to_csv({snapshot_csv!r}, index=False)
print("TRANSFORM_SUCCESS")
"""

    # Save script for debugging/replay
    with open(script_path, "w") as f:
        f.write(full_script)

    # Execute in subprocess — use sys.executable to ensure same Python/venv
    import sys as _sys
    try:
        result = subprocess.run(
            [_sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=sandbox_dir,
        )

        if result.returncode == 0 and "TRANSFORM_SUCCESS" in result.stdout:
            return ExecutionResult(
                success=True,
                stdout=result.stdout.replace("TRANSFORM_SUCCESS", "").strip(),
                stderr=result.stderr.strip(),
            )
        else:
            return ExecutionResult(
                success=False,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                error=result.stderr.strip() or f"Exit code: {result.returncode}",
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            error=f"Code execution timed out after {timeout}s",
        )
    except Exception as e:
        return ExecutionResult(success=False, error=str(e))


def execute_explore(
    query: str,
    sandbox_dir: str,
    step_idx: int,
    timeout: int = 10,
) -> ExecutionResult:
    """Execute a read-only explore query against the current data."""
    current_csv = os.path.join(sandbox_dir, "current.csv")
    artifact_path = os.path.join(sandbox_dir, "artifacts", f"explore_{step_idx:03d}.txt")

    # Build explore script — read-only, captures output
    full_script = f"""\
import pandas as pd
import numpy as np
import io
import sys

df = pd.read_csv({current_csv!r})

# Capture output
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()

try:
    result = {query!r}
    # Try to eval the query as a pandas expression
    try:
        output = eval(result, {{"df": df, "pd": pd, "np": np}})
        if output is not None:
            print(output)
    except Exception:
        # If eval fails, just print the query string
        print(result)
except Exception as e:
    print(f"Error: {{e}}")

sys.stdout = old_stdout
output_text = buffer.getvalue()

# Save to artifact
with open({artifact_path!r}, "w") as f:
    f.write(output_text)

print(output_text)
"""

    import sys as _sys
    try:
        result = subprocess.run(
            [_sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=sandbox_dir,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output = result.stderr.strip() or f"Explore failed with exit code {result.returncode}"
            return ExecutionResult(success=False, error=output)
        return ExecutionResult(success=True, stdout=output)

    except subprocess.TimeoutExpired:
        return ExecutionResult(success=False, error=f"Explore timed out after {timeout}s")
    except Exception as e:
        return ExecutionResult(success=False, error=str(e))
