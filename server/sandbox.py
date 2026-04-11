"""
Sandboxed code execution for agent transform/explore actions.

Uses a persistent worker process per episode so pandas/numpy are loaded
once and kept in memory — eliminates per-step import cold-start overhead.

Safety: agent code is AST-scanned before being sent to the worker.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import select
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("sandbox")

# ── Safety ───────────────────────────────────────────────────────────────────

BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "importlib",
    "ctypes", "signal", "multiprocessing", "threading",
    "pickle", "shelve", "tempfile", "glob",
    "webbrowser", "code", "codeop",
})

BLOCKED_NAMES = frozenset({
    "exec", "eval", "compile", "__import__",
    "open", "breakpoint", "input",
})


class UnsafeCodeError(Exception):
    pass


def check_code_safety(code: str) -> None:
    """AST-scan agent code for blocked imports and calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    raise UnsafeCodeError(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in BLOCKED_MODULES:
                    raise UnsafeCodeError(f"Blocked import: from {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked call: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked call: .{node.func.attr}()")


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


# ── Worker management ─────────────────────────────────────────────────────────

_WORKER_SCRIPT = str(Path(__file__).parent / "worker.py")


def _send(proc: subprocess.Popen, obj: dict) -> bool:
    """Write a JSON line to the worker. Returns False if pipe is broken."""
    try:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()
        return True
    except (BrokenPipeError, OSError):
        logger.warning("Worker pipe broken — process may have died")
        return False


def _recv(proc: subprocess.Popen, timeout: float) -> Optional[dict]:
    """Read one JSON line from the worker with a timeout (Unix select)."""
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        return None
    line = proc.stdout.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Malformed worker response: %s", line[:200])
        return None


def create_sandbox(
    episode_id: str,
    dirty_csv_path: str,
    base_dir: str = "outputs/sandbox",
    dirty_content: str | bytes = "",
    file_format: str = "csv",
) -> tuple[str, subprocess.Popen]:
    """Create an isolated sandbox and spawn the persistent worker.

    Returns (sandbox_dir, worker_proc).
    """
    sandbox_dir = os.path.abspath(os.path.join(base_dir, episode_id))
    artifacts_dir = os.path.join(sandbox_dir, "artifacts")
    scripts_dir = os.path.join(sandbox_dir, "scripts")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    current_csv = os.path.join(sandbox_dir, "current.csv")
    shutil.copy2(dirty_csv_path, os.path.join(sandbox_dir, "input.csv"))
    shutil.copy2(dirty_csv_path, current_csv)

    if dirty_content:
        ext_map = {
            "csv": "csv",
            "json": "json",
            "jsonl": "jsonl",
            "excel": "xlsx",
            "tsv": "tsv",
            "xml": "xml",
            "fixed_width": "txt",
            "html_table": "html",
            "sql_dump": "sql",
            "yaml": "yaml",
        }
        ext = ext_map.get(file_format, "csv")
        raw_path = os.path.join(sandbox_dir, f"input.{ext}")
        mode = "wb" if isinstance(dirty_content, bytes) else "w"
        with open(raw_path, mode) as f:
            f.write(dirty_content)

    # Spawn persistent worker
    worker_proc = subprocess.Popen(
        [sys.executable, _WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=sandbox_dir,
    )

    # Send setup
    _send(worker_proc, {
        "current_csv": current_csv,
        "artifacts_dir": artifacts_dir,
        "scripts_dir": scripts_dir,
    })

    # Wait for ready signal (10s timeout)
    ready = _recv(worker_proc, timeout=10.0)
    if not ready or not ready.get("ready"):
        terminate_worker(worker_proc)
        raise RuntimeError(f"Worker failed to start for episode {episode_id}")

    return sandbox_dir, worker_proc


def save_checkpoint(sandbox_dir: str, step: int) -> str:
    """Copy current.csv to checkpoints/step_NNN.csv. Returns checkpoint path."""
    ckpt_dir = os.path.join(sandbox_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    src = os.path.join(sandbox_dir, "current.csv")
    dst = os.path.join(ckpt_dir, f"step_{step:03d}.csv")
    shutil.copy2(src, dst)
    return dst


def restore_checkpoint(sandbox_dir: str, step: int) -> bool:
    """Restore current.csv from checkpoints/step_NNN.csv. Returns True if found."""
    ckpt_dir = os.path.join(sandbox_dir, "checkpoints")
    src = os.path.join(ckpt_dir, f"step_{step:03d}.csv")
    if not os.path.exists(src):
        return False
    dst = os.path.join(sandbox_dir, "current.csv")
    shutil.copy2(src, dst)
    return True


def terminate_worker(worker_proc: subprocess.Popen) -> None:
    """Gracefully shut down the worker process and close all pipes."""
    if worker_proc is None:
        return
    try:
        _send(worker_proc, {"type": "exit"})
        worker_proc.wait(timeout=3)
    except Exception:
        pass
    # Close pipes explicitly
    for pipe in (worker_proc.stdin, worker_proc.stdout, worker_proc.stderr):
        try:
            if pipe and not pipe.closed:
                pipe.close()
        except Exception:
            pass
    # Force kill if still alive
    try:
        worker_proc.kill()
        worker_proc.wait(timeout=1)
    except Exception:
        pass


# ── Execution ─────────────────────────────────────────────────────────────────

def execute_transform(
    code: str,
    worker_proc: subprocess.Popen,
    step_idx: int,
    timeout: int = 30,
) -> ExecutionResult:
    """Send a transform to the persistent worker and return the result."""
    try:
        check_code_safety(code)
    except UnsafeCodeError as e:
        return ExecutionResult(success=False, error=str(e))

    if not _send(worker_proc, {"type": "transform", "code": code, "step": step_idx}):
        return ExecutionResult(success=False, error="Worker process is not running")

    result = _recv(worker_proc, timeout=float(timeout))
    if result is None:
        return ExecutionResult(success=False, error=f"Worker timed out after {timeout}s")

    return ExecutionResult(
        success=result.get("success", False),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        error=result.get("error") or None,
    )


def execute_explore(
    query: str,
    worker_proc: subprocess.Popen,
    step_idx: int,
    timeout: int = 10,
) -> ExecutionResult:
    """Send an explore query to the persistent worker and return the result."""
    if not _send(worker_proc, {"type": "explore", "query": query, "step": step_idx}):
        return ExecutionResult(success=False, error="Worker process is not running")

    result = _recv(worker_proc, timeout=float(timeout))
    if result is None:
        return ExecutionResult(success=False, error=f"Worker timed out after {timeout}s")

    return ExecutionResult(
        success=result.get("success", False),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        error=result.get("error") or None,
    )
