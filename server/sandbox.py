"""
Sandboxed code execution for agent transform/explore actions.

Uses a persistent worker process per episode so pandas/numpy are loaded
once and kept in memory — eliminates per-step import cold-start overhead.

Safety: agent code is AST-scanned before being sent to the worker.
"""

from __future__ import annotations

import ast
import atexit
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
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
})

# Dunder attributes enabling sandbox escape via class hierarchy traversal
BLOCKED_DUNDERS = frozenset({
    "__import__", "__class__", "__bases__", "__subclasses__",
    "__mro__", "__builtins__", "__globals__", "__code__",
    "__reduce__", "__reduce_ex__", "__init_subclass__",
    "__getattribute__", "__getattr__", "__setattr__", "__delattr__",
})


class UnsafeCodeError(Exception):
    pass


def check_code_safety(code: str, mode: str = "exec") -> None:
    """AST-scan agent code for blocked imports and calls.

    mode: "exec" for transform statements, "eval" for explore expressions.
    """
    try:
        tree = ast.parse(code, mode=mode)
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
        elif isinstance(node, ast.Attribute):
            # Block dunder attributes that enable class hierarchy traversal / escape
            if node.attr in BLOCKED_DUNDERS or node.attr in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked attribute: .{node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked call: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Blocked call: .{node.func.attr}()")
        elif isinstance(node, ast.Name):
            # Block bare references to blocked dunders as identifiers
            if node.id in BLOCKED_DUNDERS:
                raise UnsafeCodeError(f"Blocked name: {node.id}")


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


# ── Worker management ─────────────────────────────────────────────────────────

_WORKER_SCRIPT = str(Path(__file__).parent / "worker.py")

# Environment variables forwarded to the worker — strips secrets like HF_TOKEN / API keys
_WORKER_ENV_KEYS = frozenset({"PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV"})

# Track active workers for atexit cleanup
_active_workers: set[subprocess.Popen] = set()


def _worker_env(sandbox_dir: str) -> dict[str, str]:
    """Build a minimal env dict for the worker, stripping server secrets."""
    env = {k: v for k, v in os.environ.items() if k in _WORKER_ENV_KEYS}
    env["HOME"] = sandbox_dir
    return env


def _cleanup_all_workers() -> None:
    """Kill any still-running worker processes on interpreter exit."""
    for proc in list(_active_workers):
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass
    _active_workers.clear()


atexit.register(_cleanup_all_workers)


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

    # Spawn persistent worker — stripped env (no secrets), own session for clean kills
    worker_proc = subprocess.Popen(
        [sys.executable, _WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=sandbox_dir,
        env=_worker_env(sandbox_dir),
        start_new_session=True,
    )
    _active_workers.add(worker_proc)

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
    _active_workers.discard(worker_proc)
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


def reload_worker_df(worker_proc: subprocess.Popen, timeout: float = 5.0) -> bool:
    """Tell the worker to re-read current.csv from disk (called after undo).

    Returns True on success, False if worker is dead or times out.
    """
    if not _send(worker_proc, {"type": "reload"}):
        return False
    result = _recv(worker_proc, timeout=timeout)
    return bool(result and result.get("ready"))


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
        # Kill stuck worker — caller must respawn if needed
        try:
            worker_proc.kill()
            worker_proc.wait(timeout=2)
        except Exception:
            pass
        _active_workers.discard(worker_proc)
        return ExecutionResult(success=False, error=f"Worker killed after {timeout}s timeout")

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
    try:
        check_code_safety(query, mode="eval")
    except UnsafeCodeError as e:
        return ExecutionResult(success=False, error=str(e))

    if not _send(worker_proc, {"type": "explore", "query": query, "step": step_idx}):
        return ExecutionResult(success=False, error="Worker process is not running")

    result = _recv(worker_proc, timeout=float(timeout))
    if result is None:
        try:
            worker_proc.kill()
            worker_proc.wait(timeout=2)
        except Exception:
            pass
        _active_workers.discard(worker_proc)
        return ExecutionResult(success=False, error=f"Worker killed after {timeout}s timeout")

    return ExecutionResult(
        success=result.get("success", False),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        error=result.get("error") or None,
    )
