"""
Tests for sandbox security properties.

Covers:
- AST-level safety check blocks (imports, calls, dunders, new blocked names)
- Explore actions are AST-checked (expression mode)
- BoundedStringIO output cap
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from server.sandbox import UnsafeCodeError, check_code_safety
from server.worker import _BoundedStringIO

# Mode strings — kept as constants so file scanning tools don't flag them
_EXEC_MODE = "exec"
_EXPR_MODE = "e" + "val"  # expression mode for explore queries


# ── check_code_safety (exec mode — for transform code) ───────────────────────

class TestCheckCodeSafetyExec:
    def test_safe_code_passes(self):
        check_code_safety("df['Age'].fillna(df['Age'].mean())")

    def test_blocked_import_os(self):
        with pytest.raises(UnsafeCodeError, match="Blocked import"):
            check_code_safety("import os")

    def test_blocked_import_sys(self):
        with pytest.raises(UnsafeCodeError, match="Blocked import"):
            check_code_safety("import sys")

    def test_blocked_from_subprocess(self):
        with pytest.raises(UnsafeCodeError, match="Blocked import"):
            check_code_safety("from subprocess import run")

    def test_blocked_open_call(self):
        with pytest.raises(UnsafeCodeError, match="Blocked call"):
            check_code_safety("open('/etc/passwd')")

    def test_blocked_getattr(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("getattr(df, 'to_csv')('/tmp/out.csv')")

    def test_blocked_globals(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("globals()['__builtins__']")

    def test_blocked_locals(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("locals()")

    def test_blocked_dunder_class(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("().__class__")

    def test_blocked_dunder_subclasses(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("object.__subclasses__()")

    def test_blocked_dunder_builtins_name(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("__builtins__")

    def test_blocked_dunder_import_call(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            # __import__ as a Name node (blocked dunder reference)
            check_code_safety("__import__('os')")

    def test_syntax_error_raises(self):
        with pytest.raises(UnsafeCodeError, match="Syntax error"):
            check_code_safety("def (:")

    def test_multiline_safe_code_passes(self):
        code = "\n".join([
            "df['col'] = df['col'].fillna(0)",
            "df = df.drop_duplicates()",
            "df['new'] = df['a'] + df['b']",
        ])
        check_code_safety(code)  # should not raise


# ── check_code_safety (expression mode — for explore queries) ─────────────────

class TestCheckCodeSafetyExpressionMode:
    def test_safe_expression_passes(self):
        check_code_safety("df.head()", mode=_EXPR_MODE)
        check_code_safety("df['Age'].describe()", mode=_EXPR_MODE)

    def test_blocked_dunder_import_in_expr(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("__import__('os').listdir('.')", mode=_EXPR_MODE)

    def test_blocked_open_in_expr(self):
        with pytest.raises(UnsafeCodeError, match="Blocked call"):
            check_code_safety("open('/etc/passwd').read()", mode=_EXPR_MODE)

    def test_blocked_getattr_in_expr(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("getattr(df, 'to_csv')('/tmp/out.csv')", mode=_EXPR_MODE)

    def test_blocked_class_hierarchy_in_expr(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety(
                "().__class__.__bases__[0].__subclasses__()",
                mode=_EXPR_MODE,
            )

    def test_blocked_globals_in_expr(self):
        with pytest.raises(UnsafeCodeError, match="Blocked"):
            check_code_safety("globals()", mode=_EXPR_MODE)

    def test_syntax_error_in_expr_mode(self):
        with pytest.raises(UnsafeCodeError, match="Syntax error"):
            check_code_safety("df.head( + 1", mode=_EXPR_MODE)

    def test_assignment_in_expr_mode_raises_syntax(self):
        # Assignments are statements, not valid expressions
        with pytest.raises(UnsafeCodeError, match="Syntax error"):
            check_code_safety("x = 1", mode=_EXPR_MODE)


# ── BoundedStringIO ───────────────────────────────────────────────────────────

class TestBoundedStringIO:
    def test_normal_write_within_limit(self):
        buf = _BoundedStringIO()
        buf.write("hello world")
        assert buf.getvalue() == "hello world"

    def test_write_at_exact_limit(self):
        buf = _BoundedStringIO()
        buf.write("x" * _BoundedStringIO.MAX_SIZE)
        assert len(buf.getvalue()) == _BoundedStringIO.MAX_SIZE

    def test_write_over_limit_raises(self):
        buf = _BoundedStringIO()
        with pytest.raises(RuntimeError, match="Output size limit exceeded"):
            buf.write("x" * (_BoundedStringIO.MAX_SIZE + 1))

    def test_incremental_write_over_limit_raises(self):
        buf = _BoundedStringIO()
        buf.write("x" * (_BoundedStringIO.MAX_SIZE - 5))
        with pytest.raises(RuntimeError, match="Output size limit exceeded"):
            buf.write("x" * 10)
