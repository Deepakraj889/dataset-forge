"""Regression tests for console encoding.

History: the whole CLI crashed on Windows with
``UnicodeEncodeError: 'charmap' codec can't encode character '\\u2713'``
because Rich writes ``✓`` / ``⚡`` straight to a cp1252 console. These tests pin
the fix so the failure can never come back unnoticed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cli(env_extra: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dataset_forge.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ, **env_extra},
        timeout=120,
    )


@pytest.mark.parametrize("encoding", ["cp1252", "cp437", "ascii"])
def test_help_survives_legacy_console_encoding(encoding: str) -> None:
    """--help must not die on a console that cannot encode ✓ or ⚡."""
    result = _run_cli({"PYTHONIOENCODING": encoding}, "--help")
    assert result.returncode == 0, f"--help crashed under {encoding}:\n{result.stderr[-2000:]}"
    assert "UnicodeEncodeError" not in result.stderr


def test_console_module_forces_utf8() -> None:
    """Importing the package must leave stdout UTF-8 capable."""
    import dataset_forge  # noqa: F401  (import triggers _console hardening)

    assert (sys.stdout.encoding or "").lower().replace("-", "") in {"utf8", "cp65001"}
    assert "✓⚡".encode(sys.stdout.encoding)  # no UnicodeEncodeError
