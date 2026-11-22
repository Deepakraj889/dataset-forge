"""Console hardening.

Windows consoles still default to a legacy code page (cp437/cp1252), which cannot
encode the glyphs this tool prints (``✓`` U+2713, ``⚡`` U+26A1). Rich writes them
through ``sys.stdout``, so the whole CLI dies with ``UnicodeEncodeError`` before it
does any work.

Importing this module switches the Windows console to UTF-8 and forces stdout and
stderr into UTF-8 with ``errors="replace"`` as a hard backstop, so a glyph can never
abort a run again.
"""

from __future__ import annotations

import sys


def _switch_console_to_utf8() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:  # no console attached (pythonw, redirected, service)
        pass


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            if encoding in {"utf8", "cp65001"}:
                continue
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # stream without reconfigure, or already-detached pipe
            pass


_switch_console_to_utf8()
_force_utf8_streams()
