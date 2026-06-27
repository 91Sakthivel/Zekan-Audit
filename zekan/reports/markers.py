"""Terminal Unicode marker helper.

Provides ASCII-safe, capability-detecting markers for terminal output.
HTML output paths write UTF-8 files directly and do not use this module.

Usage::

    from zekan.reports.markers import marker, supports_unicode
    m = marker("failed", stream=sys.stdout)  # "✗ FAILED" or "[FAIL] FAILED"
"""

from __future__ import annotations

import sys
from typing import Optional

# Probe string: all three distinct glyphs used across the four states.
_PROBE = "✓⚠✗"

_GLYPHS: dict[str, str] = {
    "trusted":      "✓ TRUSTED",
    "risky":        "⚠ RISKY",
    "failed":       "✗ FAILED",
    "inconclusive": "⚠ INCONCLUSIVE",
}

_ASCII: dict[str, str] = {
    "trusted":      "[PASS] TRUSTED",
    "risky":        "[RISK] RISKY",
    "failed":       "[FAIL] FAILED",
    "inconclusive": "[????] INCONCLUSIVE",
}


def supports_unicode(stream) -> bool:
    """True if stream can encode all marker glyphs.

    Checks stream.encoding then test-encodes the probe string.
    Returns False for: None stream, missing/None/empty encoding attribute,
    LookupError (unknown codec), or any UnicodeEncodeError.
    """
    if stream is None:
        return False
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        _PROBE.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_ASCII_MAP = {"—": "-", "−": "-", "─": "-"}
_ASCII_TRANS = str.maketrans(_ASCII_MAP)


def sanitize(text: str, stream=None) -> str:
    """Replace body-text Unicode chars with ASCII equivalents when stream cannot encode them.

    Reuses supports_unicode() — same detection path as marker().
    Idempotent on already-ASCII text.  stream=None defaults to sys.stdout.
    """
    if stream is None:
        stream = sys.stdout
    if supports_unicode(stream):
        return text
    return text.translate(_ASCII_TRANS)


def marker(kind: str, stream=None) -> str:
    """Return the glyph marker or its ASCII fallback for the given verdict kind.

    kind: one of "trusted", "risky", "failed", "inconclusive"
    stream: stream whose encoding is checked; defaults to sys.stdout when None.
    """
    if stream is None:
        stream = sys.stdout
    if supports_unicode(stream):
        return _GLYPHS[kind]
    return _ASCII[kind]
