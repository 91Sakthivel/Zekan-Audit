"""Tests for zekan.reports.markers — Unicode capability detection and ASCII fallback."""

from __future__ import annotations

import pytest

from zekan.reports.markers import _ASCII, _GLYPHS, marker, sanitize, supports_unicode


# ── Fake streams ──────────────────────────────────────────────────────────────

class _Utf8Stream:
    encoding = "utf-8"


class _Cp1252Stream:
    encoding = "cp1252"


class _AsciiStream:
    encoding = "ascii"


class _NoEncodingStream:
    pass


class _BlankEncodingStream:
    encoding = ""


class _UnknownCodecStream:
    encoding = "nonexistent_codec_xyz"


# ── supports_unicode ──────────────────────────────────────────────────────────

def test_none_returns_false():
    assert supports_unicode(None) is False


def test_missing_encoding_attr_returns_false():
    assert supports_unicode(_NoEncodingStream()) is False


def test_blank_encoding_returns_false():
    assert supports_unicode(_BlankEncodingStream()) is False


def test_utf8_returns_true():
    assert supports_unicode(_Utf8Stream()) is True


def test_cp1252_returns_false():
    assert supports_unicode(_Cp1252Stream()) is False


def test_ascii_returns_false():
    assert supports_unicode(_AsciiStream()) is False


def test_unknown_codec_returns_false():
    assert supports_unicode(_UnknownCodecStream()) is False


# ── marker — glyph path ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["trusted", "risky", "failed", "inconclusive"])
def test_marker_utf8_returns_glyph(kind):
    assert marker(kind, _Utf8Stream()) == _GLYPHS[kind]


# ── marker — ASCII fallback path ──────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["trusted", "risky", "failed", "inconclusive"])
def test_marker_cp1252_returns_ascii(kind):
    assert marker(kind, _Cp1252Stream()) == _ASCII[kind]


@pytest.mark.parametrize("kind", ["trusted", "risky", "failed", "inconclusive"])
def test_marker_ascii_stream_returns_ascii(kind):
    assert marker(kind, _AsciiStream()) == _ASCII[kind]


@pytest.mark.parametrize("kind", ["trusted", "risky", "failed", "inconclusive"])
def test_ascii_fallback_is_encodable_in_cp1252(kind):
    _ASCII[kind].encode("cp1252")  # must not raise


# ── regression: render_verdict with cp1252 stream uses ASCII markers ──────────

def _make_fail_report():
    from zekan.severity.engine import PerFoldSeverity, SeverityResult
    from zekan.severity.verdict import build_verdict
    sr = SeverityResult(
        status="pass",
        metric="roc_auc",
        naive_auc=0.90,
        estimated_deployable_auc=0.70,
        total_optimism=0.20,
        fixable_leakage=0.20,
        fixable_leakage_range=(0.16, 0.24),
        nonfixable_optimism=0.00,
        per_fold=[
            PerFoldSeverity(
                fold_idx=0,
                auc_with_forbidden=0.90,
                auc_without_forbidden=0.70,
                is_terminal=False,
            ),
        ],
        caveat="test",
        null_iqr=0.01,
        null_99th=0.09,
        p_value=0.001,
        nsl=2.5,
        n_permutations_run=100,
    )
    report = build_verdict(sr)
    assert report.policy_decision.verdict == "FAIL"
    return report


def test_render_verdict_fail_cp1252_uses_ascii_marker():
    """Regression for the UnicodeEncodeError crash on Windows cp1252 consoles."""
    from zekan.reports.text_view import render_verdict
    out = render_verdict(_make_fail_report(), stream=_Cp1252Stream())
    assert "[FAIL]" in out
    assert "✗" not in out
    assert "⚠" not in out
    assert "✓" not in out


# ── sanitize: body-text Unicode fallback ─────────────────────────────────────

def test_sanitize_cp1252_no_unicode_body_chars():
    """cp1252 stream: em-dash, minus sign, and box-drawing are transliterated; no '?' in output."""
    from zekan.reports.text_view import render_verdict
    out = render_verdict(_make_fail_report(), stream=_Cp1252Stream())
    assert "?" not in out
    assert "—" not in out   # em dash
    assert "−" not in out   # minus sign
    assert "─" not in out   # box drawing


def test_sanitize_utf8_preserves_body_chars():
    """UTF-8 stream: em-dash, minus sign, and box-drawing are kept unchanged."""
    from zekan.reports.text_view import render_verdict
    out = render_verdict(_make_fail_report(), stream=_Utf8Stream())
    assert "—" in out   # em dash in "inflated — real performance"
    assert "−" in out   # minus sign in "B−C fixable leakage"
    assert "─" in out   # box drawing in divider
