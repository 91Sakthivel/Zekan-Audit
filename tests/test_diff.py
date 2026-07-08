"""Unit and CLI tests for Upgrade C: zekan diff."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from zekan.cli import app
from zekan.reports.diff import diff_reports

runner = CliRunner()


# ── Fixture builders ──────────────────────────────────────────────────────────

def _artifact(
    fixable_leakage,
    verdict: str = "PASS",
    headline: str = "TRUSTED",
    top_feature=None,
    schema_version: str = "1",
) -> dict:
    """Minimal schema-1 artifact dict for diff testing."""
    return {
        "schema_version": schema_version,
        "summary": {
            "fixable_leakage": fixable_leakage,
            "headline": headline,
            "top_feature": top_feature,
            "verdict": verdict,
        },
    }


def _write(path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# ── Unit tests: diff_reports ──────────────────────────────────────────────────

def test_improved_direction_and_delta():
    old = _artifact(0.2100, verdict="FAIL", headline="FAILED")
    new = _artifact(0.0400, verdict="PASS", headline="TRUSTED")
    d = diff_reports(old, new)
    assert d["direction"] == "IMPROVED"
    assert d["fl_old"] == pytest.approx(0.2100)
    assert d["fl_new"] == pytest.approx(0.0400)
    assert d["fl_delta"] == pytest.approx(-0.1700)


def test_regressed_direction_and_delta():
    old = _artifact(0.0400, verdict="PASS", headline="TRUSTED")
    new = _artifact(0.2100, verdict="FAIL", headline="FAILED")
    d = diff_reports(old, new)
    assert d["direction"] == "REGRESSED"
    assert d["fl_delta"] == pytest.approx(0.1700)


def test_unchanged_direction():
    old = _artifact(0.0400)
    new = _artifact(0.0400)
    d = diff_reports(old, new)
    assert d["direction"] == "UNCHANGED"
    assert d["fl_delta"] == pytest.approx(0.0)


def test_verdict_transition_captured():
    old = _artifact(0.2100, verdict="FAIL", headline="FAILED")
    new = _artifact(0.0400, verdict="PASS", headline="TRUSTED")
    d = diff_reports(old, new)
    assert d["verdict_old"] == "FAIL"
    assert d["verdict_new"] == "PASS"
    assert d["verdict_changed"] is True


def test_verdict_unchanged_flagged():
    old = _artifact(0.0500, verdict="WARN", headline="RISKY")
    new = _artifact(0.0300, verdict="WARN", headline="RISKY")
    d = diff_reports(old, new)
    assert d["verdict_changed"] is False


def test_top_feature_fixed():
    old = _artifact(0.2100, top_feature="next_cycle_value")
    new = _artifact(0.0400, top_feature=None)
    d = diff_reports(old, new)
    assert d["top_feature_status"] == "fixed"
    assert d["top_feature_old"] == "next_cycle_value"
    assert d["top_feature_new"] is None


def test_top_feature_changed():
    old = _artifact(0.2100, top_feature="feature_a")
    new = _artifact(0.1800, top_feature="feature_b")
    d = diff_reports(old, new)
    assert d["top_feature_status"] == "changed"
    assert d["top_feature_old"] == "feature_a"
    assert d["top_feature_new"] == "feature_b"


def test_top_feature_same():
    old = _artifact(0.2100, top_feature="feature_a")
    new = _artifact(0.1500, top_feature="feature_a")
    d = diff_reports(old, new)
    assert d["top_feature_status"] == "same"


def test_top_feature_appeared():
    old = _artifact(0.0200, top_feature=None)
    new = _artifact(0.1500, top_feature="feature_a")
    d = diff_reports(old, new)
    assert d["top_feature_status"] == "appeared"


def test_top_feature_none_both():
    old = _artifact(0.0100, top_feature=None)
    new = _artifact(0.0050, top_feature=None)
    d = diff_reports(old, new)
    assert d["top_feature_status"] == "none"


def test_null_fl_old_gives_unverifiable_no_delta():
    old = _artifact(None)
    new = _artifact(0.0400)
    d = diff_reports(old, new)
    assert d["direction"] == "UNVERIFIABLE_CHANGE"
    assert d["fl_delta"] is None


def test_null_fl_new_gives_unverifiable_no_delta():
    old = _artifact(0.2100)
    new = _artifact(None)
    d = diff_reports(old, new)
    assert d["direction"] == "UNVERIFIABLE_CHANGE"
    assert d["fl_delta"] is None


def test_null_fl_both_gives_unverifiable_no_delta():
    old = _artifact(None)
    new = _artifact(None)
    d = diff_reports(old, new)
    assert d["direction"] == "UNVERIFIABLE_CHANGE"
    assert d["fl_delta"] is None


def test_schema_mismatch_flagged():
    old = _artifact(0.1000, schema_version="1")
    new = _artifact(0.0500, schema_version="2")
    d = diff_reports(old, new)
    assert "schema_mismatch" in d
    assert "'1'" in d["schema_mismatch"]
    assert "'2'" in d["schema_mismatch"]


def test_no_schema_mismatch_key_when_versions_match():
    old = _artifact(0.1000, schema_version="1")
    new = _artifact(0.0500, schema_version="1")
    d = diff_reports(old, new)
    assert "schema_mismatch" not in d


def test_determinism_same_inputs_identical_output():
    old = _artifact(0.2100, verdict="FAIL", headline="FAILED", top_feature="x")
    new = _artifact(0.0400, verdict="PASS", headline="TRUSTED")
    assert diff_reports(old, new) == diff_reports(old, new)


# ── CLI tests ─────────────────────────────────────────────────────────────────

def test_cli_diff_renders_human_output(tmp_path):
    """zekan diff --old a.json --new b.json exits 0 and shows direction."""
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.2100, verdict="FAIL", headline="FAILED", top_feature="next_cycle_value"))
    _write(b, _artifact(0.0400, verdict="PASS", headline="TRUSTED"))

    result = runner.invoke(app, ["diff", "--old", str(a), "--new", str(b)])

    assert result.exit_code == 0, result.output
    assert "IMPROVED" in result.output
    assert "0.2100" in result.output
    assert "0.0400" in result.output


def test_cli_diff_json_flag_produces_parseable_json(tmp_path):
    """--json emits valid JSON with required keys."""
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.2100, verdict="FAIL", headline="FAILED"))
    _write(b, _artifact(0.0400, verdict="PASS", headline="TRUSTED"))

    result = runner.invoke(app, ["diff", "--old", str(a), "--new", str(b), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["direction"] == "IMPROVED"
    assert "fl_delta" in parsed
    assert "verdict_changed" in parsed


def test_cli_diff_fail_on_regression_exits_1(tmp_path):
    """--fail-on-regression exits 1 when leakage increased."""
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.0400, verdict="PASS", headline="TRUSTED"))
    _write(b, _artifact(0.2100, verdict="FAIL", headline="FAILED"))

    result = runner.invoke(app, [
        "diff", "--old", str(a), "--new", str(b), "--fail-on-regression",
    ])

    assert result.exit_code == 1


def test_cli_diff_fail_on_regression_exits_0_on_improvement(tmp_path):
    """--fail-on-regression exits 0 when leakage decreased."""
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.2100, verdict="FAIL", headline="FAILED"))
    _write(b, _artifact(0.0400, verdict="PASS", headline="TRUSTED"))

    result = runner.invoke(app, [
        "diff", "--old", str(a), "--new", str(b), "--fail-on-regression",
    ])

    assert result.exit_code == 0, result.output


def test_cli_diff_missing_file_exits_nonzero(tmp_path):
    """Missing old file → non-zero exit with error message."""
    b = tmp_path / "new.json"
    _write(b, _artifact(0.0400))

    result = runner.invoke(app, [
        "diff", "--old", str(tmp_path / "ghost.json"), "--new", str(b),
    ])

    assert result.exit_code != 0


def test_cli_diff_malformed_json_exits_nonzero(tmp_path):
    """Malformed JSON → non-zero exit."""
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    a.write_text("not json", encoding="utf-8")
    _write(b, _artifact(0.0400))

    result = runner.invoke(app, ["diff", "--old", str(a), "--new", str(b)])

    assert result.exit_code != 0
