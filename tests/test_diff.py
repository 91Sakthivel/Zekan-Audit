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
    null_scheme: str | None = None,
    null_stopping: str | None = None,
    estimator_identity: str | None = None,
) -> dict:
    """Minimal schema-1 artifact dict for diff testing.

    null_scheme, when given, is nested under provenance.seed.null_scheme,
    mirroring build_provenance's real shape (F2a).  None (default) omits
    "provenance" entirely, matching artifacts predating F2a.

    null_stopping, when given, is nested under provenance.seed.null_stopping,
    mirroring build_provenance's real shape (Tier 2). None (default) omits
    it, matching artifacts predating Tier 2. Mirrors null_scheme's pattern
    exactly.

    estimator_identity, when given, is set at provenance.estimator_identity
    (top-level of the provenance dict, matching build_provenance's real shape)
    -- Tier 3 Phase C.
    """
    d: dict = {
        "schema_version": schema_version,
        "summary": {
            "fixable_leakage": fixable_leakage,
            "headline": headline,
            "top_feature": top_feature,
            "verdict": verdict,
        },
    }
    if null_scheme is not None:
        d.setdefault("provenance", {}).setdefault("seed", {})["null_scheme"] = null_scheme
    if null_stopping is not None:
        d.setdefault("provenance", {}).setdefault("seed", {})["null_stopping"] = null_stopping
    if estimator_identity is not None:
        d.setdefault("provenance", {})["estimator_identity"] = estimator_identity
    return d


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


# ── F2a: null-scheme guard ───────────────────────────────────────────────────
# fixable_leakage does not depend on the permutation-null stream, so a scheme
# difference must never suppress or alter the fl comparison — it only adds an
# informational notice.  diff_reports does not compare null-derived scalars at
# all today, so there is nothing to suppress; these tests lock the notice.

def test_null_scheme_differs_sets_notice():
    old = _artifact(0.0400, null_scheme="serial_v1")
    new = _artifact(0.0400, null_scheme="spawn_v2")
    d = diff_reports(old, new)
    assert "null_scheme_notice" in d
    assert "serial_v1" in d["null_scheme_notice"]
    assert "spawn_v2" in d["null_scheme_notice"]
    assert "not comparable" in d["null_scheme_notice"]


def test_null_scheme_same_no_notice():
    old = _artifact(0.0400, null_scheme="spawn_v2")
    new = _artifact(0.0300, null_scheme="spawn_v2")
    d = diff_reports(old, new)
    assert "null_scheme_notice" not in d


def test_null_scheme_missing_both_treated_as_serial_v1_no_notice():
    """Two pre-F2a artifacts (no provenance at all) → both default to serial_v1
    → schemes match → no notice, even though neither ever declared a scheme."""
    old = _artifact(0.0400)
    new = _artifact(0.0300)
    d = diff_reports(old, new)
    assert "null_scheme_notice" not in d


def test_null_scheme_missing_on_one_side_sets_notice():
    """Old artifact predates F2a (no null_scheme) → treated as serial_v1; new
    is spawn_v2 → schemes differ → notice fires."""
    old = _artifact(0.0400)  # no provenance at all
    new = _artifact(0.0300, null_scheme="spawn_v2")
    d = diff_reports(old, new)
    assert "null_scheme_notice" in d
    assert "serial_v1" in d["null_scheme_notice"]
    assert "spawn_v2" in d["null_scheme_notice"]


def test_null_scheme_notice_does_not_affect_fl_comparison():
    """A scheme difference must not change direction/fl_delta — fl doesn't
    depend on the null stream."""
    old = _artifact(0.2100, verdict="FAIL", null_scheme="serial_v1")
    new = _artifact(0.0400, verdict="PASS", null_scheme="spawn_v2")
    d = diff_reports(old, new)
    assert "null_scheme_notice" in d
    assert d["direction"] == "IMPROVED"
    assert d["fl_delta"] == pytest.approx(-0.1700)


def test_cli_diff_prints_null_scheme_notice(tmp_path):
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.0400, null_scheme="serial_v1"))
    _write(b, _artifact(0.0300, null_scheme="spawn_v2"))

    result = runner.invoke(app, ["diff", "--old", str(a), "--new", str(b)])

    assert result.exit_code == 0, result.output
    assert "null scheme differs" in result.output
    assert "not comparable across schemes" in result.output


def test_cli_diff_fail_on_regression_silent_across_schemes_on_improvement(tmp_path):
    """Schemes differ AND leakage improved -> --fail-on-regression must exit 0.

    A trust tool must not cry regression at its own upgrade: the scheme
    difference alone (unrelated to fl) must never be the reason this fires.
    """
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.2100, verdict="FAIL", headline="FAILED", null_scheme="serial_v1"))
    _write(b, _artifact(0.0400, verdict="PASS", headline="TRUSTED", null_scheme="spawn_v2"))

    result = runner.invoke(app, [
        "diff", "--old", str(a), "--new", str(b), "--fail-on-regression",
    ])

    assert result.exit_code == 0, result.output
    assert "null scheme differs" in result.output


def test_cli_diff_fail_on_regression_still_fires_on_real_regression_across_schemes(tmp_path):
    """Schemes differing must not SUPPRESS a genuine fl regression either —
    the gate is fl-only and stays fl-only regardless of scheme notices."""
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.0400, verdict="PASS", headline="TRUSTED", null_scheme="serial_v1"))
    _write(b, _artifact(0.2100, verdict="FAIL", headline="FAILED", null_scheme="spawn_v2"))

    result = runner.invoke(app, [
        "diff", "--old", str(a), "--new", str(b), "--fail-on-regression",
    ])

    assert result.exit_code == 1


# ── Tier 2: null-stopping guard ──────────────────────────────────────────────
# Mirrors the F2a null-scheme guard section above exactly: fixable_leakage does
# not depend on the stopping scheme (fixed_v1 vs sequential_v1), so a stopping
# difference must never suppress or alter the fl comparison -- only add a
# notice. This coverage and the CLI wiring below were a pre-existing gap
# (Tier 2b-final left null_stopping_notice computed in diff_reports but never
# tested or rendered) -- fixed here as debt cleanup alongside Upgrade 1 1b.

def test_null_stopping_differs_sets_notice():
    old = _artifact(0.0400, null_stopping="fixed_v1")
    new = _artifact(0.0400, null_stopping="sequential_v1")
    d = diff_reports(old, new)
    assert "null_stopping_notice" in d
    assert "fixed_v1" in d["null_stopping_notice"]
    assert "sequential_v1" in d["null_stopping_notice"]
    assert "not directly comparable" in d["null_stopping_notice"]


def test_null_stopping_same_no_notice():
    old = _artifact(0.0400, null_stopping="sequential_v1")
    new = _artifact(0.0300, null_stopping="sequential_v1")
    d = diff_reports(old, new)
    assert "null_stopping_notice" not in d


def test_null_stopping_missing_both_treated_as_fixed_v1_no_notice():
    """Two pre-Tier-2 artifacts (no provenance at all) -> both default to
    fixed_v1 -> stopping schemes match -> no notice."""
    old = _artifact(0.0400)
    new = _artifact(0.0300)
    d = diff_reports(old, new)
    assert "null_stopping_notice" not in d


def test_null_stopping_missing_on_one_side_sets_notice():
    """Old artifact predates Tier 2 (no null_stopping) -> treated as
    fixed_v1; new is sequential_v1 -> stopping schemes differ -> notice fires."""
    old = _artifact(0.0400)  # no provenance at all
    new = _artifact(0.0300, null_stopping="sequential_v1")
    d = diff_reports(old, new)
    assert "null_stopping_notice" in d
    assert "fixed_v1" in d["null_stopping_notice"]
    assert "sequential_v1" in d["null_stopping_notice"]


def test_null_stopping_notice_does_not_affect_fl_comparison():
    """A stopping-scheme difference must not change direction/fl_delta -- fl
    doesn't depend on how many permutations were drawn or how."""
    old = _artifact(0.2100, verdict="FAIL", null_stopping="fixed_v1")
    new = _artifact(0.0400, verdict="PASS", null_stopping="sequential_v1")
    d = diff_reports(old, new)
    assert "null_stopping_notice" in d
    assert d["direction"] == "IMPROVED"
    assert d["fl_delta"] == pytest.approx(-0.1700)


def test_cli_diff_prints_null_stopping_notice(tmp_path):
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.0400, null_stopping="fixed_v1"))
    _write(b, _artifact(0.0300, null_stopping="sequential_v1"))

    result = runner.invoke(app, ["diff", "--old", str(a), "--new", str(b)])

    assert result.exit_code == 0, result.output
    assert "null stopping scheme differs" in result.output
    assert "not directly comparable" in result.output


# ── Tier 3 Phase C: estimator-identity guard ─────────────────────────────────
# Unlike null_scheme/null_stopping (which only affect null-derived stats),
# a different estimator changes what fixable_leakage itself means -- so this
# guard goes further than a notice: it also refuses fl_delta/direction.

def test_estimator_identity_differs_sets_notice_and_refuses_delta():
    old = _artifact(0.2100, verdict="FAIL", estimator_identity="rf")
    new = _artifact(0.0400, verdict="PASS", estimator_identity="histgb")
    d = diff_reports(old, new)
    assert "estimator_identity_notice" in d
    assert "rf" in d["estimator_identity_notice"]
    assert "histgb" in d["estimator_identity_notice"]
    assert "not comparable" in d["estimator_identity_notice"]
    assert d["fl_delta"] is None
    assert d["direction"] == "UNVERIFIABLE_CHANGE"


def test_estimator_identity_same_no_notice():
    old = _artifact(0.0400, estimator_identity="histgb")
    new = _artifact(0.0300, estimator_identity="histgb")
    d = diff_reports(old, new)
    assert "estimator_identity_notice" not in d
    assert d["direction"] == "IMPROVED"
    assert d["fl_delta"] == pytest.approx(-0.0100)


def test_estimator_identity_missing_both_treated_as_unknown_no_notice():
    """Two artifacts with no provenance at all -> both default to "unknown"
    -> identities match -> no notice."""
    old = _artifact(0.0400)
    new = _artifact(0.0300)
    d = diff_reports(old, new)
    assert "estimator_identity_notice" not in d


def test_estimator_identity_missing_on_one_side_sets_notice():
    old = _artifact(0.0400)  # no provenance at all -> "unknown"
    new = _artifact(0.0300, estimator_identity="histgb")
    d = diff_reports(old, new)
    assert "estimator_identity_notice" in d
    assert "unknown" in d["estimator_identity_notice"]
    assert "histgb" in d["estimator_identity_notice"]


def test_cli_diff_prints_estimator_identity_notice(tmp_path):
    a = tmp_path / "old.json"
    b = tmp_path / "new.json"
    _write(a, _artifact(0.0400, estimator_identity="rf"))
    _write(b, _artifact(0.0300, estimator_identity="histgb"))

    result = runner.invoke(app, ["diff", "--old", str(a), "--new", str(b)])

    assert result.exit_code == 0, result.output
    assert "estimator differs" in result.output
    assert "not comparable" in result.output


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
