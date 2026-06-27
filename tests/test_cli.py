"""Tests for the zekan CLI."""

import textwrap

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from typer.testing import CliRunner

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_label_proxy
from zekan.cli import app

runner = CliRunner()


def test_help_exits_zero_and_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.output
    assert "benchmark" in result.output
    assert "report" in result.output


# ── Helpers ───────────────────────────────────────────────────────────────────

_CLEAN_CONFIG = textwrap.dedent("""\
    contract:
      prediction_problem: cli-test-clean
      entity_id: entity_id
      prediction_time: prediction_time
      target: target
      available_features_until: prediction_time
""")

_LEAKY_CONFIG = textwrap.dedent("""\
    contract:
      prediction_problem: cli-test-leaky
      entity_id: entity_id
      prediction_time: prediction_time
      target: target
      available_features_until: prediction_time
      forbidden_after_prediction:
        - leaky_label_proxy
""")

# 1200 rows / 6 periods satisfies row_count_and_folds (>=1000) and
# temporal_periods_count (>=6) so can_compute_severity=True with defaults.
_N_ENTITIES = 200
_SNAPSHOTS = 6


def _fast_clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


# ── End-to-end audit tests ────────────────────────────────────────────────────

def test_audit_clean_dataset_shows_trusted(tmp_path, monkeypatch):
    """Clean data → exit 0, ✓ TRUSTED in output."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert "✓ TRUSTED" in result.output


def test_audit_leaky_dataset_shows_risky_or_failed(tmp_path, monkeypatch):
    """Label-proxy leak → exit 0, RISKY or FAILED marker and WHAT TO FIX in output."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert ("⚠ RISKY" in result.output or "✗ FAILED" in result.output), result.output
    assert "WHAT TO FIX" in result.output


def test_audit_dry_run_stops_before_audit(tmp_path, monkeypatch):
    """--dry-run validates contract and stops; no verdict marker appears."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "READY" in result.output
    assert "✓ TRUSTED" not in result.output
    assert "⚠ RISKY" not in result.output
    assert "✗ FAILED" not in result.output
    assert "⚠ INCONCLUSIVE" not in result.output


# ── End-to-end report tests ───────────────────────────────────────────────────

def test_report_clean_writes_html_trusted(tmp_path, monkeypatch):
    """Clean data → exit 0, HTML file exists, contains TRUSTED and a <div."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    out = tmp_path / "report.html"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(
        app, ["report", "--data", str(csv), "--config", str(cfg), "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<div" in html
    assert "TRUSTED" in html


def test_report_leaky_writes_html_risky_or_failed(tmp_path, monkeypatch):
    """Label-proxy leak → HTML file contains RISKY or FAILED and WHAT TO FIX."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    out = tmp_path / "report.html"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(
        app, ["report", "--data", str(csv), "--config", str(cfg), "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert ("RISKY" in html or "FAILED" in html), html[:500]
    assert "WHAT TO FIX" in html


def test_report_and_audit_same_verdict(tmp_path, monkeypatch):
    """audit stdout and report HTML agree on verdict for the same input."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    out = tmp_path / "report.html"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    audit_result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])
    report_result = runner.invoke(
        app, ["report", "--data", str(csv), "--config", str(cfg), "--output", str(out)]
    )

    assert audit_result.exit_code == 0, audit_result.output
    assert report_result.exit_code == 0, report_result.output

    html = out.read_text(encoding="utf-8")
    # The verdict word that audit echoed must also appear in the HTML file.
    for word in ("TRUSTED", "RISKY", "FAILED", "INCONCLUSIVE"):
        if word in audit_result.output:
            assert word in html, (
                f"audit showed {word!r} but report HTML does not contain it"
            )
            break


# ── Inflation gate tests ──────────────────────────────────────────────────────

# 800 rows (80 × 10) → row_count_and_folds WARN → can_compute_severity=False
# contract still passes (no FAIL checks) → cannot-compute branch, not contract FAIL
_CANNOT_COMPUTE_N = 80


def test_gate_fires_on_leaky_above_threshold(tmp_path, monkeypatch):
    """Gate FAIL: leaky fl≈0.48 exceeds threshold 0.20 → exit 1, 'Inflation gate: FAIL'."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--fail-if-inflation-greater-than", "0.20",
    ])

    assert result.exit_code == 1, result.output
    assert "Inflation gate: FAIL" in result.output


def test_gate_passes_on_leaky_below_threshold(tmp_path, monkeypatch):
    """Gate PASS: leaky fl≈0.48 is within threshold 0.90 → exit 0, 'Inflation gate: PASS'."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--fail-if-inflation-greater-than", "0.90",
    ])

    assert result.exit_code == 0, result.output
    assert "Inflation gate: PASS" in result.output


def test_gate_passes_on_clean(tmp_path, monkeypatch):
    """Gate PASS: clean dataset fl≈0 is within threshold 0.20 → exit 0, 'Inflation gate: PASS'."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--fail-if-inflation-greater-than", "0.20",
    ])

    assert result.exit_code == 0, result.output
    assert "Inflation gate: PASS" in result.output


def test_gate_unverifiable_on_cannot_compute(tmp_path):
    """Cannot-compute + gate flag → exit 1, 'UNVERIFIABLE' (no monkeypatch; never trains)."""
    df = make_clean_dataset(n_entities=_CANNOT_COMPUTE_N, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--fail-if-inflation-greater-than", "0.20",
    ])

    # Confirm the fixture landed on cannot-compute, not a contract FAIL
    assert "CONTRACT VALID" in result.output, (
        f"Expected cannot-compute branch ('CONTRACT VALID'), got:\n{result.output}"
    )
    assert result.exit_code == 1, result.output
    assert "UNVERIFIABLE" in result.output


def test_gate_absent_on_dry_run(tmp_path):
    """--dry-run must NOT trigger the gate even when flag is set (no monkeypatch; never trains)."""
    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--dry-run", "--fail-if-inflation-greater-than", "0.20",
    ])

    assert result.exit_code == 0, result.output
    assert "Inflation gate:" not in result.output


def test_gate_absent_without_flag(tmp_path, monkeypatch):
    """Absent --fail-if-inflation-greater-than → no 'Inflation gate:' line at all."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert "Inflation gate:" not in result.output


def test_no_consecutive_blank_lines_with_gate(tmp_path, monkeypatch):
    """Verdict → gate transition must have no double-blank lines (FIX 4)."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--fail-if-inflation-greater-than", "0.20",
    ])

    assert result.exit_code == 0, result.output
    assert "Inflation gate: PASS" in result.output

    lines = result.output.split("\n")
    consecutive_blanks = sum(
        1 for i in range(len(lines) - 1)
        if lines[i] == "" and lines[i + 1] == ""
    )
    assert consecutive_blanks == 0, (
        f"Found {consecutive_blanks} consecutive blank line pair(s) in output:\n{result.output!r}"
    )


# ── Config error tests ────────────────────────────────────────────────────────

def test_audit_wrong_top_level_key_exits_1_with_helpful_message(tmp_path):
    """Config with 'prediction_contract:' key → exit 1, message names 'contract'."""
    df = make_clean_dataset(n_entities=20, snapshots_per_entity=3, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(
        "prediction_contract:\n"
        "  prediction_problem: test\n"
        "  entity_id: entity_id\n"
        "  prediction_time: prediction_time\n"
        "  target: target\n"
        "  available_features_until: prediction_time\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert result.exit_code == 1
    assert "did you mean 'contract'" in result.output
