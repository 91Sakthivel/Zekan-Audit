"""Tests for the zekan CLI."""

import json
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


# ── Config-declared data path tests ───────────────────────────────────────────

def test_data_flag_overrides_config_data(tmp_path, monkeypatch):
    """Explicit --data wins over cfg.data even when cfg.data points elsewhere."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    real_csv = tmp_path / "real_data.csv"
    df.to_csv(real_csv, index=False)

    cfg = tmp_path / "zekan.yml"
    cfg.write_text(_CLEAN_CONFIG + "data: does_not_exist.csv\n")

    result = runner.invoke(app, [
        "audit", "--data", str(real_csv), "--config", str(cfg), "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert "READY" in result.output
    assert "does_not_exist.csv" not in result.output


def test_config_data_used_when_flag_absent(tmp_path, monkeypatch):
    """cfg.data is used to locate the dataset when --data is not passed."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    cfg = tmp_path / "zekan.yml"
    cfg.write_text(_CLEAN_CONFIG + "data: data.csv\n")

    result = runner.invoke(app, ["audit", "--config", str(cfg), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "READY" in result.output


def test_neither_data_flag_nor_config_data_teaching_error(tmp_path):
    """No --data and no cfg.data → teaching error to stderr, exit 1."""
    cfg = tmp_path / "zekan.yml"
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--config", str(cfg)])

    assert result.exit_code == 1
    assert "no data file specified" in result.stderr
    assert "--data" in result.stderr
    assert "zekan.yml" in result.stderr


def test_audit_auto_finds_zekan_yml_in_cwd(tmp_path, monkeypatch):
    """Bare `zekan audit` (no --config, no --data) uses zekan.yml + its data: in cwd."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    monkeypatch.chdir(tmp_path)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df.to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "zekan.yml").write_text(_CLEAN_CONFIG + "data: data.csv\n")

    result = runner.invoke(app, ["audit", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "READY" in result.output


def test_config_data_resolved_against_config_dir_not_cwd(tmp_path, monkeypatch):
    """cfg.data is resolved relative to the config file's directory, not cwd."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df.to_csv(cfg_dir / "data.csv", index=False)
    (cfg_dir / "zekan.yml").write_text(_CLEAN_CONFIG + "data: data.csv\n")

    monkeypatch.chdir(other_dir)

    result = runner.invoke(app, [
        "audit", "--config", str(cfg_dir / "zekan.yml"), "--dry-run",
    ])

    assert result.exit_code == 0, result.output
    assert "READY" in result.output


def test_echo_reports_data_source_when_from_config(tmp_path, monkeypatch):
    """When data comes from cfg.data, stderr echoes which file was used."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = tmp_path / "zekan.yml"
    cfg.write_text(_CLEAN_CONFIG + "data: data.csv\n")

    result = runner.invoke(app, ["audit", "--config", str(cfg), "--dry-run"])

    assert result.exit_code == 0, result.output
    # No --json: human text (including the echo) goes to stdout, per err=json_mode.
    assert "(from zekan.yml)" in result.output
    assert "data.csv" in result.output


def test_echo_absent_when_data_from_flag(tmp_path, monkeypatch):
    """When --data is given explicitly, no '(from zekan.yml)' echo appears."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = tmp_path / "zekan.yml"
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "(from zekan.yml)" not in result.output


def test_json_stdout_clean_when_data_from_config(tmp_path, monkeypatch):
    """--json with cfg.data: stdout parses as JSON; the echo line stays on stderr."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    cfg = tmp_path / "zekan.yml"
    cfg.write_text(_CLEAN_CONFIG + "data: data.csv\n")

    result = runner.invoke(app, ["audit", "--config", str(cfg), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    assert "(from zekan.yml)" not in result.stdout
    assert "(from zekan.yml)" in result.stderr
    assert "data.csv" in result.stderr


def test_explicit_data_config_byte_identical_json(tmp_path, monkeypatch):
    """With --data and --config both given explicitly, --json output is unchanged (regression guard)."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    monkeypatch.setattr("zekan.severity.ablation._default_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    args = ["audit", "--data", str(csv), "--config", str(cfg), "--json"]
    r1 = runner.invoke(app, args)
    r2 = runner.invoke(app, args)

    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert r1.stdout == r2.stdout


# ── --json flag tests ─────────────────────────────────────────────────────────

def test_json_stdout_is_parseable(tmp_path, monkeypatch):
    """--json: stdout is valid JSON on a clean dataset."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)


def test_json_stdout_has_no_human_text(tmp_path, monkeypatch):
    """--json: stdout contains no contract table, READY, or verdict banner lines."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg), "--json"])

    stdout = result.stdout
    assert "[PASS]" not in stdout
    assert "READY:" not in stdout
    assert "Zekan audit:" not in stdout
    assert "TRUSTED" not in stdout  # verdict banner must not appear as human text in stdout


def test_json_exit_code_unchanged_vs_non_json(tmp_path, monkeypatch):
    """--json does not change exit code relative to a non-json run."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    plain = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])
    json_run = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg), "--json"])

    assert plain.exit_code == json_run.exit_code


def test_json_schema_version(tmp_path, monkeypatch):
    """--json: schema_version == '1' in the output."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg), "--json"])
    parsed = json.loads(result.stdout)
    assert parsed["schema_version"] == "1"


def test_json_gate_null_without_flag(tmp_path, monkeypatch):
    """--json without --fail-if-inflation-greater-than: gate is null."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg), "--json"])
    parsed = json.loads(result.stdout)
    assert parsed["gate"] is None


def test_json_gate_triggered(tmp_path, monkeypatch):
    """--json with gate triggered: gate.triggered=true, exit=1, gate.exit_code=1."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--json", "--fail-if-inflation-greater-than", "0.20",
    ])

    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert parsed["gate"]["triggered"] is True
    assert parsed["gate"]["exit_code"] == 1
    assert parsed["gate"]["threshold"] == pytest.approx(0.20)


def test_json_gate_not_triggered(tmp_path, monkeypatch):
    """--json with gate not triggered: gate.triggered=false, exit=0, gate.exit_code=0."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--json", "--fail-if-inflation-greater-than", "0.90",
    ])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["gate"]["triggered"] is False
    assert parsed["gate"]["exit_code"] == 0


def test_json_gate_human_text_not_in_stdout(tmp_path, monkeypatch):
    """Gate human line ('Inflation gate:') goes to stderr, not stdout in --json mode."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df_clean = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df_leaky.to_csv(csv, index=False)
    cfg.write_text(_LEAKY_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--json", "--fail-if-inflation-greater-than", "0.20",
    ])

    stdout = result.stdout
    parsed = json.loads(stdout)
    assert "Inflation gate:" not in stdout
    assert parsed["gate"]["triggered"] is True


# ── --estimator flag tests ────────────────────────────────────────────────────

def test_audit_unknown_estimator_exits_with_error(tmp_path, monkeypatch):
    """--estimator with an unsupported name → non-zero exit, error lists valid choices."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg), "--estimator", "xgboost",
    ])

    assert result.exit_code != 0
    combined = (result.output or "") + str(result.exception or "")
    assert "xgboost" in combined or "Valid choices" in combined


def test_audit_estimator_logistic_completes(tmp_path):
    """--estimator logistic runs the full audit and exits 0 on clean data."""
    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    result = runner.invoke(app, [
        "audit", "--data", str(csv), "--config", str(cfg), "--estimator", "logistic",
    ])

    assert result.exit_code == 0, result.output
    assert "✓ TRUSTED" in result.output


def test_audit_estimator_histgb_deterministic_byte_identical(tmp_path):
    """--estimator histgb is accepted and produces byte-identical JSON across
    two consecutive runs on a small fixture -- the hard prerequisite for using
    it in any trust gate. early_stopping is pinned off in the factory
    (see estimators.py), so this must hold regardless of row count."""
    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    args = [
        "audit", "--data", str(csv), "--config", str(cfg),
        "--estimator", "histgb", "--json",
    ]
    r1 = runner.invoke(app, args)
    r2 = runner.invoke(app, args)

    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert r1.stdout == r2.stdout


def test_audit_no_estimator_json_is_deterministic(tmp_path, monkeypatch):
    """Omitting --estimator produces byte-identical JSON across two consecutive runs."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    monkeypatch.setattr("zekan.severity.ablation._default_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG)

    args = ["audit", "--data", str(csv), "--config", str(cfg), "--json"]
    r1 = runner.invoke(app, args)
    r2 = runner.invoke(app, args)

    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert r1.stdout == r2.stdout


# ── Categorical support Part B: gates on contract failure ──────────────────────
# CATEGORICAL_SUPPORT_PREREGISTRATION.md section 2/3(c): a structural probe
# whose own precondition is just a dataframe and a contract (Upgrade H) must
# still run when the contract fails feature_columns_numeric; a probe that
# needs temporal folds (Upgrade 1) must not; the exit code stays non-zero;
# and no VerdictReport -- so no PASS-shaped verdict -- is ever built on this
# path.

_FAILED_NUMERIC_CONFIG = textwrap.dedent("""\
    contract:
      prediction_problem: cli-test-failed-numeric
      entity_id: entity_id
      prediction_time: prediction_time
      target: target
      available_features_until: prediction_time
""")


def _failed_contract_dataset():
    """A clean panel plus one extra RAW STRING feature column that is a
    deterministic, noise-free copy of `target` -- undeclared (no
    categorical_features in _FAILED_NUMERIC_CONFIG), so it both fails
    feature_columns_numeric (raw text, not castable to float) and gives
    Upgrade H's near-bijection probe a genuine Theil's U ~= 1.0 to find."""
    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df = df.copy()
    df["leak_as_text"] = df["target"].map({0: "class_zero", 1: "class_one"})
    return df


def _write_failed_contract_fixture(tmp_path):
    df = _failed_contract_dataset()
    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_FAILED_NUMERIC_CONFIG)
    return csv, cfg


def test_failed_contract_still_runs_upgrade_h(tmp_path, monkeypatch):
    """feature_columns_numeric FAILS on leak_as_text, but Upgrade H's own
    precondition (a dataframe and a contract) is unaffected -- it must still
    run and report the near-bijection column by name."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    csv, cfg = _write_failed_contract_fixture(tmp_path)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert "CONTRACT FAILED" in result.output
    assert "feature_columns_numeric" in result.output
    assert "NEAR_BIJECTION_UNDECLARED_LEAK" in result.output, result.output
    assert "leak_as_text" in result.output


def test_failed_contract_does_not_run_upgrade_1(tmp_path, monkeypatch):
    """Same failed contract: probe_undeclared_feature_screen (Upgrade 1) must
    never be CALLED -- proven by spying on the function itself, since "ran
    and found nothing" and "was skipped" both produce zero visible output,
    so scanning result.output alone couldn't tell them apart."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    calls: list = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(
        "zekan.detectors.undeclared_feature_probe.probe_undeclared_feature_screen", _spy
    )
    csv, cfg = _write_failed_contract_fixture(tmp_path)

    runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert calls == [], (
        "probe_undeclared_feature_screen (Upgrade 1) was called despite folds=None "
        "-- it needs numeric features via temporal folds, which a failed contract "
        "cannot provide"
    )


def test_failed_contract_exits_nonzero(tmp_path, monkeypatch):
    """Exit code stays non-zero on contract failure, exactly as before this
    change -- running structural probes must not soften the failure."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    csv, cfg = _write_failed_contract_fixture(tmp_path)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert result.exit_code == 1, result.output


def test_failed_contract_never_builds_verdict_report(tmp_path, monkeypatch):
    """No PASS-shaped verdict is ever emitted on a failed contract. Proven two
    ways: (1) run_audit -- the only function that can construct a
    VerdictReport -- is spied on and must never be called, a stronger
    guarantee than scanning text, since build_verdict's own unavailable-status
    branch sets policy_decision.verdict="PASS" for OTHER callers (pinned by
    test_verdict.py::test_unavailable_engine_result) and that string must
    never reach this path; (2) none of the verdict marker strings a real
    TRUSTED/RISKY/FAILED render_verdict() call would print appear in output
    (a bare "PASS" substring check would be defeated by the per-check
    "[PASS]" lines the contract table already prints for every check that DID
    pass, e.g. entity_id_exists -- so the marker strings are checked
    specifically, not the bare word)."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    run_audit_calls: list = []

    def _spy_run_audit(*args, **kwargs):
        run_audit_calls.append((args, kwargs))
        raise RuntimeError("run_audit must not be reached on a failed contract")

    monkeypatch.setattr("zekan.severity.audit.run_audit", _spy_run_audit)
    csv, cfg = _write_failed_contract_fixture(tmp_path)

    result = runner.invoke(app, ["audit", "--data", str(csv), "--config", str(cfg)])

    assert run_audit_calls == [], (
        "run_audit() was called on a failed contract -- a VerdictReport "
        "(and its PASS-shaped policy_decision.verdict) could have been built"
    )
    assert "✓ TRUSTED" not in result.output
    assert "TRUSTED" not in result.output
    assert "⚠ RISKY" not in result.output
    assert "✗ FAILED" not in result.output
