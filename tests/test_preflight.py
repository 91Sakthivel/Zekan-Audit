"""Tests for zekan/reports/preflight.py — unit (scan_dataframe, format_preflight_lines) + CLI."""

from __future__ import annotations

import json
import textwrap

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from typer.testing import CliRunner

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.cli import app
from zekan.reports.preflight import format_preflight_lines, scan_dataframe

runner = CliRunner()

_CLEAN_CONFIG = textwrap.dedent("""\
    contract:
      prediction_problem: preflight-test
      entity_id: entity_id
      prediction_time: prediction_time
      target: target
      available_features_until: prediction_time
      forbidden_after_prediction: []
""")

_N_ENTITIES = 200
_SNAPSHOTS = 6


def _fast_clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


# ── Unit: scan_dataframe ──────────────────────────────────────────────────────

def test_constant_col_detected():
    # b has 2 distinct values across 4 rows — neither constant nor all_unique
    df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 1, 2]})
    notes = scan_dataframe(df, {})
    assert len(notes) == 1
    assert notes[0]["column"] == "a"
    assert notes[0]["kind"] == "constant"


def test_constant_detail_string():
    df = pd.DataFrame({"a": [99, 99, 99]})
    notes = scan_dataframe(df, {})
    assert notes[0]["detail"] == "only 1 distinct value"


def test_all_unique_col_detected():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]})
    notes = scan_dataframe(df, {})
    assert len(notes) == 1
    assert notes[0]["column"] == "a"
    assert notes[0]["kind"] == "all_unique"


def test_all_unique_detail_contains_n():
    df = pd.DataFrame({"a": [10, 20, 30]})
    notes = scan_dataframe(df, {})
    assert "3 of 3" in notes[0]["detail"]


def test_normal_varied_numeric_no_note():
    df = pd.DataFrame({"a": [1, 2, 3, 1, 2]})
    notes = scan_dataframe(df, {})
    assert notes == []


def test_object_col_cardinality_note():
    df = pd.DataFrame({"a": ["x", "y", "x", "z", "y"]})
    notes = scan_dataframe(df, {})
    assert len(notes) == 1
    assert notes[0]["kind"] == "cardinality"


def test_cardinality_detail_correct_k_n_p():
    # 5 distinct values across 10 rows → 50.0%
    df = pd.DataFrame({"a": [str(i // 2) for i in range(10)]})
    notes = scan_dataframe(df, {})
    assert notes[0]["kind"] == "cardinality"
    assert "5 distinct values" in notes[0]["detail"]
    assert "10 rows" in notes[0]["detail"]
    assert "50.0%" in notes[0]["detail"]


def test_cardinality_only_for_object_not_numeric():
    # Numeric column with many but repeated values — not all_unique, not constant, not object
    df = pd.DataFrame({"a": [i % 5 for i in range(100)]})
    notes = scan_dataframe(df, {})
    assert notes == []


def test_role_attached_to_constant_col():
    df = pd.DataFrame({"target": [0, 0, 0]})
    notes = scan_dataframe(df, {"target": "target"})
    assert notes[0]["role"] == "target"


def test_no_role_when_not_declared():
    df = pd.DataFrame({"a": [1, 1, 1]})
    notes = scan_dataframe(df, {})
    assert notes[0]["role"] is None


def test_deterministic_order_matches_df_columns():
    df = pd.DataFrame({"b": [1, 1, 1], "a": [1, 1, 1], "c": [1, 1, 1]})
    notes = scan_dataframe(df, {})
    assert [n["column"] for n in notes] == ["b", "a", "c"]


def test_empty_notes_on_all_numeric_varied():
    # All numeric, all varied, none constant, none all_unique
    df = pd.DataFrame({"x": [1, 2, 3, 1], "y": [4.0, 5.0, 6.0, 4.0]})
    assert scan_dataframe(df, {}) == []


def test_single_row_df_triggers_constant_not_all_unique():
    # len==1 → nunique==1 → constant fires; all_unique requires n > 1
    df = pd.DataFrame({"a": [42]})
    notes = scan_dataframe(df, {})
    assert len(notes) == 1
    assert notes[0]["kind"] == "constant"


def test_constant_takes_priority_not_all_unique_when_len1():
    df = pd.DataFrame({"a": [7]})
    notes = scan_dataframe(df, {})
    assert notes[0]["kind"] != "all_unique"


def test_cardinality_not_emitted_when_constant():
    # Object dtype, but constant → only constant note, not cardinality
    df = pd.DataFrame({"a": ["x", "x", "x"]})
    notes = scan_dataframe(df, {})
    assert len(notes) == 1
    assert notes[0]["kind"] == "constant"


def test_cardinality_not_emitted_when_all_unique():
    # Object dtype, all unique → only all_unique note, not cardinality
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    notes = scan_dataframe(df, {})
    assert len(notes) == 1
    assert notes[0]["kind"] == "all_unique"


# ── Unit: format_preflight_lines ──────────────────────────────────────────────

def test_constant_target_mentions_role_and_audit_warning():
    note = {
        "column": "target",
        "kind": "constant",
        "role": "target",
        "detail": "only 1 distinct value",
    }
    lines = format_preflight_lines([note])
    assert "(your target)" in lines[0]
    assert "audit cannot" in lines[0].lower() or "constant target" in lines[0].lower()


def test_constant_no_role_generic_message():
    note = {
        "column": "some_col",
        "kind": "constant",
        "role": None,
        "detail": "only 1 distinct value",
    }
    lines = format_preflight_lines([note])
    assert "1 distinct value" in lines[0]
    assert "(your" not in lines[0]


def test_constant_with_non_target_role_shows_role():
    note = {
        "column": "entity_id",
        "kind": "constant",
        "role": "entity_id",
        "detail": "only 1 distinct value",
    }
    lines = format_preflight_lines([note])
    assert "(your entity id)" in lines[0]


def test_all_unique_entity_id_panel_caveat():
    note = {
        "column": "entity_id",
        "kind": "all_unique",
        "role": "entity_id",
        "detail": "every value distinct (5 of 5 rows)",
    }
    lines = format_preflight_lines([note])
    assert "(your entity id)" in lines[0]
    assert "panel" in lines[0].lower() or "repeating entities" in lines[0].lower()


def test_all_unique_no_role_generic_message():
    note = {
        "column": "row_key",
        "kind": "all_unique",
        "role": None,
        "detail": "every value distinct (5 of 5 rows)",
    }
    lines = format_preflight_lines([note])
    assert "unique on every row" in lines[0]
    assert "(your" not in lines[0]


def test_cardinality_line_contains_distinct_count_and_ratio():
    note = {
        "column": "description",
        "kind": "cardinality",
        "role": None,
        "detail": "87 distinct values across 200 rows (43.5% unique)",
    }
    lines = format_preflight_lines([note])
    assert "87" in lines[0]
    assert "200" in lines[0]
    assert "43.5%" in lines[0]


def test_all_lines_are_single_line():
    notes = [
        {"column": "target", "kind": "constant", "role": "target", "detail": "only 1 distinct value"},
        {"column": "eid", "kind": "all_unique", "role": "entity_id", "detail": "every value distinct (10 of 10 rows)"},
        {"column": "desc", "kind": "cardinality", "role": None, "detail": "50 distinct values across 100 rows (50.0% unique)"},
    ]
    for line in format_preflight_lines(notes):
        assert "\n" not in line, f"Line contains newline: {line!r}"


def test_every_line_starts_with_note():
    notes = [
        {"column": "a", "kind": "constant", "role": None, "detail": "only 1 distinct value"},
        {"column": "b", "kind": "all_unique", "role": None, "detail": "every value distinct (3 of 3 rows)"},
        {"column": "c", "kind": "cardinality", "role": None, "detail": "2 distinct values across 5 rows (40.0% unique)"},
    ]
    for line in format_preflight_lines(notes):
        assert line.startswith("NOTE:"), f"Line does not start with 'NOTE:': {line!r}"


def test_empty_notes_returns_empty_lines():
    assert format_preflight_lines([]) == []


# ── CLI tests ─────────────────────────────────────────────────────────────────

def test_preflight_note_appears_for_constant_col(tmp_path):
    """Constant column → NOTE on stdout (non-json), exit code 0."""
    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df["const_col"] = 42

    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG, encoding="utf-8")

    result = runner.invoke(
        app,
        ["audit", "--data", str(csv), "--config", str(cfg), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "const_col" in result.output
    assert "only 1 distinct value" in result.output


def test_preflight_constant_col_does_not_change_exit_code(tmp_path):
    """A constant column must not cause a non-zero exit — warn-only."""
    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df["const_col"] = "fixed"

    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG, encoding="utf-8")

    result = runner.invoke(
        app,
        ["audit", "--data", str(csv), "--config", str(cfg), "--dry-run"],
    )
    # Pre-flight note must not alter the exit code
    assert result.exit_code == 0


def test_preflight_json_stdout_contains_no_note_text(tmp_path, monkeypatch):
    """Under --json, pre-flight notes go to stderr; stdout remains clean JSON."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    monkeypatch.setattr("zekan.severity.ablation._default_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df["const_col"] = 99  # will produce a NOTE

    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG, encoding="utf-8")

    result = runner.invoke(
        app,
        ["audit", "--data", str(csv), "--config", str(cfg), "--json"],
    )

    assert result.exit_code == 0, result.output
    assert "NOTE" not in result.stdout
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)


def test_preflight_json_stdout_still_byte_deterministic(tmp_path, monkeypatch):
    """With a constant column present, --json stdout is still byte-identical across runs."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    monkeypatch.setattr("zekan.severity.ablation._default_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    df["const_col"] = 0

    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG, encoding="utf-8")

    args = ["audit", "--data", str(csv), "--config", str(cfg), "--json"]
    r1 = runner.invoke(app, args)
    r2 = runner.invoke(app, args)

    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert r1.stdout == r2.stdout


def test_preflight_no_constant_or_all_unique_on_clean_data(tmp_path):
    """Dataset with binned integer features produces no CONSTANT or ALL_UNIQUE notes.

    Uses integer features (10 distinct values, 1200 rows) so no column is all_unique.
    entity_id and prediction_time (object dtype, low-cardinality) produce only
    informational cardinality notes, which are expected and not alarming.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    periods = [f"2022-{m:02d}-01" for m in range(1, 7)]
    rows = []
    for period in periods:
        for i in range(200):
            rows.append({
                "entity_id": f"entity_{i:03d}",
                "prediction_time": period,
                "feature1": int(rng.integers(1, 11)),  # 10 distinct int values
                "feature2": int(rng.integers(1, 11)),
                "target": int(rng.integers(0, 2)),
            })
    df = pd.DataFrame(rows)

    csv = tmp_path / "data.csv"
    cfg = tmp_path / "zekan.yml"
    df.to_csv(csv, index=False)
    cfg.write_text(_CLEAN_CONFIG, encoding="utf-8")

    result = runner.invoke(
        app,
        ["audit", "--data", str(csv), "--config", str(cfg), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "only 1 distinct value" not in result.output
    assert "is unique on every row" not in result.output
