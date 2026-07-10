"""Tests for the zekan init wizard — unit (pure helpers) + CLI (interactive)."""

from __future__ import annotations

import textwrap

import pytest
from typer.testing import CliRunner

from zekan.cli import app
from zekan.config.schema import load_config
from zekan.init_wizard import build_contract_mapping, validate_mapping, write_config

runner = CliRunner()

# ── Shared fixture data ───────────────────────────────────────────────────────

_COLS = ["entity_id", "prediction_time", "target", "feature1", "leaky_col"]

# Tiny CSV whose column names match _COLS
_CSV_CONTENT = textwrap.dedent("""\
    entity_id,prediction_time,target,feature1,leaky_col
    e1,2023-01-01,0,1.0,0.5
    e2,2023-01-01,1,2.0,0.8
""")

# Input sequence for the happy-path wizard run (all valid, no forbidden features):
#   prediction_problem → "test-problem"
#   entity_id          → col 0
#   prediction_time    → col 1
#   target             → col 2
#   available_features_until → col 1
#   forbidden_after_prediction → empty (Enter)
_HAPPY_INPUT = "test-problem\n0\n1\n2\n1\n\n"


# ── Unit: build_contract_mapping ──────────────────────────────────────────────

def test_build_contract_mapping_correct_keys_and_values():
    m = build_contract_mapping("my-problem", "eid", "pt", "tgt", "afu", ["bad_col"])
    assert list(m.keys()) == ["contract"]
    c = m["contract"]
    assert c["prediction_problem"] == "my-problem"
    assert c["entity_id"] == "eid"
    assert c["prediction_time"] == "pt"
    assert c["target"] == "tgt"
    assert c["available_features_until"] == "afu"
    assert c["forbidden_after_prediction"] == ["bad_col"]


def test_build_contract_mapping_field_order():
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", [])
    keys = list(m["contract"].keys())
    assert keys == [
        "prediction_problem",
        "entity_id",
        "prediction_time",
        "target",
        "available_features_until",
        "forbidden_after_prediction",
    ]


def test_build_contract_mapping_empty_forbidden_included():
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", [])
    assert "forbidden_after_prediction" in m["contract"]
    assert m["contract"]["forbidden_after_prediction"] == []


def test_build_contract_mapping_with_forbidden():
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", ["x", "y"])
    assert m["contract"]["forbidden_after_prediction"] == ["x", "y"]


def test_build_contract_mapping_data_path_omitted_by_default():
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", [])
    assert "data" not in m


def test_build_contract_mapping_data_path_included_when_given():
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", [], data_path="data.csv")
    assert m["data"] == "data.csv"
    assert list(m.keys()) == ["contract", "data"]


# ── Unit: validate_mapping ────────────────────────────────────────────────────

def test_validate_mapping_accepts_good_mapping():
    m = build_contract_mapping("p", "entity", "time", "label", "time", [])
    validate_mapping(m)  # must not raise


def test_validate_mapping_raises_on_missing_required_field():
    from pydantic import ValidationError
    bad = {"contract": {"prediction_problem": "p"}}  # missing entity_id etc.
    with pytest.raises(ValidationError):
        validate_mapping(bad)


def test_validate_mapping_raises_on_missing_contract_key():
    from pydantic import ValidationError
    with pytest.raises((ValidationError, TypeError)):
        validate_mapping({})  # no 'contract' key at all


# ── Unit: write_config ────────────────────────────────────────────────────────

def test_write_config_bom_free(tmp_path):
    out = tmp_path / "zekan.yml"
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", [])
    write_config(m, out)
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "File must not have a UTF-8 BOM"


def test_write_config_round_trip(tmp_path):
    out = tmp_path / "zekan.yml"
    m = build_contract_mapping("round-trip", "entity_id", "prediction_time", "target", "prediction_time", [])
    write_config(m, out)
    cfg = load_config(out)
    assert cfg.contract.prediction_problem == "round-trip"
    assert cfg.contract.entity_id == "entity_id"
    assert cfg.contract.forbidden_after_prediction == []


def test_write_config_empty_forbidden_renders_as_list(tmp_path):
    out = tmp_path / "zekan.yml"
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", [])
    write_config(m, out)
    text = out.read_text(encoding="utf-8")
    assert "forbidden_after_prediction" in text
    # PyYAML renders empty list as [] in flow style
    assert "forbidden_after_prediction: []" in text


def test_write_config_field_order_preserved(tmp_path):
    out = tmp_path / "zekan.yml"
    m = build_contract_mapping("p", "e", "t", "tgt", "afu", [])
    write_config(m, out)
    text = out.read_text(encoding="utf-8")
    # prediction_problem must appear before entity_id in the file
    assert text.index("prediction_problem") < text.index("entity_id")
    assert text.index("entity_id") < text.index("prediction_time")


# ── CLI: happy path ───────────────────────────────────────────────────────────

def test_init_happy_path_exit_0_file_created(tmp_path):
    """Full wizard run on a tiny CSV: exit 0, file created, loads cleanly."""
    csv = tmp_path / "data.csv"
    out = tmp_path / "zekan.yml"
    csv.write_text(_CSV_CONTENT, encoding="utf-8")

    result = runner.invoke(
        app,
        ["init", "--data", str(csv), "--out", str(out)],
        input=_HAPPY_INPUT,
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    cfg = load_config(out)
    assert cfg.contract.prediction_problem == "test-problem"
    assert cfg.contract.entity_id == "entity_id"
    assert cfg.contract.prediction_time == "prediction_time"
    assert cfg.contract.target == "target"
    assert cfg.contract.forbidden_after_prediction == []


def test_init_happy_path_prints_done_and_audit_hint(tmp_path):
    """Done message includes 'zekan audit' hint."""
    csv = tmp_path / "data.csv"
    out = tmp_path / "zekan.yml"
    csv.write_text(_CSV_CONTENT, encoding="utf-8")

    result = runner.invoke(
        app,
        ["init", "--data", str(csv), "--out", str(out)],
        input=_HAPPY_INPUT,
    )

    assert "Done" in result.output
    assert "zekan audit" in result.output


def test_init_with_forbidden_selection(tmp_path):
    """Selecting a forbidden column is captured correctly."""
    csv = tmp_path / "data.csv"
    out = tmp_path / "zekan.yml"
    csv.write_text(_CSV_CONTENT, encoding="utf-8")

    # forbidden = col 4 (leaky_col)
    wizard_input = "test-forbidden\n0\n1\n2\n1\n4\n"
    result = runner.invoke(
        app,
        ["init", "--data", str(csv), "--out", str(out)],
        input=wizard_input,
    )

    assert result.exit_code == 0, result.output
    cfg = load_config(out)
    assert cfg.contract.forbidden_after_prediction == ["leaky_col"]


# ── CLI: no-clobber ───────────────────────────────────────────────────────────

def test_init_no_clobber_declined_file_unchanged(tmp_path):
    """When the output file already exists and user declines, it is not overwritten."""
    csv = tmp_path / "data.csv"
    out = tmp_path / "zekan.yml"
    csv.write_text(_CSV_CONTENT, encoding="utf-8")

    original_content = "# sentinel\n"
    out.write_text(original_content, encoding="utf-8")

    # wizard answers + "n" to the overwrite confirm
    no_clobber_input = _HAPPY_INPUT + "n\n"
    result = runner.invoke(
        app,
        ["init", "--data", str(csv), "--out", str(out)],
        input=no_clobber_input,
    )

    assert out.read_text(encoding="utf-8") == original_content, (
        "File must not have been changed when overwrite was declined"
    )
    assert "Aborted" in result.output


# ── CLI: init → audit round trip ──────────────────────────────────────────────

def test_init_writes_relative_data_path(tmp_path):
    """init writes a 'data:' key relative to the config's own directory."""
    csv = tmp_path / "data.csv"
    out = tmp_path / "zekan.yml"
    csv.write_text(_CSV_CONTENT, encoding="utf-8")

    result = runner.invoke(
        app,
        ["init", "--data", str(csv), "--out", str(out)],
        input=_HAPPY_INPUT,
    )

    assert result.exit_code == 0, result.output
    cfg = load_config(out)
    assert cfg.data is not None
    # Resolved against the config's own directory, it must land back on the same file.
    resolved = (out.parent / cfg.data).resolve()
    assert resolved == csv.resolve()


def test_init_then_bare_audit_uses_same_file(tmp_path, monkeypatch):
    """The full zero-flag arc: init writes data:, then a bare `zekan audit` finds it."""
    from zekan.benchmark.fixtures import make_clean_dataset
    from sklearn.ensemble import RandomForestClassifier

    def _fast_clf():
        return RandomForestClassifier(n_estimators=5, random_state=0)

    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    monkeypatch.chdir(tmp_path)

    df = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=1)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    # Columns: entity_id(0), prediction_time(1), feature_0..feature_5(2-7), target(8).
    assert list(df.columns)[-1] == "target"
    target_idx = len(df.columns) - 1
    init_result = runner.invoke(
        app,
        ["init", "--data", "data.csv"],
        input=f"round-trip-problem\n0\n1\n{target_idx}\n1\n\n",
    )
    assert init_result.exit_code == 0, init_result.output
    assert (tmp_path / "zekan.yml").exists()

    audit_result = runner.invoke(app, ["audit", "--dry-run"])

    assert audit_result.exit_code == 0, audit_result.output
    assert "READY" in audit_result.output
    # No --json: the echo goes to stdout (err=json_mode), and confirms which file was used.
    assert "(from zekan.yml)" in audit_result.output
    assert "data.csv" in audit_result.output
