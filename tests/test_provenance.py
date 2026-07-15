"""Tests for zekan/reports/provenance.py — unit + CLI integration."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from typer.testing import CliRunner

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.cli import app
from zekan.reports.provenance import (
    build_manifest,
    build_provenance,
    capture_versions,
    hash_contract,
    hash_dataframe,
    read_estimator_random_state,
    write_manifest,
)

runner = CliRunner()

_CLEAN_CONFIG = textwrap.dedent("""\
    contract:
      prediction_problem: prov-test
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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


@pytest.fixture()
def simple_provenance() -> dict:
    return build_provenance(
        data_hash="abc123",
        contract_hash="def456",
        versions={"zekan": "0.1.0", "numpy": "1.26.0", "pandas": "2.0.0", "scikit_learn": "1.3.0"},
        null_seed=0,
        estimator_identity="default",
        estimator_random_state=None,
    )


# ── hash_dataframe ─────────────────────────────────────────────────────────────

def test_hash_dataframe_returns_hex_string(tiny_df):
    h = hash_dataframe(tiny_df)
    assert isinstance(h, str)
    assert len(h) == 64
    int(h, 16)  # must be valid hex


def test_hash_dataframe_is_deterministic(tiny_df):
    assert hash_dataframe(tiny_df) == hash_dataframe(tiny_df)


def test_hash_dataframe_changes_on_row_change(tiny_df):
    df2 = tiny_df.copy()
    df2.loc[0, "a"] = 999
    assert hash_dataframe(tiny_df) != hash_dataframe(df2)



# ── hash_contract ──────────────────────────────────────────────────────────────

def test_hash_contract_returns_hex_string(tmp_path):
    from zekan.config.schema import load_config

    cfg_text = textwrap.dedent("""\
        contract:
          prediction_problem: test-problem
          entity_id: entity_id
          prediction_time: prediction_time
          target: target
          available_features_until: prediction_time
          forbidden_after_prediction: []
    """)
    cfg_path = tmp_path / "zekan.yml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(cfg_path)

    h = hash_contract(cfg.contract)
    assert isinstance(h, str)
    assert len(h) == 64


def test_hash_contract_is_deterministic(tmp_path):
    from zekan.config.schema import load_config

    cfg_text = textwrap.dedent("""\
        contract:
          prediction_problem: test-problem
          entity_id: entity_id
          prediction_time: prediction_time
          target: target
          available_features_until: prediction_time
          forbidden_after_prediction: []
    """)
    cfg_path = tmp_path / "zekan.yml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(cfg_path)

    assert hash_contract(cfg.contract) == hash_contract(cfg.contract)


# ── capture_versions ───────────────────────────────────────────────────────────

def test_capture_versions_has_required_keys():
    v = capture_versions()
    assert set(v) == {"numpy", "pandas", "scikit_learn", "zekan"}


def test_capture_versions_values_are_strings():
    v = capture_versions()
    for val in v.values():
        assert isinstance(val, str)


def test_capture_versions_zekan_matches_package():
    import zekan as _zekan
    v = capture_versions()
    assert v["zekan"] == _zekan.__version__


# ── read_estimator_random_state ────────────────────────────────────────────────

def test_read_estimator_random_state_none_for_none():
    assert read_estimator_random_state(None) is None


def test_read_estimator_random_state_reads_rf():
    from sklearn.ensemble import RandomForestClassifier
    factory = lambda: RandomForestClassifier(n_estimators=10, random_state=42)
    assert read_estimator_random_state(factory) == 42


def test_read_estimator_random_state_reads_logistic():
    from sklearn.linear_model import LogisticRegression
    factory = lambda: LogisticRegression(random_state=7, max_iter=100)
    assert read_estimator_random_state(factory) == 7


def test_read_estimator_random_state_returns_none_on_no_attr():
    factory = lambda: object()  # no random_state
    assert read_estimator_random_state(factory) is None


def test_read_estimator_random_state_returns_none_on_exception():
    def bad_factory():
        raise RuntimeError("boom")
    assert read_estimator_random_state(bad_factory) is None


# ── build_provenance ───────────────────────────────────────────────────────────

def test_build_provenance_keys(simple_provenance):
    assert set(simple_provenance) == {
        "data_sha256", "contract_sha256", "estimator_identity", "seed", "versions"
    }


def test_build_provenance_seed_subkeys(simple_provenance):
    # re-baselined: F2a adds null_scheme (additive) to the seed sub-dict
    assert set(simple_provenance["seed"]) == {
        "null_seed", "estimator_random_state", "null_scheme",
    }


def test_build_provenance_no_timestamp(simple_provenance):
    text = json.dumps(simple_provenance)
    assert "created_at" not in text
    assert "timestamp" not in text


def test_build_provenance_estimator_identity(simple_provenance):
    assert simple_provenance["estimator_identity"] == "default"


def test_build_provenance_custom_estimator():
    p = build_provenance("h1", "h2", {}, 0, "rf", 42)
    assert p["estimator_identity"] == "rf"
    assert p["seed"]["estimator_random_state"] == 42


def test_build_provenance_is_json_serialisable(simple_provenance):
    text = json.dumps(simple_provenance, sort_keys=True)
    assert json.loads(text) == simple_provenance


def test_build_provenance_null_scheme_default(simple_provenance):
    assert simple_provenance["seed"]["null_scheme"] == "spawn_v2"


def test_build_provenance_null_scheme_custom():
    p = build_provenance("h1", "h2", {}, 0, "rf", 42, null_scheme="serial_v1")
    assert p["seed"]["null_scheme"] == "serial_v1"


# ── build_manifest ─────────────────────────────────────────────────────────────

def test_build_manifest_has_created_at(simple_provenance):
    m = build_manifest(simple_provenance)
    assert "created_at" in m


def test_build_manifest_created_at_is_utc_iso(simple_provenance):
    m = build_manifest(simple_provenance)
    ts = m["created_at"]
    assert "T" in ts
    assert "+00:00" in ts or ts.endswith("Z") or "UTC" in ts


def test_build_manifest_provenance_unchanged(simple_provenance):
    m = build_manifest(simple_provenance)
    assert m["provenance"] == simple_provenance


# ── write_manifest ─────────────────────────────────────────────────────────────

def test_write_manifest_creates_file(tmp_path, simple_provenance):
    out = tmp_path / "manifest.json"
    write_manifest(build_manifest(simple_provenance), out)
    assert out.exists()


def test_write_manifest_bom_free(tmp_path, simple_provenance):
    out = tmp_path / "manifest.json"
    write_manifest(build_manifest(simple_provenance), out)
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")


def test_write_manifest_valid_json(tmp_path, simple_provenance):
    out = tmp_path / "manifest.json"
    write_manifest(build_manifest(simple_provenance), out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "created_at" in loaded
    assert "provenance" in loaded


# ── CLI: --manifest flag ───────────────────────────────────────────────────────

@pytest.fixture()
def audit_assets(tmp_path, monkeypatch):
    """1200-row CSV (200 entities × 6 snapshots) + config; model factory patched to 5-tree RF."""
    monkeypatch.setattr("zekan.severity.metrics._default_model_factory", _fast_clf)
    monkeypatch.setattr("zekan.severity.ablation._default_factory", _fast_clf)

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)
    csv_path = tmp_path / "data.csv"
    cfg_path = tmp_path / "zekan.yml"
    df.to_csv(csv_path, index=False)
    cfg_path.write_text(_CLEAN_CONFIG, encoding="utf-8")
    return str(csv_path), str(cfg_path)


def test_audit_json_contains_provenance_key(audit_assets):
    """--json output includes a 'provenance' key."""
    data, config = audit_assets
    result = runner.invoke(
        app,
        ["audit", "--data", data, "--config", config, "--json"],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert "provenance" in parsed


def test_audit_json_provenance_has_expected_keys(audit_assets):
    data, config = audit_assets
    result = runner.invoke(
        app,
        ["audit", "--data", data, "--config", config, "--json"],
    )
    assert result.exit_code == 0, result.output
    prov = json.loads(result.stdout)["provenance"]
    assert set(prov) == {"data_sha256", "contract_sha256", "estimator_identity", "seed", "versions"}


def test_audit_json_provenance_no_timestamp(audit_assets):
    """The --json body must never contain a timestamp."""
    data, config = audit_assets
    result = runner.invoke(
        app,
        ["audit", "--data", data, "--config", config, "--json"],
    )
    body = result.stdout
    assert "created_at" not in body


def test_audit_json_is_deterministic(audit_assets):
    """Running --json twice produces byte-identical stdout."""
    data, config = audit_assets
    r1 = runner.invoke(app, ["audit", "--data", data, "--config", config, "--json"])
    r2 = runner.invoke(app, ["audit", "--data", data, "--config", config, "--json"])
    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert r1.stdout == r2.stdout


def test_audit_manifest_file_created(tmp_path, audit_assets):
    data, config = audit_assets
    manifest_file = str(tmp_path / "prov.json")
    result = runner.invoke(
        app,
        ["audit", "--data", data, "--config", config, "--json", "--manifest", manifest_file],
    )
    assert result.exit_code == 0, result.output
    assert Path(manifest_file).exists()


def test_audit_manifest_contains_timestamp(tmp_path, audit_assets):
    data, config = audit_assets
    manifest_file = str(tmp_path / "prov.json")
    runner.invoke(
        app,
        ["audit", "--data", data, "--config", config, "--json", "--manifest", manifest_file],
    )
    loaded = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    assert "created_at" in loaded
    assert "T" in loaded["created_at"]


def test_audit_manifest_provenance_matches_json_body(tmp_path, audit_assets):
    """Manifest's provenance block must equal the provenance in --json output."""
    data, config = audit_assets
    manifest_file = str(tmp_path / "prov.json")
    result = runner.invoke(
        app,
        ["audit", "--data", data, "--config", config, "--json", "--manifest", manifest_file],
    )
    json_prov = json.loads(result.stdout)["provenance"]
    manifest_prov = json.loads(Path(manifest_file).read_text(encoding="utf-8"))["provenance"]
    assert json_prov == manifest_prov


def test_audit_no_manifest_flag_no_file(tmp_path, audit_assets):
    """Without --manifest, no manifest file is written."""
    data, config = audit_assets
    manifest_file = tmp_path / "should_not_exist.json"
    runner.invoke(app, ["audit", "--data", data, "--config", config, "--json"])
    assert not manifest_file.exists()
