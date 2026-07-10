"""Tests for load_config: BOM stripping, unknown-key error, and missing-field error."""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from zekan.config.schema import ZekanConfig, load_config


_MINIMAL_VALID = textwrap.dedent("""\
    contract:
      prediction_problem: test-problem
      entity_id: customer_id
      prediction_time: snapshot_date
      target: churned
      available_features_until: snapshot_date
""")


def test_bom_stripped_before_parse(tmp_path):
    """UTF-8 BOM prefix must be silently stripped — config loads without error."""
    cfg_file = tmp_path / "zekan.yml"
    cfg_file.write_bytes(b"\xef\xbb\xbf" + _MINIMAL_VALID.encode("utf-8"))
    cfg = load_config(cfg_file)
    assert isinstance(cfg, ZekanConfig)
    assert cfg.contract.prediction_problem == "test-problem"


def test_wrong_top_level_key_raises_helpful_error(tmp_path):
    """'prediction_contract' key → ValueError naming 'contract' in the message."""
    cfg_file = tmp_path / "zekan.yml"
    cfg_file.write_text(
        textwrap.dedent("""\
            prediction_contract:
              prediction_problem: test-problem
              entity_id: customer_id
              prediction_time: snapshot_date
              target: churned
              available_features_until: snapshot_date
        """),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did you mean 'contract'"):
        load_config(cfg_file)


def test_data_field_absent_defaults_to_none(tmp_path):
    """A config without 'data:' still loads cleanly, with cfg.data == None."""
    cfg_file = tmp_path / "zekan.yml"
    cfg_file.write_text(_MINIMAL_VALID, encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.data is None


def test_data_field_present_populates_cfg_data(tmp_path):
    """A config with 'data:' populates cfg.data with the given string."""
    cfg_file = tmp_path / "zekan.yml"
    cfg_file.write_text(_MINIMAL_VALID + "data: mydata.csv\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.data == "mydata.csv"


def test_missing_prediction_problem_names_field(tmp_path):
    """Valid 'contract' key but no prediction_problem → ValidationError naming the field."""
    cfg_file = tmp_path / "zekan.yml"
    cfg_file.write_text(
        textwrap.dedent("""\
            contract:
              entity_id: customer_id
              prediction_time: snapshot_date
              target: churned
              available_features_until: snapshot_date
        """),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as exc_info:
        load_config(cfg_file)
    assert "prediction_problem" in str(exc_info.value)
