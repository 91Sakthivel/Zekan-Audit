"""Tests for the gotcha CLI."""

from typer.testing import CliRunner

from gotcha.cli import app

runner = CliRunner()


def test_help_exits_zero_and_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.output
    assert "benchmark" in result.output
    assert "report" in result.output
