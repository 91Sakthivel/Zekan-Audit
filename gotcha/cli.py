"""Gotcha CLI — entry point for audit, benchmark, and report subcommands."""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(
    name="gotcha",
    help="Gotcha - local-first ML trust-audit tool.",
    no_args_is_help=True,
)

_WIDTH = 64


@app.command()
def audit(
    data: str = typer.Option(..., "--data", help="Path to dataset (CSV or Parquet)."),
    config: str = typer.Option(..., "--config", help="Path to gotcha config YAML."),
    model: Optional[str] = typer.Option(None, "--model", help="Path to serialised model artifact (Phase 2+)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate contract and stop; do not train or score."),
    fail_if_inflation_greater_than: Optional[float] = typer.Option(
        None,
        "--fail-if-inflation-greater-than",
        help="Exit non-zero if AUC inflation exceeds this threshold (Phase 2+).",
    ),
) -> None:
    """Audit a model for data-leakage and trust issues."""
    # Heavy imports are lazy so `gotcha --help` stays instant.
    from pathlib import Path

    import pandas as pd
    from pydantic import ValidationError

    from gotcha.config.schema import load_config
    from gotcha.contract.contract_checks import CheckStatus, validate_contract

    # ── load config ───────────────────────────────────────────────────────────
    config_path = Path(config)
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        typer.echo(f"ERROR: config not found: {config_path}", err=True)
        raise typer.Exit(1)
    except ValidationError as exc:
        typer.echo(f"ERROR: invalid config:\n{exc}", err=True)
        raise typer.Exit(1)

    # ── load data ─────────────────────────────────────────────────────────────
    data_path = Path(data)
    try:
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(data_path)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(data_path)
        else:
            typer.echo(f"ERROR: unsupported format '{data_path.suffix}' (use .csv or .parquet)", err=True)
            raise typer.Exit(1)
    except FileNotFoundError:
        typer.echo(f"ERROR: data file not found: {data_path}", err=True)
        raise typer.Exit(1)

    # ── validate contract ─────────────────────────────────────────────────────
    result = validate_contract(cfg.contract, df)

    _icon = {CheckStatus.PASS: "PASS", CheckStatus.FAIL: "FAIL", CheckStatus.WARN: "WARN"}
    typer.echo(f"\nGotcha audit: {cfg.contract.prediction_problem}")
    typer.echo("=" * _WIDTH)
    for check in result.checks:
        typer.echo(f"  [{_icon[check.status]}]  {check.name}: {check.message}")
    typer.echo("=" * _WIDTH)

    if result.passed and result.can_compute_severity:
        typer.echo("READY: contract valid, severity computable")
    elif result.passed:
        typer.echo(
            "CONTRACT VALID - severity cannot be computed "
            "(data too small or too few time periods; see warnings above)."
        )
    else:
        failed = [c.name for c in result.checks if c.status == CheckStatus.FAIL]
        typer.echo(f"CONTRACT FAILED: {', '.join(failed)}, severity will not be computed.")
        raise typer.Exit(1)

    if dry_run:
        return

    typer.echo("\n[Phase 2] Model training and severity detection not yet implemented.")


@app.command()
def benchmark() -> None:
    """Run the benchmark suite against known leakage fixtures."""
    typer.echo("not implemented yet")


@app.command()
def report() -> None:
    """Generate a trust-audit report from a previous audit run."""
    typer.echo("not implemented yet")
