"""Zekan CLI — entry point for audit, benchmark, and report subcommands."""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(
    name="zekan",
    help="Zekan - local-first ML trust-audit tool.",
    no_args_is_help=True,
)

_WIDTH = 64


# ── Shared pipeline ───────────────────────────────────────────────────────────

def _run_audit_pipeline(
    data: str,
    config: str,
    dry_run: bool = False,
    json_mode: bool = False,
    model_factory=None,
) -> Optional[object]:  # VerdictReport | None; VerdictReport imported lazily
    """Load config+data, validate contract, run audit. Returns None on early stop.

    Prints the contract check table and status messages in all cases.
    Raises typer.Exit(1) on contract failure or load errors.
    Returns None when cannot-compute or dry_run stops early (caller should return).
    """
    from pathlib import Path

    import pandas as pd
    from pydantic import ValidationError

    from zekan.config.schema import load_config
    from zekan.contract.contract_checks import CheckStatus, validate_contract

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
    except ValueError as e:
        typer.echo(f"ERROR: invalid config: {e}", err=json_mode)
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
    typer.echo(f"\nZekan audit: {cfg.contract.prediction_problem}", err=json_mode)
    typer.echo("=" * _WIDTH, err=json_mode)
    for check in result.checks:
        typer.echo(f"  [{_icon[check.status]}]  {check.name}: {check.message}", err=json_mode)
    typer.echo("=" * _WIDTH, err=json_mode)

    if result.passed and result.can_compute_severity:
        typer.echo("READY: contract valid, severity computable", err=json_mode)
    elif result.passed:
        typer.echo(
            "CONTRACT VALID - severity cannot be computed "
            "(data too small or too few time periods; see warnings above).",
            err=json_mode,
        )
        return None
    else:
        failed = [c.name for c in result.checks if c.status == CheckStatus.FAIL]
        typer.echo(
            f"CONTRACT FAILED: {', '.join(failed)}, severity will not be computed.",
            err=json_mode,
        )
        raise typer.Exit(1)

    if dry_run:
        return None

    from zekan.severity.audit import run_audit

    return run_audit(df, cfg.contract, cfg, model_factory=model_factory)


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def audit(
    data: str = typer.Option(..., "--data", help="Path to dataset (CSV or Parquet)."),
    config: str = typer.Option(..., "--config", help="Path to zekan config YAML."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate contract and stop; do not train or score."),
    fail_if_inflation_greater_than: Optional[float] = typer.Option(
        None,
        "--fail-if-inflation-greater-than",
        help="Exit non-zero if AUC inflation exceeds this threshold.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output structured JSON to stdout; send all human-readable text to stderr.",
    ),
    estimator: Optional[str] = typer.Option(
        None,
        "--estimator",
        help="Classifier for leakage detection. Choices: extra_trees, gbm, logistic, rf. Default: rf (200-tree random forest).",
    ),
) -> None:
    """Audit a model for data-leakage and trust issues."""
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    model_factory = None
    if estimator is not None:
        from zekan.severity.estimators import _build_factory
        model_factory = _build_factory(estimator)

    audit_report = _run_audit_pipeline(data, config, dry_run=dry_run, json_mode=json_output, model_factory=model_factory)
    if audit_report is None:
        if fail_if_inflation_greater_than is not None and not dry_run:
            # None + not dry_run == cannot-compute → UNVERIFIABLE, fail-safe
            typer.echo(
                "Inflation gate: UNVERIFIABLE — inflation threshold was requested "
                "but leakage could not be computed (data too small / too few periods); "
                "cannot certify build.",
                err=json_output,
            )
            raise typer.Exit(1)
        return  # dry-run, or no gate requested → unchanged soft stop

    from zekan.reports.text_view import render_verdict

    if not json_output:
        typer.echo("")
        typer.echo(render_verdict(audit_report, stream=sys.stdout).rstrip())

    gate_block = None
    should_exit_1 = False

    if fail_if_inflation_greater_than is not None:
        import math
        fl = audit_report.measured_damage.fixable_leakage
        if fl is None or math.isnan(fl):
            typer.echo(
                "Inflation gate: UNVERIFIABLE — leakage not computed; cannot "
                "certify build.",
                err=json_output,
            )
            gate_block = {
                "exit_code": 1,
                "threshold": fail_if_inflation_greater_than,
                "triggered": None,
            }
            should_exit_1 = True
        elif fl > fail_if_inflation_greater_than:
            typer.echo(
                f"Inflation gate: FAIL — inflation {fl:.4f} exceeds threshold "
                f"{fail_if_inflation_greater_than}.",
                err=json_output,
            )
            gate_block = {
                "exit_code": 1,
                "threshold": fail_if_inflation_greater_than,
                "triggered": True,
            }
            should_exit_1 = True
        else:
            typer.echo(
                f"Inflation gate: PASS — inflation {fl:.4f} within threshold "
                f"{fail_if_inflation_greater_than}.",
                err=json_output,
            )
            gate_block = {
                "exit_code": 0,
                "threshold": fail_if_inflation_greater_than,
                "triggered": False,
            }

    if json_output:
        import json as _json
        from zekan.reports.json_export import verdict_to_dict
        d = verdict_to_dict(audit_report)
        d["gate"] = gate_block
        typer.echo(_json.dumps(d, sort_keys=True, indent=2))

    if should_exit_1:
        raise typer.Exit(1)


@app.command()
def report(
    data: str = typer.Option(..., "--data", help="Path to dataset (CSV or Parquet)."),
    config: str = typer.Option(..., "--config", help="Path to zekan config YAML."),
    output: str = typer.Option(..., "--output", help="Path to write the HTML report to."),
    estimator: Optional[str] = typer.Option(
        None,
        "--estimator",
        help="Classifier for leakage detection. Choices: extra_trees, gbm, logistic, rf. Default: rf (200-tree random forest).",
    ),
) -> None:
    """Run the audit and write an HTML report to a file."""
    from pathlib import Path

    model_factory = None
    if estimator is not None:
        from zekan.severity.estimators import _build_factory
        model_factory = _build_factory(estimator)

    audit_report = _run_audit_pipeline(data, config, model_factory=model_factory)
    if audit_report is None:
        return

    from zekan.reports.html_view import render_verdict_html

    output_path = Path(output)
    output_path.write_text(render_verdict_html(audit_report), encoding="utf-8")
    typer.echo(f"Report written to {output_path}")


@app.command()
def benchmark() -> None:
    """Run the benchmark suite against known leakage fixtures."""
    typer.echo("not implemented yet")
