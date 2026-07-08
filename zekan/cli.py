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
def diff(
    old: str = typer.Option(..., "--old", help="Path to old audit JSON artifact."),
    new: str = typer.Option(..., "--new", help="Path to new audit JSON artifact."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output structured JSON to stdout; send human-readable text to stderr.",
    ),
    fail_on_regression: bool = typer.Option(
        False,
        "--fail-on-regression",
        help="Exit non-zero if leakage increased between old and new.",
    ),
) -> None:
    """Compare two audit JSON artifacts and show the leakage delta."""
    import json as _json
    import sys
    from pathlib import Path

    from zekan.reports.diff import diff_reports

    def _load(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            typer.echo(f"ERROR: file not found: {p}", err=True)
            raise typer.Exit(1)
        try:
            return _json.loads(p.read_text(encoding="utf-8"))
        except _json.JSONDecodeError as exc:
            typer.echo(f"ERROR: invalid JSON in {p}: {exc}", err=True)
            raise typer.Exit(1)

    old_data = _load(old)
    new_data = _load(new)

    result = diff_reports(old_data, new_data)

    if json_output:
        typer.echo(_json.dumps(result, sort_keys=True, indent=2))
    else:
        _render_diff(result, err=False)

    if "schema_mismatch" in result:
        typer.echo(f"  Note: {result['schema_mismatch']}", err=json_output)

    if fail_on_regression and result["direction"] == "REGRESSED":
        raise typer.Exit(1)


def _render_diff(d: dict, *, err: bool = False) -> None:
    """Human-readable diff output: verdict-first, one-screen."""
    direction = d["direction"]
    fl_old = d["fl_old"]
    fl_new = d["fl_new"]
    fl_delta = d["fl_delta"]

    if direction == "UNVERIFIABLE_CHANGE":
        header = "UNVERIFIABLE_CHANGE — leakage comparison not possible (null on one or both sides)"
    elif fl_delta is not None:
        sign = "+" if fl_delta > 0 else ""
        header = (
            f"{direction} — leakage {fl_old:.4f} → {fl_new:.4f} "
            f"({sign}{fl_delta:.4f})"
        )
    else:
        header = direction

    typer.echo(header, err=err)

    headline_old = d.get("headline_old") or d.get("verdict_old") or "?"
    headline_new = d.get("headline_new") or d.get("verdict_new") or "?"
    verdict_tag = "" if d["verdict_changed"] else " (no change)"
    typer.echo(f"  Verdict: {headline_old} → {headline_new}{verdict_tag}", err=err)

    tf_old = d["top_feature_old"] or "(none)"
    tf_new = d["top_feature_new"] or "(none)"
    status = d["top_feature_status"]
    if status == "none":
        typer.echo("  Top cause: (none)", err=err)
    elif status == "fixed":
        typer.echo(f"  Fixed: {tf_old} (was top cause)", err=err)
    elif status == "appeared":
        typer.echo(f"  New top cause: {tf_new}", err=err)
    elif status == "same":
        typer.echo(f"  Top cause: {tf_old} (still present)", err=err)
    else:
        typer.echo(f"  Top cause: {tf_old} → {tf_new}", err=err)


@app.command()
def init(
    data: str = typer.Option(..., "--data", help="Path to dataset (CSV or Parquet)."),
    out: str = typer.Option("zekan.yml", "--out", help="Output path for the generated config YAML."),
) -> None:
    """Interactive wizard: inspect your dataset and write a zekan.yml contract."""
    from pathlib import Path

    import pandas as pd
    from pydantic import ValidationError

    from zekan.init_wizard import build_contract_mapping, validate_mapping, write_config

    # ── load data (same block as _run_audit_pipeline) ─────────────────────────
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

    cols = list(df.columns)
    if not cols:
        typer.echo("ERROR: dataset has no columns.", err=True)
        raise typer.Exit(1)

    # ── interactive helpers ────────────────────────────────────────────────────
    def _show_menu() -> None:
        typer.echo("  Columns:")
        for i, c in enumerate(cols):
            typer.echo(f"    [{i}] {c}")

    def _pick_one(field_name: str) -> str:
        while True:
            _show_menu()
            raw = typer.prompt(f"  Select {field_name} (0-{len(cols) - 1})")
            try:
                idx = int(raw.strip())
                if 0 <= idx < len(cols):
                    return cols[idx]
            except ValueError:
                pass
            typer.echo(f"  ERROR: enter a number between 0 and {len(cols) - 1}.")

    def _pick_many(field_name: str) -> list[str]:
        while True:
            _show_menu()
            raw = typer.prompt(
                f"  Select {field_name} (comma-separated numbers, empty for none)",
                default="",
            )
            raw = raw.strip()
            if not raw:
                return []
            try:
                parts = [x.strip() for x in raw.split(",") if x.strip()]
                indices = [int(x) for x in parts]
                if all(0 <= idx < len(cols) for idx in indices):
                    return [cols[idx] for idx in indices]
            except ValueError:
                pass
            typer.echo(
                f"  ERROR: enter comma-separated numbers between 0 and {len(cols) - 1},"
                " or leave empty."
            )

    # ── prompts ───────────────────────────────────────────────────────────────
    typer.echo(f"\nZekan init — {len(cols)} columns found in {data_path.name}")
    prediction_problem = typer.prompt("  Describe your prediction problem (free text)")
    entity_id = _pick_one("entity_id")
    prediction_time = _pick_one("prediction_time")
    target = _pick_one("target")
    available_features_until = _pick_one("available_features_until")
    forbidden_after_prediction = _pick_many("forbidden_after_prediction")

    # ── build + validate ───────────────────────────────────────────────────────
    mapping = build_contract_mapping(
        prediction_problem=prediction_problem,
        entity_id=entity_id,
        prediction_time=prediction_time,
        target=target,
        available_features_until=available_features_until,
        forbidden_after_prediction=forbidden_after_prediction,
    )
    try:
        validate_mapping(mapping)
    except ValidationError as exc:
        typer.echo(f"ERROR: contract validation failed:\n{exc}", err=True)
        raise typer.Exit(1)

    # ── no-clobber ────────────────────────────────────────────────────────────
    out_path = Path(out)
    if out_path.exists():
        overwrite = typer.confirm(f"  {out_path} already exists. Overwrite?", default=False)
        if not overwrite:
            typer.echo("Aborted. Existing file not changed.")
            raise typer.Exit(0)

    # ── write ─────────────────────────────────────────────────────────────────
    write_config(mapping, out_path)
    typer.echo(f"\nDone. Run:  zekan audit --data {data} --config {out}")


@app.command()
def benchmark() -> None:
    """Run the benchmark suite against known leakage fixtures."""
    typer.echo("not implemented yet")
