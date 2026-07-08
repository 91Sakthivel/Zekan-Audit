"""Pre-flight data sanity check — pure, no typer, no side effects.

Returns data; the caller is responsible for all I/O.
Three deterministic checks, warn-only — never changes exit code or verdict.
"""

from __future__ import annotations

from typing import Optional


def scan_dataframe(df, declared_columns: dict[str, str]) -> list[dict]:
    """Scan df columns for structural anomalies.

    declared_columns: {column_name -> role_label}.  Built from cfg.contract by the caller.
    Returns notes in df.columns order; at most one note per column.

    Checks (in priority order, mutually exclusive per column):
      constant    — nunique(dropna=False) == 1
      all_unique  — nunique(dropna=False) == len(df)  (only when len(df) > 1)
      cardinality — object/string dtype only; not constant and not all_unique
    """
    notes: list[dict] = []
    n = len(df)

    for col in df.columns:
        role: Optional[str] = declared_columns.get(col)
        nunique = int(df[col].nunique(dropna=False))

        if nunique == 1:
            notes.append({
                "column": col,
                "kind": "constant",
                "role": role,
                "detail": "only 1 distinct value",
            })

        elif n > 1 and nunique == n:
            notes.append({
                "column": col,
                "kind": "all_unique",
                "role": role,
                "detail": f"every value distinct ({n} of {n} rows)",
            })

        else:
            # Cardinality note for object/string-like dtype only.
            # dtype.kind == "O" covers numpy object arrays and all pandas
            # StringDtype variants (pandas 2.x/3.x where str(dtype) == "str").
            if df[col].dtype.kind != "O":
                continue
            pct = round(100.0 * nunique / n, 1) if n > 0 else 0.0
            notes.append({
                "column": col,
                "kind": "cardinality",
                "role": role,
                "detail": f"{nunique} distinct values across {n} rows ({pct}% unique)",
            })

    return notes


def format_preflight_lines(notes: list[dict]) -> list[str]:
    """Turn note dicts into single human-readable lines starting with 'NOTE:'."""
    lines: list[str] = []

    for note in notes:
        col: str = note["column"]
        kind: str = note["kind"]
        role: Optional[str] = note.get("role")
        detail: str = note["detail"]
        role_display: Optional[str] = role.replace("_", " ") if role else None

        if kind == "constant":
            if role == "target":
                line = (
                    f"NOTE: column '{col}' (your target) has only 1 distinct value"
                    f" — the audit cannot learn anything from a constant target."
                    f" Check your data."
                )
            elif role_display:
                line = (
                    f"NOTE: column '{col}' (your {role_display}) has only 1 distinct value"
                    f" (constant). It carries no signal; consider removing it."
                )
            else:
                line = (
                    f"NOTE: column '{col}' has only 1 distinct value (constant)."
                    f" It carries no signal; consider removing it."
                )

        elif kind == "all_unique":
            if role == "entity_id":
                line = (
                    f"NOTE: column '{col}' (your entity id) is unique on every row"
                    f" — expected for a pure ID, but if this is meant to be panel data"
                    f" with repeating entities, your entity column may be wrong."
                )
            elif role_display:
                line = (
                    f"NOTE: column '{col}' (your {role_display}) is unique on every row"
                    f" (ID-like). If it's an identifier it should likely be excluded from features."
                )
            else:
                line = (
                    f"NOTE: column '{col}' is unique on every row (ID-like)."
                    f" If it's an identifier it should likely be excluded from features."
                )

        else:  # cardinality
            line = f"NOTE: column '{col}' has {detail}."

        lines.append(line)

    return lines
