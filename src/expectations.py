"""
expectations.py
---------------
Great Expectations (GX) Data Quality suite for geospatial asset data.

What Great Expectations does
-----------------------------
Where Pydantic validates ONE ROW at a time (schema enforcement),
Great Expectations validates the WHOLE DATASET at once (statistical
& completeness assertions). Think of it as the difference between
a bouncer checking each person's ID vs. an auditor reviewing the
entire guest list.

Checks implemented
------------------
COMPLETENESS  → no nulls in capacity_mw
UNIQUENESS    → no duplicate asset IDs
VALIDITY      → latitude ∈ [-90, 90], longitude ∈ [-180, 180]
CONSISTENCY   → capacity_mw ≥ 0, commissioning_year ∈ [1900, 2100]
VOLUME        → dataset contains at least 1 row

Each check maps to a named GX Expectation and feeds into the final
Data Quality Dimensions report.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpectationResult:
    """Result of a single GX expectation."""
    expectation:  str
    dimension:    str          # Completeness | Uniqueness | Validity | Consistency
    success:      bool
    column:       str
    observed:     Any          # what we actually measured
    threshold:    Any          # what we required
    detail:       str = ""


@dataclass
class QAReport:
    """Aggregated report across all expectations."""
    run_time:     str
    total_rows:   int
    results:      list[ExpectationResult] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    def by_dimension(self) -> dict[str, dict]:
        """Group results by Data Quality Dimension."""
        dims: dict[str, list] = {}
        for r in self.results:
            dims.setdefault(r.dimension, []).append(r)
        return {
            dim: {
                "checks":  len(items),
                "passed":  sum(1 for i in items if i.success),
                "failed":  sum(1 for i in items if not i.success),
                "score":   sum(1 for i in items if i.success) / len(items),
                "details": items,
            }
            for dim, items in dims.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# GX-style expectation runners
# (each function mirrors a real GX expectation by name)
# ─────────────────────────────────────────────────────────────────────────────

def expect_column_values_to_not_be_null(
    df: pd.DataFrame, column: str, threshold: float = 1.0
) -> ExpectationResult:
    """
    COMPLETENESS check.
    Passes if the fraction of non-null values ≥ threshold (default: 100%).

    GX equivalent: expect_column_values_to_not_be_null
    """
    if column not in df.columns:
        return ExpectationResult(
            expectation="expect_column_values_to_not_be_null",
            dimension="Completeness",
            success=False,
            column=column,
            observed="column missing",
            threshold=threshold,
            detail=f"Column '{column}' does not exist in the dataset.",
        )

    null_count   = df[column].isna().sum()
    total        = len(df)
    completeness = (total - null_count) / total if total > 0 else 0.0
    success      = completeness >= threshold

    return ExpectationResult(
        expectation="expect_column_values_to_not_be_null",
        dimension="Completeness",
        success=success,
        column=column,
        observed=round(completeness, 4),
        threshold=threshold,
        detail=(
            f"{null_count:,} / {total:,} values are null "
            f"({100*(1-completeness):.1f}% missing)."
        ),
    )


def expect_column_values_to_be_unique(
    df: pd.DataFrame, column: str
) -> ExpectationResult:
    """
    UNIQUENESS check.
    Passes if every value in `column` is unique (no duplicates).

    GX equivalent: expect_column_values_to_be_unique
    """
    if column not in df.columns:
        return ExpectationResult(
            expectation="expect_column_values_to_be_unique",
            dimension="Uniqueness",
            success=False,
            column=column,
            observed="column missing",
            threshold="100% unique",
            detail=f"Column '{column}' does not exist in the dataset.",
        )

    total      = len(df)
    n_dupes    = df[column].duplicated(keep=False).sum()
    n_unique   = df[column].nunique()
    uniqueness = 1 - (n_dupes / total) if total > 0 else 1.0
    success    = n_dupes == 0

    # Surface the actual duplicate values (up to 5) for the report
    dup_vals = (
        df[df[column].duplicated(keep=False)][column]
        .value_counts()
        .head(5)
        .to_dict()
    )
    detail = (
        f"{n_dupes:,} duplicate rows across {total - n_unique:,} non-unique values."
        + (f" Sample duplicates: {dup_vals}" if dup_vals else "")
    )

    return ExpectationResult(
        expectation="expect_column_values_to_be_unique",
        dimension="Uniqueness",
        success=success,
        column=column,
        observed=round(uniqueness, 4),
        threshold=1.0,
        detail=detail,
    )


def expect_column_values_to_be_between(
    df: pd.DataFrame,
    column: str,
    min_value: float,
    max_value: float,
    dimension: str = "Validity",
) -> ExpectationResult:
    """
    VALIDITY / CONSISTENCY check.
    Passes if all non-null values in `column` fall within [min_value, max_value].

    GX equivalent: expect_column_values_to_be_between
    """
    if column not in df.columns:
        return ExpectationResult(
            expectation="expect_column_values_to_be_between",
            dimension=dimension,
            success=False,
            column=column,
            observed="column missing",
            threshold=f"[{min_value}, {max_value}]",
            detail=f"Column '{column}' does not exist in the dataset.",
        )

    sub      = df[column].dropna()
    out_mask = (sub < min_value) | (sub > max_value)
    n_out    = out_mask.sum()
    success  = n_out == 0

    return ExpectationResult(
        expectation="expect_column_values_to_be_between",
        dimension=dimension,
        success=success,
        column=column,
        observed=f"min={sub.min():.2f}, max={sub.max():.2f}",
        threshold=f"[{min_value}, {max_value}]",
        detail=(
            f"{n_out:,} values outside [{min_value}, {max_value}]."
            if n_out > 0
            else f"All {len(sub):,} values within range."
        ),
    )


def expect_table_row_count_to_be_between(
    df: pd.DataFrame, min_value: int = 1, max_value: int = 10_000_000
) -> ExpectationResult:
    """
    VOLUME check.
    Passes if the row count falls within [min_value, max_value].

    GX equivalent: expect_table_row_count_to_be_between
    """
    n      = len(df)
    success = min_value <= n <= max_value
    return ExpectationResult(
        expectation="expect_table_row_count_to_be_between",
        dimension="Completeness",
        success=success,
        column="(table)",
        observed=n,
        threshold=f"[{min_value}, {max_value}]",
        detail=f"Dataset contains {n:,} rows.",
    )


def expect_column_values_to_not_be_null_pct(
    df: pd.DataFrame, column: str, mostly: float = 0.95
) -> ExpectationResult:
    """
    COMPLETENESS check with a tolerance.
    Passes if at least `mostly` fraction of values are non-null.
    Useful for optional fields like commissioning_year.
    """
    result = expect_column_values_to_not_be_null(df, column, threshold=mostly)
    result.threshold = mostly
    result.detail    = f"Required ≥ {mostly*100:.0f}% complete. " + result.detail
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Suite runner — runs all expectations and builds a QAReport
# ─────────────────────────────────────────────────────────────────────────────

def run_expectation_suite(
    df:          pd.DataFrame,
    id_col:      str = "gppd_idnr",
    cap_col:     str = "capacity_mw",
    lat_col:     str = "latitude",
    lon_col:     str = "longitude",
    year_col:    str = "commissioning_year",
) -> QAReport:
    """
    Run the full GX-style expectation suite on `df` and return a QAReport.

    Expectations
    ------------
    Completeness
        1. capacity_mw has no nulls
        2. commissioning_year is ≥ 95% complete
        3. Table has at least 1 row

    Uniqueness
        4. Asset ID column has no duplicates

    Validity
        5. latitude ∈ [-90, 90]
        6. longitude ∈ [-180, 180]

    Consistency
        7. capacity_mw ≥ 0 (no negative capacity)
        8. commissioning_year ∈ [1900, 2100]
    """
    print("[expectations] Running Great Expectations suite …")

    report = QAReport(
        run_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        total_rows=len(df),
    )

    checks = [
        # ── Completeness ──────────────────────────────────────────────────
        expect_column_values_to_not_be_null(df, cap_col),
        expect_column_values_to_not_be_null_pct(df, year_col, mostly=0.95),
        expect_table_row_count_to_be_between(df, min_value=1),

        # ── Uniqueness ────────────────────────────────────────────────────
        expect_column_values_to_be_unique(df, id_col),

        # ── Validity ──────────────────────────────────────────────────────
        expect_column_values_to_be_between(df, lat_col,  -90,   90,   "Validity"),
        expect_column_values_to_be_between(df, lon_col, -180,  180,   "Validity"),

        # ── Consistency ───────────────────────────────────────────────────
        expect_column_values_to_be_between(df, cap_col,    0, 25_000, "Consistency"),
        expect_column_values_to_be_between(df, year_col, 1900,  2100, "Consistency"),
    ]

    report.results = checks

    passed = sum(1 for c in checks if c.success)
    print(f"[expectations] ✓ {passed}/{len(checks)} expectations passed.\n")
    return report
