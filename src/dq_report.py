"""
dq_report.py
------------
Builds and renders the final Data Quality Dimensions report.

The four DQ dimensions used here map directly to industry standards
(DAMA-DMBOK, MSCI internal QA frameworks, ISO 8000):

  ACCURACY     → Does the data correctly represent reality?
                 Source: geometric country-border check (schema.py)

  COMPLETENESS → Is all required data present?
                 Source: Great Expectations null checks (expectations.py)

  CONSISTENCY  → Does the data obey internal business rules?
                 Source: GX range checks + Pydantic validators

  UNIQUENESS   → Are there duplicate records?
                 Source: GX uniqueness check + duplicate ID detection

Each dimension gets a 0–100% score, a pass/fail status, and a list
of contributing issues with row-level detail.
"""

from __future__ import annotations

import pandas as pd

from expectations import QAReport, ExpectationResult
from schema import validate_assets


# ─────────────────────────────────────────────────────────────────────────────
# Dimension score calculator
# ─────────────────────────────────────────────────────────────────────────────

def _score(items: list[ExpectationResult]) -> float:
    if not items:
        return 1.0
    return sum(1 for i in items if i.success) / len(items)


# ─────────────────────────────────────────────────────────────────────────────
# Main report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dq_report(
    df:         pd.DataFrame,
    gx_report:  QAReport,
    asset_df:   pd.DataFrame | None = None,
) -> dict:
    """
    Combine outputs from Great Expectations + Pydantic schema validation
    into a single Data Quality Dimensions report.

    Parameters
    ----------
    df         : The dirty / raw DataFrame being assessed.
    gx_report  : QAReport returned by run_expectation_suite().
    asset_df   : DataFrame with 'asset_valid' and 'geometry_consistent'
                 columns (returned by validate_assets()). If None,
                 validate_assets() is called automatically.

    Returns
    -------
    dict with keys: dimensions, summary, failing_records, run_metadata
    """
    print("[dq_report] Building Data Quality Dimensions report …")

    # ── Run asset validation if not already done ──────────────────────────
    if asset_df is None:
        asset_df = validate_assets(df)

    by_dim = gx_report.by_dimension()

    # ── ACCURACY ─────────────────────────────────────────────────────────
    # Source: geometry_consistent flag from Pydantic model_validator
    n_total         = len(asset_df)
    n_geo_consistent = asset_df["geometry_consistent"].sum()
    accuracy_score  = n_geo_consistent / n_total if n_total > 0 else 1.0

    accuracy_issues = asset_df[~asset_df["geometry_consistent"]][[
        "gppd_idnr" if "gppd_idnr" in asset_df.columns else asset_df.columns[0],
        "latitude", "longitude", "country", "asset_errors"
    ]].head(20).to_dict("records")

    # ── COMPLETENESS ─────────────────────────────────────────────────────
    completeness_checks = by_dim.get("Completeness", {})
    completeness_score  = completeness_checks.get("score", 1.0)

    # ── CONSISTENCY ───────────────────────────────────────────────────────
    consistency_checks = by_dim.get("Consistency", {})
    consistency_score  = consistency_checks.get("score", 1.0)

    # ── UNIQUENESS ────────────────────────────────────────────────────────
    uniqueness_checks = by_dim.get("Uniqueness", {})
    uniqueness_score  = uniqueness_checks.get("score", 1.0)

    # ── Overall score (simple average of the 4 dimensions) ────────────────
    overall = (accuracy_score + completeness_score +
               consistency_score + uniqueness_score) / 4

    # ── Assemble report ───────────────────────────────────────────────────
    report = {
        "run_metadata": {
            "run_time":   gx_report.run_time,
            "total_rows": n_total,
            "gx_checks":  len(gx_report.results),
            "gx_passed":  gx_report.passed,
            "gx_failed":  gx_report.failed,
        },
        "dimensions": {
            "ACCURACY": {
                "score":       round(accuracy_score, 4),
                "pct":         f"{accuracy_score*100:.1f}%",
                "status":      "✅ PASS" if accuracy_score >= 0.95 else "❌ FAIL",
                "description": "Fraction of assets whose coordinates fall within "
                               "their reported country's borders.",
                "source":      "Pydantic model_validator → Shapely spatial check",
                "issues":      accuracy_issues,
            },
            "COMPLETENESS": {
                "score":       round(completeness_score, 4),
                "pct":         f"{completeness_score*100:.1f}%",
                "status":      "✅ PASS" if completeness_score >= 0.95 else "❌ FAIL",
                "description": "Fraction of required fields (capacity_mw, "
                               "commissioning_year) that are non-null.",
                "source":      "Great Expectations → expect_column_values_to_not_be_null",
                "checks":      [
                    {
                        "expectation": r.expectation,
                        "column":      r.column,
                        "success":     r.success,
                        "detail":      r.detail,
                    }
                    for r in gx_report.results
                    if r.dimension == "Completeness"
                ],
            },
            "CONSISTENCY": {
                "score":       round(consistency_score, 4),
                "pct":         f"{consistency_score*100:.1f}%",
                "status":      "✅ PASS" if consistency_score >= 0.95 else "❌ FAIL",
                "description": "Fraction of business-rule checks passed "
                               "(capacity ≥ 0, year ∈ [1900–2100]).",
                "source":      "Great Expectations → expect_column_values_to_be_between",
                "checks":      [
                    {
                        "expectation": r.expectation,
                        "column":      r.column,
                        "success":     r.success,
                        "observed":    r.observed,
                        "threshold":   r.threshold,
                        "detail":      r.detail,
                    }
                    for r in gx_report.results
                    if r.dimension == "Consistency"
                ],
            },
            "UNIQUENESS": {
                "score":       round(uniqueness_score, 4),
                "pct":         f"{uniqueness_score*100:.1f}%",
                "status":      "✅ PASS" if uniqueness_score >= 0.99 else "❌ FAIL",
                "description": "Fraction of asset IDs that are unique "
                               "(no duplicate gppd_idnr values).",
                "source":      "Great Expectations → expect_column_values_to_be_unique",
                "checks":      [
                    {
                        "expectation": r.expectation,
                        "column":      r.column,
                        "success":     r.success,
                        "observed":    r.observed,
                        "detail":      r.detail,
                    }
                    for r in gx_report.results
                    if r.dimension == "Uniqueness"
                ],
            },
        },
        "summary": {
            "overall_score": round(overall, 4),
            "overall_pct":   f"{overall*100:.1f}%",
            "overall_status": "✅ PASS" if overall >= 0.90 else "❌ FAIL",
            "dimension_scores": {
                "ACCURACY":     f"{accuracy_score*100:.1f}%",
                "COMPLETENESS": f"{completeness_score*100:.1f}%",
                "CONSISTENCY":  f"{consistency_score*100:.1f}%",
                "UNIQUENESS":   f"{uniqueness_score*100:.1f}%",
            },
        },
    }

    print(f"[dq_report] ✓ Report built. Overall DQ score: "
          f"{overall*100:.1f}%\n")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Console printer
# ─────────────────────────────────────────────────────────────────────────────

def print_dq_report(report: dict) -> None:
    """Pretty-print the Data Quality Dimensions report to stdout."""

    meta = report["run_metadata"]
    summ = report["summary"]
    dims = report["dimensions"]

    width = 65
    bar   = "=" * width

    print()
    print(bar)
    print("  DATA QUALITY DIMENSIONS REPORT")
    print(f"  Run time : {meta['run_time']}")
    print(f"  Dataset  : {meta['total_rows']:,} rows")
    print(f"  GX suite : {meta['gx_passed']}/{meta['gx_checks']} expectations passed")
    print(bar)
    print()

    for dim_name, dim in dims.items():
        print(f"  {'─'*58}")
        print(f"  {dim_name:<15} {dim['pct']:>8}   {dim['status']}")
        print(f"  {'─'*58}")
        print(f"  {dim['description']}")
        print(f"  Source: {dim['source']}")

        # Print per-check detail
        for check_key in ("checks", "issues"):
            items = dim.get(check_key, [])
            if items and isinstance(items[0], dict):
                for item in items[:3]:          # cap at 3 items for readability
                    detail = item.get("detail") or item.get("asset_errors", "")
                    col    = item.get("column", "")
                    status = "✅" if item.get("success", False) else "❌"
                    if col:
                        print(f"    {status} [{col}] {detail[:80]}")
                    else:
                        print(f"    ❌ {str(item)[:80]}")
        print()

    print(bar)
    print(f"  OVERALL DQ SCORE : {summ['overall_pct']}  {summ['overall_status']}")
    print(bar)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV exporter
# ─────────────────────────────────────────────────────────────────────────────

def export_dq_report_csv(report: dict, path: str = "data/dq_report.csv") -> None:
    """Flatten the report into a tidy CSV for further analysis."""
    rows = []
    for dim_name, dim in report["dimensions"].items():
        rows.append({
            "dimension":   dim_name,
            "score":       dim["score"],
            "pct":         dim["pct"],
            "status":      dim["status"],
            "description": dim["description"],
            "source":      dim["source"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[dq_report] Saved: {path}")
