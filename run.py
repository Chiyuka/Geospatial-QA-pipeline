"""
run.py
------
End-to-end Geo-Integrity Engine pipeline.

Steps
-----
1.  Load GPPD data                    (src/loader.py)
2.  Create dirty dataset              (src/dirty.py)
3.  Pydantic Asset schema + geometry  (src/schema.py)
4.  Great Expectations suite          (src/expectations.py)
5.  Train Isolation Forest            (src/anomalies.py)
6.  Score anomalies                   (src/anomalies.py)
7.  Rule-based QA detection           (src/anomalies.py)
8.  Build & print DQ Dimensions report(src/dq_report.py)
9.  Save all outputs to data/

Usage
-----
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from loader       import load_gppd
from dirty        import inject_errors
from schema       import validate_assets
from expectations import run_expectation_suite
from anomalies    import train_isolation_forest, score_anomalies, detect_qa_errors
from dq_report    import build_dq_report, print_dq_report, export_dq_report_csv

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 65)
    print("  Geo-Integrity Engine — MSCI Climate Risk QA Pipeline")
    print("=" * 65)
    print()

    # 1. Load
    clean_gdf = load_gppd(n_rows=2_000)
    clean_gdf.drop(columns="geometry").to_csv(DATA_DIR / "gppd_clean.csv", index=False)
    print(f"  -> Saved: data/gppd_clean.csv\n")

    # 2. Dirty
    dirty_gdf = inject_errors(
        clean_gdf,
        pct_ocean=0.05, pct_duplicate=0.04, pct_missing=0.08,
    )
    dirty_df = dirty_gdf.drop(columns="geometry")
    dirty_df.to_csv(DATA_DIR / "gppd_dirty.csv", index=False)
    print(f"  -> Saved: data/gppd_dirty.csv\n")

    # 3. Pydantic Asset schema + geometric country check
    asset_df = validate_assets(dirty_df)

    # 4. Great Expectations suite
    gx_report = run_expectation_suite(dirty_df)

    # 5 & 6. Isolation Forest
    model     = train_isolation_forest(dirty_gdf, contamination=0.08)
    scored_df = score_anomalies(dirty_gdf, model=model)
    scored_df.drop(columns="geometry").to_csv(DATA_DIR / "gppd_scored.csv", index=False)
    print(f"  -> Saved: data/gppd_scored.csv\n")

    # 7. Rule-based QA
    qa_report = detect_qa_errors(dirty_gdf)
    qa_report.to_csv(DATA_DIR / "qa_report.csv", index=False)
    print(f"  -> Saved: data/qa_report.csv\n")

    # 8. DQ Dimensions report
    dq = build_dq_report(dirty_df, gx_report, asset_df=asset_df)
    print_dq_report(dq)
    export_dq_report_csv(dq, str(DATA_DIR / "dq_dimensions.csv"))

    print()
    print("  Pipeline complete.")


if __name__ == "__main__":
    main()
