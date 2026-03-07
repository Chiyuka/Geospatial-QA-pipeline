"""
run.py
------
Entry point for the Geo-Integrity Engine pipeline.

Steps
-----
1. Load GPPD data            (src/loader.py)
2. Create dirty dataset      (src/dirty.py)
3. Validate schema           (src/validation.py)
4. Train Isolation Forest    (src/anomalies.py)
5. Score anomalies           (src/anomalies.py)
6. Rule-based QA detection   (src/anomalies.py)
7. Save outputs to data/

Usage
-----
    python run.py
"""

import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from loader    import load_gppd
from dirty     import inject_errors
from validation import validate_dataframe
from anomalies import train_isolation_forest, score_anomalies, detect_qa_errors

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("  Geo-Integrity Engine — MSCI Climate Risk QA Pipeline")
    print("=" * 60)
    print()

    # 1. Load
    clean_gdf = load_gppd(n_rows=2_000)
    clean_gdf.drop(columns="geometry").to_csv(DATA_DIR / "gppd_clean.csv", index=False)
    print(f"  → Saved: data/gppd_clean.csv\n")

    # 2. Dirty
    dirty_gdf = inject_errors(clean_gdf, pct_ocean=0.05, pct_duplicate=0.04, pct_missing=0.08)
    dirty_gdf.drop(columns="geometry").to_csv(DATA_DIR / "gppd_dirty.csv", index=False)
    print(f"  → Saved: data/gppd_dirty.csv\n")

    # 3. Pydantic validation
    validated_df = validate_dataframe(dirty_gdf)
    invalid_count = (~validated_df["pydantic_valid"]).sum()
    print(f"  Pydantic: {invalid_count:,} rows failed schema validation.\n")

    # 4. Train Isolation Forest on dirty data (simulates real-world training)
    model = train_isolation_forest(dirty_gdf, contamination=0.08)

    # 5. Score
    scored_df = score_anomalies(dirty_gdf, model=model)
    scored_df.drop(columns="geometry").to_csv(DATA_DIR / "gppd_scored.csv", index=False)
    print(f"  → Saved: data/gppd_scored.csv\n")

    # 6. Rule-based QA report
    report = detect_qa_errors(dirty_gdf)
    report.to_csv(DATA_DIR / "qa_report.csv", index=False)
    print(f"  → Saved: data/qa_report.csv\n")

    # 7. Summary
    print("=" * 60)
    print("  QA Summary")
    print("=" * 60)
    print(f"  Clean records   : {len(clean_gdf):>6,}")
    print(f"  Dirty records   : {len(dirty_gdf):>6,}")
    print(f"  Schema failures : {invalid_count:>6,}")
    print(f"  ML anomalies    : {scored_df['is_anomaly'].sum():>6,}")
    print()
    if not report.empty:
        for issue, cnt in report["issue_type"].value_counts().items():
            print(f"  {issue:<22}: {cnt:>5,} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
