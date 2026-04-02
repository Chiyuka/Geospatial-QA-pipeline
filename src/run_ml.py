"""
run_ml.py
---------
Standalone ML anomaly detection runner.
Loads the dirty dataset produced by run.py, runs the full Isolation
Forest pipeline, generates a Folium map and Plotly dashboard, and
prints the final anomaly summary.

Usage
-----
    # Run main pipeline first to generate data/gppd_dirty.csv
    python run.py

    # Then run the ML visualisation pipeline
    python src/run_ml.py

Outputs
-------
    reports/anomaly_map.html      ← Folium interactive map (open in browser)
    reports/anomaly_charts.html   ← Plotly 4-panel dashboard
    data/gppd_ml_scored.csv       ← Full scored dataset
"""

import sys
import os
from pathlib import Path

# Allow running from project root or src/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pandas as pd
from ml_anomaly import run_ml_anomaly_detection
from visualise  import build_folium_map, build_plotly_charts

DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 65)
    print("  Geo-Integrity Engine — ML Anomaly Detection")
    print("=" * 65)
    print()

    # ── Load dirty dataset ────────────────────────────────────────────────
    dirty_path = DATA_DIR / "gppd_dirty.csv"
    if not dirty_path.exists():
        print(f"[run_ml] ERROR: {dirty_path} not found.")
        print("[run_ml] Please run 'python run.py' first to generate the dataset.")
        sys.exit(1)

    df = pd.read_csv(dirty_path)
    print(f"[run_ml] Loaded dirty dataset: {len(df):,} rows\n")

    # ── ML pipeline ───────────────────────────────────────────────────────
    scored_df = run_ml_anomaly_detection(df, contamination=0.08)

    # ── Save scored CSV ───────────────────────────────────────────────────
    out_cols = [
        "gppd_idnr", "name", "country", "primary_fuel",
        "latitude", "longitude", "capacity_mw",
        "ANOMALY_SCORE", "IS_ANOMALY",
        "global_score", "country_score", "model_used",
        "cap_log", "country_cap_zscore", "error_flags",
    ]
    save_cols = [c for c in out_cols if c in scored_df.columns]
    scored_df[save_cols].to_csv(DATA_DIR / "gppd_ml_scored.csv", index=False)
    print(f"[run_ml] Saved: data/gppd_ml_scored.csv\n")

    # ── Visualisations ────────────────────────────────────────────────────
    map_path    = build_folium_map(scored_df,    str(REPORTS_DIR / "anomaly_map.html"))
    charts_path = build_plotly_charts(scored_df, str(REPORTS_DIR / "anomaly_charts.html"))

    # ── Final summary ─────────────────────────────────────────────────────
    n_total   = len(scored_df)
    n_anom    = scored_df["IS_ANOMALY"].sum()
    n_clean   = n_total - n_anom
    avg_score = scored_df["ANOMALY_SCORE"].mean()

    # Cross-reference with injected error flags (if column exists)
    if "error_flags" in scored_df.columns:
        known_errors = scored_df["error_flags"].str.len() > 0
        caught = (known_errors & scored_df["IS_ANOMALY"]).sum()
        total_known = known_errors.sum()
        recall = caught / total_known if total_known > 0 else 0
    else:
        caught = total_known = recall = None

    print("=" * 65)
    print("  ML ANOMALY DETECTION SUMMARY")
    print("=" * 65)
    print(f"  Total assets      : {n_total:>6,}")
    print(f"  Clean             : {n_clean:>6,}  ({100*n_clean/n_total:.1f}%)")
    print(f"  Anomalous         : {n_anom:>6,}  ({100*n_anom/n_total:.1f}%)")
    print(f"  Avg anomaly score : {avg_score:>8.3f}")

    if recall is not None:
        print(f"\n  Known injected errors : {total_known:,}")
        print(f"  Caught by ML          : {caught:,}")
        print(f"  Recall                : {100*recall:.1f}%")

    print(f"\n  Country model coverage:")
    if "model_used" in scored_df.columns:
        for model_type, count in scored_df["model_used"].value_counts().items():
            pct = 100 * count / n_total
            print(f"    {model_type:<20}: {count:>5,} assets ({pct:.1f}%)")

    print(f"\n  Top 5 most anomalous assets:")
    top5_cols = ["gppd_idnr", "country", "latitude", "longitude",
                 "capacity_mw", "ANOMALY_SCORE", "IS_ANOMALY"]
    top5_cols = [c for c in top5_cols if c in scored_df.columns]
    top5 = scored_df.nlargest(5, "ANOMALY_SCORE")[top5_cols]
    print(top5.to_string(index=False))

    print()
    print("=" * 65)
    print(f"  Reports saved to: reports/")
    print(f"    → anomaly_map.html     (open in browser — Folium map)")
    print(f"    → anomaly_charts.html  (open in browser — Plotly dashboard)")
    print("=" * 65)


if __name__ == "__main__":
    main()