"""
anomalies.py
------------
Unsupervised anomaly detection using Isolation Forest.
Flags power-plant records whose numeric feature combination
(lat, lon, capacity) is statistically unusual — catching errors
that rule-based checks (ocean bbox, null checks) might miss.

Typical use-cases
-----------------
- A plant with valid land coordinates but capacity = 0.001 MW (data entry error).
- A plant misattributed to the wrong country hemisphere.
- Capacity values that are plausible in isolation but anomalous for the fuel type.
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["latitude", "longitude", "capacity_mw"]
MODEL_PATH   = Path(__file__).parent.parent / "models" / "isolation_forest.pkl"
SCALER_PATH  = Path(__file__).parent.parent / "models" / "scaler.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and scale numeric features.
    Rows with NaN in any feature column are dropped before fitting/scoring,
    but NaN rows are re-joined at the end with anomaly_score = NaN.
    """
    sub = df[FEATURE_COLS].copy()
    sub["capacity_mw"] = sub["capacity_mw"].fillna(sub["capacity_mw"].median())
    return sub.values


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(
    df:              pd.DataFrame,
    contamination:   float = 0.08,
    n_estimators:    int   = 200,
    random_state:    int   = 42,
    save_model:      bool  = True,
) -> IsolationForest:
    """
    Fit an Isolation Forest on FEATURE_COLS and optionally persist it.

    Parameters
    ----------
    df            : Training DataFrame (dirty or clean, your choice).
    contamination : Expected fraction of anomalies (mirrors pct_ocean + pct_missing).
    n_estimators  : Number of trees in the forest.
    save_model    : Persist model + scaler to models/ directory.

    Returns
    -------
    Fitted IsolationForest instance.
    """
    print("[anomalies] Training Isolation Forest …")
    X = _prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    if save_model:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model,  MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[anomalies] ✓ Model saved  → {MODEL_PATH}")
        print(f"[anomalies] ✓ Scaler saved → {SCALER_PATH}")

    print("[anomalies] Training complete.\n")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Score
# ─────────────────────────────────────────────────────────────────────────────

def score_anomalies(
    df:    pd.DataFrame,
    model: IsolationForest | None = None,
) -> pd.DataFrame:
    """
    Attach anomaly scores and labels to every row in `df`.

    If `model` is None, loads from models/isolation_forest.pkl.

    New columns
    -----------
    anomaly_score  : float  – Raw decision function score (lower = more anomalous).
    is_anomaly     : bool   – True if model predicts -1 (anomaly).
    """
    print("[anomalies] Scoring records …")

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model at {MODEL_PATH}. Run train_isolation_forest() first."
            )
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else StandardScaler()

    X        = _prepare_features(df)
    X_scaled = scaler.transform(X)

    scores      = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)          # 1 = normal, -1 = anomaly

    result = df.copy()
    result["anomaly_score"] = scores.round(4)
    result["is_anomaly"]    = predictions == -1

    n_flagged = result["is_anomaly"].sum()
    print(f"[anomalies] ✓ {n_flagged:,} / {len(df):,} records flagged as anomalies.\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: detect_qa_errors  (ocean + duplicate + missing — rule-based)
# ─────────────────────────────────────────────────────────────────────────────

def detect_qa_errors(gdf) -> pd.DataFrame:
    """
    Rule-based QA detection: ocean coordinates, duplicate IDs, missing capacity.
    Returns a report DataFrame (one issue row per flagged record).
    """
    import geopandas as gpd

    print("[anomalies] Running rule-based QA checks …")
    issues = []
    df     = gdf.copy()

    # ── Ocean coordinates (spatial join) ──────────────────────────────────
    try:
        land   = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).dissolve()
        joined = gpd.sjoin(gdf, land, how="left", predicate="within")
        ocean_mask = joined["index_right"].isna()
        for idx, row in gdf[ocean_mask.values].iterrows():
            issues.append(_issue(row, "OCEAN_COORDS",
                                 f"lat={row['latitude']}, lon={row['longitude']}"))
        print(f"[anomalies] ✓ Ocean check (spatial join): {ocean_mask.sum()} flagged.")
    except Exception:
        # Fallback bbox check
        lat, lon = df["latitude"], df["longitude"]
        mask = (
            (lat.between(-10, 10) & lon.between(-170, -120)) |
            (lat.between(  5, 20) & lon.between(-40,  -20))  |
            (lat.between(-30,-10) & lon.between( 60,   90))
        )
        for _, row in df[mask].iterrows():
            issues.append(_issue(row, "OCEAN_COORDS",
                                 f"lat={row['latitude']}, lon={row['longitude']}"))
        print(f"[anomalies] ✓ Ocean check (bbox fallback): {mask.sum()} flagged.")

    # ── Duplicate IDs ─────────────────────────────────────────────────────
    counts  = df["gppd_idnr"].value_counts()
    dup_ids = counts[counts > 1].index
    for _, row in df[df["gppd_idnr"].isin(dup_ids)].iterrows():
        issues.append(_issue(row, "DUPLICATE_ID",
                             f"appears {counts[row['gppd_idnr']]}x"))
    print(f"[anomalies] ✓ Duplicate-ID check: {len(dup_ids)} IDs duplicated.")

    # ── Missing capacity ──────────────────────────────────────────────────
    for _, row in df[df["capacity_mw"].isna()].iterrows():
        issues.append(_issue(row, "MISSING_CAPACITY", "capacity_mw is NaN"))
    print(f"[anomalies] ✓ Missing-capacity check: {df['capacity_mw'].isna().sum()} rows.\n")

    return pd.DataFrame(issues)


def _issue(row, issue_type: str, detail: str) -> dict:
    return {
        "gppd_idnr":   row["gppd_idnr"],
        "plant_name":  row.get("name", ""),
        "issue_type":  issue_type,
        "detail":      detail,
        "capacity_mw": row.get("capacity_mw"),
    }
