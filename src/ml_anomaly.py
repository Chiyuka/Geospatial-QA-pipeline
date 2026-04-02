"""
ml_anomaly.py
-------------
Isolation Forest anomaly detection for geospatial asset data.

What this module does
---------------------
Standard Isolation Forest treats all assets as one pool. That's too
blunt for MSCI's use case — a 3,000 MW coal plant is normal in China
but anomalous in Luxembourg. This module runs TWO detection layers:

  Layer 1 — GLOBAL model
      Trained on all assets. Catches extreme outliers: ocean-placed
      plants, impossibly high capacities, coordinate inversions.

  Layer 2 — PER-COUNTRY model
      One Isolation Forest per country (min 10 assets). Catches
      subtle within-country anomalies: a solar plant with coal-plant
      capacity, a plant whose coordinates are in the right country
      but the wrong region.

Both scores are combined into a final ANOMALY_SCORE (0–1, higher = 
more suspicious) and a binary IS_ANOMALY flag.

Features used
-------------
  latitude, longitude             — spatial position
  capacity_mw                     — generation scale
  lat_lon_ratio                   — proxy for coordinate swap errors
  cap_log                         — log-scaled capacity (reduces MW skew)
  country_cap_zscore              — capacity z-score within country
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that help Isolation Forest detect geospatial
    and capacity anomalies more precisely.

    New columns
    -----------
    cap_filled          : capacity_mw with NaN replaced by country median
    cap_log             : log1p(cap_filled) — reduces MW scale skew
    lat_lon_ratio       : lat / (|lon| + 1e-6) — detects coordinate swaps
    country_cap_median  : median capacity for the asset's country
    country_cap_zscore  : how many std deviations from country median
    """
    out = df.copy()

    # Fill missing capacity with country-level median, then global median
    country_med = out.groupby("country")["capacity_mw"].transform("median")
    global_med  = out["capacity_mw"].median()
    out["cap_filled"] = out["capacity_mw"].fillna(country_med).fillna(global_med)

    # Log scale capacity — a 10 MW plant and a 10,000 MW plant differ by 3
    # orders of magnitude; log squashes this so both get fair attention
    out["cap_log"] = np.log1p(out["cap_filled"])

    # Coordinate swap detector: if lat and lon are accidentally swapped,
    # this ratio will be very different from nearby assets in the same country
    out["lat_lon_ratio"] = out["latitude"] / (out["longitude"].abs() + 1e-6)

    # Country-level capacity z-score: how unusual is this capacity
    # compared to other plants in the same country?
    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else pd.Series(0, index=s.index)

    out["country_cap_zscore"] = (
        out.groupby("country")["cap_filled"].transform(zscore).fillna(0)
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Global Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_FEATURES = [
    "latitude",
    "longitude",
    "cap_log",
    "lat_lon_ratio",
    "country_cap_zscore",
]


def train_global_model(
    df:            pd.DataFrame,
    contamination: float = 0.08,
    random_state:  int   = 42,
) -> tuple[IsolationForest, StandardScaler]:
    """
    Train a single Isolation Forest on all assets (global scope).

    Returns fitted (model, scaler) tuple.
    """
    print("[ml_anomaly] Training global Isolation Forest …")
    X      = df[GLOBAL_FEATURES].fillna(0).values
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(Xs)
    print(f"[ml_anomaly] ✓ Global model trained on {len(df):,} assets.\n")
    return model, scaler


def score_global(
    df:     pd.DataFrame,
    model:  IsolationForest,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply global model. Adds:
        global_score   : raw decision function (lower = more anomalous)
        global_anomaly : bool flag (-1 from model → True)
    """
    X  = df[GLOBAL_FEATURES].fillna(0).values
    Xs = scaler.transform(X)

    out = df.copy()
    out["global_score"]   = model.decision_function(Xs).round(4)
    out["global_anomaly"] = model.predict(Xs) == -1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-country Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

COUNTRY_FEATURES = [
    "latitude",
    "longitude",
    "cap_log",
    "country_cap_zscore",
]

MIN_COUNTRY_ASSETS = 10   # need at least this many assets to train a country model


def train_and_score_per_country(
    df:            pd.DataFrame,
    contamination: float = 0.08,
    random_state:  int   = 42,
) -> pd.DataFrame:
    """
    For each country with ≥ MIN_COUNTRY_ASSETS plants, train a separate
    Isolation Forest and score those plants. Countries with fewer assets
    fall back to global_anomaly.

    Adds:
        country_score   : per-country decision function score
        country_anomaly : bool flag from per-country model
        model_used      : 'country' or 'global_fallback'
    """
    print("[ml_anomaly] Training per-country models …")
    out = df.copy()
    out["country_score"]   = np.nan
    out["country_anomaly"] = False
    out["model_used"]      = "global_fallback"

    countries       = df["country"].value_counts()
    eligible        = countries[countries >= MIN_COUNTRY_ASSETS].index
    n_country_models = 0

    for country in eligible:
        mask = df["country"] == country
        sub  = df[mask].copy()

        X      = sub[COUNTRY_FEATURES].fillna(0).values
        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(Xs)

        scores = model.decision_function(Xs)
        preds  = model.predict(Xs) == -1

        out.loc[mask, "country_score"]   = scores.round(4)
        out.loc[mask, "country_anomaly"] = preds
        out.loc[mask, "model_used"]      = "country"
        n_country_models += 1

    print(f"[ml_anomaly] ✓ Per-country models: {n_country_models} countries "
          f"| {(out['model_used'] == 'global_fallback').sum()} assets used "
          f"global fallback.\n")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Combined anomaly score
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine global and per-country signals into a single ANOMALY_SCORE
    (0.0 = perfectly normal, 1.0 = highly anomalous) and IS_ANOMALY flag.

    Scoring logic
    -------------
    - decision_function scores are negative-is-bad; invert and normalise
      to [0, 1] so higher always means more suspicious.
    - Final score = 0.4 × global_norm + 0.6 × country_norm
      (country model weighted higher as it's more contextually aware)
    - IS_ANOMALY = True if ANOMALY_SCORE > 0.6 OR either model flags it
    """
    out = df.copy()

    def norm(series: pd.Series) -> pd.Series:
        """Invert (lower raw = more anomalous) and min-max normalise to [0,1]."""
        s   = series.fillna(series.median())
        inv = -s                          # invert: higher = more anomalous
        mn, mx = inv.min(), inv.max()
        return (inv - mn) / (mx - mn + 1e-9)

    global_norm  = norm(out["global_score"])
    country_norm = norm(out["country_score"].fillna(out["global_score"]))

    out["ANOMALY_SCORE"] = (0.4 * global_norm + 0.6 * country_norm).round(4)
    out["IS_ANOMALY"]    = (
        (out["ANOMALY_SCORE"] > 0.60) |
        out["global_anomaly"] |
        out["country_anomaly"]
    )

    n_anomalies = out["IS_ANOMALY"].sum()
    pct         = 100 * n_anomalies / len(out)
    print(f"[ml_anomaly] ✓ Final scores computed.")
    print(f"[ml_anomaly]   Anomalies flagged: {n_anomalies:,} / {len(out):,} "
          f"({pct:.1f}%)\n")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_anomaly_detection(
    df:            pd.DataFrame,
    contamination: float = 0.08,
    random_state:  int   = 42,
) -> pd.DataFrame:
    """
    Full ML anomaly detection pipeline.

    Parameters
    ----------
    df            : DataFrame with columns latitude, longitude,
                    capacity_mw, country.
    contamination : Expected fraction of anomalies.
    random_state  : Seed for reproducibility.

    Returns
    -------
    DataFrame with added columns:
        ANOMALY_SCORE  : float [0, 1]
        IS_ANOMALY     : bool
        global_score   : raw global IF decision score
        country_score  : raw country IF decision score
        model_used     : 'country' or 'global_fallback'
        cap_log        : engineered feature
        country_cap_zscore : engineered feature
    """
    print("[ml_anomaly] Starting ML anomaly detection pipeline …")
    print(f"[ml_anomaly] Input: {len(df):,} assets\n")

    # 1. Feature engineering
    featured = engineer_features(df)

    # 2. Global model
    g_model, g_scaler = train_global_model(
        featured, contamination=contamination, random_state=random_state
    )
    featured = score_global(featured, g_model, g_scaler)

    # 3. Per-country models
    featured = train_and_score_per_country(
        featured, contamination=contamination, random_state=random_state
    )

    # 4. Final combined score
    result = compute_final_score(featured)

    return result