"""
tests/test_ml_anomaly.py
------------------------
Pytest suite for ml_anomaly.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from ml_anomaly import (
    engineer_features,
    train_global_model,
    score_global,
    train_and_score_per_country,
    compute_final_score,
    run_ml_anomaly_detection,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Reproducible asset DataFrame large enough for country models."""
    rng = np.random.default_rng(42)
    n   = 300

    countries = (["USA"] * 100) + (["DEU"] * 80) + (["CHN"] * 70) + (["IND"] * 50)
    countries = countries[:n]

    return pd.DataFrame({
        "gppd_idnr":   [f"WRI{i:06d}" for i in range(n)],
        "name":        [f"Plant_{i}"  for i in range(n)],
        "capacity_mw": rng.uniform(10, 3000, n).round(1),
        "latitude":    rng.uniform(20, 60, n).round(4),
        "longitude":   rng.uniform(-120, 140, n).round(4),
        "country":     countries,
        "primary_fuel": rng.choice(["Coal", "Gas", "Wind", "Solar"], n),
        "error_flags": [""] * n,
    })


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Minimal 5-row DataFrame for edge-case tests."""
    return pd.DataFrame({
        "gppd_idnr":   ["A", "B", "C", "D", "E"],
        "capacity_mw": [100.0, 200.0, None, 50.0, 9999.0],
        "latitude":    [40.0, 51.5, 35.0, 28.6, -100.0],  # -100 is anomalous
        "longitude":   [-74.0, -0.1, 139.0, 77.2, 0.0],
        "country":     ["USA", "GBR", "JPN", "IND", "ZZZ"],
        "error_flags": ["", "", "", "", "OCEAN_COORDS"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# engineer_features
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_adds_cap_filled(self, sample_df):
        out = engineer_features(sample_df)
        assert "cap_filled" in out.columns

    def test_cap_filled_has_no_nulls(self, sample_df):
        # Introduce some nulls first
        sample_df.loc[:5, "capacity_mw"] = None
        out = engineer_features(sample_df)
        assert out["cap_filled"].isna().sum() == 0

    def test_adds_cap_log(self, sample_df):
        out = engineer_features(sample_df)
        assert "cap_log" in out.columns
        assert (out["cap_log"] >= 0).all()

    def test_adds_lat_lon_ratio(self, sample_df):
        out = engineer_features(sample_df)
        assert "lat_lon_ratio" in out.columns

    def test_adds_country_cap_zscore(self, sample_df):
        out = engineer_features(sample_df)
        assert "country_cap_zscore" in out.columns

    def test_row_count_unchanged(self, sample_df):
        out = engineer_features(sample_df)
        assert len(out) == len(sample_df)


# ─────────────────────────────────────────────────────────────────────────────
# train_global_model + score_global
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalModel:

    def test_returns_model_and_scaler(self, sample_df):
        featured = engineer_features(sample_df)
        model, scaler = train_global_model(featured)
        assert model is not None
        assert scaler is not None

    def test_score_global_adds_columns(self, sample_df):
        featured      = engineer_features(sample_df)
        model, scaler = train_global_model(featured)
        scored        = score_global(featured, model, scaler)
        assert "global_score"   in scored.columns
        assert "global_anomaly" in scored.columns

    def test_global_anomaly_is_bool(self, sample_df):
        featured      = engineer_features(sample_df)
        model, scaler = train_global_model(featured)
        scored        = score_global(featured, model, scaler)
        assert scored["global_anomaly"].dtype == bool

    def test_score_length_matches_input(self, sample_df):
        featured      = engineer_features(sample_df)
        model, scaler = train_global_model(featured)
        scored        = score_global(featured, model, scaler)
        assert len(scored) == len(sample_df)


# ─────────────────────────────────────────────────────────────────────────────
# train_and_score_per_country
# ─────────────────────────────────────────────────────────────────────────────

class TestCountryModel:

    def test_adds_country_score_column(self, sample_df):
        featured = engineer_features(sample_df)
        model, scaler = train_global_model(featured)
        featured = score_global(featured, model, scaler)
        result   = train_and_score_per_country(featured)
        assert "country_score"   in result.columns
        assert "country_anomaly" in result.columns
        assert "model_used"      in result.columns

    def test_model_used_values_are_valid(self, sample_df):
        featured = engineer_features(sample_df)
        model, scaler = train_global_model(featured)
        featured = score_global(featured, model, scaler)
        result   = train_and_score_per_country(featured)
        valid = {"country", "global_fallback"}
        assert set(result["model_used"].unique()).issubset(valid)

    def test_small_country_gets_fallback(self, tiny_df):
        # tiny_df has only 1 asset per country — all should be global_fallback
        featured = engineer_features(tiny_df)
        model, scaler = train_global_model(featured)
        featured = score_global(featured, model, scaler)
        result   = train_and_score_per_country(featured)
        assert (result["model_used"] == "global_fallback").all()


# ─────────────────────────────────────────────────────────────────────────────
# compute_final_score
# ─────────────────────────────────────────────────────────────────────────────

class TestFinalScore:

    @pytest.fixture
    def fully_scored(self, sample_df):
        featured = engineer_features(sample_df)
        model, scaler = train_global_model(featured)
        featured = score_global(featured, model, scaler)
        featured = train_and_score_per_country(featured)
        return compute_final_score(featured)

    def test_anomaly_score_in_range(self, fully_scored):
        assert (fully_scored["ANOMALY_SCORE"] >= 0).all()
        assert (fully_scored["ANOMALY_SCORE"] <= 1).all()

    def test_is_anomaly_is_bool(self, fully_scored):
        assert fully_scored["IS_ANOMALY"].dtype == bool

    def test_some_anomalies_detected(self, fully_scored):
        assert fully_scored["IS_ANOMALY"].sum() > 0

    def test_not_all_flagged(self, fully_scored):
        assert fully_scored["IS_ANOMALY"].sum() < len(fully_scored)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    def test_run_returns_dataframe(self, sample_df):
        result = run_ml_anomaly_detection(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, sample_df):
        result = run_ml_anomaly_detection(sample_df)
        for col in ["ANOMALY_SCORE", "IS_ANOMALY", "global_score",
                    "country_score", "model_used"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_length_matches_input(self, sample_df):
        result = run_ml_anomaly_detection(sample_df)
        assert len(result) == len(sample_df)

    def test_reproducible_with_same_seed(self, sample_df):
        r1 = run_ml_anomaly_detection(sample_df, random_state=0)
        r2 = run_ml_anomaly_detection(sample_df, random_state=0)
        assert list(r1["IS_ANOMALY"]) == list(r2["IS_ANOMALY"])

    def test_missing_capacity_handled(self, sample_df):
        sample_df.loc[:20, "capacity_mw"] = None
        result = run_ml_anomaly_detection(sample_df)
        assert result["ANOMALY_SCORE"].isna().sum() == 0