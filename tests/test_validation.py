"""
tests/test_validation.py
------------------------
Pytest suite for the validation.py Pydantic schema module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from validation import PowerPlantRecord, validate_dataframe
from pydantic import ValidationError


# ─────────────────────────────────────────────────────────────────────────────
# PowerPlantRecord unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPowerPlantRecord:

    def _valid(self, **overrides) -> dict:
        base = dict(
            gppd_idnr="WRI000001",
            name="Test Plant",
            capacity_mw=500.0,
            latitude=40.0,
            longitude=-74.0,
            country="USA",
            primary_fuel="Coal",
            commissioning_year=2005,
        )
        return {**base, **overrides}

    def test_valid_record_passes(self):
        rec = PowerPlantRecord(**self._valid())
        assert rec.gppd_idnr == "WRI000001"

    def test_latitude_out_of_range_fails(self):
        with pytest.raises(ValidationError):
            PowerPlantRecord(**self._valid(latitude=91))

    def test_longitude_out_of_range_fails(self):
        with pytest.raises(ValidationError):
            PowerPlantRecord(**self._valid(longitude=-181))

    def test_negative_capacity_fails(self):
        with pytest.raises(ValidationError):
            PowerPlantRecord(**self._valid(capacity_mw=-10))

    def test_blank_id_fails(self):
        with pytest.raises(ValidationError):
            PowerPlantRecord(**self._valid(gppd_idnr="   "))

    def test_nan_capacity_coerced_to_none(self):
        rec = PowerPlantRecord(**self._valid(capacity_mw=float("nan")))
        assert rec.capacity_mw is None

    def test_none_capacity_allowed(self):
        rec = PowerPlantRecord(**self._valid(capacity_mw=None))
        assert rec.capacity_mw is None

    def test_country_code_too_short_fails(self):
        with pytest.raises(ValidationError):
            PowerPlantRecord(**self._valid(country="US"))

    def test_commissioning_year_far_future_fails(self):
        with pytest.raises(ValidationError):
            PowerPlantRecord(**self._valid(commissioning_year=2200))


# ─────────────────────────────────────────────────────────────────────────────
# validate_dataframe tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateDataframe:

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            dict(gppd_idnr="WRI000001", name="Plant A", capacity_mw=100.0,
                 latitude=40.0, longitude=-74.0, country="USA",
                 primary_fuel="Coal", commissioning_year=2000),
            dict(gppd_idnr="WRI000002", name="Plant B", capacity_mw=float("nan"),
                 latitude=999.0, longitude=0.0, country="DEU",   # bad lat
                 primary_fuel="Wind", commissioning_year=2010),
        ])

    def test_returns_dataframe(self, sample_df):
        result = validate_dataframe(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_valid_and_errors_columns(self, sample_df):
        result = validate_dataframe(sample_df)
        assert "pydantic_valid" in result.columns
        assert "pydantic_errors" in result.columns

    def test_valid_row_flagged_true(self, sample_df):
        result = validate_dataframe(sample_df)
        assert result.loc[0, "pydantic_valid"] == True

    def test_invalid_row_flagged_false(self, sample_df):
        result = validate_dataframe(sample_df)
        assert result.loc[1, "pydantic_valid"] == False

    def test_invalid_row_has_error_message(self, sample_df):
        result = validate_dataframe(sample_df)
        assert len(result.loc[1, "pydantic_errors"]) > 0

    def test_output_length_matches_input(self, sample_df):
        result = validate_dataframe(sample_df)
        assert len(result) == len(sample_df)