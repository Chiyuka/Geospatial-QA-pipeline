"""
tests/test_dirty.py
-------------------
Pytest suite for the dirty.py error-injection module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point
from dirty import inject_errors


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_gdf() -> gpd.GeoDataFrame:
    """Small, reproducible clean GeoDataFrame for testing."""
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "gppd_idnr":          [f"WRI{i:06d}" for i in range(n)],
        "name":               [f"Plant_{i}"  for i in range(n)],
        "capacity_mw":        rng.uniform(10, 3000, n).round(1),
        "latitude":           rng.uniform(-60, 75, n).round(4),
        "longitude":          rng.uniform(-180, 180, n).round(4),
        "country":            ["USA"] * n,
        "primary_fuel":       ["Coal"] * n,
        "commissioning_year": [2000] * n,
    })
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInjectErrors:

    def test_output_is_geodataframe(self, clean_gdf):
        dirty = inject_errors(clean_gdf, seed=1)
        assert isinstance(dirty, gpd.GeoDataFrame)

    def test_crs_preserved(self, clean_gdf):
        dirty = inject_errors(clean_gdf, seed=1)
        assert dirty.crs.to_epsg() == 4326

    def test_ocean_coords_injected(self, clean_gdf):
        pct = 0.10
        dirty = inject_errors(clean_gdf, pct_ocean=pct, pct_duplicate=0, pct_missing=0, seed=1)
        ocean_rows = dirty[dirty["error_flags"].str.contains("OCEAN_COORDS")]
        expected = int(len(clean_gdf) * pct)
        # Allow ±1 due to rounding
        assert abs(len(ocean_rows) - expected) <= 1

    def test_duplicate_ids_appended(self, clean_gdf):
        pct = 0.05
        dirty = inject_errors(clean_gdf, pct_ocean=0, pct_duplicate=pct, pct_missing=0, seed=1)
        dup_count = dirty["gppd_idnr"].duplicated(keep=False).sum()
        assert dup_count > 0
        # Total rows should be larger than clean
        assert len(dirty) > len(clean_gdf)

    def test_missing_capacity_injected(self, clean_gdf):
        pct = 0.10
        dirty = inject_errors(clean_gdf, pct_ocean=0, pct_duplicate=0, pct_missing=pct, seed=1)
        null_count = dirty["capacity_mw"].isna().sum()
        expected   = int(len(clean_gdf) * pct)
        assert abs(null_count - expected) <= 2

    def test_error_flags_column_exists(self, clean_gdf):
        dirty = inject_errors(clean_gdf, seed=1)
        assert "error_flags" in dirty.columns

    def test_clean_rows_have_empty_flag(self, clean_gdf):
        dirty = inject_errors(clean_gdf, pct_ocean=0, pct_duplicate=0, pct_missing=0, seed=1)
        assert (dirty["error_flags"] == "").all()

    def test_no_trailing_pipe_in_flags(self, clean_gdf):
        dirty = inject_errors(clean_gdf, seed=1)
        trailing = dirty["error_flags"].str.endswith("|")
        assert not trailing.any()

    def test_reproducible_with_same_seed(self, clean_gdf):
        d1 = inject_errors(clean_gdf, seed=42)
        d2 = inject_errors(clean_gdf, seed=42)
        assert list(d1["error_flags"]) == list(d2["error_flags"])
