"""
schema.py
---------
Pydantic v2 schema for a geospatial 'Asset' record.

Responsibilities
----------------
1. Type & range enforcement   — id, lat, lon, capacity, country
2. Geometric country check    — does the (lat, lon) point actually fall
                                inside the reported country's borders?
                                Uses Shapely + Natural Earth polygons.

Why separate from validation.py?
  validation.py holds the original PowerPlantRecord (full GPPD schema).
  This module defines a leaner 'Asset' model that maps directly to the
  5 fields the MSCI job description specifies, and adds the geometric
  consistency check as a model-level validator.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Optional

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from shapely.geometry import Point

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Land / country polygon cache  (loaded once, reused for every record)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_country_polygons() -> gpd.GeoDataFrame:
    """
    Load Natural Earth low-resolution country polygons.
    Cached with @lru_cache so we only hit disk once per Python session.

    Returns GeoDataFrame indexed by ISO-A3 country code (e.g. 'USA', 'DEU').
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # naturalearth_lowres uses 'iso_a3' — drop rows where it's missing (-99)
    world = world[world["iso_a3"] != "-99"].copy()
    world = world.set_index("iso_a3")
    return world


def point_in_country(lat: float, lon: float, iso_a3: str) -> bool:
    """
    Return True if Point(lon, lat) falls within the borders of `iso_a3`.

    Uses a 0.5-degree buffer to absorb coastline inaccuracies in the
    low-resolution Natural Earth dataset (so coastal plants aren't
    incorrectly flagged).

    Parameters
    ----------
    lat    : Latitude  (WGS-84)
    lon    : Longitude (WGS-84)
    iso_a3 : ISO 3166-1 alpha-3 country code (e.g. 'USA')
    """
    try:
        world = _load_country_polygons()
        if iso_a3 not in world.index:
            return True          # unknown country code → skip spatial check
        country_geom = world.loc[iso_a3, "geometry"]
        plant_point  = Point(lon, lat)
        # Buffer by ~0.5° to handle coastline tolerance
        return plant_point.within(country_geom.buffer(0.5))
    except Exception:
        return True              # if spatial check fails, don't block the record


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Asset schema
# ─────────────────────────────────────────────────────────────────────────────

class Asset(BaseModel):
    """
    Schema for a single geospatial asset record.

    Fields
    ------
    id          : Unique string identifier (non-blank).
    latitude    : WGS-84 latitude  ∈ [-90,  90].
    longitude   : WGS-84 longitude ∈ [-180, 180].
    capacity_mw : Installed capacity in MW (≥ 0, may be None/missing).
    country     : ISO 3166-1 alpha-3 country code (exactly 3 letters).

    Validators (run in order)
    -------------------------
    1. id_not_blank        — field-level, rejects whitespace-only IDs.
    2. coerce_nan_capacity — field-level, converts float NaN → None.
    3. check_point_in_country — model-level, geometric border check.
    """

    id:          str
    latitude:    float          = Field(..., ge=-90,  le=90)
    longitude:   float          = Field(..., ge=-180, le=180)
    capacity_mw: Optional[float] = Field(None, ge=0)
    country:     str            = Field(..., min_length=3, max_length=3)

    # ── Field validators ──────────────────────────────────────────────────

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Asset id must not be blank or whitespace.")
        return v.strip()

    @field_validator("capacity_mw", mode="before")
    @classmethod
    def coerce_nan_capacity(cls, v):
        """Convert float NaN to None so Optional[float] path is used."""
        if v is None:
            return None
        try:
            return None if math.isnan(float(v)) else float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("country")
    @classmethod
    def country_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    # ── Model validator (runs after all field validators pass) ────────────

    @model_validator(mode="after")
    def check_point_in_country(self) -> "Asset":
        """
        Geometric consistency check:
        Does the (lat, lon) coordinate actually fall inside `country`?

        This catches the classic MSCI data-quality error: a plant whose
        country field says 'USA' but whose coordinates are in Europe,
        or in the middle of the ocean.
        """
        inside = point_in_country(self.latitude, self.longitude, self.country)
        if not inside:
            raise ValueError(
                f"Coordinates ({self.latitude}, {self.longitude}) do not fall "
                f"within the borders of '{self.country}'. "
                f"Possible country mis-assignment or coordinate error."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-level validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_assets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate every row in `df` against the Asset schema.

    Expects columns: id, latitude, longitude, capacity_mw, country.
    Maps from GPPD column names automatically if needed.

    Returns
    -------
    Original DataFrame plus:
        asset_valid         : bool — passed all Pydantic + geometric checks.
        asset_errors        : str  — pipe-separated error messages.
        geometry_consistent : bool — specifically the country-border check.
    """
    print("[schema] Running Asset schema + geometric validation …")

    # ── Column name normalisation (GPPD → Asset field names) ─────────────
    rename_map = {
        "gppd_idnr":   "id",
        "capacity_mw": "capacity_mw",   # same, kept for clarity
    }
    working = df.rename(columns=rename_map).copy()

    # Ensure required columns exist
    required = {"id", "latitude", "longitude", "capacity_mw", "country"}
    missing  = required - set(working.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    results = []
    for _, row in working.iterrows():
        record = {
            "id":          str(row["id"]),
            "latitude":    row["latitude"],
            "longitude":   row["longitude"],
            "capacity_mw": row["capacity_mw"],
            "country":     str(row["country"]),
        }
        try:
            Asset(**record)
            results.append({
                "asset_valid":         True,
                "asset_errors":        "",
                "geometry_consistent": True,
            })
        except Exception as exc:
            errors   = str(exc)
            geo_fail = "do not fall within" in errors or "border" in errors
            results.append({
                "asset_valid":         False,
                "asset_errors":        errors,
                "geometry_consistent": not geo_fail,
            })

    out = df.copy().reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(results)], axis=1)

    n_invalid = (~out["asset_valid"]).sum()
    n_geo     = (~out["geometry_consistent"]).sum()
    print(f"[schema] ✓ {len(df):,} rows | {n_invalid:,} failed schema | "
          f"{n_geo:,} geometry inconsistencies.\n")
    return out
