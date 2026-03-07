"""
validation.py
-------------
Pydantic-based schema validation for individual power-plant records.
Catches type errors, out-of-range coordinates, and negative capacity
before any ML or spatial analysis runs.

Why Pydantic?
  - Validates one record at a time (row-level schema enforcement).
  - Raises clear errors with field names — easy to log per asset.
  - Complements Great Expectations (dataset-level stats) nicely.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

class PowerPlantRecord(BaseModel):
    """Pydantic schema for a single GPPD row."""

    gppd_idnr:          str
    name:               str
    capacity_mw:        Optional[float] = Field(None, ge=0)
    latitude:           float           = Field(..., ge=-90,  le=90)
    longitude:          float           = Field(..., ge=-180, le=180)
    country:            str             = Field(..., min_length=3, max_length=3)
    primary_fuel:       str
    commissioning_year: Optional[int]   = Field(None, ge=1900, le=2100)

    @field_validator("gppd_idnr")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("gppd_idnr must not be blank.")
        return v

    @field_validator("capacity_mw", mode="before")
    @classmethod
    def coerce_nan_to_none(cls, v):
        """Treat NaN as None so the Optional[float] path is used."""
        if v is None:
            return None
        try:
            import math
            return None if math.isnan(float(v)) else float(v)
        except (TypeError, ValueError):
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Row-level validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate every row in `df` against PowerPlantRecord.

    Returns
    -------
    pd.DataFrame with two extra columns:
        pydantic_valid  : bool   – True if the row passed validation.
        pydantic_errors : str    – Comma-separated error messages (or "").
    """
    print("[validation] Running Pydantic schema checks …")
    results = []

    for _, row in df.iterrows():
        record = row.to_dict()
        try:
            PowerPlantRecord(**record)
            results.append({"pydantic_valid": True, "pydantic_errors": ""})
        except ValidationError as exc:
            msgs = ", ".join(
                f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()
            )
            results.append({"pydantic_valid": False, "pydantic_errors": msgs})

    result_df = df.copy().reset_index(drop=True)
    result_df = pd.concat([result_df, pd.DataFrame(results)], axis=1)

    invalid = (~result_df["pydantic_valid"]).sum()
    print(f"[validation] ✓ {len(df):,} rows checked | {invalid:,} failed schema.\n")
    return result_df
