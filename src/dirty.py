"""
dirty.py
--------
Intentionally injects geospatial data-quality errors into a clean GeoDataFrame
to simulate the real-world issues MSCI flags in asset databases.

Error types
-----------
A. OCEAN_COORDS     → coordinates placed in mid-ocean (geospatial integrity)
B. DUPLICATE_ID     → same gppd_idnr appears in multiple rows (provenance)
C. MISSING_CAPACITY → capacity_mw is NaN (exposure modelling gaps)
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# Mid-ocean bounding boxes: (lon_range, lat_range)
OCEAN_ZONES = [
    ((-170, -120), (-10,  10)),   # Central Pacific
    (( -40,  -20), (  5,  20)),   # Mid-Atlantic
    ((  60,   90), (-30, -10)),   # Indian Ocean
]


def inject_errors(
    gdf: gpd.GeoDataFrame,
    pct_ocean:     float = 0.05,
    pct_duplicate: float = 0.04,
    pct_missing:   float = 0.08,
    seed:          int   = 99,
) -> gpd.GeoDataFrame:
    """
    Return a corrupted copy of `gdf` with three injected error types.

    Parameters
    ----------
    gdf           : Clean input GeoDataFrame.
    pct_ocean     : Fraction of rows to move into mid-ocean coordinates.
    pct_duplicate : Fraction of rows to duplicate (same ID appended).
    pct_missing   : Fraction of capacity_mw values to null out.
    seed          : Random seed for reproducibility.
    """
    print("[dirty] Injecting errors …")
    rng   = np.random.default_rng(seed)
    dirty = gdf.copy()
    dirty["error_flags"] = ""
    n = len(dirty)

    # ── A: Ocean coordinates ──────────────────────────────────────────────
    ocean_idx = rng.choice(n, size=int(n * pct_ocean), replace=False)
    for i, idx in enumerate(ocean_idx):
        zone = OCEAN_ZONES[i % len(OCEAN_ZONES)]
        lon  = round(rng.uniform(*zone[0]), 4)
        lat  = round(rng.uniform(*zone[1]), 4)
        dirty.at[idx, "longitude"] = lon
        dirty.at[idx, "latitude"]  = lat
        dirty.at[idx, "geometry"]  = Point(lon, lat)
        dirty.at[idx, "error_flags"] += "OCEAN_COORDS|"
    print(f"[dirty] ✓ Ocean coordinates injected into {len(ocean_idx)} rows.")

    # ── B: Duplicate IDs ──────────────────────────────────────────────────
    dup_idx  = rng.choice(n, size=int(n * pct_duplicate), replace=False)
    dup_rows = dirty.iloc[dup_idx].copy()
    dup_rows["error_flags"] = dup_rows["error_flags"] + "DUPLICATE_ID|"
    dirty = gpd.GeoDataFrame(
        gpd.pd.concat([dirty, dup_rows], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    print(f"[dirty] ✓ Duplicate-ID rows appended: {len(dup_rows)}.")

    # ── C: Missing capacity ───────────────────────────────────────────────
    n2       = len(dirty)
    miss_idx = rng.choice(n2, size=int(n2 * pct_missing), replace=False)
    dirty.loc[miss_idx, "capacity_mw"] = float("nan")
    for idx in miss_idx:
        if "MISSING_CAPACITY" not in dirty.at[idx, "error_flags"]:
            dirty.at[idx, "error_flags"] += "MISSING_CAPACITY|"
    print(f"[dirty] ✓ capacity_mw nulled in {len(miss_idx)} rows.")

    dirty["error_flags"] = dirty["error_flags"].str.rstrip("|")
    print(f"[dirty] Dirty dataset: {len(dirty):,} rows total.\n")
    return dirty
