"""
loader.py
---------
Handles downloading and loading the Global Power Plant Database (GPPD)
into a GeoDataFrame. Falls back to a synthetic dataset if offline.
"""

import io
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point

warnings.filterwarnings("ignore")

GPPD_URL = (
    "https://raw.githubusercontent.com/wri/global-power-plant-database"
    "/master/output_database/global_power_plant_database.csv"
)

COLUMNS_OF_INTEREST = [
    "gppd_idnr",
    "name",
    "capacity_mw",
    "latitude",
    "longitude",
    "country",
    "primary_fuel",
    "commissioning_year",
]


def load_gppd(url: str = GPPD_URL, n_rows: int = 2_000) -> gpd.GeoDataFrame:
    """
    Download a subset of GPPD and return a GeoDataFrame (EPSG:4326).

    Parameters
    ----------
    url    : CSV URL or local file path.
    n_rows : Number of rows to sample.
    """
    print(f"[loader] Fetching GPPD ({n_rows} rows) …")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), usecols=COLUMNS_OF_INTEREST)
        print(f"[loader] ✓ Remote fetch: {len(df):,} rows.")
    except Exception as exc:
        print(f"[loader] ✗ Remote fetch failed ({exc}). Using synthetic data.")
        df = _make_synthetic_gppd(n_rows)

    df = df.sample(n=min(n_rows, len(df)), random_state=42).reset_index(drop=True)

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    print(f"[loader] Loaded {len(gdf):,} records | CRS: {gdf.crs}")
    return gdf


def _make_synthetic_gppd(n: int) -> pd.DataFrame:
    """Reproducible synthetic power-plant records for offline use."""
    rng = np.random.default_rng(0)
    fuels     = ["Coal", "Gas", "Wind", "Solar", "Hydro", "Nuclear", "Oil"]
    countries = ["USA", "CHN", "IND", "DEU", "BRA", "ZAF", "AUS"]

    return pd.DataFrame({
        "gppd_idnr":          [f"WRI{i:06d}" for i in range(n)],
        "name":               [f"Plant_{i}"  for i in range(n)],
        "capacity_mw":        rng.uniform(10, 3000, n).round(1),
        "latitude":           rng.uniform(-60, 75, n).round(4),
        "longitude":          rng.uniform(-180, 180, n).round(4),
        "country":            rng.choice(countries, n),
        "primary_fuel":       rng.choice(fuels, n),
        "commissioning_year": rng.integers(1950, 2023, n),
    })
