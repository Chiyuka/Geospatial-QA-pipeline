# 🌍 Geo-Integrity Engine

> **Geospatial Asset Data Quality Assurance** — a production-style Python pipeline for detecting coordinate errors, duplicate IDs, missing values, and statistical anomalies in energy asset databases. Built to mirror the QA workflows used in climate risk platforms like MSCI.

---

## 📌 Overview

Physical climate risk models — for flood, heatwave, wildfire, and hurricane exposure — depend entirely on the **accuracy of asset coordinates and capacity data**. A single mislocated power plant can suppress or inflate a portfolio's risk score by millions of dollars.

This project builds an end-to-end QA engine on the [Global Power Plant Database (GPPD)](https://datasets.wri.org/dataset/globalpowerplantdatabase) (~35,000 real-world energy assets) with five layers of quality assurance:

1. **Loads** real-world assets as a `GeoDataFrame` (EPSG:4326)
2. **Injects** realistic data-quality errors for benchmark testing
3. **Validates** each record with a Pydantic schema + Shapely geometric border check
4. **Profiles** the dataset with a Great Expectations suite across 4 DQ dimensions
5. **Detects** anomalies using a two-layer Isolation Forest (global + per-country)
6. **Visualises** results as an interactive Folium map and Plotly dashboard

---

## 🗂 Project Structure

```
geo-integrity-engine/
│
├── data/                        # Generated datasets (git-ignored)
│   ├── gppd_clean.csv
│   ├── gppd_dirty.csv
│   ├── gppd_scored.csv
│   ├── gppd_ml_scored.csv
│   ├── qa_report.csv
│   └── dq_dimensions.csv
│
├── models/                      # Serialised ML artifacts (git-ignored)
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── notebooks/                   # Exploratory Data Analysis
│
├── reports/                     # Generated HTML visualisations (git-ignored)
│   ├── anomaly_map.html         # Folium interactive map
│   └── anomaly_charts.html      # Plotly 4-panel dashboard
│
├── src/                         # Core pipeline modules
│   ├── loader.py                # GPPD download → GeoDataFrame
│   ├── dirty.py                 # Error injection (3 error types)
│   ├── validation.py            # Pydantic PowerPlantRecord schema
│   ├── schema.py                # Pydantic Asset schema + Shapely country check
│   ├── expectations.py          # Great Expectations QA suite
│   ├── anomalies.py             # Rule-based QA + basic Isolation Forest
│   ├── ml_anomaly.py            # Two-layer Isolation Forest (global + per-country)
│   ├── visualise.py             # Folium map + Plotly dashboard
│   ├── dq_report.py             # Data Quality Dimensions report builder
│   └── run_ml.py                # ML pipeline entry point
│
├── tests/
│   ├── test_dirty.py            # 9 tests for error injection
│   ├── test_validation.py       # 15 tests for Pydantic schema
│   └── test_ml_anomaly.py       # 20 tests for ML pipeline
│
├── run.py                       # Main QA pipeline entry point
├── conftest.py                  # Pytest path configuration
├── pyrightconfig.json           # VS Code / Pylance configuration
├── requirements.txt
└── README.md
```

---

## ⚙️ Quickstart

### 1. Clone & set up environment

```bash
git clone https://github.com/<your-username>/geo-integrity-engine.git
cd geo-integrity-engine

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the main QA pipeline

```bash
python run.py
```

Downloads GPPD, injects errors, validates schema, runs Great Expectations, trains Isolation Forest, builds the DQ Dimensions report.

### 3. Run the ML anomaly detection + visualisation

```bash
python src/run_ml.py
```

Trains global + per-country Isolation Forest models, generates `reports/anomaly_map.html` and `reports/anomaly_charts.html`.

### 4. Open the visualisations

```bash
open reports/anomaly_map.html      # Interactive Folium map
open reports/anomaly_charts.html   # Plotly dashboard
```

### 5. Run tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔍 Error Types Detected

| # | Error Type | Detection Method | Climate Risk Impact |
|---|---|---|---|
| 1 | **Ocean coordinates** — asset placed in water | Spatial join vs. Natural Earth land polygons | Zero physical risk score → exposure under-reported |
| 2 | **Duplicate asset IDs** — same `gppd_idnr` in multiple rows | `value_counts()` deduplication | Asset weight doubled → climate exposure overstated |
| 3 | **Missing capacity** — `capacity_mw` is `NaN` | `isnull()` scan | MW-weighted risk averaging breaks → gaps in transition metrics |
| 4 | **Schema violations** — invalid types, out-of-range lat/lon | Pydantic `Asset` model | Bad records propagate silently into downstream models |
| 5 | **Geometric inconsistency** — coordinates outside reported country | Shapely `Point.within(polygon)` | Country mis-assignment corrupts regional risk aggregation |
| 6 | **Statistical anomalies** — unusual (lat, lon, capacity) combinations | Two-layer Isolation Forest | Catches subtle errors rule-based checks miss |

---

## 📊 Data Quality Dimensions Report

The pipeline outputs a scored report across four industry-standard DQ dimensions (DAMA-DMBOK):

```
=================================================================
  DATA QUALITY DIMENSIONS REPORT
  Dataset  : 2,080 rows  |  GX suite: 5/8 expectations passed
=================================================================

  ACCURACY          100.0%   ✅ PASS
  Coordinates fall within reported country borders (Shapely check)

  COMPLETENESS       33.3%   ❌ FAIL
  capacity_mw: 166 / 2,080 null (8.0% injected + real GPPD gaps)
  commissioning_year: 1,063 / 2,080 null (51.1% — real GPPD issue)

  CONSISTENCY       100.0%   ✅ PASS
  All capacity values ≥ 0, all years within [1900, 2100]

  UNIQUENESS          0.0%   ❌ FAIL
  160 duplicate rows across 80 non-unique IDs

=================================================================
  OVERALL DQ SCORE : 58.3%   ❌ FAIL
=================================================================
```

---

## 🤖 ML Anomaly Detection

The ML layer uses a **two-tier Isolation Forest** approach:

**Layer 1 — Global model** catches extreme outliers regardless of context: ocean-placed assets, impossible capacities, swapped lat/lon coordinates.

**Layer 2 — Per-country models** trains a separate Isolation Forest for each country with ≥ 10 assets (26 countries in the sample). This catches subtle within-country anomalies — a capacity value that is plausible globally but statistically unusual for its country and fuel type.

```
  Total assets      :  2,080
  Clean             :  1,825  (87.7%)
  Anomalous         :    255  (12.3%)

  Country model coverage:
    country          : 1,876 assets (90.2%)
    global_fallback  :   204 assets  (9.8%)

  Top anomalies (ocean-coordinate errors correctly identified):
  GBR1000500   GBR   lat=-2.78   lon=-141.69   score=0.91  🔴
  WRI1007821   ESP   lat=-7.77   lon=-130.37   score=0.85  🔴
```

---

## 🧠 Technical Design Decisions

**Why two-layer Isolation Forest?**
A global model treats all assets equally — a 10 MW solar plant in Germany and a 10 MW solar plant in China get the same treatment. But country context matters: median installed capacity, geographic clustering, and fuel mix vary significantly by region. Per-country models catch anomalies that are invisible globally but obvious locally.

**Why Pydantic + Great Expectations together?**
Pydantic validates one row at a time (schema enforcement at ingestion). Great Expectations validates the whole dataset at once (statistical and completeness assertions). They operate at different granularities and catch different failure modes — using both mirrors production-grade data pipeline architecture.

**Why Shapely spatial join over bounding-box checks?**
Bounding boxes misclassify coastal assets and produce false positives near borders. `Point.within(polygon)` against Natural Earth land polygons is geometrically correct and generalises to any asset class or geography without manual tuning.

**Why 25.8% recall is the right answer, not a problem**
Isolation Forest flags statistical outliers. Duplicate rows are statistically identical to valid rows — same coordinates, same capacity — so they are undetectable by any unsupervised model. Rule-based checks catch duplicates perfectly. The two methods are complementary, not competing, and together cover the full error surface.

---

## 📦 Stack

| Library | Role |
|---|---|
| `geopandas` | GeoDataFrame, spatial joins, CRS management |
| `shapely` | Point geometry + country border checks |
| `pandas` | Tabular data handling |
| `scikit-learn` | Isolation Forest, StandardScaler |
| `pydantic` | Row-level schema validation |
| `great-expectations` | Dataset-level statistical profiling |
| `folium` | Interactive HTML map |
| `plotly` | Anomaly score dashboard |
| `requests` | GPPD remote download |
| `joblib` | Model serialisation |
| `pytest` | 44-test suite with coverage reporting |

---

## 🧪 Test Coverage

```
Name                  Stmts   Cover
------------------------------------
src/dirty.py             35   100%
src/validation.py        45    96%
src/ml_anomaly.py       ~80    85%
------------------------------------
44 tests | 0 failures
```

---

## 📁 Data Source

**Global Power Plant Database v1.3.0**
World Resources Institute (WRI) · License: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
https://datasets.wri.org/dataset/globalpowerplantdatabase

> Downloaded automatically at runtime — no manual download required.

---

## 👤 Author

Built as a portfolio project for the **Geospatial Asset Data Quality Assurance**

CS Student · 
[github.com/Chiyuka](https://github.com/Chiyuka) · [www.linkedin.com/in/phannarong-tuon-734267296](https://www.linkedin.com/in/phannarong-tuon-734267296)
