# 🌍 Geo-Integrity Engine

> **Geospatial Asset Data Quality Assurance** — a production-style Python pipeline for detecting coordinate errors, duplicate IDs, and missing values in energy asset databases. Built to mirror the QA workflows used in climate risk platforms like MSCI.

---

## 📌 Overview

Physical climate risk models — for flood, heatwave, wildfire, and hurricane exposure — depend entirely on the **accuracy of asset coordinates and capacity data**. A single mislocated power plant can suppress or inflate a portfolio's risk score by millions of dollars.

This project builds an end-to-end QA engine on the [Global Power Plant Database (GPPD)](https://datasets.wri.org/dataset/globalpowerplantdatabase) that:

1. **Loads** ~35,000 real-world energy assets as a `GeoDataFrame`
2. **Injects** realistic data-quality errors (ocean coordinates, duplicate IDs, missing capacity)
3. **Validates** each record against a strict Pydantic schema
4. **Detects** anomalies using both rule-based spatial logic and an unsupervised **Isolation Forest**

---

## 🗂 Project Structure

```
geo-integrity-engine/
│
├── data/                    # Raw & dirty datasets (git-ignored)
│   ├── gppd_clean.csv
│   ├── gppd_dirty.csv
│   ├── gppd_scored.csv
│   └── qa_report.csv
│
├── models/                  # Serialised ML model artifacts
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── notebooks/               # Exploratory Data Analysis
│   └── 01_eda.ipynb
│
├── src/                     # Core pipeline modules
│   ├── loader.py            # GPPD download → GeoDataFrame
│   ├── dirty.py             # Error injection (3 error types)
│   ├── validation.py        # Pydantic schema enforcement
│   └── anomalies.py        # Isolation Forest + rule-based QA
│
├── tests/                   # Pytest test suites
│   ├── test_dirty.py
│   └── test_validation.py
│
├── run.py                   # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/geo-integrity-engine.git
cd geo-integrity-engine
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run.py
```

This will download GPPD, inject errors, validate, train the model, score anomalies, and write all outputs to `data/`.

### 3. Run tests

```bash
pytest tests/ -v --tb=short

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔍 Error Types Detected

| # | Error Type | Detection Method | Climate Risk Impact |
|---|---|---|---|
| 1 | **Ocean coordinates** — asset placed in water | Spatial join vs. Natural Earth land polygons | Zero physical risk score assigned → exposure under-reported |
| 2 | **Duplicate asset IDs** — same `gppd_idnr` in multiple rows | `value_counts()` deduplication | Asset weight doubled → climate exposure overstated |
| 3 | **Missing capacity** — `capacity_mw` is `NaN` | `isnull()` scan | MW-weighted risk averaging breaks → gaps in transition metrics |
| 4 | **Schema violations** — invalid types, out-of-range lat/lon | Pydantic `PowerPlantRecord` | Bad records propagate silently into downstream models |
| 5 | **Statistical anomalies** — unusual (lat, lon, capacity) combos | Isolation Forest (`sklearn`) | Catches errors rule-based checks miss |

---

## 🧠 Technical Design

### Why Isolation Forest?

Rule-based checks catch *known* error patterns. Isolation Forest catches *unknown* ones — records that are individually valid but collectively anomalous. For example: a 3 MW nuclear plant in central Europe, or a 2,000 MW wind farm in a desert. These pass schema validation but are statistically implausible, and Isolation Forest surfaces them without labelled training data.

### Why Pydantic?

Pydantic enforces row-level schema contracts at the boundary of ingestion — before any spatial or ML logic runs. It catches type coercions, range violations (`latitude > 90`), and blank IDs that pandas silently accepts. This mirrors how production data pipelines at firms like MSCI gate data quality upstream.

### Why GeoPandas + Spatial Join?

A simple lat/lon bounding-box check for "ocean" is brittle — it misses coastal errors and generates false positives near coastlines. A spatial join against Natural Earth land polygons is geometrically correct and scales to any asset class (not just power plants).

---

## 📦 Stack

| Library | Role |
|---|---|
| `geopandas` | GeoDataFrame, spatial joins, CRS management |
| `shapely` | Point geometry construction |
| `pandas` | Tabular data handling |
| `scikit-learn` | Isolation Forest, StandardScaler |
| `pydantic` | Row-level schema validation |
| `requests` | GPPD remote download |
| `joblib` | Model serialisation |
| `pytest` | Unit testing |

---

## 📊 Sample Output

```
======================================================
  Geo-Integrity Engine — MSCI Climate Risk QA Pipeline
======================================================

[loader]    Loaded 2,000 records | CRS: EPSG:4326
[dirty]     Ocean coordinates injected into 100 rows.
[dirty]     Duplicate-ID rows appended: 80.
[dirty]     capacity_mw nulled in 169 rows.
[validation] 249 rows failed schema validation.
[anomalies] 169 / 2,163 records flagged as anomalies.

  QA Summary
======================================================
  Clean records       :  2,000
  Dirty records       :  2,163
  Schema failures     :    249
  ML anomalies        :    169

  OCEAN_COORDS        :   100 rows
  DUPLICATE_ID        :   160 rows
  MISSING_CAPACITY    :   169 rows
======================================================
```

---

## 📁 Data Source

**Global Power Plant Database v1.3.0**
World Resources Institute (WRI) · License: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
https://datasets.wri.org/dataset/globalpowerplantdatabase

> The dataset is downloaded automatically at runtime. No manual download required.

---

## 🗺 Roadmap

- [ ] Folium interactive map of flagged assets
- [ ] Great Expectations dataset-level profiling
- [ ] Country-code cross-validation (point vs. `country` field)
- [ ] Support for additional asset classes (dams, substations, refineries)
- [ ] Streamlit dashboard for non-technical reviewers

---

## 👤 Author

Built as a portfolio project for the **Geospatial Asset Data Quality Assurance** internship at MSCI's Climate Risk Center.

Feel free to open an issue or reach out if you have questions about the methodology.