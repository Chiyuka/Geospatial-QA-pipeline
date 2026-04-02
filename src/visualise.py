"""
visualise.py
------------
Generates two visualisations from the ML anomaly detection results:

  1. Folium interactive HTML map
       Blue markers  → clean assets
       Red markers   → anomalies flagged by Isolation Forest
       Marker size   → proportional to capacity_mw
       Popup         → asset ID, country, capacity, anomaly score

  2. Plotly static charts (saved as HTML)
       a) Anomaly score distribution histogram
       b) Scatter: capacity_mw vs latitude, coloured by anomaly status
       c) Top 20 countries by anomaly rate bar chart

Both outputs are saved to the reports/ directory.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Folium map
# ─────────────────────────────────────────────────────────────────────────────

def build_folium_map(df: pd.DataFrame, output_path: str = "reports/anomaly_map.html") -> str:
    """
    Build an interactive Folium map.

    Clean assets   → blue CircleMarker
    Anomalous      → red CircleMarker
    Marker radius  → scaled by log(capacity_mw)
    Popup          → full asset detail on click

    Parameters
    ----------
    df          : Scored DataFrame with IS_ANOMALY, ANOMALY_SCORE columns.
    output_path : Where to save the HTML file.

    Returns
    -------
    Path to saved HTML file.
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        raise ImportError("Install folium: pip install folium")

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("[visualise] Building Folium map …")

    # Centre map on data centroid
    centre_lat = df["latitude"].median()
    centre_lon = df["longitude"].median()

    fmap = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=3,
        tiles="CartoDB positron",   # clean light basemap
    )

    # ── Layer groups so user can toggle clean/anomalous ───────────────────
    clean_layer   = folium.FeatureGroup(name="✅ Clean Assets",    show=True)
    anomaly_layer = folium.FeatureGroup(name="🔴 Anomalous Assets", show=True)

    # ── Normalise radius: log(capacity) mapped to [4, 18] pixels ─────────
    cap   = df["cap_filled"] if "cap_filled" in df.columns else df["capacity_mw"].fillna(100)
    r_raw = np.log1p(cap)
    r_min, r_max = r_raw.min(), r_raw.max()
    radii = 4 + 14 * (r_raw - r_min) / (r_max - r_min + 1e-9)

    n_clean   = 0
    n_anomaly = 0

    for idx, row in df.iterrows():
        is_anom = bool(row["IS_ANOMALY"])
        score   = float(row["ANOMALY_SCORE"])
        radius  = float(radii.iloc[idx] if hasattr(radii, 'iloc') else radii[idx])

        color      = "#e74c3c" if is_anom else "#2980b9"   # red or blue
        fill_color = "#ff6b6b" if is_anom else "#5dade2"
        opacity    = 0.85      if is_anom else 0.55

        # Build popup HTML
        name     = row.get("name", row.get("gppd_idnr", "Unknown"))
        asset_id = row.get("gppd_idnr", "N/A")
        country  = row.get("country", "N/A")
        fuel     = row.get("primary_fuel", "N/A")
        capacity = row.get("capacity_mw", "N/A")
        model    = row.get("model_used", "N/A")

        status_icon = "🔴 ANOMALY" if is_anom else "✅ CLEAN"
        popup_html  = f"""
        <div style="font-family: Arial; font-size: 13px; width: 220px;">
            <b>{name}</b><br>
            <hr style="margin:4px 0">
            <b>Status:</b> {status_icon}<br>
            <b>ID:</b> {asset_id}<br>
            <b>Country:</b> {country}<br>
            <b>Fuel:</b> {fuel}<br>
            <b>Capacity:</b> {capacity} MW<br>
            <b>Anomaly Score:</b>
            <span style="color:{'#e74c3c' if is_anom else '#27ae60'}">
                <b>{score:.3f}</b>
            </span><br>
            <b>Model:</b> {model}<br>
            <b>Lat/Lon:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
        </div>
        """

        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=opacity,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{'⚠ ' if is_anom else ''}{name} | Score: {score:.2f}",
        )

        if is_anom:
            marker.add_to(anomaly_layer)
            n_anomaly += 1
        else:
            marker.add_to(clean_layer)
            n_clean += 1

    clean_layer.add_to(fmap)
    anomaly_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_html = f"""
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 9999;
        background: white; padding: 14px 18px; border-radius: 8px;
        border: 1px solid #ccc; font-family: Arial; font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b>Geo-Integrity Engine</b><br>
        <b>Isolation Forest Anomaly Detection</b>
        <hr style="margin: 6px 0">
        <span style="color:#2980b9">&#11044;</span> Clean asset
            &nbsp;&nbsp;<b>{n_clean:,}</b><br>
        <span style="color:#e74c3c">&#11044;</span> Anomalous asset
            &nbsp;<b>{n_anomaly:,}</b><br>
        <hr style="margin: 6px 0">
        <small>Marker size ∝ log(capacity MW)<br>
        Click marker for details</small>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(output_path)
    print(f"[visualise] ✓ Folium map saved → {output_path}")
    print(f"[visualise]   Clean: {n_clean:,}  |  Anomalous: {n_anomaly:,}\n")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Plotly charts
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_charts(
    df: pd.DataFrame, output_path: str = "reports/anomaly_charts.html"
) -> str:
    """
    Build a multi-panel Plotly dashboard saved as a single HTML file.

    Panel 1 — Anomaly score distribution (histogram)
    Panel 2 — Capacity vs latitude scatter (coloured by anomaly)
    Panel 3 — Top 20 countries by anomaly rate (bar chart)

    Parameters
    ----------
    df          : Scored DataFrame.
    output_path : Where to save the HTML file.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Install plotly: pip install plotly")

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("[visualise] Building Plotly dashboard …")

    clean   = df[~df["IS_ANOMALY"]]
    anomaly = df[ df["IS_ANOMALY"]]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Anomaly Score Distribution",
            "Capacity (MW) vs Latitude",
            "Anomaly Rate by Country (Top 20)",
            "Global Scatter: Longitude vs Latitude",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    BLUE = "#2980b9"
    RED  = "#e74c3c"

    # ── Panel 1: Score histogram ──────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=clean["ANOMALY_SCORE"],
        name="Clean",
        marker_color=BLUE,
        opacity=0.7,
        nbinsx=40,
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=anomaly["ANOMALY_SCORE"],
        name="Anomalous",
        marker_color=RED,
        opacity=0.7,
        nbinsx=40,
        showlegend=True,
    ), row=1, col=1)

    # Threshold line
    fig.add_vline(x=0.60, line_dash="dash", line_color="orange",
                  annotation_text="Threshold 0.6", row=1, col=1)

    # ── Panel 2: Capacity vs latitude ─────────────────────────────────────
    cap_col = "cap_filled" if "cap_filled" in df.columns else "capacity_mw"

    for subset, color, name in [(clean, BLUE, "Clean"), (anomaly, RED, "Anomalous")]:
        fig.add_trace(go.Scatter(
            x=subset["latitude"],
            y=subset[cap_col].fillna(0),
            mode="markers",
            name=name,
            marker=dict(color=color, size=4, opacity=0.6),
            showlegend=False,
            text=subset.get("country", pd.Series([""] * len(subset))),
        ), row=1, col=2)

    # ── Panel 3: Anomaly rate by country ──────────────────────────────────
    country_stats = (
        df.groupby("country")["IS_ANOMALY"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "anomalies", "count": "total"})
    )
    country_stats["rate"] = country_stats["anomalies"] / country_stats["total"]
    top20 = country_stats.nlargest(20, "anomalies").sort_values("rate", ascending=True)

    fig.add_trace(go.Bar(
        x=top20["rate"],
        y=top20.index,
        orientation="h",
        marker_color=RED,
        opacity=0.8,
        showlegend=False,
        text=[f"{r:.0%}" for r in top20["rate"]],
        textposition="outside",
    ), row=2, col=1)

    # ── Panel 4: Global lon/lat scatter ───────────────────────────────────
    for subset, color, name in [(clean, BLUE, "Clean"), (anomaly, RED, "Anomalous")]:
        fig.add_trace(go.Scatter(
            x=subset["longitude"],
            y=subset["latitude"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=4, opacity=0.5),
            showlegend=False,
        ), row=2, col=2)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>Geo-Integrity Engine — Isolation Forest Anomaly Detection</b>",
            font=dict(size=18),
            x=0.5,
        ),
        height=750,
        barmode="overlay",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
        ),
        font=dict(family="Arial", size=12),
    )

    fig.update_xaxes(title_text="Anomaly Score", row=1, col=1)
    fig.update_yaxes(title_text="Count",         row=1, col=1)
    fig.update_xaxes(title_text="Latitude",      row=1, col=2)
    fig.update_yaxes(title_text="Capacity (MW)", row=1, col=2)
    fig.update_xaxes(title_text="Anomaly Rate",  row=2, col=1)
    fig.update_xaxes(title_text="Longitude",     row=2, col=2)
    fig.update_yaxes(title_text="Latitude",      row=2, col=2)

    fig.write_html(output_path)
    print(f"[visualise] ✓ Plotly dashboard saved → {output_path}\n")
    return output_path