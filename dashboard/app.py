"""
A-PNT Dashboard (Streamlit)

- Tails logs/metrics.jsonl written by System C pipeline
- Shows live KPIs: GeoFix rate, Confidence, Inliers, RMSE, Latency
- Plots time series (latency, confidence, inliers)
- Renders a pydeck map of recent GeoFix locations with confidence coloring
- Optional: loads AOI polygon from config/area_of_interest.geojson

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pydeck as pdk
import streamlit as st

# Optional nicer tables/plots if pandas is available
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# -------------------------
# Config
# -------------------------
LOG_PATH_DEFAULT = Path("logs/metrics.jsonl")
AOI_GEOJSON = Path("config/area_of_interest.geojson")
MAX_ROWS = 500  # how many recent rows to load
MAP_POINTS = 250  # how many points to show on the map


# -------------------------
# Helpers
# -------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def load_last_jsonl(path: Path, max_rows: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    # Read tail efficiently
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # heuristic: read last ~256 KB to avoid huge files
        read_back = min(size, 256 * 1024)
        f.seek(size - read_back)
        chunk = f.read().decode("utf-8", errors="ignore")
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    rows: List[Dict[str, Any]] = []
    for ln in lines[-max_rows:]:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    return rows


def load_aoi_polygon(path: Path) -> Optional[List[List[float]]]:
    """Returns outer ring [[lon,lat], ...] if present."""
    try:
        g = json.loads(path.read_text())
        feats = g.get("features") or []
        if not feats:
            return None
        geom = feats[0].get("geometry", {})
        if geom.get("type") != "Polygon":
            return None
        coords = geom.get("coordinates", [])
        if not coords:
            return None
        # first ring
        ring = coords[0]
        return [[float(x), float(y)] for x, y in ring]
    except Exception:
        return None


def rate_hz_from_timestamps(ts_list: List[str]) -> float:
    """Compute approximate rate (Hz) from ISO timestamps (list must be ordered)."""
    if len(ts_list) < 2:
        return 0.0
    t0 = _parse_iso(ts_list[0])
    t1 = _parse_iso(ts_list[-1])
    if not t0 or not t1:
        return 0.0
    span_s = (t1 - t0).total_seconds()
    if span_s <= 0:
        return 0.0
    return (len(ts_list) - 1) / span_s


def latency_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    vals = [r.get("latency_ms", 0) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    if not vals:
        return {"latest": 0.0, "p50": 0.0, "p95": 0.0}
    latest = float(vals[-1])
    arr = np.array(vals, dtype=float)
    return {"latest": latest, "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def to_dataframe(rows: List[Dict[str, Any]]) -> Optional["pd.DataFrame"]:
    if pd is None:
        return None
    # Normalize keys we care about; ignore others
    cols = ["ts", "lat", "lon", "conf", "inliers", "rmse_px", "latency_ms"]
    # Filter out incomplete rows
    filtered = [{k: r.get(k, None) for k in cols} for r in rows if "lat" in r and "lon" in r]
    if not filtered:
        return None
    df = pd.DataFrame(filtered)
    # Ensure numeric types
    for c in ["lat", "lon", "conf", "inliers", "rmse_px", "latency_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def conf_to_color(conf: float) -> List[int]:
    """
    Map confidence [0..1] → RGBA color.
    Low = red, mid = amber, high = green.
    """
    c = max(0.0, min(1.0, float(conf)))
    # simple red→green gradient with some alpha
    r = int(255 * (1.0 - c))
    g = int(255 * (0.2 + 0.8 * c))
    b = int(60 * (1.0 - c))
    return [r, g, b, 180]


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="A-PNT Dashboard", layout="wide")
st.title("A-PNT: Georegistration & Fusion Dashboard")

with st.sidebar:
    st.subheader("Data Sources")
    log_path = st.text_input("Metrics JSONL", str(LOG_PATH_DEFAULT))
    log_path = str(Path(log_path))
    aoi_path = st.text_input("AOI GeoJSON (optional)", str(AOI_GEOJSON))
    refresh = st.button("Refresh now")
    st.caption("Tip: Keep this page open; click Refresh to pull the latest metrics.")

# Load data
rows = load_last_jsonl(Path(log_path), MAX_ROWS)
if not rows:
    st.warning(
        "No metrics found yet. Start System B (imagery API), System C (pipeline), "
        "and let it write to `logs/metrics.jsonl`."
    )
    st.stop()

# Compute KPIs
rate_hz = rate_hz_from_timestamps([r.get("ts", "") for r in rows])
lat_stats = latency_stats(rows)
latest = rows[-1]
latest_conf = float(latest.get("conf", 0.0))
latest_inliers = int(latest.get("inliers", 0) or 0)
latest_rmse = float(latest.get("rmse_px", 0.0))
latest_lat = latest.get("lat", None)
latest_lon = latest.get("lon", None)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("GeoFix Rate (Hz)", f"{rate_hz:.2f}")
k2.metric("Confidence (latest)", f"{latest_conf:.2f}")
k3.metric("Inliers (latest)", f"{latest_inliers}")
k4.metric("RMSE (px, latest)", f"{latest_rmse:.2f}")
k5.metric("Latency (ms, latest)", f"{lat_stats['latest']:.0f}")
k6.metric("Latency p95 (ms)", f"{lat_stats['p95']:.0f}")

# Charts
left, right = st.columns(2)
with left:
    st.subheader("Latency (ms)")
    lat_series = np.array([r.get("latency_ms", 0.0) for r in rows], dtype=float)
    st.line_chart(lat_series, height=220)
    st.caption(f"Median: {lat_stats['p50']:.0f} ms · p95: {lat_stats['p95']:.0f} ms")

with right:
    st.subheader("Confidence & Inliers")
    conf_series = np.array([r.get("conf", 0.0) for r in rows], dtype=float)
    inl_series = np.array([r.get("inliers", 0) for r in rows], dtype=float)
    st.line_chart(
        np.vstack([conf_series, inl_series]).T,  # 2 columns (conf, inliers)
        height=220,
    )
    st.caption("Blue: confidence (0..1), Orange: inliers (count)")

# Map
st.subheader("GeoFix Map (recent)")
aoi_ring = load_aoi_polygon(Path(aoi_path)) if aoi_path else None

# Prepare features for pydeck
pts = []
for r in rows[-MAP_POINTS:]:
    lat = r.get("lat", None)
    lon = r.get("lon", None)
    conf = float(r.get("conf", 0.0) or 0.0)
    if lat is None or lon is None:
        continue
    pts.append({"position": [float(lon), float(lat)], "color": conf_to_color(conf), "conf": conf})

if pts:
    # View state: center on latest point (fallback to AOI centroid)
    if latest_lat is not None and latest_lon is not None:
        center_lat, center_lon = float(latest_lat), float(latest_lon)
    elif aoi_ring:
        xs = [p[0] for p in aoi_ring]
        ys = [p[1] for p in aoi_ring]
        center_lon = float(sum(xs) / len(xs))
        center_lat = float(sum(ys) / len(ys))
    else:
        center_lat, center_lon = 38.8895, -77.0352  # default (DC Mall)

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position="position",
            get_fill_color="color",
            get_radius=8,
            pickable=True,
        )
    ]
    if aoi_ring:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": aoi_ring}],
                get_polygon="polygon",
                get_fill_color=[0, 128, 255, 30],
                get_line_color=[0, 128, 255, 160],
                line_width_min_pixels=1,
            )
        )

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=16,
                pitch=0,
                bearing=0,
            ),
            layers=layers,
            tooltip={"text": "Confidence: {conf}"},
        )
    )
else:
    st.info("No lat/lon points available yet to render on the map.")

# Table of recent rows (if pandas available)
st.subheader("Recent GeoFix Records")
df = to_dataframe(rows[-50:])  # last 50 rows
if df is not None:
    st.dataframe(df, use_container_width=True, height=300)
else:
    # Fallback: show the latest JSON if pandas is not installed
    st.json(latest)

st.caption(
    "Source: logs/metrics.jsonl · "
    f"Last refresh: {_now_iso()} · Rows loaded: {len(rows)} (showing up to {MAX_ROWS})"
)
