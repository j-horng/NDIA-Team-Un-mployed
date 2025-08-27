from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import requests
import yaml

from common.logging_setup import get_logger, setup_logging
from common.types import ImageFrame
from common.geo import pix2geo
from system_c.correlate import orb_ransac_georeg


log = get_logger("system_c")


def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _aoi_center(geojson_path: str) -> Tuple[float, float]:
    g = json.loads(Path(geojson_path).read_text())
    ring = g["features"][0]["geometry"]["coordinates"][0]
    xs = [float(x) for x, _ in ring]
    ys = [float(y) for _, y in ring]
    return (float(sum(ys) / len(ys)), float(sum(xs) / len(xs)))  # (lat, lon)


def _fetch_tile(base_url: str, lat: float, lon: float, zoom: int, timeout: float = 5.0) -> Tuple[np.ndarray, Dict]:
    """
    Request PNG + X-Geo-Metadata from System B.
    Returns (sat_bgr, meta_dict).
    """
    r = requests.get(base_url, params={"lat": lat, "lon": lon, "zoom": zoom}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Imagery API error {r.status_code}: {r.text[:200]}")
    meta = {}
    if "X-Geo-Metadata" in r.headers:
        try:
            meta = json.loads(r.headers["X-Geo-Metadata"])
        except Exception:
            meta = {}
    arr = np.frombuffer(r.content, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode PNG from imagery API")
    return bgr, meta


def _sensor_crop_stream(
    sat_bgr: np.ndarray,
    *,
    window: Tuple[int, int] = (640, 480),
    stride: Tuple[int, int] = (8, 6),
    blur_sigma: float = 1.2,
    rotate_deg: float = 5.0,
):
    """
    Generator: sliding/Jittered crops from a satellite tile to simulate a live camera.
    """
    H, W = sat_bgr.shape[:2]
    w, h = window
    if w >= W or h >= H:
        scale = max(w / W, h / H) * 1.1
        sat_bgr = cv2.resize(sat_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_CUBIC)
        H, W = sat_bgr.shape[:2]

    angle_step = rotate_deg / 200.0 if rotate_deg else 0.0
    k = 0
    while True:
        x = (k * stride[0]) % max(1, W - w)
        y = (k * stride[1]) % max(1, H - h)
        crop = sat_bgr[y : y + h, x : x + w].copy()

        if blur_sigma and blur_sigma > 0:
            crop = cv2.GaussianBlur(crop, (0, 0), blur_sigma)

        if rotate_deg != 0.0:
            ang = -rotate_deg / 2.0 + k * angle_step
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
            crop = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
        yield ImageFrame(ts=ts, width=w, height=h, frame=crop)
        k += 1


def _sat_pix2geo_factory(meta: Dict):
    """
    Returns a function x,y -> lon,lat using our tile metadata format.
    """
    if not all(k in meta for k in ("top_left_lon", "top_left_lat", "px_size_lon", "px_size_lat", "width", "height")):
        raise ValueError("Imagery metadata missing required keys")
    def f(x: float, y: float) -> Tuple[float, float]:
        return pix2geo(x, y, meta)
    return f


def _write_metrics_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="System C — Correlation Pipeline")
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--zoom", type=int, default=None, help="Override zoom level")
    ap.add_argument("--rate", type=float, default=5.0, help="Target GeoFix rate (Hz) after initial warmup")
    ap.add_argument("--window", type=str, default="640x480", help="Camera crop WxH from tile")
    ap.add_argument("--stride", type=str, default="8,6", help="Crop stride dx,dy pixels")
    ap.add_argument("--rotate", type=float, default=5.0, help="Crop rotation amplitude (deg)")
    ap.add_argument("--blur", type=float, default=1.2, help="Gaussian blur sigma")
    args = ap.parse_args()

    P = _load_yaml(args.config)
    setup_logging(P.get("logging", {}).get("level", "INFO"))

    # Config pulls
    base_url = P["imagery_api"]["base_url"]
    zooms = P["tiles"].get("zooms", [18])
    zoom = int(args.zoom or (zooms[0] if zooms else 18))
    aoi_path = P["aoi"]["geojson_path"]
    conf = P["correlation"]
    gate = float(conf.get("conf_gate", 0.6))
    nfeatures = int(conf.get("orb_nfeatures", 2000))
    fast = int(conf.get("orb_fast_threshold", 12))
    ransac_px = float(conf.get("ransac_reproj_px", 3.0))
    min_inliers = int(conf.get("min_inliers", 40))

    # Window/stride parsing
    W, H = [int(x) for x in args.window.lower().replace(",", "x").split("x")]
    dx, dy = [int(x) for x in args.stride.split(",")]

    # Find AOI center and fetch a single tile (cheap, local)
    lat0, lon0 = _aoi_center(aoi_path)
    log.info("Fetching reference tile", extra={"extra": {"lat": lat0, "lon": lon0, "zoom": zoom}})
    sat_bgr, meta = _fetch_tile(base_url, lat0, lon0, zoom)
    sat_gray = cv2.cvtColor(sat_bgr, cv2.COLOR_BGR2GRAY)
    sat_p2g = _sat_pix2geo_factory(meta)

    # Sensor crop generator
    cam_gen = _sensor_crop_stream(sat_bgr, window=(W, H), stride=(dx, dy), blur_sigma=args.blur, rotate_deg=args.rotate)

    # Metrics path
    metrics_path = Path(P["logging"]["metrics_file"])

    # Main loop: correlate → GeoFix → log
    period = 1.0 / max(0.5, float(args.rate))
    last_pub = time.perf_counter()

    log.info("System C pipeline started", extra={"extra": {"rate_hz": 1.0 / period, "gate": gate}})
    while True:
        frame = next(cam_gen)

        t0 = time.perf_counter()
        fix = orb_ransac_georeg(
            frame,
            sat_gray,
            sat_p2g,
            nfeatures=nfeatures,
            fast=fast,
            ransac_px=ransac_px,
            min_inliers=min_inliers,
        )
        dt_ms = int(1000.0 * (time.perf_counter() - t0))

        if fix is None:
            # Log a failure record (useful for dashboard awareness)
            _write_metrics_row(
                metrics_path,
                {
                    "ts": frame.ts,
                    "status": "no_fix",
                    "latency_ms": dt_ms,
                    "conf": 0.0,
                    "inliers": 0,
                    "rmse_px": None,
                },
            )
        else:
            # Respect gate before publishing; dashboard still wants to see everything
            row = {
                "ts": fix.ts,
                "lat": fix.lat,
                "lon": fix.lon,
                "alt_m": fix.alt_m,
                "conf": fix.confidence,
                "inliers": fix.inliers,
                "rmse_px": fix.rmse_px,
                "latency_ms": dt_ms,
                "R": fix.R.tolist(),
            }
            _write_metrics_row(metrics_path, row)

        # Pace output rate a bit so the dashboard/System D can keep up
        # (The correlation itself often dominates; this just prevents tight loops.)
        now = time.perf_counter()
        sleep_for = period - (now - last_pub)
        if sleep_for > 0:
            time.sleep(sleep_for)
        last_pub = time.perf_counter()


if __name__ == "__main__":
    main()
