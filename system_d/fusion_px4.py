from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import yaml
from pymavlink import mavutil

from common.geo import lla_to_ned
from system_d.mavlink_out import (
    open_px4_connection,
    send_gps_input,
    send_vision_position_estimate,
    accuracies_from_geofix,
)

# ---------------------------
# GeoFix source (tail JSONL)
# ---------------------------

def geofix_iter_from_metrics(
    path: str,
    gate: float,
) -> Iterator[Dict]:
    """
    Tail a JSONL metrics file produced by System C and yield gated GeoFix dicts.
    Expects rows with keys: ts, lat, lon, conf, inliers, rmse_px (alt optional).
    """
    p = Path(path)
    last_pos = 0
    while True:
        if not p.exists():
            time.sleep(0.1)
            continue
        with p.open("r") as f:
            # seek to last known
            f.seek(last_pos)
            for line in f:
                last_pos = f.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                conf = float(j.get("conf", 0.0) or 0.0)
                if conf >= gate and "lat" in j and "lon" in j:
                    yield j
        time.sleep(0.1)


# ---------------------------
# Publisher loops
# ---------------------------

def run_gps_input_loop(
    m: mavutil.mavfile,
    fixes: Iterable[Dict],
    *,
    rate_hz: int,
    base_alt_m: float,
    horiz_min: float,
    horiz_max: float,
    vert_floor: float,
    sacc_mps: float,
    sats: int,
) -> None:
    period = 1.0 / max(1, rate_hz)
    for j in fixes:
        lat = float(j["lat"])
        lon = float(j["lon"])
        alt = float(j.get("alt_m", base_alt_m))
        inliers = int(j.get("inliers", 0) or 0)
        rmse_px = float(j.get("rmse_px", 0.0) or 0.0)

        # Default covariance if not present
        R = np.diag([25.0, 25.0, 100.0])
        if "R" in j:
            # R may be logged as list; keep safe conversion
            try:
                R = np.asarray(j["R"], dtype=float).reshape(3, 3)
            except Exception:
                pass

        hacc, vacc = accuracies_from_geofix(
            R, inliers, rmse_px, horiz_min=horiz_min, horiz_max=horiz_max, vert_floor=vert_floor
        )
        send_gps_input(
            m,
            lat_deg=lat,
            lon_deg=lon,
            alt_m=alt,
            hacc_m=hacc,
            vacc_m=vacc,
            sacc_mps=sacc_mps,
            sats=sats,
        )
        time.sleep(period)


def run_vision_pose_loop(
    m: mavutil.mavfile,
    fixes: Iterable[Dict],
    *,
    rate_hz: int,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> None:
    """
    Publish VISION_POSITION_ESTIMATE in PX4's local frame approximation (NED).
    z is down: z = -(alt - ref_alt).
    """
    period = 1.0 / max(1, rate_hz)
    ref = (ref_lat, ref_lon, ref_alt)
    for j in fixes:
        lat = float(j["lat"])
        lon = float(j["lon"])
        alt = float(j.get("alt_m", ref_alt))
        # LLA -> local NED
        ned = lla_to_ned((lat, lon, alt), ref)  # [N, E, D]
        x, y, z = float(ned[0]), float(ned[1]), float(ned[2])
        # Covariance: 6x6 (pos xyz, rot rpy)
        rmse_px = float(j.get("rmse_px", 3.0))
        inliers = int(j.get("inliers", 20))
        pos_var = max(0.25, min(100.0, 0.04 * rmse_px + 10.0 / max(1, inliers)))  # m^2
        cov = np.diag([pos_var, pos_var, max(1.0, 2.0 * pos_var),  # xyz
                       0.05, 0.05, 0.05])                           # rpy rad^2
        ts_usec = None
        send_vision_position_estimate(m, ts_usec=ts_usec, x=x, y=y, z=z, yaw=0.0, cov=cov)
        time.sleep(period)


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="System D — PX4 Fusion Bridge")
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--method", default="gps_input", choices=["gps_input", "vision_pose"])
    ap.add_argument("--rate", type=int, default=None, help="Publish rate Hz (overrides config)")
    ap.add_argument("--px4", default=None, help="PX4 MAVLink URL (overrides config fusion.px4_url)")
    ap.add_argument("--metrics-file", default=None, help="Path to metrics JSONL (overrides config)")
    ap.add_argument("--gate", type=float, default=None, help="Confidence gate (overrides config)")
    args = ap.parse_args()

    P = yaml.safe_load(open(args.config, "r"))

    # Config pulls
    px4_url = args.px4 or P["fusion"].get("px4_url", "udpout:127.0.0.1:14540")
    rate_hz = int(args.rate or P["fusion"].get("publish_rate_hz", 10))
    conf_gate = float(args.gate or P["fusion"].get("conf_gate", 0.6))
    metrics_file = args.metrics_file or P["logging"].get("metrics_file", "logs/metrics.jsonl")

    # Accuracy mapping params
    horiz_min = float(P["fusion"].get("horiz_acc_min_m", 3.0))
    horiz_max = float(P["fusion"].get("horiz_acc_max_m", 25.0))
    vert_floor = float(P["fusion"].get("vert_acc_floor_m", 6.0))
    sacc_mps = float(P["fusion"].get("speed_accuracy_mps", 0.5))
    sats = int(P["fusion"].get("satellites_visible", 10))

    # AOI ref for local frames
    ref_lat = float(P["aoi"]["home_lat"])
    ref_lon = float(P["aoi"]["home_lon"])
    ref_alt = float(P["aoi"]["home_alt_m"])

    # Connect to PX4
    print(f"[D] Connecting MAVLink → {px4_url}")
    m = open_px4_connection(px4_url)
    print("[D] PX4 heartbeat OK")

    # GeoFix stream
    fixes = geofix_iter_from_metrics(metrics_file, gate=conf_gate)

    # Publish
    print(f"[D] Method={args.method}, rate={rate_hz} Hz, conf_gate={conf_gate}")
    if args.method == "gps_input":
        run_gps_input_loop(
            m,
            fixes,
            rate_hz=rate_hz,
            base_alt_m=ref_alt,
            horiz_min=horiz_min,
            horiz_max=horiz_max,
            vert_floor=vert_floor,
            sacc_mps=sacc_mps,
            sats=sats,
        )
    else:
        run_vision_pose_loop(
            m,
            fixes,
            rate_hz=rate_hz,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            ref_alt=ref_alt,
        )


if __name__ == "__main__":
    main()
