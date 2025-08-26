from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from pymavlink import mavutil

# ---------------------------
# Connection helpers
# ---------------------------

def open_px4_connection(px4_url: str, timeout_s: float = 10.0) -> mavutil.mavfile:
    """
    Open MAVLink connection to PX4 and wait for heartbeat.
    px4_url examples:
      - "udpout:127.0.0.1:14540" (PX4 SITL default)
      - "udp:0.0.0.0:14550"
      - "tcp:127.0.0.1:5760"
      - "serial:/dev/ttyACM0:57600"
    """
    m = mavutil.mavlink_connection(px4_url)
    m.wait_heartbeat(timeout=timeout_s)
    return m


# ---------------------------
# Message senders
# ---------------------------

def send_gps_input(
    m: mavutil.mavfile,
    *,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    hacc_m: float,
    vacc_m: float,
    sacc_mps: float = 0.5,
    yaw_deg: Optional[float] = None,
    fix_type: int = 3,        # 3D fix
    sats: int = 10,
    ignore_flags: int = 0,
) -> None:
    """
    Send MAVLink GPS_INPUT. PX4 EKF2 can fuse this as GNSS.
    Note: lat/lon in 1E7 scaled integers, alt in meters AMSL.
    """
    t_usec = int(time.time() * 1e6)
    yaw = float("nan") if yaw_deg is None else float(np.deg2rad(yaw_deg))
    m.mav.gps_input_send(
        t_usec,            # time_usec
        0,                 # gps_id
        ignore_flags,      # ignore_flags
        0, 0,              # time_week_ms, time_week (unused)
        fix_type,          # fix_type
        int(lat_deg * 1e7),
        int(lon_deg * 1e7),
        float(alt_m),
        float(hacc_m),     # h_acc
        float(vacc_m),     # v_acc
        0.0, 0.0, 0.0,     # vel NED (unused if ignored)
        float(sacc_mps),   # speed_accuracy
        float(hacc_m),     # horiz_acc (compat)
        float(vacc_m),     # vert_acc  (compat)
        int(sats),         # satellites_visible
        yaw,               # yaw
    )


def send_vision_position_estimate(
    m: mavutil.mavfile,
    *,
    ts_usec: Optional[int],
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    cov: Optional[np.ndarray] = None,
) -> None:
    """
    Send VISION_POSITION_ESTIMATE in a local frame (NED preferred for PX4).
    PX4 interprets x,y,z in meters; use z-down (NED) for consistency.
    """
    if ts_usec is None:
        ts_usec = int(time.time() * 1e6)
    if cov is None:
        cov = np.diag([0.5, 0.5, 1.0, 0.05, 0.05, 0.05])  # 6x6 default covariance

    cov = np.asarray(cov, dtype=float)
    if cov.shape != (6, 6):
        raise ValueError("cov must be 6x6")

    m.mav.vision_position_estimate_send(
        ts_usec,
        float(x), float(y), float(z),
        float(roll), float(pitch), float(yaw),
        cov.flatten().astype(np.float32).tolist()
    )


# ---------------------------
# Heuristics / Mapping
# ---------------------------

def accuracies_from_geofix(
    R_neu_m2: np.ndarray,
    inliers: int,
    rmse_px: float,
    horiz_min: float,
    horiz_max: float,
    vert_floor: float,
) -> Tuple[float, float]:
    """
    Map C's covariance/inlier stats into GPS_INPUT hacc/vacc meters.
    - Start from covariance (N/E), clamp into [horiz_min, horiz_max].
    - Inflate with rmse_px and (1/inliers) to penalize small or noisy solutions.
    """
    R = np.asarray(R_neu_m2, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3 (N/E/U)")

    sN = float(np.sqrt(max(R[0, 0], 0.0)))
    sE = float(np.sqrt(max(R[1, 1], 0.0)))
    base_h = 0.5 * (sN + sE)

    # Penalize with rmse_px and small inlier sets
    infl = 1.0 + 0.02 * max(0.0, rmse_px) + (3.0 / max(1.0, float(inliers)))
    hacc = max(horiz_min, min(horiz_max, base_h * infl))

    sU = float(np.sqrt(max(R[2, 2], 0.0)))
    vacc = max(vert_floor, sU * infl * 1.3)

    return hacc, vacc
