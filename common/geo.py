from __future__ import annotations

from typing import Tuple, Sequence, Dict
import math
import numpy as np


# --- WGS84 constants ---
_WGS84_A = 6378137.0              # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563    # flattening
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)  # first eccentricity squared


# -------------------------
# Pixel/Geo helpers for tiles
# -------------------------
def pix2geo(x: float, y: float, meta: Dict) -> Tuple[float, float]:
    """
    Convert pixel (x,y) in a *single* georeferenced tile to lon/lat (deg).

    Expected metadata keys (example schema used in this project):
      - top_left_lon, top_left_lat: upper-left corner lon/lat (deg)
      - px_size_lon, px_size_lat: degrees per pixel (lat step is typically negative)
      - width, height: image dimensions (pixels)

    NOTE: No bounds checking here; caller should ensure x,y in range.
    """
    lon = float(meta["top_left_lon"]) + x * float(meta["px_size_lon"])
    lat = float(meta["top_left_lat"]) + y * float(meta["px_size_lat"])
    return lon, lat


def geo2pix(lon: float, lat: float, meta: Dict) -> Tuple[float, float]:
    """
    Convert lon/lat (deg) to pixel (x,y) for the same metadata schema as pix2geo().
    """
    dx = lon - float(meta["top_left_lon"])
    dy = lat - float(meta["top_left_lat"])
    x = dx / float(meta["px_size_lon"])
    y = dy / float(meta["px_size_lat"])
    return x, y


# -------------------------
# Great-circle & bearings
# -------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance on WGS84 sphere approximation."""
    R = 6371008.8  # mean Earth radius (m)
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2 (degrees, 0..360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    b = math.degrees(math.atan2(y, x))
    return (b + 360.0) % 360.0


# -------------------------
# LLA <-> ECEF <-> NED/ENU
# -------------------------
def lla_to_ecef(lat: float, lon: float, alt_m: float) -> np.ndarray:
    """WGS84 geodetic to ECEF (x,y,z) meters."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    sinp = math.sin(phi)
    cosp = math.cos(phi)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sinp * sinp)
    x = (N + alt_m) * cosp * math.cos(lam)
    y = (N + alt_m) * cosp * math.sin(lam)
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sinp
    return np.array([x, y, z], dtype=float)


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    ECEF (x,y,z) to WGS84 geodetic (lat, lon, alt). Bowring's method.
    Accurate to <1e-11 rad typical.
    """
    # Longitude
    lon = math.atan2(y, x)
    # Reduced latitude
    b = _WGS84_A * (1 - _WGS84_F)
    ep2 = (_WGS84_A ** 2 - b ** 2) / (b ** 2)
    p = math.hypot(x, y)
    th = math.atan2(_WGS84_A * z, b * p)
    cth, sth = math.cos(th), math.sin(th)
    lat = math.atan2(z + ep2 * b * sth ** 3, p - _WGS84_E2 * _WGS84_A * cth ** 3)
    sinp = math.sin(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sinp * sinp)
    alt = p / math.cos(lat) - N
    return (math.degrees(lat), math.degrees(lon), float(alt))


def ned_rotation(ref_lat_deg: float, ref_lon_deg: float) -> np.ndarray:
    """
    Rotation matrix R_e2n that maps ECEF vectors into local NED at ref (lat, lon).
    """
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sL, cL = math.sin(lat), math.cos(lat)
    sO, cO = math.sin(lon), math.cos(lon)
    # ENU first
    R_e2e = np.array(
        [
            [-sO, cO, 0],
            [-sL * cO, -sL * sO, cL],
            [cL * cO, cL * sO, sL],
        ],
        dtype=float,
    )
    # NED is a permutation/sign flip of ENU: N=E_y, E=E_x, D=-E_z
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    return P @ R_e2e


def lla_to_ned(
    lla: Tuple[float, float, float],
    ref_lla: Tuple[float, float, float],
) -> np.ndarray:
    """
    Convert LLA to local NED (meters) around reference LLA.

    For small areas this is equivalent to a local tangent plane; for large distances
    ENU/NED via ECEF is preferable to simple flat-earth scaling.
    """
    x = lla_to_ecef(*lla)
    x0 = lla_to_ecef(*ref_lla)
    R = ned_rotation(ref_lla[0], ref_lla[1])
    return R @ (x - x0)


def ned_to_lla(
    ned_m: Sequence[float],
    ref_lla: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Convert local NED (meters) back to LLA at reference origin.
    """
    ned = np.asarray(ned_m, dtype=float).reshape(3)
    x0 = lla_to_ecef(*ref_lla)
    R = ned_rotation(ref_lla[0], ref_lla[1])
    x = x0 + R.T @ ned  # invert R
    return ecef_to_lla(float(x[0]), float(x[1]), float(x[2]))


# -------------------------
# Covariance & metrics
# -------------------------
def covariance_to_cep50_m(R_ne: np.ndarray) -> float:
    """
    Approximate 50% Circular Error Probable (CEP50) from a 2x2 N/E covariance.
    Uses a common approximation CEP50 ≈ 0.589 * (sigma_x + sigma_y).
    """
    if R_ne.shape != (2, 2):
        raise ValueError("R_ne must be 2x2 for N/E")
    sN = math.sqrt(max(R_ne[0, 0], 0.0))
    sE = math.sqrt(max(R_ne[1, 1], 0.0))
    return 0.589 * (sN + sE)


def confidence_score(
    inliers: int,
    total_matches: int,
    rmse_px: float,
    gsd_m: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.0,
) -> float:
    """
    A simple, bounded confidence score in [0,1] similar to the design in the plan:
      conf = σ( α * (inliers/total)  - β * rmse_px  - γ * ΔGSD )
    where σ is a logistic squashing to [0,1]. If gsd_m is None, the last term is ignored.
    """
    if total_matches <= 0 or inliers <= 0:
        return 0.0
    ratio = inliers / float(total_matches)
    gsd_term = 0.0 if gsd_m is None else gsd_m
    x = alpha * ratio - beta * rmse_px - gamma * gsd_term
    # logistic to [0..1]
    return float(1.0 / (1.0 + math.exp(-x)))
