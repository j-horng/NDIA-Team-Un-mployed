from __future__ import annotations

from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np

from common.types import ImageFrame, GeoFix
from common.geo import confidence_score
from common.utils import to_numpy_3x3


@dataclass(slots=True)
class MatchResult:
    ok: bool
    H_cam_to_sat: Optional[np.ndarray]
    inliers: int
    total_matches: int
    rmse_px: float
    # diagnostics
    good_matches: Optional[List[cv2.DMatch]] = None
    mask_inliers: Optional[np.ndarray] = None


def _detect_orb(gray: np.ndarray, nfeatures: int, fast: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=fast)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None:
        des = np.zeros((0, 32), dtype=np.uint8)
    return kps, des


def _match_hamming(des_cam: np.ndarray, des_sat: np.ndarray, ratio: float = 0.8) -> List[cv2.DMatch]:
    """
    Lowe ratio on Hamming-2 nearest neighbors. Returns 'good' one-to-one matches.
    """
    if len(des_cam) == 0 or len(des_sat) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des_cam, des_sat, k=2)
    good = []
    used_train = set()
    for m, n in knn:
        if m.distance < ratio * n.distance:
            if m.trainIdx not in used_train:  # one-to-one on sat side
                good.append(m)
                used_train.add(m.trainIdx)
    return good


def _homography_ransac(
    kps_cam: List[cv2.KeyPoint],
    kps_sat: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_px: float,
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    Estimate H: cam→sat via RANSAC. Returns (H, inlier_mask, rmse_px).
    """
    if len(matches) < 4:
        return None, np.zeros((0, 1), dtype=np.uint8), float("inf")

    pts_cam = np.float32([kps_cam[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_sat = np.float32([kps_sat[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_cam, pts_sat, cv2.RANSAC, ransacReprojThreshold=ransac_px, maxIters=2000, confidence=0.999)
    if H is None:
        return None, np.zeros((len(matches), 1), dtype=np.uint8), float("inf")

    # Compute reprojection RMSE (pixels) on inliers
    inlier_idx = mask.ravel().astype(bool)
    if inlier_idx.sum() == 0:
        return None, mask, float("inf")

    proj = cv2.perspectiveTransform(pts_cam[inlier_idx], H)
    err = np.linalg.norm(proj.reshape(-1, 2) - pts_sat[inlier_idx].reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return H, mask, rmse


def _meters_per_pixel(meta: dict, at_lat_deg: float) -> Tuple[float, float]:
    """
    Convert per-pixel angular sizes from tile metadata to meters per pixel at the given latitude.
    """
    # Angular step per pixel (deg/px). Note Y step will be negative; use abs for scale.
    dlon_deg = abs(float(meta["px_size_lon"]))
    dlat_deg = abs(float(meta["px_size_lat"]))
    # meters per degree
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.0001, np.cos(np.radians(at_lat_deg)))  # avoid zero at poles
    return (dlon_deg * m_per_deg_lon, dlat_deg * m_per_deg_lat)  # (E, N)


def _covariance_from_residuals(
    residuals_px: np.ndarray,
    mpp_E: float,
    mpp_N: float,
) -> np.ndarray:
    """
    Build a 3x3 covariance (N/E/U) from pixel residuals (dx,dy in pixels).
    """
    if residuals_px.size == 0:
        return np.diag([100.0, 100.0, 400.0])  # very uncertain
    dx = residuals_px[:, 0]
    dy = residuals_px[:, 1]
    # Std of inlier reprojection errors (pixel)
    sx_px = float(np.std(dx)) if dx.size else 5.0
    sy_px = float(np.std(dy)) if dy.size else 5.0
    # Convert to meters
    sE_m = max(0.5, sx_px * mpp_E)
    sN_m = max(0.5, sy_px * mpp_N)
    sU_m = max(2.0, 2.0 * max(sN_m, sE_m))
    R = np.diag([sN_m ** 2, sE_m ** 2, sU_m ** 2])
    return R


def orb_ransac_georeg(
    cam_frame: ImageFrame,
    sat_gray: np.ndarray,
    sat_pix2geo: Callable[[float, float], Tuple[float, float]],
    *,
    nfeatures: int = 2000,
    fast: int = 12,
    ransac_px: float = 3.0,
    min_inliers: int = 40,
) -> Optional[GeoFix]:
    """
    Georegister camera frame against a satellite image:

    Args:
        cam_frame: ImageFrame with .frame (BGR or gray)
        sat_gray: reference satellite image (grayscale)
        sat_pix2geo: function (x_px, y_px) -> (lon_deg, lat_deg) for the sat image
        nfeatures, fast: ORB detector params
        ransac_px: inlier threshold (pixels)
        min_inliers: minimum inliers to accept a solution

    Returns:
        GeoFix or None if correlation fails or inliers are below threshold.
    """
    # grayscale inputs
    cam_img = cam_frame.frame
    if cam_img.ndim == 3:
        cam_gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
    else:
        cam_gray = cam_img
    if sat_gray.ndim != 2:
        raise ValueError("sat_gray must be single-channel grayscale")

    # 1) ORB detect/compute
    kps_cam, des_cam = _detect_orb(cam_gray, nfeatures, fast)
    kps_sat, des_sat = _detect_orb(sat_gray, nfeatures, fast)

    # 2) Match with Lowe ratio
    good = _match_hamming(des_cam, des_sat, ratio=0.8)
    if len(good) < max(8, min_inliers // 2):
        return None

    # 3) RANSAC homography
    H, mask, rmse = _homography_ransac(kps_cam, kps_sat, good, ransac_px)
    if H is None:
        return None

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < min_inliers:
        return None

    # 4) Map camera center pixel -> satellite pixel
    cx = cam_frame.width / 2.0
    cy = cam_frame.height / 2.0
    pt_cam = np.float32([[[cx, cy]]])  # shape (1,1,2)
    pt_sat = cv2.perspectiveTransform(pt_cam, H).reshape(2)
    x_sat, y_sat = float(pt_sat[0]), float(pt_sat[1])

    # 5) Pixel residuals of inliers for covariance
    pts_cam = np.float32([kps_cam[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_sat = np.float32([kps_sat[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts_cam[inlier_mask], H).reshape(-1, 2)
    res = (proj - pts_sat[inlier_mask].reshape(-1, 2))

    # 6) Convert pixel to lon/lat
    lon, lat = sat_pix2geo(x_sat, y_sat)

    # 7) Covariance in meters using local scale at lat & sat metadata (caller knows px->deg)
    # Estimate meters per px from two nearby geo samples around the mapped point.
    # We get dlon/dlat per pixel from local differential using sat_pix2geo around x_sat,y_sat.
    eps = 1.0
    lon_dx, lat_dx = sat_pix2geo(x_sat + eps, y_sat)
    lon_dy, lat_dy = sat_pix2geo(x_sat, y_sat + eps)
    # per-pixel angular steps near the mapped point
    dlon_per_px = abs(lon_dx - lon)
    dlat_per_px = abs(lat_dy - lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.0001, np.cos(np.radians(lat)))
    mpp_E = max(1e-3, dlon_per_px * m_per_deg_lon)
    mpp_N = max(1e-3, dlat_per_px * m_per_deg_lat)
    R_neu = _covariance_from_residuals(res, mpp_E=mpp_E, mpp_N=mpp_N)

    # 8) Confidence
    conf = confidence_score(inliers=inliers, total_matches=len(good), rmse_px=rmse, gsd_m=0.5 * (mpp_E + mpp_N))

    # 9) Build GeoFix
    fix = GeoFix(
        ts=cam_frame.ts,
        lat=float(lat),
        lon=float(lon),
        alt_m=None,
        R=to_numpy_3x3(R_neu),
        confidence=float(conf),
        inliers=int(inliers),
        rmse_px=float(rmse),
    )
    return fix
