from __future__ import annotations
"""
Preprocessing utilities for System C:
- Load camera intrinsics from YAML (OpenCV pinhole + radtan or fisheye)
- Undistort
- Contrast/denoise/sharpen
- Gaussian pyramid
- Turn-key preprocessors for sensor and satellite images
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import yaml


# -----------------------------
# Camera model
# -----------------------------

@dataclass
class CameraModel:
    width: int
    height: int
    K: np.ndarray            # 3x3
    dist: np.ndarray         # (k1,k2,p1,p2,k3) for radtan OR (k1,k2,k3,k4) for fisheye
    model: str = "radtan"    # "radtan" | "fisheye"

    @classmethod
    def from_yaml(cls, path: str) -> "CameraModel":
        """
        Load a simple camera model from config/camera_intrinsics.yaml
        Expected fields (radtan example):
            resolution: {width, height}
            fx, fy, cx, cy, skew
            model: "radtan" (default) or "fisheye"
            k1, k2, p1, p2, k3 (for radtan)
            OR k1, k2, k3, k4 (for fisheye)
        """
        D = yaml.safe_load(open(path, "r"))
        W = int(D.get("resolution", {}).get("width", 640))
        H = int(D.get("resolution", {}).get("height", 480))
        fx = float(D.get("fx", 930.0))
        fy = float(D.get("fy", 930.0))
        cx = float(D.get("cx", W / 2.0))
        cy = float(D.get("cy", H / 2.0))
        skew = float(D.get("skew", 0.0))
        model = str(D.get("model", "radtan")).lower()

        K = np.array([[fx, skew, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        if model == "fisheye":
            k1 = float(D.get("k1", 0.0))
            k2 = float(D.get("k2", 0.0))
            k3 = float(D.get("k3", 0.0))
            k4 = float(D.get("k4", 0.0))
            dist = np.array([k1, k2, k3, k4], dtype=np.float32)
        else:  # radtan
            k1 = float(D.get("k1", 0.0))
            k2 = float(D.get("k2", 0.0))
            p1 = float(D.get("p1", 0.0))
            p2 = float(D.get("p2", 0.0))
            k3 = float(D.get("k3", 0.0))
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        return cls(width=W, height=H, K=K, dist=dist, model=model)

    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


# -----------------------------
# Basic image ops
# -----------------------------

def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def clahe(gray_u8: np.ndarray, clip_limit: float = 3.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    cl = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
    return cl.apply(gray_u8)


def unsharp_mask(gray_u8: np.ndarray, amount: float = 0.6, radius: float = 1.0) -> np.ndarray:
    """
    Simple unsharp mask: result = (1+amount)*img - amount*gaussian(img)
    """
    if amount <= 0.0:
        return gray_u8
    blur = cv2.GaussianBlur(gray_u8, (0, 0), max(1e-6, float(radius)))
    f = cv2.addWeighted(gray_u8, 1.0 + amount, blur, -amount, 0)
    return np.clip(f, 0, 255).astype(np.uint8)


def gaussian_pyramid(gray_u8: np.ndarray, levels: int = 3) -> List[np.ndarray]:
    pyr = [gray_u8]
    cur = gray_u8
    for _ in range(1, max(1, levels)):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def canny_edges(gray_u8: np.ndarray, lo: int = 50, hi: int = 150) -> np.ndarray:
    return cv2.Canny(gray_u8, threshold1=int(lo), threshold2=int(hi), L2gradient=True)


def resize_keep(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = int(size[0]), int(size[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# -----------------------------
# Undistortion
# -----------------------------

def undistort_image(
    img: np.ndarray,
    cam: CameraModel,
    balance: float = 0.0,
    new_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undistort using the provided camera model.
    Returns (undistorted_img, newK).
    - For radtan: cv2.undistort with getOptimalNewCameraMatrix.
    - For fisheye: cv2.fisheye.initUndistortRectifyMap + remap.
    """
    H, W = img.shape[:2]
    if new_size is None:
        new_size = (W, H)

    if cam.model == "fisheye":
        # balance in [0..1] blends between crop and full FOV
        dim = (cam.width, cam.height)
        K = cam.K.astype(np.float64)
        D = cam.dist.astype(np.float64)
        if (W, H) != dim:
            # scale K if the input image is not the calibration resolution
            sx, sy = W / float(dim[0]), H / float(dim[1])
            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
            K = S @ K

        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (W, H), np.eye(3), balance=float(np.clip(balance, 0.0, 1.0))
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (W, H), cv2.CV_16SC2
        )
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if new_size != (W, H):
            und = cv2.resize(und, new_size, interpolation=cv2.INTER_AREA)
        return und, newK.astype(np.float32)

    # radtan / pinhole
    newK, _ = cv2.getOptimalNewCameraMatrix(cam.K, cam.dist, (W, H), alpha=float(np.clip(balance, 0.0, 1.0)))
    und = cv2.undistort(img, cam.K, cam.dist, None, newK)
    if new_size != (W, H):
        und = cv2.resize(und, new_size, interpolation=cv2.INTER_AREA)
    return und, newK.astype(np.float32)


# -----------------------------
# Turn-key preprocessors
# -----------------------------

def preprocess_sensor_frame(
    bgr: np.ndarray,
    *,
    cam: Optional[CameraModel] = None,
    out_size: Optional[Tuple[int, int]] = None,
    do_undistort: bool = True,
    clahe_clip: Optional[float] = 3.0,
    sharpen_amount: float = 0.5,
    blur_sigma: float = 0.0,
    fisheye_balance: float = 0.0,
) -> np.ndarray:
    """
    Prepare a sensor frame for feature matching:
      - optional undistort (radtan or fisheye)
      - resize
      - to gray (u8)
      - optional blur (for noise)
      - optional CLAHE
      - optional unsharp mask
    Returns a grayscale uint8 image.
    """
    img = bgr
    if cam is not None and do_undistort:
        img, _ = undistort_image(img, cam, balance=fisheye_balance, new_size=out_size)
    elif out_size is not None:
        img = resize_keep(img, out_size)

    gray = to_gray_u8(img)

    if blur_sigma and blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    if clahe_clip and clahe_clip > 0:
        gray = clahe(gray, clip_limit=float(clahe_clip), tile_grid=(8, 8))

    if sharpen_amount and sharpen_amount > 0:
        gray = unsharp_mask(gray, amount=float(sharpen_amount), radius=1.0)

    return gray


def preprocess_satellite_image(
    bgr: np.ndarray,
    *,
    out_size: Optional[Tuple[int, int]] = None,
    clahe_clip: Optional[float] = 2.0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Satellite tile preprocessing:
      - optional resize
      - to gray
      - optional mild blur
      - optional CLAHE
    Returns a grayscale uint8 image.
    """
    img = bgr
    if out_size is not None:
        img = resize_keep(img, out_size)
    gray = to_gray_u8(img)
    if blur_sigma and blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    if clahe_clip and clahe_clip > 0:
        gray = clahe(gray, clip_limit=float(clahe_clip), tile_grid=(8, 8))
    return gray
