from __future__ import annotations
"""
Feature extraction & matching helpers for System C.

- FeatureExtractor(method='orb'|'akaze') with .detect_and_compute(gray)
- Grid non-max suppression (keeps spatially well-distributed strong keypoints)
- KNN Hamming matcher + Lowe ratio + one-to-one filtering
- Homography RANSAC helper with residuals & RMSE
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


# -----------------------------
# Extractors
# -----------------------------

@dataclass
class FeatureExtractor:
    method: str = "orb"
    nfeatures: int = 2000
    fast_threshold: int = 12
    nlevels: int = 8
    scale_factor: float = 1.2

    def __post_init__(self):
        m = self.method.lower()
        if m == "orb":
            self._det = cv2.ORB_create(
                nfeatures=int(self.nfeatures),
                scaleFactor=float(self.scale_factor),
                nlevels=int(self.nlevels),
                edgeThreshold=19,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=int(self.fast_threshold),
            )
            self.descriptor_kind = "binary"
        elif m == "akaze":
            self._det = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=0.001,
                nOctaves=4,
                nOctaveLayers=4,
                diffusivity=cv2.KAZE_DIFF_PM_G2,
            )
            self.descriptor_kind = "binary"
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def detect_and_compute(self, gray_u8: np.ndarray, mask: Optional[np.ndarray] = None):
        kps, des = self._det.detectAndCompute(gray_u8, mask)
        if des is None:
            des = np.zeros((0, 32), dtype=np.uint8)
        return kps, des


# -----------------------------
# Keypoint post-processing
# -----------------------------

def grid_nms(
    kps: List[cv2.KeyPoint],
    img_size: Tuple[int, int],
    grid: Tuple[int, int] = (8, 8),
    cap_per_cell: int = 60,
) -> List[cv2.KeyPoint]:
    """
    Keep at most cap_per_cell keypoints per grid cell, sorted by response.
    """
    if not kps:
        return []
    W, H = int(img_size[0]), int(img_size[1])
    gx, gy = int(grid[0]), int(grid[1])
    cells: List[List[cv2.KeyPoint]] = [[[] for _ in range(gx)] for _ in range(gy)]
    cw = max(1, W // gx)
    ch = max(1, H // gy)
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cx = min(gx - 1, max(0, x // cw))
        cy = min(gy - 1, max(0, y // ch))
        cells[cy][cx].append(kp)
    kept: List[cv2.KeyPoint] = []
    for row in cells:
        for cell in row:
            cell.sort(key=lambda p: p.response, reverse=True)
            kept.extend(cell[:cap_per_cell])
    return kept


# -----------------------------
# Matching
# -----------------------------

def match_binary_knn_ratio(
    des1: np.ndarray,
    des2: np.ndarray,
    *,
    ratio: float = 0.8,
    enforce_uniqueness: bool = True,
) -> List[cv2.DMatch]:
    """
    KNN (k=2) + Lowe ratio (for Hamming). Optionally enforce one-to-one on train side.
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good: List[cv2.DMatch] = []
    used_train = set()
    for m, n in knn:
        if m.distance < ratio * n.distance:
            if not enforce_uniqueness or (m.trainIdx not in used_train):
                good.append(m)
                used_train.add(m.trainIdx)
    return good


# -----------------------------
# Geometry
# -----------------------------

@dataclass
class HomographyResult:
    H: Optional[np.ndarray]
    inlier_mask: np.ndarray
    rmse_px: float
    inliers: int
    total: int


def homography_ransac_from_matches(
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_px: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.999,
) -> HomographyResult:
    """
    Estimate H: image1 -> image2 using RANSAC; compute inlier RMSE.
    """
    if len(matches) < 4:
        return HomographyResult(None, np.zeros((0, 1), np.uint8), float("inf"), 0, len(matches))

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts1, pts2, cv2.RANSAC,
        ransacReprojThreshold=float(ransac_px),
        maxIters=int(max_iters),
        confidence=float(confidence)
    )
    if H is None:
        return HomographyResult(None, np.zeros((len(matches), 1), np.uint8), float("inf"), 0, len(matches))

    inlier_mask = mask.ravel().astype(bool)
    ninl = int(inlier_mask.sum())
    if ninl == 0:
        return HomographyResult(None, mask, float("inf"), 0, len(matches))

    proj = cv2.perspectiveTransform(pts1[inlier_mask], H)
    err = np.linalg.norm(proj.reshape(-1, 2) - pts2[inlier_mask].reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else float("inf")
    return HomographyResult(H, mask, rmse, ninl, len(matches))


def inlier_residuals_px(
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    H: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    """
    Return residuals (dx, dy) for inliers under homography H mapping 1->2.
    """
    if H is None or inlier_mask is None or len(matches) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    mask_b = inlier_mask.ravel().astype(bool)
    if not mask_b.any():
        return np.zeros((0, 2), dtype=np.float32)
    proj = cv2.perspectiveTransform(pts1[mask_b], H).reshape(-1, 2)
    res = proj - pts2[mask_b].reshape(-1, 2)
    return res.astype(np.float32)


# -----------------------------
# Debug/visualization helpers (optional)
# -----------------------------

def draw_matches_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inlier_mask: Optional[np.ndarray] = None,
    max_draw: int = 100,
) -> np.ndarray:
    """
    Convenience wrapper over cv2.drawMatches with optional inlier highlighting.
    """
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    if inlier_mask is not None and len(inlier_mask) == len(matches):
        mask_list = inlier_mask.ravel().tolist()
    else:
        mask_list = None
    draw = cv2.drawMatches(
        img1, kps1, img2, kps2,
        matches[:max_draw],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=mask_list[:max_draw] if mask_list else None,
        flags=flags,
    )
    return draw
