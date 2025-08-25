import cv2, numpy as np
from typing import Callable, Optional
from common.types import ImageFrame, GeoFix
def orb_ransac_georeg(cam: ImageFrame, sat_gray, sat_pix2geo: Callable[[float,float],tuple],
                      nfeatures=2000, fast=12, min_inliers=40, ransac_px=3.0) -> Optional[GeoFix]:
    cam_gray = cv2.cvtColor(cam.frame, cv2.COLOR_BGR2GRAY) if cam.frame.ndim==3 else cam.frame
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=fast)
    k1,d1 = orb.detectAndCompute(cam_gray,None); k2,d2 = orb.detectAndCompute(sat_gray,None)
    if d1 is None or d2 is None or len(k1)<50 or len(k2)<50: return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = m.knnMatch(d1,d2,k=2); good = [a for a,b in raw if a.distance < 0.7*b.distance]
    if len(good) < min_inliers: return None
    p1 = np.float32([k1[g.queryIdx].pt for g in good]); p2 = np.float32([k2[g.trainIdx].pt for g in good])
    H, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_px)
    if H is None or inliers is None: return None
    inl = inliers.ravel().astype(bool); inlier_ratio = inl.mean()
    h,w = cam_gray.shape[:2]; center = np.array([[[w/2.0, h/2.0]]], dtype=np.float32)
    sat_center = cv2.perspectiveTransform(center, H)[0,0]; lon, lat = sat_pix2geo(float(sat_center[0]), float(sat_center[1]))
    reproj = cv2.perspectiveTransform(p1[None,inl,:], H)[0]; rmse = float(np.sqrt(np.mean((reproj - p2[inl])**2)))
    conf = float(min(1.0, (0.5 + 0.5*inlier_ratio) * (1.0 / (1.0 + rmse / 3.0))))
    R = np.diag([25.0,25.0,100.0]).astype(float)
    return GeoFix(ts=cam.ts, lat=lat, lon=lon, alt_m=None, R=R, confidence=conf, inliers=int(inl.sum()), rmse_px=rmse)
