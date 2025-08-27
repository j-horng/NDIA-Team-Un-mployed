# FILE: system_c/__init__.py
"""
System C — Feature Matching & Georectification

This package provides:
- ORB-based image correlation with robust RANSAC verification
- Simple geolocation estimation (map center pixel of camera frame into the
  satellite image via homography; then pixel→WGS84 using tile metadata)
- Confidence scoring & covariance estimation for downstream fusion (System D)
- A self-contained pipeline that reads a reference tile from System B,
  synthesizes “sensor” crops, and emits GeoFixes to logs/metrics.jsonl

Entry point:
    python -m system_c.pipeline --config config/params.yaml
"""
from .correlate import orb_ransac_georeg

__all__ = ["orb_ransac_georeg"]
