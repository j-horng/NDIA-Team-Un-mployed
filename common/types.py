from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
IsoTime = str
@dataclass
class ImageFrame:
    ts: IsoTime; width: int; height: int; frame: np.ndarray; camera_id: str = "cam0"
@dataclass
class ImuDelta:
    ts: IsoTime; dt: float; gyro: Tuple[float,float,float]; accel: Tuple[float,float,float]
@dataclass
class GeoFix:
    ts: IsoTime; lat: float; lon: float; alt_m: Optional[float]; R: np.ndarray
    confidence: float; inliers: int; rmse_px: float
