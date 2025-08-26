from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, Dict
from datetime import datetime, timezone
import numpy as np


IsoTime = str


def now_iso() -> IsoTime:
    """UTC timestamp in RFC3339/ISO-8601 with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _as_float_tuple(x: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]))


@dataclass(slots=True)
class ImageFrame:
    """
    Represents a single camera image captured by System A.

    Attributes:
        ts: ISO-8601 (UTC) timestamp string.
        width, height: image dimensions in pixels.
        frame: np.ndarray of shape (H,W) or (H,W,3), dtype uint8.
        camera_id: logical ID for source camera.
    """
    ts: IsoTime
    width: int
    height: int
    frame: np.ndarray
    camera_id: str = "cam0"

    def __post_init__(self) -> None:
        if not isinstance(self.frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")
        if self.frame.ndim not in (2, 3):
            raise ValueError("frame must be 2D (gray) or 3D (BGR)")
        if self.frame.shape[0] != self.height or self.frame.shape[1] != self.width:
            raise ValueError("width/height do not match frame shape")
        if self.frame.dtype != np.uint8:
            # Keep strict to avoid slowdowns / surprises downstream
            self.frame = self.frame.astype(np.uint8, copy=False)

    @property
    def shape(self) -> Tuple[int, int, int | None]:
        if self.frame.ndim == 2:
            return (self.height, self.width, None)
        return (self.height, self.width, self.frame.shape[2])

    def to_meta(self) -> Dict[str, Any]:
        """Metadata without image bytes (safe to log/serialize)."""
        return {
            "ts": self.ts,
            "width": self.width,
            "height": self.height,
            "channels": None if self.frame.ndim == 2 else self.frame.shape[2],
            "camera_id": self.camera_id,
        }


@dataclass(slots=True)
class ImuDelta:
    """
    IMU increment at high rate (100–1000 Hz).

    Attributes:
        ts: ISO-8601 (UTC) timestamp.
        dt: sample period in seconds.
        gyro: (wx, wy, wz) rad/s
        accel: (ax, ay, az) m/s^2
    """
    ts: IsoTime
    dt: float
    gyro: Tuple[float, float, float]
    accel: Tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        self.gyro = _as_float_tuple(self.gyro)
        self.accel = _as_float_tuple(self.accel)


@dataclass(slots=True)
class GeoFix:
    """
    Geodetic fix estimated by System C (image correlation/georegistration).

    Attributes:
        ts: ISO-8601 (UTC) time of originating camera frame.
        lat, lon: WGS84 degrees.
        alt_m: optional altitude (meters); None if not solved.
        R: 3x3 covariance (m^2) for N/E/U in a local tangent frame (approx).
        confidence: [0..1] fused quality score produced by System C.
        inliers: number of inlier correspondences used in solution.
        rmse_px: reprojection RMSE in pixels.
    """
    ts: IsoTime
    lat: float
    lon: float
    alt_m: Optional[float]
    R: np.ndarray = field(repr=False)
    confidence: float
    inliers: int
    rmse_px: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat <= 90.0) or not (-180.0 <= self.lon <= 180.0):
            raise ValueError("lat/lon out of range")
        if not isinstance(self.R, np.ndarray):
            raise TypeError("R must be a numpy ndarray")
        if self.R.shape != (3, 3):
            raise ValueError("R must be 3x3")
        # ensure symmetric positive (semi)definite
        self.R = 0.5 * (self.R + self.R.T)
        if self.inliers < 0:
            raise ValueError("inliers must be >= 0")
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["R"] = self.R.tolist()
        return d
