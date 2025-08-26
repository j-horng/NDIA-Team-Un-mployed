"""
Optional lightweight local ES-EKF (position-only update from GeoFix).

This is **not required** for the demo (PX4 EKF2 is preferred). It can be useful
if you want to pre-fuse before publishing to PX4.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common.geo import lla_to_ned

@dataclass
class LocalEKF:
    """
    A toy 15-state ES-EKF skeleton collapsed to position/bias subset for demo use.
    State kept here: only position (N,E,D) for brevity.
    """
    ref_lla: tuple[float, float, float]
    P: np.ndarray

    def __init__(self, ref_lla: tuple[float, float, float]):
        self.ref_lla = ref_lla
        self.P = np.diag([100.0, 100.0, 400.0]).astype(float)  # large initial uncertainty
        self._pos_ned = np.zeros(3, dtype=float)

    @property
    def pos_ned(self) -> np.ndarray:
        return self._pos_ned.copy()

    def predict(self, Q: np.ndarray) -> None:
        """Very simple process model: inflate covariance with Q."""
        Q = np.asarray(Q, dtype=float)
        if Q.shape != (3, 3):
            raise ValueError("Q must be 3x3")
        self.P = self.P + Q

    def vision_update(self, fix: dict) -> None:
        """
        Update with a GeoFix dict:
          {lat, lon, alt_m?, R: 3x3, confidence}
        """
        if float(fix.get("conf", 0.0)) < 0.6:
            return
        lat = float(fix["lat"]); lon = float(fix["lon"])
        alt = float(fix.get("alt_m", self.ref_lla[2]))
        z = lla_to_ned((lat, lon, alt), self.ref_lla)  # measurement in NED (meters)

        # Observation model: H = [I3]
        H = np.eye(3)
        R = np.asarray(fix.get("R", np.diag([25.0, 25.0, 100.0])), dtype=float)
        if R.shape != (3, 3):
            R = np.diag([25.0, 25.0, 100.0])

        # Kalman update
        y = z - self._pos_ned
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self._pos_ned = self._pos_ned + dx
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
