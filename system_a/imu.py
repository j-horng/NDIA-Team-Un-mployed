from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Iterator, Optional, Tuple

import numpy as np

from common.types import ImuDelta
from common.utils import iso_now_ms


CSV_HEADER = ["ts", "dt", "gx", "gy", "gz", "ax", "ay", "az"]


@dataclass
class IMUCSVSource:
    """
    Replay IMU from a CSV file with columns: ts, dt, gx, gy, gz, ax, ay, az.
    If realtime=True, sleeps dt between samples; else yields as fast as possible.
    """
    path: str
    realtime: bool = True
    scale_dt: float = 1.0  # multiply dt by this factor (e.g., 0.5 = 2x speed)

    def samples(self) -> Iterator[ImuDelta]:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"IMU CSV not found: {self.path}")
        with open(self.path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                dt = float(row.get("dt", "0.01")) * float(self.scale_dt)
                ts = row.get("ts") or iso_now_ms()
                gx = float(row.get("gx", "0.0"))
                gy = float(row.get("gy", "0.0"))
                gz = float(row.get("gz", "0.0"))
                ax = float(row.get("ax", "0.0"))
                ay = float(row.get("ay", "0.0"))
                az = float(row.get("az", "0.0"))
                yield ImuDelta(ts=ts, dt=dt, gyro=(gx, gy, gz), accel=(ax, ay, az))
                if self.realtime and dt > 0:
                    time.sleep(dt)


@dataclass
class IMUSyntheticSource:
    """
    Procedural IMU generator that simulates gentle UAV turns and mild accelerations.

    Args:
        rate_hz: sample rate (e.g., 200 Hz)
        yaw_rate_dps: nominal yaw rate amplitude (deg/s)
        accel_hz: frequency of lateral accel modulation
        gyro_noise_dps: gyro white noise std (deg/s)
        accel_noise_mps2: accel white noise std (m/s^2)
        bias_walk: (gyro, accel) random walk increments per sqrt(sec)
        gravity_mps2: gravity constant to include on +Z (body frame up is negative)
    """
    rate_hz: int = 200
    yaw_rate_dps: float = 10.0
    accel_hz: float = 0.2
    gyro_noise_dps: float = 0.2
    accel_noise_mps2: float = 0.05
    bias_walk: Tuple[float, float] = (0.02, 0.01)  # (gyro dps, accel m/s^2) per sqrt(s)
    gravity_mps2: float = 9.80665

    def samples(self, duration_s: Optional[float] = None) -> Iterator[ImuDelta]:
        dt = 1.0 / max(1, self.rate_hz)
        t = 0.0
        g = self.gravity_mps2
        rng = np.random.default_rng(1234)
        gyro_bias = np.zeros(3, dtype=float)
        accel_bias = np.zeros(3, dtype=float)

        start = time.perf_counter()
        while (duration_s is None) or (t < duration_s):
            # Yaw rate: sinusoid
            wz = math.radians(self.yaw_rate_dps) * math.sin(2 * math.pi * self.accel_hz * t)
            wx = 0.02 * math.sin(2 * math.pi * 0.05 * t)
            wy = 0.02 * math.cos(2 * math.pi * 0.07 * t)

            # Lateral accel: small sinusoid in body X/Y
            ax = 0.3 * math.sin(2 * math.pi * self.accel_hz * t)
            ay = 0.3 * math.cos(2 * math.pi * (self.accel_hz * 0.8) * t)
            az = -g  # gravity (body Z up negative; sign convention matches many IMUs)

            # Add bias random walk
            gyro_bias += rng.normal(0, self.bias_walk[0] * math.sqrt(dt), size=3)
            accel_bias += rng.normal(0, self.bias_walk[1] * math.sqrt(dt), size=3)

            # Add white noise
            gyro_noise = rng.normal(0, math.radians(self.gyro_noise_dps), size=3)
            accel_noise = rng.normal(0, self.accel_noise_mps2, size=3)

            gyro = np.array([wx, wy, wz]) + gyro_bias + gyro_noise
            accel_vec = np.array([ax, ay, az]) + accel_bias + accel_noise

            ts = iso_now_ms()
            yield ImuDelta(ts=ts, dt=dt, gyro=(float(gyro[0]), float(gyro[1]), float(gyro[2])),
                           accel=(float(accel_vec[0]), float(accel_vec[1]), float(accel_vec[2])))

            if duration_s is not None:
                t += dt
            # realtime pacing
            elapsed = time.perf_counter() - start
            target = len(range(int(t / dt))) * dt if duration_s is not None else t + dt
            sleep_s = max(0.0, (target - elapsed))
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # catch up (don’t sleep)
                pass


def write_imu_csv(path: str, samples: Iterable[ImuDelta], max_rows: int = 0) -> None:
    """
    Write an IMU stream to CSV. If max_rows > 0, stops after that many rows.
    """
    n = 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for s in samples:
            w.writerow([s.ts, f"{s.dt:.6f}", f"{s.gyro[0]:.6f}", f"{s.gyro[1]:.6f}", f"{s.gyro[2]:.6f}",
                        f"{s.accel[0]:.6f}", f"{s.accel[1]:.6f}", f"{s.accel[2]:.6f}"])
            n += 1
            if max_rows > 0 and n >= max_rows:
                break
