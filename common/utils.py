from __future__ import annotations

from typing import Iterable, Optional, Deque, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
import time
import numpy as np


def iso_now_ms() -> str:
    """UTC ISO-8601 timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso8601(ts: str) -> datetime:
    """Parse a strict ISO-8601 timestamp with optional 'Z'."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


@dataclass(slots=True)
class RateTimer:
    """
    Simple rate tracker for loop diagnostics.

    Usage:
        rt = RateTimer(window=50)
        while True:
            # work...
            hz = rt.tick()
    """
    window: int = 50
    _times: Deque[float] = deque(maxlen=50)

    def __post_init__(self) -> None:
        self._times = deque(maxlen=self.window)

    def tick(self) -> float:
        t = time.perf_counter()
        self._times.append(t)
        if len(self._times) < 2:
            return 0.0
        dt = (self._times[-1] - self._times[0]) / (len(self._times) - 1)
        return 0.0 if dt <= 0 else 1.0 / dt


@dataclass(slots=True)
class RunningStats:
    """
    Online mean/std using Welford's algorithm.
    """
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.m2 += d * d2

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return self.variance ** 0.5


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def to_numpy_3x3(x) -> np.ndarray:
    """Ensure input is a 3x3 float64 numpy array (copy if necessary)."""
    a = np.asarray(x, dtype=float)
    if a.shape != (3, 3):
        raise ValueError("Expected 3x3")
    return a.copy()


def timer_ms(func):
    """
    Decorator that returns (result, elapsed_ms) for benchmarking small functions.
    """
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt_ms = (time.perf_counter() - t0) * 1e3
        return out, dt_ms
    return wrapper
