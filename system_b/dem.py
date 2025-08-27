from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rasterio


class DEM:
    """
    Lightweight DEM reader.
    - If `path` exists: use rasterio to sample elevations.
    - Else: return a flat(ish) 50 m surface for all queries.
    """

    def __init__(self, path: str = "data/dem/dem.tif"):
        self.path = path
        self._ds = rasterio.open(path) if os.path.exists(path) else None

    @property
    def exists(self) -> bool:
        return self._ds is not None

    def xyz(self, lat: float, lon: float) -> Tuple[float, float, float]:
        """
        Return (lon, lat, elev_m). If DEM missing, elev=50.
        """
        if not self._ds:
            return (lon, lat, 50.0)
        row, col = self._ds.index(lon, lat)
        band = self._ds.read(1)
        row = max(0, min(band.shape[0] - 1, row))
        col = max(0, min(band.shape[1] - 1, col))
        elev = float(band[row, col])
        return (lon, lat, elev)

    def sample_many(self, coords: Iterable[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        for lon, lat in coords:
            out.append(self.xyz(lat, lon))
        return out
