from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371008.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class TileRef:
    """Index entry representing a single tile."""
    z: int
    x: int
    y: int
    json_path: Path
    png_path: Path
    meta: Dict

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        # [lon_min, lat_min, lon_max, lat_max]
        b = self.meta.get("bbox")
        if b and len(b) == 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        # Derive from top-left & px sizes if bbox not present
        tl_lon = float(self.meta["top_left_lon"])
        tl_lat = float(self.meta["top_left_lat"])
        w = int(self.meta["width"])
        h = int(self.meta["height"])
        dx = float(self.meta["px_size_lon"])
        dy = float(self.meta["px_size_lat"])
        lon_min = min(tl_lon, tl_lon + w * dx)
        lon_max = max(tl_lon, tl_lon + w * dx)
        lat_min = min(tl_lat, tl_lat + h * dy)
        lat_max = max(tl_lat, tl_lat + h * dy)
        return (lon_min, lat_min, lon_max, lat_max)

    @property
    def center(self) -> Tuple[float, float]:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return (0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max))

    @property
    def has_image(self) -> bool:
        return self.png_path.exists() and self.png_path.is_file()

    @property
    def image_bytes(self) -> bytes:
        # Raise if missing; server will handle and return 404
        with self.png_path.open("rb") as f:
            return f.read()

    def contains(self, lat: float, lon: float) -> bool:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)


@dataclass
class Tile:
    """Thin wrapper returned to callers (server) with stable properties."""
    ref: TileRef

    @property
    def image_bytes(self) -> bytes:
        return self.ref.image_bytes

    @property
    def geo_metadata(self) -> Dict:
        # ensure zoom field matches the directory
        meta = dict(self.ref.meta)
        meta["zoom"] = int(self.ref.z)
        # include x/y for debugging/clients (non-breaking)
        meta.setdefault("_tile", {"z": self.ref.z, "x": self.ref.x, "y": self.ref.y})
        return meta

    @property
    def zxy(self) -> Tuple[int, int, int]:
        return (self.ref.z, self.ref.x, self.ref.y)


class TileCache:
    """
    Scans a TMS-like directory for tile JSON metadata and builds an in-memory index.

        root/
          └─ {z}/
              └─ {x}/
                  ├─ {y}.json   (georeferencing metadata)
                  └─ {y}.png    (imagery)

    The metadata schema is the one emitted by scripts/build_tile_cache.py.
    """
    def __init__(self, root: str = "data/tiles"):
        self.root = Path(root)
        self._index: Dict[int, List[TileRef]] = {}
        self._scan()

    # -------- public API --------

    def get_best_tile(self, lat: float, lon: float, zoom: int) -> Optional[Tile]:
        """
        Return a Tile whose bbox contains (lat,lon); fallback to nearest by center.
        If the requested zoom has no tiles, fallback to the closest available zoom.
        Note: The returned Tile may point to a missing PNG; the server will check.
        """
        # 1) same-zoom, contains
        cand = self._nearest_in_zoom(lat, lon, zoom, prefer_contains=True)
        if cand is None:
            # 2) same-zoom, nearest by center
            cand = self._nearest_in_zoom(lat, lon, zoom, prefer_contains=False)
        if cand is None:
            # 3) any-zoom fallback (nearest by center across all)
            cand = self._nearest_any_zoom(lat, lon)
        return Tile(cand) if cand else None

    def get_tile(self, z: int, x: int, y: int) -> Optional[Tile]:
        refs = self._index.get(int(z), [])
        for r in refs:
            if r.x == int(x) and r.y == int(y):
                return Tile(r)
        return None

    def stats(self) -> Dict[str, int]:
        return {
            "zooms": len(self._index),
            "tiles": sum(len(v) for v in self._index.values()),
        }

    # -------- internals --------

    def _scan(self) -> None:
        self._index.clear()
        if not self.root.exists():
            return
        for js in self.root.rglob("*.json"):
            # Expect .../{z}/{x}/{y}.json
            try:
                z = int(js.parent.parent.name)
                x = int(js.parent.name)
                y = int(js.stem)
            except Exception:
                continue
            png = js.with_suffix(".png")
            try:
                meta = json.loads(js.read_text())
                # sanity check keys
                _ = meta["top_left_lon"]; _ = meta["top_left_lat"]
                _ = meta["px_size_lon"]; _ = meta["px_size_lat"]
                _ = meta["width"]; _ = meta["height"]
            except Exception:
                continue
            ref = TileRef(z=z, x=x, y=y, json_path=js, png_path=png, meta=meta)
            self._index.setdefault(z, []).append(ref)

        # Optional: sort each zoom's list by x/y for stable ordering
        for z in self._index:
            self._index[z].sort(key=lambda r: (r.x, r.y))

    def _nearest_in_zoom(self, lat: float, lon: float, zoom: int, *, prefer_contains: bool) -> Optional[TileRef]:
        refs = self._index.get(int(zoom), [])
        if not refs:
            return None
        contains: List[TileRef] = []
        for r in refs:
            if r.contains(lat, lon):
                contains.append(r)
        if prefer_contains and contains:
            # If multiple contain (shouldn't happen with single-tile AOI), pick min-center distance
            best = min(contains, key=lambda r: _haversine_m(lat, lon, r.center[1], r.center[0]))
            return best

        # nearest by center
        best = min(refs, key=lambda r: _haversine_m(lat, lon, r.center[1], r.center[0]))
        return best

    def _nearest_any_zoom(self, lat: float, lon: float) -> Optional[TileRef]:
        all_refs = [r for refs in self._index.values() for r in refs]
        if not all_refs:
            return None
        return min(all_refs, key=lambda r: _haversine_m(lat, lon, r.center[1], r.center[0]))
