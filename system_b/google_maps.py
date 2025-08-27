from __future__ import annotations

"""
Google Static Maps adapter (optional provider for System B).

⚠️ IMPORTANT (licensing / ToS):
- Google Static Maps content is generally intended for on-screen Google map **visualization**.
- Service terms commonly **forbid caching or proxying** images from your own servers.
- This module returns image bytes only so your application can choose to **redirect**
  to Google (preferred) or proxy bytes **IFF** your license explicitly permits it.
- Default integration in System B keeps Google OFF; see config/providers.google_maps.

Usage:
    svc = GoogleMapsService()  # requires GOOGLE_MAPS_API_KEY in env or api_key=...
    tile = svc.get_static_map(lat, lon, zoom=18, size="640x640", maptype="satellite", scale=2)
    if tile:
        # tile.image_data -> PNG bytes
        # tile.meta['geotransform'] -> [x0, dx, 0, y0, 0, dy] in EPSG:4326
        # tile.meta['size'] -> "WxH" (actual pixel dimensions, includes scale)
        pass
"""

import os
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode

import requests


log = logging.getLogger(__name__)


@dataclass
class GoogleMapsTile:
    image_data: bytes
    meta: Dict


class GoogleMapsService:
    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        """
        Initialize Google Maps service.

        Params:
            api_key: Google Maps API key (falls back to env GOOGLE_MAPS_API_KEY)
            session: optional requests.Session for connection reuse
        """
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google Maps API key is required. "
                "Set GOOGLE_MAPS_API_KEY environment variable or pass api_key=..."
            )
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
        self.session = session or requests.Session()

    # ----------------------------
    # Public API
    # ----------------------------
    def build_url(
        self,
        *,
        lat: float,
        lon: float,
        zoom: int = 18,
        size: str = "640x640",
        maptype: str = "satellite",
        scale: int = 2,
        fmt: str = "png",
        extra: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Construct a Google Static Maps URL (no request performed).
        Note: `size` is in logical pixels; `scale` multiplies the returned image dimensions.

        Returns:
            Fully-qualified HTTPS URL.
        """
        params = {
            "center": f"{lat},{lon}",
            "zoom": int(zoom),
            "size": size,        # e.g., "640x640"
            "maptype": maptype,  # satellite|hybrid|terrain|roadmap
            "scale": int(scale), # 1 or 2 (2 recommended for higher DPI)
            "format": fmt,       # png, png32, jpg, etc.
            "key": self.api_key,
        }
        if extra:
            params.update(extra)
        return f"{self.base_url}?{urlencode(params)}"

    def get_static_map(
        self,
        lat: float,
        lon: float,
        zoom: int = 18,
        size: str = "640x640",
        maptype: str = "satellite",
        scale: int = 2,
        fmt: str = "png",
        timeout: float = 10.0,
        extra: Optional[Dict[str, str]] = None,
    ) -> Optional[GoogleMapsTile]:
        """
        Fetch a static map and return image bytes + georeferencing metadata.

        Meta schema (subset):
            {
              'geotransform': [x0, dx, 0, y0, 0, dy],  # lon/lat affine at top-left pixel
              'size': 'WxH',                            # actual output pixels (includes scale)
              'crs': 'EPSG:4326',
              'maptype': <str>, 'zoom': <int>,
              'center_lat': <float>, 'center_lon': <float>,
              'scale': <int>,
              'source': 'google_static_maps'
            }
        """
        url = self.build_url(
            lat=lat, lon=lon, zoom=zoom, size=size, maptype=maptype, scale=scale, fmt=fmt, extra=extra
        )
        try:
            r = self.session.get(url, timeout=timeout)
            # Google can return 200 with an error PNG overlay; we don't parse overlays here.
            if r.status_code != 200 or not r.content:
                log.warning("Google Static Maps request failed: %s %s", r.status_code, r.text[:200])
                return None

            # Compute accurate geotransform using Web Mercator formulas
            gt, out_w, out_h = self._calculate_geotransform(lat, lon, zoom, size, scale)
            meta = {
                "geotransform": gt,
                "size": f"{out_w}x{out_h}",   # include scale
                "crs": "EPSG:4326",
                "maptype": maptype,
                "zoom": int(zoom),
                "center_lat": float(lat),
                "center_lon": float(lon),
                "scale": int(scale),
                "format": fmt,
                "source": "google_static_maps",
            }
            return GoogleMapsTile(image_data=r.content, meta=meta)
        except Exception as e:
            log.exception("Error fetching Google Static Map: %s", e)
            return None

    # ----------------------------
    # Geo helpers (Web Mercator exact)
    # ----------------------------
    @staticmethod
    def _merc_world_size(zoom: int) -> float:
        # World size in "pixels" at given zoom (256px tiles)
        return 256.0 * (2 ** int(zoom))

    @staticmethod
    def _lon_to_x(lon: float, world: float) -> float:
        # Map lon [-180,180] to world pixel X [0, world]
        return (lon + 180.0) / 360.0 * world

    @staticmethod
    def _lat_to_y(lat: float, world: float) -> float:
        # Map lat [-85..85] to world pixel Y [0, world] using Web Mercator
        s = math.sin(math.radians(lat))
        y = 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
        return y * world

    @staticmethod
    def _x_to_lon(x: float, world: float) -> float:
        return x / world * 360.0 - 180.0

    @staticmethod
    def _y_to_lat(y: float, world: float) -> float:
        n = math.pi - 2.0 * math.pi * (y / world)
        return math.degrees(math.atan(math.sinh(n)))

    def _calculate_geotransform(
        self, lat: float, lon: float, zoom: int, size: str, scale: int
    ) -> Tuple[list, int, int]:
        """
        Compute an affine transform at the image's top-left pixel:
            [x0, dx, 0, y0, 0, dy]  with lon/lat in degrees.
        dy is typically negative (image coordinates increase downward).

        Returns:
            (geotransform_list, out_width_px, out_height_px)
        """
        # Parse requested size and scale -> actual image dimensions
        base_w, base_h = map(int, size.lower().split("x"))
        out_w = int(base_w * max(1, int(scale)))
        out_h = int(base_h * max(1, int(scale)))

        world = self._merc_world_size(zoom)
        cx = self._lon_to_x(lon, world)
        cy = self._lat_to_y(lat, world)

        # Top-left in world pixel coords (respecting scale-expanded output)
        x0p = cx - out_w / 2.0
        y0p = cy - out_h / 2.0

        # Convert top-left to lon/lat
        top_left_lon = self._x_to_lon(x0p, world)
        top_left_lat = self._y_to_lat(y0p, world)

        # One-pixel steps at top-left
        lon_dx = self._x_to_lon(x0p + 1.0, world) - top_left_lon
        lat_dy = self._y_to_lat(y0p + 1.0, world) - top_left_lat  # increases downward -> negative magnitude

        geotransform = [float(top_left_lon), float(lon_dx), 0.0, float(top_left_lat), 0.0, float(lat_dy)]
        return geotransform, out_w, out_h

    # ----------------------------
    # Pixel -> Geo convenience
    # ----------------------------
    @staticmethod
    def sat_pix2geo(pixel_x: float, pixel_y: float, geotransform: list) -> Tuple[float, float]:
        """
        Convert pixel (x,y) to (lon, lat) using the provided geotransform:
            [x0, dx, 0, y0, 0, dy]
        """
        x0, dx, _, y0, _, dy = geotransform
        lon = x0 + pixel_x * dx
        lat = y0 + pixel_y * dy
        return float(lon), float(lat)
