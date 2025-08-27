from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import yaml

from system_b.tile_cache import TileCache
from system_b.dem import DEM

# Optional Google provider
_GMAPS_AVAILABLE = False
_GMAPS_INIT_ERR: Optional[str] = None
try:
    from system_b.google_maps import GoogleMapsService, GoogleMapsTile  # your file
    _GMAPS_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _GMAPS_INIT_ERR = str(e)


def _load_config(path: str = "config/params.yaml") -> Dict:
    if not Path(path).exists():
        return {
            "tiles": {"cache_root": "data/tiles"},
            "fusion": {"px4_url": "udpout:127.0.0.1:14540"},
            "aoi": {"home_lat": 38.8895, "home_lon": -77.0352, "home_alt_m": 40},
            "logging": {"metrics_file": "logs/metrics.jsonl"},
            "providers": {"order": ["tiles"]},
        }
    return yaml.safe_load(open(path, "r"))


def _gmaps_to_tile_meta(tile: "GoogleMapsTile") -> Dict:
    """
    Convert GoogleMapsService meta to the project-wide X-Geo-Metadata schema.
    Expect tile.meta['geotransform'] = [x0, dx, 0, y0, 0, dy] and meta['size']="WxH".
    """
    gt = tile.meta.get("geotransform")
    size = tile.meta.get("size", "640x640")
    w, h = [int(x) for x in str(size).lower().split("x")]
    x0, dx, _, y0, _, dy = gt
    return {
        "crs": "EPSG:4326",
        "top_left_lon": float(x0),
        "top_left_lat": float(y0),
        "px_size_lon": float(dx),
        "px_size_lat": float(dy),
        "width": int(w),
        "height": int(h),
        "zoom": int(tile.meta.get("zoom", 0)),
        "_provider": "google_static_maps",
        "_source_meta": {k: v for k, v in tile.meta.items() if k not in ("geotransform",)},
    }


P = _load_config()

# Instances
cache_root = P.get("tiles", {}).get("cache_root", "data/tiles")
tile_cache = TileCache(cache_root)
dem = DEM(P.get("tiles", {}).get("dem_path", "data/dem/dem.tif"))

# Google provider
providers_cfg = P.get("providers", {})
gm_cfg = providers_cfg.get("google_maps", {})
GM_ENABLED = bool(gm_cfg.get("enabled", False))
GM_PROXY_ALLOWED = bool(gm_cfg.get("proxy_allowed", False))
GM_SIZE = str(gm_cfg.get("size", "640x640"))
GM_MAPTYPE = str(gm_cfg.get("maptype", "satellite"))
GM_ZOFF = int(gm_cfg.get("zoom_offset", 0))

_gmaps: Optional[GoogleMapsService] = None
if _GMAPS_AVAILABLE and GM_ENABLED:
    try:
        _gmaps = GoogleMapsService(api_key=gm_cfg.get("api_key"))  # falls back to env var
    except Exception as e:  # pragma: no cover
        _GMAPS_INIT_ERR = f"GoogleMapsService init failed: {e}"

app = FastAPI(title="A-PNT Imagery API", version="1.1.0")

# (Optional) CORS for local dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tiles": tile_cache.stats(),
        "dem": {"exists": dem.exists},
        "google_maps": {
            "enabled": GM_ENABLED,
            "proxy_allowed": GM_PROXY_ALLOWED,
            "available": _GMAPS_AVAILABLE,
            "init_error": _GMAPS_INIT_ERR,
        },
    }


@app.get("/stats")
def stats():
    return {"tiles": tile_cache.stats()}


@app.get("/nearest")
def nearest(lat: float = Query(...), lon: float = Query(...), zoom: int = Query(18)):
    t = tile_cache.get_best_tile(lat=lat, lon=lon, zoom=zoom)
    if t is None:
        raise HTTPException(status_code=404, detail="no_tiles_indexed")
    z, x, y = t.zxy
    return {
        "z": z,
        "x": x,
        "y": y,
        "has_image": Path(cache_root, f"{z}/{x}/{y}.png").exists(),
        "meta": t.geo_metadata,
    }


@app.get("/imagery")
def imagery(lat: float = Query(...), lon: float = Query(...), zoom: int = Query(18)):
    """
    Return PNG image bytes with `X-Geo-Metadata` header (JSON).

    Order:
      1) Local tile cache (preferred)
      2) Optional: Google Static Maps fallback (if enabled AND proxy_allowed)
    """
    # 1) local cache
    t = tile_cache.get_best_tile(lat=lat, lon=lon, zoom=zoom)
    if t is not None:
        try:
            img = t.image_bytes
            headers = {
                "X-Geo-Metadata": json.dumps(t.geo_metadata),
                "Cache-Control": "public, max-age=60",
                "X-Tile-Z": str(t.zxy[0]),
                "X-Tile-X": str(t.zxy[1]),
                "X-Tile-Y": str(t.zxy[2]),
            }
            return Response(content=img, media_type="image/png", headers=headers)
        except FileNotFoundError:
            # fall through to provider
            pass

    # 2) Google fallback (disabled by default)
    if GM_ENABLED and GM_PROXY_ALLOWED:
        if not _GMAPS_AVAILABLE or _gmaps is None:
            return JSONResponse({"error": "google_provider_unavailable", "detail": _GMAPS_INIT_ERR}, status_code=503)
        try:
            gm = _gmaps.get_static_map(lat=lat, lon=lon, zoom=zoom + GM_ZOFF, size=GM_SIZE, maptype=GM_MAPTYPE)
            if gm is None:
                return JSONResponse({"error": "google_fetch_failed"}, status_code=502)
            meta = _gmaps_to_tile_meta(gm)
            headers = {
                "X-Geo-Metadata": json.dumps(meta),
                "Cache-Control": "no-store",   # do NOT cache Google imagery
                "X-Provider": "google_static_maps",
            }
            return Response(content=gm.image_data, media_type="image/png", headers=headers)
        except Exception as e:
            return JSONResponse({"error": "google_exception", "detail": str(e)}, status_code=502)

    # no imagery
    return JSONResponse({"error": "tile_not_found"}, status_code=404)


@app.get("/dem")
def dem_endpoint(lat: float = Query(...), lon: float = Query(...)):
    """
    Return elevation for (lat,lon). If DEM missing, uses a 50 m default.
    """
    lon_, lat_, elev = dem.xyz(lat, lon)
    return {"lat": lat_, "lon": lon_, "elev_m": elev, "dem_exists": dem.exists}


# -------- local dev entrypoint --------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
