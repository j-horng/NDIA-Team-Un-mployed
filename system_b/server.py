from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import yaml

from system_b.tile_cache import TileCache
from system_b.dem import DEM


def _load_config(path: str = "config/params.yaml") -> Dict:
    if not Path(path).exists():
        # minimal defaults if config absent
        return {
            "tiles": {"cache_root": "data/tiles"},
            "fusion": {"px4_url": "udpout:127.0.0.1:14540"},
            "aoi": {"home_lat": 38.8895, "home_lon": -77.0352, "home_alt_m": 40},
            "logging": {"metrics_file": "logs/metrics.jsonl"},
        }
    return yaml.safe_load(open(path, "r"))


P = _load_config()

# Instances
cache_root = P.get("tiles", {}).get("cache_root", "data/tiles")
tile_cache = TileCache(cache_root)
dem = DEM(P.get("tiles", {}).get("dem_path", "data/dem/dem.tif"))

app = FastAPI(title="A-PNT Imagery API", version="1.0.0")

# (Optional) CORS for local dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "tiles": tile_cache.stats(), "dem": {"exists": dem.exists}}


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
        "has_image": os.path.exists(str(Path(cache_root) / f"{z}/{x}/{y}.png")),
        "meta": t.geo_metadata,
    }


@app.get("/imagery")
def imagery(lat: float = Query(...), lon: float = Query(...), zoom: int = Query(18)):
    """
    Return PNG image bytes with `X-Geo-Metadata` header (JSON).
    """
    t = tile_cache.get_best_tile(lat=lat, lon=lon, zoom=zoom)
    if t is None:
        return JSONResponse({"error": "tile_not_found"}, status_code=404)
    # Ensure the image exists
    try:
        img = t.image_bytes
    except FileNotFoundError:
        return JSONResponse({"error": "image_file_missing"}, status_code=404)

    headers = {
        "X-Geo-Metadata": json.dumps(t.geo_metadata),
        "Cache-Control": "public, max-age=60",
        "X-Tile-Z": str(t.zxy[0]),
        "X-Tile-X": str(t.zxy[1]),
        "X-Tile-Y": str(t.zxy[2]),
    }
    return Response(content=img, media_type="image/png", headers=headers)


@app.get("/dem")
def dem_endpoint(lat: float = Query(...), lon: float = Query(...)):
    """
    Return elevation for (lat,lon). If DEM missing, uses a 50 m default.
    """
    lon_, lat_, elev = dem.xyz(lat, lon)
    return {"lat": lat_, "lon": lon_, "elev_m": elev, "dem_exists": dem.exists}


# -------- local dev entrypoint --------
if __name__ == "__main__":
    # Allow `python -m system_b.server` for local runs
    uvicorn.run(app, host="0.0.0.0", port=8000)
