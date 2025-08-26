#!/usr/bin/env python3
"""
Build a tiny offline tile/DEM cache for the demo.

Creates {data/tiles/{z}/0/0.png,.json} for requested zooms and a small DEM:
- If --src-tif is given: crop/reproject to AOI bbox and rasterize into a tile.
- Else if --src-image is given: resize it to the tile size and use AOI bbox metadata.
- Else: synthesize a feature-rich tile (shapes + noise) suitable for ORB.

Also writes data/dem/dem.tif (simple gradient DEM matching the AOI bbox).

Examples:
  python scripts/build_tile_cache.py --aoi config/area_of_interest.geojson --zoom 17 18
  python scripts/build_tile_cache.py --aoi config/area_of_interest.geojson --zoom 18 --src-tif my_aoi.tif
  python scripts/build_tile_cache.py --aoi config/area_of_interest.geojson --zoom 18 --src-image my_sat.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import cv2
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling


def load_aoi_bbox(geojson_path: str) -> Tuple[float, float, float, float]:
    gj = json.loads(Path(geojson_path).read_text())
    coords = gj["features"][0]["geometry"]["coordinates"][0]
    xs = [float(x) for x, _ in coords]
    ys = [float(y) for _, y in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return minx, miny, maxx, maxy  # lon_min, lat_min, lon_max, lat_max


def ensure_dirs(root: Path, zooms: List[int]) -> None:
    for z in zooms:
        (root / f"{z}/0").mkdir(parents=True, exist_ok=True)
    (Path("data/dem")).mkdir(parents=True, exist_ok=True)


def write_metadata_json(path: Path, z: int, w: int, h: int, bbox: Tuple[float, float, float, float]) -> None:
    lon_min, lat_min, lon_max, lat_max = bbox
    top_left_lon = lon_min
    top_left_lat = lat_max
    px_size_lon = (lon_max - lon_min) / float(w)
    px_size_lat = (lat_min - lat_max) / float(h)  # negative (top-left origin)
    meta = {
        "crs": "EPSG:4326",
        "top_left_lon": top_left_lon,
        "top_left_lat": top_left_lat,
        "px_size_lon": px_size_lon,
        "px_size_lat": px_size_lat,
        "width": w,
        "height": h,
        "zoom": z,
        "bbox": [lon_min, lat_min, lon_max, lat_max],
    }
    path.write_text(json.dumps(meta, indent=2))


def load_tif_as_rgb_in_bbox(src_tif: str, bbox: Tuple[float, float, float, float], out_size: Tuple[int, int]) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = bbox
    out_w, out_h = out_size
    with rasterio.open(src_tif) as ds:
        # Reproject to EPSG:4326 if needed
        if ds.crs is None:
            raise RuntimeError("Source GeoTIFF has no CRS; cannot reproject.")
        dst_crs = "EPSG:4326"
        if str(ds.crs) != dst_crs:
            transform, width, height = calculate_default_transform(ds.crs, dst_crs, ds.width, ds.height, *ds.bounds)
            kwargs = ds.meta.copy()
            kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
            # Reproject full image (memory bounded by AOI size; AOI expected small)
            data = np.zeros((ds.count, height, width), dtype=ds.dtypes[0])
            for i in range(1, ds.count + 1):
                reproject(
                    source=rasterio.band(ds, i),
                    destination=data[i - 1],
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
            transform_used = transform
        else:
            data = ds.read()  # (bands, rows, cols)
            transform_used = ds.transform

        # Window for bbox
        left, bottom, right, top = lon_min, lat_min, lon_max, lat_max
        row_top, col_left = ~transform_used * (left, top)
        row_bot, col_right = ~transform_used * (right, bottom)
        r0, r1 = sorted([int(np.floor(row_top)), int(np.ceil(row_bot))])
        c0, c1 = sorted([int(np.floor(col_left)), int(np.ceil(col_right))])
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, data.shape[1]), min(c1, data.shape[2])
        if r1 <= r0 or c1 <= c0:
            raise RuntimeError("AOI bbox outside src_tif extent.")
        crop = data[:, r0:r1, c0:c1]

        # Normalize and stack to 3 channels
        # Prefer RGB if available; otherwise single band → gray
        if crop.shape[0] >= 3:
            img = np.stack([crop[0], crop[1], crop[2]], axis=-1)
        else:
            img = np.repeat(crop[0][..., None], 3, axis=-1)

        # Scale to 0..255 uint8
        img = img.astype(np.float32)
        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        if p99 <= p1:
            p1, p99 = float(img.min()), float(img.max() + 1e-6)
        img = np.clip((img - p1) / (p99 - p1), 0, 1) * 255.0
        img = img.astype(np.uint8)

        # Resize to tile size
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return img


def load_image_as_tile(src_image: str, out_size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(src_image, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {src_image}")
    return cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)


def synthesize_tile(out_size: Tuple[int, int], seed: int = 1234) -> np.ndarray:
    """Generate a feature-rich synthetic image (edges, corners, textures)."""
    w, h = out_size
    rng = np.random.default_rng(seed)
    base = (rng.normal(128, 25, size=(h, w, 3))).clip(0, 255).astype(np.uint8)

    # Draw grid
    for x in range(0, w, w // 16 or 1):
        cv2.line(base, (x, 0), (x, h - 1), (60, 60, 60), 1)
    for y in range(0, h, h // 16 or 1):
        cv2.line(base, (0, y), (w - 1, y), (60, 60, 60), 1)

    # Random rectangles & circles
    for _ in range(60):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = rng.integers(0, w), rng.integers(0, h)
        color = tuple(int(c) for c in rng.integers(30, 225, size=3))
        cv2.rectangle(base, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), color, 1)
    for _ in range(40):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(8, max(9, min(w, h) // 10)))
        cv2.circle(base, c, r, (255, 255, 255), 1)

    # Text
    cv2.putText(base, "A-PNT DEMO TILE", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    return base


def write_dem_tif(dem_path: Path, bbox: Tuple[float, float, float, float], size: Tuple[int, int]) -> None:
    lon_min, lat_min, lon_max, lat_max = bbox
    w, h = size
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    # Simple gradient DEM (altitude 40..60 m)
    y = np.linspace(60, 40, h, dtype=np.float32)[..., None]
    x = np.linspace(40, 60, w, dtype=np.float32)[None, ...]
    dem = (0.5 * (x + y)).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(dem, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True, help="GeoJSON polygon defining AOI")
    ap.add_argument("--zoom", nargs="+", type=int, default=[17, 18], help="Zoom levels to create")
    ap.add_argument("--tile-size", type=int, default=1024, help="Tile width/height in pixels")
    ap.add_argument("--src-tif", default="", help="Optional GeoTIFF to rasterize into tiles")
    ap.add_argument("--src-image", default="", help="Optional PNG/JPG used as base imagery")
    ap.add_argument("--seed", type=int, default=1234, help="Seed for synthetic tile generator")
    args = ap.parse_args()

    tiles_root = Path("data/tiles")
    ensure_dirs(tiles_root, args.zoom)
    bbox = load_aoi_bbox(args.aoi)  # lon_min, lat_min, lon_max, lat_max
    out_size = (args.tile_size, args.tile_size)

    # Prepare a base image for all zooms
    if args.src_tif:
        base = load_tif_as_rgb_in_bbox(args.src_tif, bbox, out_size)
    elif args.src_image:
        base = load_image_as_tile(args.src_image, out_size)
    else:
        base = synthesize_tile(out_size, seed=args.seed)

    # Save one tile per zoom (z/0/0.png + metadata)
    for z in args.zoom:
        zdir = tiles_root / f"{z}/0"
        zdir.mkdir(parents=True, exist_ok=True)
        png_path = zdir / "0.png"
        json_path = zdir / "0.json"
        # OpenCV writes BGR; our base is already BGR-ish; ensure uint8
        cv2.imwrite(str(png_path), base)
        write_metadata_json(json_path, z, args.tile_size, args.tile_size, bbox)
        print(f"[ok] wrote {png_path} and {json_path}")

    # DEM
    dem_path = Path("data/dem/dem.tif")
    write_dem_tif(dem_path, bbox, (256, 256))
    print(f"[ok] wrote DEM {dem_path}")

    print("Tile cache initialized. You can now run the imagery API:")
    print("  uvicorn system_b.server:app --port 8000")


if __name__ == "__main__":
    main()
