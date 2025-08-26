#!/usr/bin/env python3
"""
Sample ~N coordinates inside the AOI and download tiles from the local imagery API.
Saves z-specific folders with PNG + metadata, and an index.jsonl with timing + sizes.

Example:
  uvicorn system_b.server:app --port 8000
  python scripts/sample_imagery_dataset.py --n 100 --zooms 17 18 --out data/samples --dedupe
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import requests


def load_aoi_polygon(geojson_path: str) -> List[Tuple[float, float]]:
    g = json.load(open(geojson_path, "r"))
    coords = g["features"][0]["geometry"]["coordinates"][0]
    return [(float(lon), float(lat)) for lon, lat in coords]  # lon, lat


def point_in_poly(lon: float, lat: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > lat) != (y2 > lat)):
            x_int = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
            if lon < x_int:
                inside = not inside
    return inside


def sample_points(poly: List[Tuple[float, float]], n: int, seed: int) -> List[Tuple[float, float]]:
    random.seed(seed)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    pts = []
    while len(pts) < n:
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        if point_in_poly(lon, lat, poly):
            pts.append((lon, lat))
    return pts


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/imagery")
    ap.add_argument("--aoi", default="config/area_of_interest.geojson")
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--zooms", nargs="+", type=int, default=[17, 18])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.jsonl").touch(exist_ok=True)

    poly = load_aoi_polygon(args.aoi)
    pts = sample_points(poly, args.n, args.seed)
    zooms = args.zooms
    seen_hashes = set()
    written = 0

    with open(out_dir / "index.jsonl", "a", buffering=1) as idx:
        for i, (lon, lat) in enumerate(pts):
            zoom = zooms[i % len(zooms)]
            t0 = time.time()
            try:
                r = requests.get(args.api, params={"lat": lat, "lon": lon, "zoom": zoom}, timeout=args.timeout)
                latency_ms = int(1000 * (time.time() - t0))
                status = r.status_code
                meta = {}
                if "X-Geo-Metadata" in r.headers:
                    try:
                        meta = json.loads(r.headers["X-Geo-Metadata"])
                    except Exception:
                        meta = {"_raw_header": r.headers["X-Geo-Metadata"]}
                content = r.content if r.ok else b""
                digest = sha256_bytes(content) if content else ""

                if args.dedupe and digest and digest in seen_hashes:
                    idx.write(json.dumps({
                        "i": i, "lat": lat, "lon": lon, "zoom": zoom,
                        "status": status, "latency_ms": latency_ms,
                        "bytes": len(content), "sha256": digest,
                        "path_png": None, "path_meta": None, "skipped": "dedupe"
                    }) + "\n")
                    continue

                zdir = out_dir / f"z{zoom}"
                zdir.mkdir(parents=True, exist_ok=True)
                png_path = zdir / f"img_{i:04d}.png"
                meta_path = zdir / f"img_{i:04d}.meta.json"

                if r.ok and content:
                    png_path.write_bytes(content)
                    meta_out = {"api_url": args.api, "lat": lat, "lon": lon, "zoom": zoom, "returned_meta": meta}
                    meta_path.write_text(json.dumps(meta_out, indent=2))
                    if args.dedupe and digest:
                        seen_hashes.add(digest)

                idx.write(json.dumps({
                    "i": i, "lat": lat, "lon": lon, "zoom": zoom,
                    "status": status, "latency_ms": latency_ms,
                    "bytes": len(content), "sha256": digest,
                    "path_png": str(png_path if r.ok else ""),
                    "path_meta": str(meta_path if r.ok else "")
                }) + "\n")
                written += 1
                if (i + 1) % 10 == 0:
                    print(f"[{i+1}/{args.n}] status={status} ms={latency_ms}")
            except Exception as e:
                latency_ms = int(1000 * (time.time() - t0))
                idx.write(json.dumps({
                    "i": i, "lat": lat, "lon": lon, "zoom": zoom,
                    "status": "error", "error": str(e), "latency_ms": latency_ms
                }) + "\n")

    print(f"Done. Wrote {written} entries to {out_dir/'index.jsonl'}")


if __name__ == "__main__":
    main()
