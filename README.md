# A-PNT for UAS via Satellite Imagery & Sensor Correlation (ABCD)

End-to-end hackathon stack demonstrating Assured Position, Navigation & Timing using:
- A – UAS Platform (simulated inputs)
- B – Imagery API (local tile/DEM + optional Google fallback)
- C – Feature correlation & georegistration (ORB + RANSAC → GeoFix)
- D – Navigation controller (publishes GeoFix into PX4 EKF2 via MAVLink)

## Quick Start (Docker Compose)

1) Bring up the core stack (B, C, Dashboard). A is simulated inside C.

```bash
docker compose up -d imagery system_c dashboard
```

2) Open the dashboard: http://localhost:8501

3) API health check:

```bash
curl http://localhost:8000/health
```

### PX4 SITL (optional)

- Terminal A: run PX4 SITL (outside this repo, standard PX4 instructions).
- Terminal B: start the fusion bridge with the px4 profile after SITL is running:

```bash
docker compose --profile px4 up system_d
```

### Local (no Docker)

Create a venv and install deps:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Seed tiles/DEM:

```bash
python scripts/build_tile_cache.py --aoi config/area_of_interest.geojson --zoom 17 18
```

Start services (in separate terminals):

```bash
uvicorn system_b.server:app --host 0.0.0.0 --port 8000
python -m system_c.pipeline --config config/params.yaml
streamlit run dashboard/app.py
```

Optional (requires PX4 heartbeat):

```bash
python -m system_d.fusion_px4 --config config/params.yaml --method gps_input --rate 10
```

## Repository Layout

```
config/
  params.yaml
  camera_intrinsics.yaml
  area_of_interest.geojson
dashboard/
  app.py
scripts/
  build_tile_cache.py
  sample_imagery_dataset.py
  run_demo.sh
  seed_demo_data.sh
system_a/
  __init__.py, camera.py, imu.py, service.py
system_b/
  __init__.py, server.py, tile_cache.py, dem.py, google_maps.py
system_c/
  __init__.py, pipeline.py, correlate.py, preprocess.py, features.py
system_d/
  __init__.py, mavlink_out.py, fusion_px4.py, fusion_local_ekf.py
  px4_sitl/run_px4_sitl.sh
tests/
  ... (pytest suite)
```

## How it works

1. System B serves `GET /imagery?lat&lon&zoom` → PNG with X-Geo-Metadata.
2. System C fetches a reference tile, synthesizes sensor crops, runs ORB+RANSAC to geolocate the camera center pixel, and writes GeoFix rows to `logs/metrics.jsonl`.
3. System D (optional) tails that JSONL, maps covariance → GPS accuracies, and publishes `GPS_INPUT` or `VISION_POSITION_ESTIMATE` via MAVLink to PX4.
4. The dashboard visualizes metrics and the current fix over the AOI.

## Configuration

Edit `config/params.yaml`:
- `aoi.*` — AOI, reference home, altitude
- `tiles.*` — tile cache paths/zooms
- `correlation.*` — ORB, RANSAC, gates
- `fusion.*` — PX4 URL, publish rate, accuracy mapping
- `providers.google_maps` — OFF by default (see ToS note)

## Google Maps provider (optional)

We include an adapter for Google Static Maps as a fallback provider. It is disabled by default.

If you enable it in `config/params.yaml`, set your key:

```bash
export GOOGLE_MAPS_API_KEY=YOUR_KEY
```

Or create a `.env` with `GOOGLE_MAPS_API_KEY` and restart the imagery service.

ToS note: Many plans prohibit caching/proxying Static Maps and limit processing to “Google map visualizations.” Keep this OFF unless your license explicitly allows your intended use. If you enable it, the server returns Google imagery without caching.

## Testing

```bash
pytest -q
```

The suite builds a temp tile cache, patches the server to use it, and runs correlation on synthetic images (no network calls).

## Troubleshooting

- Rasterio wheels: If rasterio fails to install on your host, prefer the Docker path; wheels inside `python:3.11-slim` typically just work.
- OpenCV GUI: We use `opencv-python-headless`; no GUI dependencies required.
- PX4 heartbeat: System D waits for a MAVLink heartbeat. Use the px4 profile and start SITL first.
- Tile cache empty: Re-run `python scripts/build_tile_cache.py` (or `docker compose up seed`).

## License / Credits

This repo combines open-source libraries (OpenCV, FastAPI, Rasterio, Streamlit, PyMAVLink).

Google Maps provider usage must comply with Google’s terms.