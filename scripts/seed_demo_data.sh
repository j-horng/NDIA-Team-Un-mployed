#!/usr/bin/env bash
# Initialize folders, build a synthetic tile cache, and (optionally) generate a small sample set.
# Usage:
#   bash scripts/seed_demo_data.sh
set -euo pipefail

ZOOMS="${ZOOMS:-"17 18"}"
N="${N:-100}"
OUT="${OUT:-data/samples}"
CONF_AOI="${CONF_AOI:-config/area_of_interest.geojson}"

mkdir -p data/tiles data/dem data/video data/imu logs

echo "[seed] Building tile cache (zooms: $ZOOMS)"
python scripts/build_tile_cache.py --aoi "$CONF_AOI" --zoom $ZOOMS

echo "[seed] Starting imagery API on :8000 (background)"
uvicorn system_b.server:app --host 127.0.0.1 --port 8000 > logs/imagery_api_seed.log 2>&1 &
API_PID=$!
trap "kill $API_PID 2>/dev/null || true" EXIT INT TERM

sleep 1

echo "[seed] Sampling $N images from API -> $OUT"
python scripts/sample_imagery_dataset.py --n "$N" --zooms $ZOOMS --out "$OUT" --aoi "$CONF_AOI" --dedupe

echo "[seed] Done. Sample index at $OUT/index.jsonl"
