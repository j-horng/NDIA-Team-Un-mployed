#!/usr/bin/env bash
# End-to-end demo runner: starts Imagery API (B), System C pipeline, D fusion bridge, and the dashboard.
# Usage:
#   bash scripts/run_demo.sh
set -euo pipefail

PORT=${PORT:-8000}
API_URL="http://127.0.0.1:${PORT}/imagery"
CONF=${CONF:-config/params.yaml}

# Pre-flight checks
command -v uvicorn >/dev/null || { echo "uvicorn not found. pip install -r requirements.txt"; exit 1; }
command -v python >/dev/null || { echo "python not found"; exit 1; }
command -v streamlit >/dev/null || { echo "streamlit not found. pip install -r requirements.txt"; exit 1; }

mkdir -p logs data/tiles data/dem

# Ensure at least one tile exists; if not, synthesize one
if [ ! -f "data/tiles/18/0/0.png" ]; then
  echo "[setup] No tiles found; building synthetic cache..."
  python scripts/build_tile_cache.py --aoi config/area_of_interest.geojson --zoom 17 18
fi

pids=()
cleanup() {
  echo ""
  echo "[cleanup] Stopping services..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "[start] Imagery API on :$PORT"
uvicorn system_b.server:app --host 0.0.0.0 --port "$PORT" > logs/imagery_api.log 2>&1 &
pids+=($!)

sleep 1

echo "[start] System C pipeline"
python -m system_c.pipeline --config "$CONF" > logs/system_c.log 2>&1 &
pids+=($!)

sleep 1

echo "[start] System D fusion bridge (GPS_INPUT)"
python -m system_d.fusion_px4 --config "$CONF" --method gps_input --rate 10 > logs/system_d.log 2>&1 &
pids+=($!)

sleep 1

echo "[start] Dashboard"
streamlit run dashboard/app.py > logs/dashboard.log 2>&1 &
pids+=($!)

echo ""
echo "Demo running. Tail logs with: tail -f logs/*.log"
echo "Stop with Ctrl-C."
wait
