#!/usr/bin/env bash
#
# PX4 SITL helper. This script attempts to set the PX4 HOME from config
# and prints clear instructions to launch SITL. It does not fetch PX4.
#
# Usage:
#   system_d/px4_sitl/run_px4_sitl.sh
set -euo pipefail

CONF=${CONF:-config/params.yaml}

# Extract home from config (requires Python + PyYAML installed in your venv)
if command -v python >/dev/null 2>&1; then
  read -r HOME_LAT HOME_LON HOME_ALT <<EOF
$(python - <<'PY'
import yaml
P=yaml.safe_load(open("config/params.yaml", "r"))
print(P["aoi"]["home_lat"], P["aoi"]["home_lon"], P["aoi"]["home_alt_m"])
PY
)
  export PX4_HOME_LAT="${HOME_LAT}"
  export PX4_HOME_LON="${HOME_LON}"
  export PX4_HOME_ALT="${HOME_ALT}"
  echo "[SITL] PX4_HOME_LAT=${PX4_HOME_LAT} PX4_HOME_LON=${PX4_HOME_LON} PX4_HOME_ALT=${PX4_HOME_ALT}"
else
  echo "[SITL] Python not found; cannot parse config. Using PX4 defaults."
fi

cat <<'TXT'

PX4 SITL quickstart (choose ONE path):

1) If you have PX4 source locally:
   # From your PX4-Autopilot repo root:
   export PX4_HOME_LAT=${PX4_HOME_LAT:-38.8895}
   export PX4_HOME_LON=${PX4_HOME_LON:- -77.0352}
   export PX4_HOME_ALT=${PX4_HOME_ALT:-40}
   make px4_sitl_default gazebo

2) If you prefer Docker (example; adjust for your setup):
   # Ensure X server permissions on Linux and that the image/tag exists.
   docker run --rm -it \
     -e PX4_HOME_LAT=${PX4_HOME_LAT:-38.8895} \
     -e PX4_HOME_LON=${PX4_HOME_LON:- -77.0352} \
     -e PX4_HOME_ALT=${PX4_HOME_ALT:-40} \
     -p 14540:14540/udp \
     ghcr.io/pixhawk/containers/simulation:latest

Notes:
- System D publishes to udpout:127.0.0.1:14540 by default (see config/fusion.px4_url).
- You can run System D without SITL to validate publish timing (PX4 not required to run).
TXT
