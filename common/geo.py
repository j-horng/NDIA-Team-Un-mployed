from typing import Tuple
import numpy as np, math
def pix2geo(x: float, y: float, meta: dict) -> Tuple[float,float]:
    lon = meta["top_left_lon"] + x * meta["px_size_lon"]
    lat = meta["top_left_lat"] + y * meta["px_size_lat"]
    return float(lon), float(lat)
def lla_to_local_ned(lla, ref):
    lat, lon, alt = lla; rlat, rlon, ralt = ref
    dlat = math.radians(lat - rlat); dlon = math.radians(lon - rlon)
    R = 6378137.0; north = dlat * R; east = dlon * R * math.cos(math.radians((lat+rlat)/2))
    down = (ralt - alt); return np.array([north, east, down], dtype=float)
