import os, json
from dataclasses import dataclass
from typing import Optional
@dataclass
class Tile:
    image_path: str
    meta: dict
class TileCache:
    def __init__(self, root="data/tiles"):
        self.root = root
    def get_tile_path(self, z,x,y):
        return (f"{self.root}/{z}/{x}/{y}.png", f"{self.root}/{z}/{x}/{y}.json")
    def nearest_xyz(self, lat, lon, zoom):
        return (zoom, 0, 0)  # placeholder for tiny AOI
    def get_best_tile(self, lat, lon, zoom) -> Optional[Tile]:
        z,x,y = self.nearest_xyz(lat,lon,zoom)
        img, js = self.get_tile_path(z,x,y)
        if not (os.path.exists(img) and os.path.exists(js)): return None
        return Tile(img, json.load(open(js)))
