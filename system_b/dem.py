import os, rasterio
class DEM:
    def __init__(self, path="data/dem/dem.tif"):
        self.ds = rasterio.open(path) if os.path.exists(path) else None
    def xyz(self, lat, lon):
        if not self.ds: return (lon, lat, 50.0)
        r,c = self.ds.index(lon, lat)
        z = float(self.ds.read(1)[r,c])
        return (lon, lat, z)
