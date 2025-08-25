from fastapi import FastAPI, Query
from fastapi.responses import Response, JSONResponse
from .tile_cache import TileCache
import json
app = FastAPI(); cache = TileCache()
@app.get("/imagery")
def imagery(lat: float = Query(...), lon: float = Query(...), zoom: int = Query(17)):
    t = cache.get_best_tile(lat, lon, zoom)
    if t is None: return JSONResponse({"error":"tile_not_found"}, status_code=404)
    return Response(open(t.image_path,"rb").read(), media_type="image/png",
                    headers={"X-Geo-Metadata": json.dumps(t.meta)})
