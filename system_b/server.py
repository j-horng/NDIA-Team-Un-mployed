from fastapi import FastAPI, Query
from fastapi.responses import Response, JSONResponse
from .google_maps import GoogleMapsService
import json

app = FastAPI()

# Initialize Google Maps service
try:
    google_maps = GoogleMapsService()
    print("  Google Maps API service initialized successfully")
except ValueError as e:
    print(f"   Google Maps API not available: {e}")
    google_maps = None

@app.get("/imagery")
def imagery(lat: float = Query(...), lon: float = Query(...), zoom: int = Query(15)):
    """
    Get satellite imagery from Google Static Maps API
    
    Args:
        lat: Latitude
        lon: Longitude
        zoom: Zoom level (0-20, default 15)
    
    Returns:
        PNG image with X-Geo-Metadata header
    """
    if google_maps is None:
        return JSONResponse(
            {"error": "Google Maps API not configured. Set GOOGLE_MAPS_API_KEY environment variable."}, 
            status_code=503
        )
    
    # Get static map from Google Maps API
    tile = google_maps.get_static_map(lat=lat, lon=lon, zoom=zoom)
    
    if tile is None:
        return JSONResponse({"error": "Failed to fetch imagery from Google Maps API"}, status_code=404)
    
    # Return image with metadata
    return Response(
        tile.image_data, 
        media_type="image/png",
        headers={"X-Geo-Metadata": json.dumps(tile.meta)}
    )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "google_maps_available": google_maps is not None,
        "service": "System B - Imagery Server"
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "System B - Imagery Server",
        "endpoints": {
            "/imagery": "GET satellite imagery (lat, lon, zoom)",
            "/health": "GET service health status"
        },
        "google_maps_available": google_maps is not None
    }
