import requests
import json
from dataclasses import dataclass
from typing import Optional, Tuple
import os
from urllib.parse import urlencode

@dataclass
class GoogleMapsTile:
    image_data: bytes
    meta: dict

class GoogleMapsService:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google Maps service with API key"""
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError("Google Maps API key is required. Set GOOGLE_MAPS_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    def get_static_map(self, lat: float, lon: float, zoom: int = 15, 
                      size: str = "512x512", maptype: str = "satellite") -> Optional[GoogleMapsTile]:
        """
        Get static map from Google Maps API
        
        Args:
            lat: Latitude
            lon: Longitude  
            zoom: Zoom level (0-20)
            size: Image size (e.g., "512x512")
            maptype: Map type (satellite, roadmap, terrain, hybrid)
            
        Returns:
            GoogleMapsTile with image data and metadata, or None if failed
        """
        try:
            # Build query parameters
            params = {
                'center': f"{lat},{lon}",
                'zoom': zoom,
                'size': size,
                'maptype': maptype,
                'key': self.api_key
            }
            
            # Make request to Google Static Maps API
            url = f"{self.base_url}?{urlencode(params)}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Create metadata similar to tile cache format
                meta = {
                    'geotransform': self._calculate_geotransform(lat, lon, zoom, size),
                    'size': size,
                    'crs': 'EPSG:4326',
                    'maptype': maptype,
                    'zoom': zoom,
                    'center_lat': lat,
                    'center_lon': lon,
                    'source': 'google_static_maps'
                }
                
                return GoogleMapsTile(
                    image_data=response.content,
                    meta=meta
                )
            else:
                print(f"Google Maps API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error fetching Google Static Map: {e}")
            return None
    
    def _calculate_geotransform(self, lat: float, lon: float, zoom: int, size: str) -> list:
        """
        Calculate geotransform parameters for the image
        Returns: [top_left_x, pixel_width, 0, top_left_y, 0, pixel_height]
        """
        # Parse size string (e.g., "512x512")
        width, height = map(int, size.split('x'))
        
        # Calculate pixel resolution at this zoom level
        # At zoom level 0, one pixel represents ~156543 meters at equator
        meters_per_pixel = 156543.03392 / (2 ** zoom)
        
        # Convert to degrees (approximate)
        degrees_per_pixel_lat = meters_per_pixel / 111320.0  # meters per degree latitude
        degrees_per_pixel_lon = meters_per_pixel / (111320.0 * abs(lat) / 90.0)  # varies by latitude
        
        # Calculate top-left corner
        top_left_lat = lat + (height / 2) * degrees_per_pixel_lat
        top_left_lon = lon - (width / 2) * degrees_per_pixel_lon
        
        return [
            top_left_lon,  # top_left_x
            degrees_per_pixel_lon,  # pixel_width
            0,  # rotation (0 for static maps)
            top_left_lat,  # top_left_y
            0,  # rotation (0 for static maps)
            -degrees_per_pixel_lat  # pixel_height (negative because image coordinates are top-down)
        ]
    
    def sat_pix2geo(self, pixel_x: float, pixel_y: float, geotransform: list) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            geotransform: Geotransform parameters [x0, dx, 0, y0, 0, dy]
            
        Returns:
            Tuple of (longitude, latitude)
        """
        x0, dx, _, y0, _, dy = geotransform
        
        lon = x0 + pixel_x * dx
        lat = y0 + pixel_y * dy
        
        return (lon, lat)
