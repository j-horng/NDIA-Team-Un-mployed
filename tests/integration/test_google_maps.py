#!/usr/bin/env python3
"""
Integration test for Google Maps API integration in System B
"""

import os
import sys
import requests
from PIL import Image
import io

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from system_b.google_maps import GoogleMapsService

def test_google_maps_api():
    """Test the Google Maps API integration"""
    
    # Set the API key from the URL you provided
    api_key = "AIzaSyA4CrTjx8pIVmQ3yBgN8wVDSGQbIm00bW8"
    
    # Test coordinates from your URL
    lat = 38.872
    lon = -77.058
    zoom = 15
    
    print(f"Testing Google Maps API with coordinates: {lat}, {lon}, zoom: {zoom}")
    
    try:
        # Initialize the service
        google_maps = GoogleMapsService(api_key=api_key)
        print("✅ Google Maps service initialized")
        
        # Get static map
        tile = google_maps.get_static_map(lat=lat, lon=lon, zoom=zoom)
        
        if tile:
            print("✅ Successfully retrieved static map")
            print(f"   Image size: {len(tile.image_data)} bytes")
            print(f"   Metadata: {tile.meta}")
            
            # Test pixel to geo conversion
            geotransform = tile.meta['geotransform']
            center_x, center_y = 256, 256  # Center of 512x512 image
            
            geo_lon, geo_lat = google_maps.sat_pix2geo(center_x, center_y, geotransform)
            print(f"   Center pixel (256,256) -> ({geo_lon:.6f}, {geo_lat:.6f})")
            print(f"   Expected center: ({lon:.6f}, {lat:.6f})")
            
            # Save test image to fixtures folder
            fixtures_dir = os.path.join(project_root, "tests", "fixtures", "images")
            img = Image.open(io.BytesIO(tile.image_data))
            img.save(os.path.join(fixtures_dir, "test_google_maps.png"))
            print("✅ Saved test image as 'tests/fixtures/images/test_google_maps.png'")
            
        else:
            print("❌ Failed to retrieve static map")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_server_endpoint():
    """Test the FastAPI server endpoint"""
    
    print("\nTesting FastAPI server endpoint...")
    
    # Start the server (this would normally be done separately)
    # For now, we'll test the direct API call
    api_key = "AIzaSyA4CrTjx8pIVmQ3yBgN8wVDSGQbIm00bW8"
    
    # Test the exact URL format from your example
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        'center': '38.872,-77.058',
        'zoom': 15,
        'size': '512x512',
        'maptype': 'satellite',
        'key': api_key
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            print("✅ Direct API call successful")
            print(f"   Response size: {len(response.content)} bytes")
            print(f"   Content type: {response.headers.get('content-type', 'unknown')}")
            
            # Save the image to fixtures folder
            fixtures_dir = os.path.join(project_root, "tests", "fixtures", "images")
            with open(os.path.join(fixtures_dir, "test_direct_api.png"), "wb") as f:
                f.write(response.content)
            print("✅ Saved direct API response as 'tests/fixtures/images/test_direct_api.png'")
            
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error making direct API call: {e}")

def test_system_b_server():
    """Test the System B server endpoint"""
    
    print("\nTesting System B server endpoint...")
    
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Health endpoint working")
            health_data = health_response.json()
            print(f"   Google Maps available: {health_data.get('google_maps_available', False)}")
        else:
            print(f"❌ Health endpoint failed: {health_response.status_code}")
            return
        
        # Test imagery endpoint
        imagery_response = requests.get(
            "http://localhost:8000/imagery?lat=38.872&lon=-77.058&zoom=15",
            timeout=10
        )
        
        if imagery_response.status_code == 200:
            print("✅ Imagery endpoint working")
            print(f"   Response size: {len(imagery_response.content)} bytes")
            
            # Check metadata header
            metadata_header = imagery_response.headers.get('X-Geo-Metadata')
            if metadata_header:
                print("✅ Metadata header present")
                import json
                metadata = json.loads(metadata_header)
                print(f"   CRS: {metadata.get('crs', 'unknown')}")
                print(f"   Size: {metadata.get('size', 'unknown')}")
            
            # Save the image to fixtures folder
            fixtures_dir = os.path.join(project_root, "tests", "fixtures", "images")
            with open(os.path.join(fixtures_dir, "test_server_response.png"), "wb") as f:
                f.write(imagery_response.content)
            print("✅ Saved server response as 'tests/fixtures/images/test_server_response.png'")
            
        else:
            print(f"❌ Imagery endpoint failed: {imagery_response.status_code}")
            print(f"   Response: {imagery_response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: uvicorn system_b.server:app --port 8000")
    except Exception as e:
        print(f"❌ Error testing server: {e}")

if __name__ == "__main__":
    print("🧪 Testing Google Maps API Integration")
    print("=" * 50)
    
    test_google_maps_api()
    test_server_endpoint()
    test_system_b_server()
    
    print("\n" + "=" * 50)
    print("📝 To use with System B server:")
    print("1. Set environment variable: export GOOGLE_MAPS_API_KEY='your_api_key'")
    print("2. Start server: uvicorn system_b.server:app --port 8000")
    print("3. Test endpoint: curl 'http://localhost:8000/imagery?lat=38.872&lon=-77.058&zoom=15'")
    print("4. Run tests: python -m pytest tests/")
