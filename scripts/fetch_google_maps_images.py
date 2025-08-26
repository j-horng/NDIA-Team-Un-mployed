#!/usr/bin/env python3
"""
Script to fetch 100 random images from Google Maps API
Focused on USA locations, with preference for Northern Virginia area
"""

import os
import sys
import json
import time
import hashlib
import random
from pathlib import Path
from typing import Dict, Any
import requests
from urllib.parse import urlencode

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_b.google_maps import GoogleMapsService

def generate_usa_coordinates() -> tuple[float, float]:
    """Generate random latitude and longitude coordinates within the USA"""
    
    # Define regions with weights (higher weight = more likely to be selected)
    regions = [
        # Northern Virginia area (highest weight)
        {
            'name': 'Northern Virginia',
            'weight': 40,
            'lat_range': (38.5, 39.2),  # Northern VA latitude range
            'lon_range': (-77.8, -77.0)  # Northern VA longitude range
        },
        # Washington DC area
        {
            'name': 'Washington DC',
            'weight': 20,
            'lat_range': (38.8, 39.0),
            'lon_range': (-77.2, -76.8)
        },
        # Maryland suburbs
        {
            'name': 'Maryland Suburbs',
            'weight': 15,
            'lat_range': (38.8, 39.3),
            'lon_range': (-77.5, -76.5)
        },
        # Rest of Virginia
        {
            'name': 'Virginia',
            'weight': 10,
            'lat_range': (36.5, 39.5),
            'lon_range': (-83.7, -75.0)
        },
        # Rest of USA (lower weight)
        {
            'name': 'USA',
            'weight': 15,
            'lat_range': (24.0, 49.0),  # Continental USA latitude
            'lon_range': (-125.0, -66.0)  # Continental USA longitude
        }
    ]
    
    # Select region based on weights
    total_weight = sum(region['weight'] for region in regions)
    rand = random.uniform(0, total_weight)
    current_weight = 0
    
    selected_region = regions[0]  # Default to first region
    for region in regions:
        current_weight += region['weight']
        if rand <= current_weight:
            selected_region = region
            break
    
    # Generate coordinates within the selected region
    lat = random.uniform(selected_region['lat_range'][0], selected_region['lat_range'][1])
    lon = random.uniform(selected_region['lon_range'][0], selected_region['lon_range'][1])
    
    return lat, lon

def calculate_sha256(data: bytes) -> str:
    """Calculate SHA256 hash of data"""
    return hashlib.sha256(data).hexdigest()

def fetch_image_with_metadata(google_maps: GoogleMapsService, lat: float, lon: float, zoom: int = 15) -> Dict[str, Any]:
    """
    Fetch an image from Google Maps API with timing and metadata
    
    Returns:
        Dictionary with image data, metadata, timing info, and status
    """
    start_time = time.time()
    
    try:
        # Get static map from Google Maps API
        tile = google_maps.get_static_map(lat=lat, lon=lon, zoom=zoom)
        
        if tile is None:
            return {
                "status": "error",
                "lat": lat,
                "lon": lon,
                "zoom": zoom,
                "latency_ms": int((time.time() - start_time) * 1000),
                "bytes": 0,
                "sha256": "",
                "path": "",
                "error": "Failed to fetch image"
            }
        
        latency_ms = int((time.time() - start_time) * 1000)
        bytes_size = len(tile.image_data)
        sha256_hash = calculate_sha256(tile.image_data)
        
        return {
            "status": "success",
            "lat": lat,
            "lon": lon,
            "zoom": zoom,
            "latency_ms": latency_ms,
            "bytes": bytes_size,
            "sha256": sha256_hash,
            "path": "",
            "image_data": tile.image_data,
            "metadata": tile.meta
        }
        
    except Exception as e:
        return {
            "status": "error",
            "lat": lat,
            "lon": lon,
            "zoom": zoom,
            "latency_ms": int((time.time() - start_time) * 1000),
            "bytes": 0,
            "sha256": "",
            "path": "",
            "error": str(e)
        }

def main():
    """Main function to fetch 100 random images from USA locations"""
    
    # Check if Google Maps API key is available
    if not os.getenv('GOOGLE_MAPS_API_KEY'):
        print(" Error: GOOGLE_MAPS_API_KEY environment variable is required")
        print("Please set it with: export GOOGLE_MAPS_API_KEY='your_api_key_here'")
        sys.exit(1)
    
    # Initialize Google Maps service
    try:
        google_maps = GoogleMapsService()
        print(" Google Maps API service initialized successfully")
    except Exception as e:
        print(f" Error initializing Google Maps service: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("data/google_maps_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Output directory: {output_dir.absolute()}")
    print("   Focusing on USA locations with preference for Northern Virginia area")
    
    # Fetch 100 random images
    num_images = 100
    results = []
    
    print(f"  Fetching {num_images} random images from USA locations...")
    
    for i in range(num_images):
        print(f"  [{i+1}/{num_images}] Fetching image...", end=" ")
        
        # Generate random coordinates in USA
        lat, lon = generate_usa_coordinates()
        zoom = random.randint(12, 18)  # Higher zoom levels for urban areas
        
        # Fetch image
        result = fetch_image_with_metadata(google_maps, lat, lon, zoom)
        
        if result["status"] == "success":
            # Save image file
            image_filename = f"image_{i+1:03d}.png"
            image_path = output_dir / image_filename
            with open(image_path, "wb") as f:
                f.write(result["image_data"])
            
            # Save metadata file
            meta_filename = f"image_{i+1:03d}.meta.json"
            meta_path = output_dir / meta_filename
            with open(meta_path, "w") as f:
                json.dump(result["metadata"], f, indent=2)
            
            # Update result with file path
            result["path"] = str(image_path)
            
            print(f"  Saved {image_filename} ({result['bytes']} bytes, {result['latency_ms']}ms)")
            print(f"       Location: {lat:.4f}, {lon:.4f} (zoom: {zoom})")
        else:
            print(f"  Failed: {result.get('error', 'Unknown error')}")
        
        results.append(result)
        
        # Add a small delay to be respectful to the API
        time.sleep(0.1)
    
    # Save index.jsonl file
    index_path = output_dir / "index.jsonl"
    with open(index_path, "w") as f:
        for result in results:
            # Remove image_data and metadata from the index (they're saved separately)
            index_entry = {k: v for k, v in result.items() 
                          if k not in ["image_data", "metadata"]}
            f.write(json.dumps(index_entry) + "\n")
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = num_images - successful
    
    print(f"\n  Summary:")
    print(f"    Successful: {successful}")
    print(f"    Failed: {failed}")
    print(f"    Images saved to: {output_dir}")
    print(f"    Index file: {index_path}")
    
    if successful > 0:
        avg_latency = sum(r["latency_ms"] for r in results if r["status"] == "success") / successful
        total_bytes = sum(r["bytes"] for r in results if r["status"] == "success")
        print(f"     Average latency: {avg_latency:.1f}ms")
        print(f"    Total data: {total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MB)")
    
    print(f"\n   Regional Distribution:")
    print(f"  • Northern Virginia: ~40% of images")
    print(f"  • Washington DC: ~20% of images")
    print(f"  • Maryland Suburbs: ~15% of images")
    print(f"  • Virginia: ~10% of images")
    print(f"  • Rest of USA: ~15% of images")

if __name__ == "__main__":
    main()
