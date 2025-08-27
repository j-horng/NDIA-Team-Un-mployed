#!/usr/bin/env python3
"""
Script to fetch first 1000 images of Kaggle Dataset from Google Maps API
Uses the UAV navigation dataset coordinates to fetch corresponding satellite imagery
"""

import os
import sys
import json
import time
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import requests

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_b.google_maps import GoogleMapsService

def calculate_sha256(data: bytes) -> str:
    """Calculate SHA256 hash of data"""
    return hashlib.sha256(data).hexdigest()
def fetch_image_with_metadata(google_maps: GoogleMapsService, lat: float, lon: float, zoom: int = 15) -> Dict[str, Any]:
    """Fetch image and metadata for given coordinates"""

    try:
        # Fetch satellite image
        result = google_maps.get_static_map(lat, lon, zoom, maptype="satellite")

        if result:
            # Calculate hash
            sha256_hash = calculate_sha256(result.image_data)

            return {
                "status": "success",
                "lat": lat,
                "lon": lon,
                "zoom": zoom,
                "latency_ms": 0,  # Not tracked in this implementation
                "bytes": len(result.image_data),
                "sha256": sha256_hash,
                "image_data": result.image_data,
                "metadata": result.meta
            }
        else:
            return {
                "status": "error",
                "lat": lat,
                "lon": lon,
                "zoom": zoom,
                "error": "Failed to fetch image",
                "image_data": None,
                "metadata": {}
            }

    except Exception as e:
        return {
            "status": "error",
            "lat": lat,
            "lon": lon,
            "zoom": zoom,
            "error": str(e),
            "image_data": None,
            "metadata": {}
        }

def load_kaggle_dataset(file_path: str, max_rows: int = 1000) -> List[Dict[str, Any]]:
    """Load the Kaggle UAV navigation dataset"""
    print(f"  Loading Kaggle dataset from {file_path}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Take only the first max_rows
        df = df.head(max_rows)
        
        # Convert to list of dictionaries
        rows = df.to_dict('records')
        
        print(f"  Loaded {len(rows)} rows from dataset")
        
        # Print some statistics
        if len(rows) > 0:
            lats = [row['latitude'] for row in rows]
            lons = [row['longitude'] for row in rows]
            altitudes = [row['altitude'] for row in rows]
            speeds = [row['speed'] for row in rows]
            
            print(f"    Location range: {min(lats):.4f} to {max(lats):.4f} lat, {min(lons):.4f} to {max(lons):.4f} lon")
            print(f"    Altitude range: {min(altitudes):.1f}m - {max(altitudes):.1f}m")
            print(f"    Speed range: {min(speeds):.1f}m/s - {max(speeds):.1f}m/s")
        
        return rows
    
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return []

def main():
    """Main function to fetch Kaggle dataset images"""
    
    # Configuration
    dataset_file = "data/uav_navigation_dataset.csv"
    output_dir = Path("data/kaggle_dataset_images")
    max_images = 1000
    
    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f" Error: Dataset file not found at {dataset_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f" Fetching first {max_images} images from Kaggle UAV navigation dataset...")
    print(f"  Dataset file: {dataset_file}")
    print(f"  Output directory: {output_dir}")
    
    # Load dataset
    dataset_rows = load_kaggle_dataset(dataset_file, max_images)
    
    if not dataset_rows:
        print(" Error: No data loaded from dataset")
        sys.exit(1)
    
    # Initialize Google Maps service
    google_maps = GoogleMapsService()
    
    # Process each row
    successful_fetches = 0
    failed_fetches = 0
    index_entries = []
    
    print(f"  Fetching images for {len(dataset_rows)} coordinates...")
    
    for i, row in enumerate(dataset_rows):
        print(f"    Processing image {i+1}/{len(dataset_rows)}: {row['latitude']:.4f}, {row['longitude']:.4f}")
        
        # Fetch image
        result = fetch_image_with_metadata(
            google_maps, 
            row['latitude'], 
            row['longitude']
        )
        
        if result["status"] == "success":
            # Save image file
            image_filename = f"kaggle_dataset_{i+1:04d}.png"
            image_path = output_dir / image_filename
            with open(image_path, "wb") as f:
                f.write(result["image_data"])
            
            # Save metadata file (Google Maps metadata)
            meta_filename = f"kaggle_dataset_{i+1:04d}.meta.json"
            meta_path = output_dir / meta_filename
            with open(meta_path, "w") as f:
                json.dump(result["metadata"], f, indent=2)
            
            # Save dataset row data
            dataset_filename = f"kaggle_dataset_{i+1:04d}.dataset.json"
            dataset_path = output_dir / dataset_filename
            with open(dataset_path, "w") as f:
                json.dump(row, f, indent=2)
            
            # Add to index
            index_entry = {
                "status": "success",
                "lat": result["lat"],
                "lon": result["lon"],
                "zoom": result["zoom"],
                "latency_ms": result["latency_ms"],
                "bytes": result["bytes"],
                "sha256": result["sha256"],
                "path": str(image_path.relative_to(Path("data")))
            }
            index_entries.append(index_entry)
            
            successful_fetches += 1
            print(f"      ✓ Saved {image_filename} ({result['bytes']:,} bytes)")
        
        else:
            # Save error info
            error_filename = f"kaggle_dataset_{i+1:04d}.error.json"
            error_path = output_dir / error_filename
            with open(error_path, "w") as f:
                json.dump({
                    "status": "error",
                    "lat": result["lat"],
                    "lon": result["lon"],
                    "error": result["error"],
                    "dataset_row": row
                }, f, indent=2)
            
            # Add to index
            index_entry = {
                "status": "error",
                "lat": result["lat"],
                "lon": result["lon"],
                "error": result["error"],
                "path": str(error_path.relative_to(Path("data")))
            }
            index_entries.append(index_entry)
            
            failed_fetches += 1
            print(f"      ✗ Error: {result['error']}")
        
        # Rate limiting - be nice to the API
        time.sleep(0.1)
    
    # Save index file
    index_path = output_dir / "index.jsonl"
    with open(index_path, "w") as f:
        for entry in index_entries:
            f.write(json.dumps(entry) + "\n")
    
    # Print summary
    print(f"\n Fetching complete!")
    print(f"  Successful: {successful_fetches}")
    print(f"  Failed: {failed_fetches}")
    print(f"  Total: {len(dataset_rows)}")
    print(f"  Index saved to: {index_path}")
    
    if successful_fetches > 0:
        total_size = sum(entry.get("bytes", 0) for entry in index_entries if entry["status"] == "success")
        print(f"  Total size: {total_size / (1024*1024):.1f} MB")
    
    return successful_fetches > 0

if __name__ == "__main__":
    main()
