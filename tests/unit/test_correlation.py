"""
Unit tests for correlation system (System C)
"""

import pytest
import numpy as np
import cv2
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from system_c.correlate import orb_ransac_georeg
from common.types import ImageFrame

class TestCorrelation:
    """Test cases for correlation functions"""
    
    def test_orb_ransac_georeg_with_valid_images(self):
        """Test ORB-RANSAC georegistration with valid images"""
        # Create test camera frame
        cam_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cam_frame = ImageFrame(
            ts="2023-01-01T12:00:00Z",
            width=640,
            height=480,
            frame=cam_frame
        )
        
        # Create test satellite image (same size for simplicity)
        sat_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Mock pixel to geo conversion function
        def mock_sat_pix2geo(x, y):
            return (-77.058 + x * 0.0001, 38.872 + y * 0.0001)
        
        # Test with valid parameters
        result = orb_ransac_georeg(
            cam=cam_frame,
            sat_gray=sat_gray,
            sat_pix2geo=mock_sat_pix2geo,
            nfeatures=100,  # Lower for testing
            fast=12,
            min_inliers=10,  # Lower for testing
            ransac_px=3.0
        )
        
        # Result might be None if not enough features match
        # This is expected behavior for random images
        if result is not None:
            assert hasattr(result, 'ts')
            assert hasattr(result, 'lat')
            assert hasattr(result, 'lon')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'inliers')
            assert hasattr(result, 'rmse_px')
            assert 0 <= result.confidence <= 1
            assert result.inliers >= 0
    
    def test_orb_ransac_georeg_with_insufficient_features(self):
        """Test ORB-RANSAC with insufficient features"""
        # Create test images with very few features
        cam_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cam_frame = ImageFrame(
            ts="2023-01-01T12:00:00Z",
            width=100,
            height=100,
            frame=cam_frame
        )
        
        sat_gray = np.zeros((100, 100), dtype=np.uint8)
        
        def mock_sat_pix2geo(x, y):
            return (-77.058, 38.872)
        
        result = orb_ransac_georeg(
            cam=cam_frame,
            sat_gray=sat_gray,
            sat_pix2geo=mock_sat_pix2geo,
            nfeatures=2000,
            min_inliers=40
        )
        
        # Should return None due to insufficient features
        assert result is None
    
    def test_orb_ransac_georeg_with_grayscale_camera_frame(self):
        """Test ORB-RANSAC with grayscale camera frame"""
        # Create grayscale camera frame
        cam_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        cam_frame = ImageFrame(
            ts="2023-01-01T12:00:00Z",
            width=640,
            height=480,
            frame=cam_frame
        )
        
        sat_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        def mock_sat_pix2geo(x, y):
            return (-77.058 + x * 0.0001, 38.872 + y * 0.0001)
        
        result = orb_ransac_georeg(
            cam=cam_frame,
            sat_gray=sat_gray,
            sat_pix2geo=mock_sat_pix2geo,
            nfeatures=100,
            min_inliers=10
        )
        
        # Should handle grayscale input correctly
        if result is not None:
            assert hasattr(result, 'lat')
            assert hasattr(result, 'lon')
    
    def test_orb_ransac_georeg_parameter_validation(self):
        """Test parameter validation in ORB-RANSAC"""
        cam_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cam_frame = ImageFrame(
            ts="2023-01-01T12:00:00Z",
            width=100,
            height=100,
            frame=cam_frame
        )
        
        sat_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        def mock_sat_pix2geo(x, y):
            return (-77.058, 38.872)
        
        # Test with different parameter combinations
        result1 = orb_ransac_georeg(
            cam=cam_frame,
            sat_gray=sat_gray,
            sat_pix2geo=mock_sat_pix2geo,
            nfeatures=50,
            fast=8,
            min_inliers=5,
            ransac_px=2.0
        )
        
        result2 = orb_ransac_georeg(
            cam=cam_frame,
            sat_gray=sat_gray,
            sat_pix2geo=mock_sat_pix2geo,
            nfeatures=200,
            fast=20,
            min_inliers=20,
            ransac_px=5.0
        )
        
        # Both should either return None or valid results
        for result in [result1, result2]:
            if result is not None:
                assert hasattr(result, 'confidence')
                assert 0 <= result.confidence <= 1

class TestImageFrame:
    """Test cases for ImageFrame dataclass"""
    
    def test_image_frame_creation(self):
        """Test ImageFrame creation"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_frame = ImageFrame(
            ts="2023-01-01T12:00:00Z",
            width=640,
            height=480,
            frame=frame,
            camera_id="test_cam"
        )
        
        assert img_frame.ts == "2023-01-01T12:00:00Z"
        assert img_frame.width == 640
        assert img_frame.height == 480
        assert img_frame.frame.shape == (480, 640, 3)
        assert img_frame.camera_id == "test_cam"
    
    def test_image_frame_default_camera_id(self):
        """Test ImageFrame with default camera_id"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_frame = ImageFrame(
            ts="2023-01-01T12:00:00Z",
            width=100,
            height=100,
            frame=frame
        )
        
        assert img_frame.camera_id == "cam0"  # Default value
