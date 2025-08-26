"""
Unit tests for Google Maps service
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from system_b.google_maps import GoogleMapsService, GoogleMapsTile

class TestGoogleMapsService:
    """Test cases for GoogleMapsService"""
    
    def test_init_with_api_key(self):
        """Test initialization with API key"""
        api_key = "test_api_key"
        service = GoogleMapsService(api_key=api_key)
        assert service.api_key == api_key
        assert service.base_url == "https://maps.googleapis.com/maps/api/staticmap"
    
    def test_init_with_env_var(self):
        """Test initialization with environment variable"""
        api_key = "env_api_key"
        with patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': api_key}):
            service = GoogleMapsService()
            assert service.api_key == api_key
    
    def test_init_no_api_key(self):
        """Test initialization without API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google Maps API key is required"):
                GoogleMapsService()
    
    def test_calculate_geotransform(self):
        """Test geotransform calculation"""
        service = GoogleMapsService(api_key="test")
        lat, lon, zoom = 38.872, -77.058, 15
        size = "512x512"
        
        geotransform = service._calculate_geotransform(lat, lon, zoom, size)
        
        assert len(geotransform) == 6
        assert geotransform[0] == pytest.approx(-77.083, abs=0.001)  # top_left_lon
        assert geotransform[3] == pytest.approx(38.883, abs=0.001)   # top_left_lat
        assert geotransform[1] > 0  # pixel_width (positive)
        assert geotransform[5] < 0  # pixel_height (negative)
    
    def test_sat_pix2geo(self):
        """Test pixel to geographic coordinate conversion"""
        service = GoogleMapsService(api_key="test")
        
        # Test geotransform parameters
        geotransform = [-77.083, 0.0001, 0, 38.883, 0, -0.00004]
        
        # Test center pixel (256, 256) should give center coordinates
        lon, lat = service.sat_pix2geo(256, 256, geotransform)
        
        assert lon == pytest.approx(-77.058, abs=0.001)
        assert lat == pytest.approx(38.872, abs=0.001)
    
    @patch('requests.get')
    def test_get_static_map_success(self, mock_get):
        """Test successful static map retrieval"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response
        
        service = GoogleMapsService(api_key="test")
        tile = service.get_static_map(lat=38.872, lon=-77.058, zoom=15)
        
        assert tile is not None
        assert isinstance(tile, GoogleMapsTile)
        assert tile.image_data == b"fake_image_data"
        assert tile.meta['crs'] == 'EPSG:4326'
        assert tile.meta['maptype'] == 'satellite'
        assert tile.meta['source'] == 'google_static_maps'
    
    @patch('requests.get')
    def test_get_static_map_failure(self, mock_get):
        """Test failed static map retrieval"""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_get.return_value = mock_response
        
        service = GoogleMapsService(api_key="test")
        tile = service.get_static_map(lat=38.872, lon=-77.058, zoom=15)
        
        assert tile is None
    
    @patch('requests.get')
    def test_get_static_map_exception(self, mock_get):
        """Test exception handling in static map retrieval"""
        # Mock exception
        mock_get.side_effect = Exception("Network error")
        
        service = GoogleMapsService(api_key="test")
        tile = service.get_static_map(lat=38.872, lon=-77.058, zoom=15)
        
        assert tile is None

class TestGoogleMapsTile:
    """Test cases for GoogleMapsTile dataclass"""
    
    def test_tile_creation(self):
        """Test GoogleMapsTile creation"""
        image_data = b"test_image"
        meta = {"test": "metadata"}
        
        tile = GoogleMapsTile(image_data=image_data, meta=meta)
        
        assert tile.image_data == image_data
        assert tile.meta == meta
