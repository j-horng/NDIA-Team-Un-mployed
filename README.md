# A-PNT for UAS via Satellite Imagery & Sensor Correlation (ABCD)

**A:** UAS Platform (camera/IMU, sim or replay)  
**B:** Imagery Server (offline tiles/DEM + `/imagery`)  
**C:** Correlation (ORBâ†’RANSAC, optional PnP+DEM) â†’ GeoFix  
**D:** Nav Controller (publish GeoFix to PX4 EKF2 via MAVLink)

Quickstart
----------
1) `pip install -r requirements.txt`  
2) Start imagery API: `uvicorn system_b.server:app --port 8000`  
3) Run PX4 SITL, C pipeline, then D fusion bridge.
