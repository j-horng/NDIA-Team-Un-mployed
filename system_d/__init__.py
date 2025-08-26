"""
System D — Navigation Controller

Bridges System C's GeoFixes into PX4 EKF2 via MAVLink:
- Preferred: GPS_INPUT (pseudo-GNSS fixes)
- Optional: VISION_POSITION_ESTIMATE (vision pose updates)

Also hosts an (optional) local EKF for pre-fusion before publishing.
"""
