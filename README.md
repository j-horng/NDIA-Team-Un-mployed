# NDIA-Team-Un-mployed
A-PNT for UAS via Satellite Imagery &amp; Sensor Correlation

System-of-Systems (SoS) Overview

System A: UAS Platform
Unmanned Aerial System equipped with GPS, onboard camera/sensor, and compute module. Captures real-time ground imagery and performs onboard processing for geolocation.

System B: Satellite Imagery Server
Provides high-resolution georeferenced satellite imagery based on GPS coordinates. Accessible via HTTP API for imagery download.

System C: Feature Matching & Georectification Module
Software module (onboard or remote) that performs image correlation using feature detectors (e.g., SIFT, ORB). Estimates platform position using matched features and camera intrinsics.

System D: Navigation Controller
Consumes estimated position data to update UAS navigation and timing. Ensures continuity of navigation even in GPS-denied environments.
