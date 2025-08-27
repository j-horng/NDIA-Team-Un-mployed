"""
System A — UAS Platform (Sim/Replay)

Provides:
- Camera frame sources:
    - VideoFrameSource: replay from a video file
    - TilePanZoomSource: synthetic motion over a satellite tile PNG
- IMU sources:
    - IMUCSVSource: replay from CSV (ts, dt, gx, gy, gz, ax, ay, az)
    - IMUSyntheticSource: procedural IMU generator for quick tests
- A small CLI service in service.py to preview or dump data.

Usage examples:
    from system_a.camera import VideoFrameSource, TilePanZoomSource
    from system_a.imu import IMUCSVSource, IMUSyntheticSource
"""
