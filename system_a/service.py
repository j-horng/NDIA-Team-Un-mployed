from __future__ import annotations

"""
System A service: emit camera frames to disk and IMU deltas to JSONL.

This is optional for the demo (System C can read directly from a video or tile),
but helpful when you want an external producer.

Examples:
  # Synthetic frames from tile + synthetic IMU (circle), 20 FPS, 200 Hz, 30s
  python -m system_a.service --tile data/tiles/18/0/0.png --fps 20 \
      --imu-synth circle --imu-rate 200 --duration 30 \
      --out-frames runtime/frames --out-imu logs/imu.jsonl

  # Replay from video + IMU CSV
  python -m system_a.service --video data/video/aoi.mp4 --fps 15 \
      --imu data/imu/demo.csv --out-frames runtime/frames --out-imu logs/imu.jsonl
"""

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

from system_a.camera import frames_from_video, frames_from_tile, frames_from_webcam
from system_a.imu import imu_from_csv, imu_synthetic


def _write_frames(
    src_iter,
    out_dir: Path,
    stop_event: threading.Event,
    limit: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in src_iter:
        if stop_event.is_set():
            break
        fn = out_dir / f"frame_{frame.ts.replace(':','').replace('-','').replace('Z','Z')}_{count:06d}.png"
        cv2.imwrite(str(fn), frame.frame)
        count += 1
        if limit is not None and count >= limit:
            break


def _write_imu_jsonl(
    src_iter,
    out_file: Path,
    stop_event: threading.Event,
    limit: Optional[int] = None,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a", buffering=1) as f:
        count = 0
        for imu in src_iter:
            if stop_event.is_set():
                break
            f.write(
                f'{{"ts":"{imu.ts}","dt":{imu.dt:.6f},"gx":{imu.gyro[0]:.6f},"gy":{imu.gyro[1]:.6f},"gz":{imu.gyro[2]:.6f},'
                f'"ax":{imu.accel[0]:.6f},"ay":{imu.accel[1]:.6f},"az":{imu.accel[2]:.6f}}}\n'
            )
            count += 1
            if limit is not None and count >= limit:
                break


def parse_size(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    if "x" in s.lower():
        w, h = s.lower().split("x")
    else:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError("Size must be WxH or W,H")
        w, h = parts
    return (int(w), int(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    gsrc = ap.add_mutually_exclusive_group()
    gsrc.add_argument("--video", help="Path to video file")
    gsrc.add_argument("--tile", help="Path to satellite tile PNG")
    gsrc.add_argument("--webcam", type=int, help="Webcam index (e.g., 0)")

    ap.add_argument("--fps", type=float, default=20.0, help="Camera frame rate cap (Hz)")
    ap.add_argument("--size", type=str, default="640x480", help="Resize output frames WxH (or omit)")
    ap.add_argument("--steps", type=int, default=600, help="#frames for tile synth (frames_from_tile)")
    ap.add_argument("--stride", type=str, default="6,4", help="Tile synth per-frame dx,dy")
    ap.add_argument("--rotate", type=float, default=5.0, help="Tile synth rotation amplitude (deg)")
    ap.add_argument("--blur", type=float, default=1.2, help="Tile synth Gaussian sigma")

    gimu = ap.add_mutually_exclusive_group()
    gimu.add_argument("--imu", help="Path to IMU CSV file to replay")
    gimu.add_argument("--imu-synth", choices=["hover", "circle", "figure8"], help="Synthetic IMU pattern")

    ap.add_argument("--imu-rate", type=float, default=200.0, help="IMU sample rate (Hz) for synthetic IMU")
    ap.add_argument("--duration", type=float, default=None, help="Stop after N seconds (both streams)")

    ap.add_argument("--out-frames", default="runtime/frames", help="Directory to write frames (PNG)")
    ap.add_argument("--out-imu", default="logs/imu.jsonl", help="File to append IMU JSONL")

    args = ap.parse_args()

    # Camera source selection
    size = parse_size(args.size) if args.size else None

    if args.video:
        cam_iter = frames_from_video(args.video, loop=True, size=size, max_fps=args.fps)
    elif args.webcam is not None:
        cam_iter = frames_from_webcam(args.webcam, size=size, max_fps=args.fps)
    else:
        # Default to tile synth; pick a default tile if not provided
        tile = args.tile or "data/tiles/18/0/0.png"
        if not os.path.exists(tile):
            raise SystemExit(f"Tile not found: {tile}. Run scripts/build_tile_cache.py first.")
        dx, dy = (int(x) for x in args.stride.split(","))
        cam_iter = frames_from_tile(
            tile,
            window=size or (640, 480),
            steps=args.steps,
            stride=(dx, dy),
            add_blur_sigma=args.blur,
            rotate_deg=args.rotate,
            max_fps=args.fps,
        )

    # IMU source selection
    if args.imu:
        imu_iter = imu_from_csv(args.imu)
    elif args.imu_synth:
        imu_iter = imu_synthetic(pattern=args.imu_synth, sample_rate_hz=args.imu_rate, duration_s=args.duration)
    else:
        imu_iter = None

    stop = threading.Event()
    threads = []

    # Duration handling (optional global stop)
    if args.duration is not None:
        def _timer():
            time.sleep(float(args.duration))
            stop.set()
        t_timer = threading.Thread(target=_timer, daemon=True)
        t_timer.start()

    # Start camera writer
    t_cam = threading.Thread(
        target=_write_frames,
        args=(cam_iter, Path(args.out_frames), stop, None),
        daemon=True,
    )
    t_cam.start()
    threads.append(t_cam)

    # Start IMU writer (if any)
    if imu_iter is not None:
        t_imu = threading.Thread(
            target=_write_imu_jsonl,
            args=(imu_iter, Path(args.out_imu), stop, None),
            daemon=True,
        )
        t_imu.start()
        threads.append(t_imu)

    try:
        # Wait for camera thread; IMU thread will also stop when Event is set (or duration elapsed)
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop.set()
    finally:
        for t in threads:
            t.join(timeout=1.0)

    print("System A service finished.")


if __name__ == "__main__":
    main()
