from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple

import cv2
import numpy as np

from common.types import ImageFrame
from common.utils import RateTimer, iso_now_ms


@dataclass
class VideoFrameSource:
    """
    Replay frames from a video file.

    Args:
        path: path to video file
        target_fps: if set, throttles output to this FPS (sleeping between frames)
        loop: restart when reaching EOF (handy for endless demo)
        resize: (width, height) to resize frames, or None to keep native
        grayscale: if True, convert to gray before emitting
        blur_sigma: Gaussian blur sigma (0 disables)
    """
    path: str
    target_fps: Optional[float] = None
    loop: bool = False
    resize: Optional[Tuple[int, int]] = None
    grayscale: bool = False
    blur_sigma: float = 0.0

    def frames(self) -> Iterator[ImageFrame]:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")

        dt_target = None if not self.target_fps or self.target_fps <= 0 else (1.0 / self.target_fps)
        while True:
            t_start = time.perf_counter()
            ok, img = cap.read()
            if not ok:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if self.resize:
                w, h = self.resize
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if self.grayscale and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.blur_sigma and self.blur_sigma > 0:
                img = cv2.GaussianBlur(img, (0, 0), self.blur_sigma)

            H, W = img.shape[:2]
            ts = iso_now_ms()
            yield ImageFrame(ts=ts, width=W, height=H, frame=img)

            if dt_target:
                elapsed = time.perf_counter() - t_start
                sleep_s = max(0.0, dt_target - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        cap.release()


@dataclass
class TilePanZoomSource:
    """
    Synthesize frames by panning/zooming/rotating over a PNG tile.

    Args:
        path: PNG (e.g., data/tiles/18/0/0.png)
        size: emitted frame size (width, height)
        fps: output rate
        steps: number of frames to generate (ignored if loop=True)
        loop: loop indefinitely
        pan_px: (dx, dy) pixels per frame
        zoom_range: (min, max) zoom factor range (random walk within)
        rot_deg_per_s: rotation speed in degrees/sec
        blur_sigma: Gaussian blur sigma to mimic motion blur
        noise_std: additive Gaussian noise std (0 disables)
    """
    path: str
    size: Tuple[int, int] = (640, 480)
    fps: float = 20.0
    steps: int = 300
    loop: bool = False
    pan_px: Tuple[int, int] = (6, 4)
    zoom_range: Tuple[float, float] = (0.9, 1.05)
    rot_deg_per_s: float = 6.0
    blur_sigma: float = 1.0
    noise_std: float = 2.0

    def frames(self) -> Iterator[ImageFrame]:
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Tile image not found: {self.path}")
        base = img.copy()
        W, H = base.shape[1], base.shape[0]
        w, h = self.size
        dt = 1.0 / max(1.0, self.fps)
        rt = RateTimer()

        kx, ky = 0, 0
        zx = zy = 1.0
        zmin, zmax = self.zoom_range
        z_dir = 1

        t0 = time.perf_counter()
        n = 0
        while self.loop or n < self.steps:
            n += 1
            t_frame = time.perf_counter() - t0
            # Pan ROI
            kx = (kx + self.pan_px[0]) % max(1, W - w)
            ky = (ky + self.pan_px[1]) % max(1, H - h)
            crop = base[ky:ky + h, kx:kx + w].copy()

            # Zoom random walk within [zmin,zmax]
            zx += 0.01 * z_dir
            zy += 0.01 * z_dir
            if zx > zmax or zx < zmin:
                z_dir *= -1
                zx = np.clip(zx, zmin, zmax)
                zy = np.clip(zy, zmin, zmax)
            if zx != 1.0 or zy != 1.0:
                crop = cv2.resize(crop, (int(w * zx), int(h * zy)), interpolation=cv2.INTER_LINEAR)
                # center-crop back to (w,h)
                ch, cw = crop.shape[:2]
                y0 = max(0, (ch - h) // 2)
                x0 = max(0, (cw - w) // 2)
                crop = crop[y0:y0 + h, x0:x0 + w]

            # Rotate
            angle = self.rot_deg_per_s * t_frame
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            crop = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Blur + noise
            if self.blur_sigma and self.blur_sigma > 0:
                crop = cv2.GaussianBlur(crop, (0, 0), self.blur_sigma)
            if self.noise_std and self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, size=crop.shape).astype(np.float32)
                crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            ts = iso_now_ms()
            out = ImageFrame(ts=ts, width=w, height=h, frame=crop)
            yield out

            # Throttle to fps
            hz = rt.tick()
            sleep_s = dt
            time.sleep(sleep_s)


def annotate_frame(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay readable text on a frame (for debug/preview)."""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (5, 5), (460, 40), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out
