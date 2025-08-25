import cv2, time
from typing import Iterator
import numpy as np
from common.types import ImageFrame
def frames_from_video(path: str) -> Iterator[ImageFrame]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {path}")
    while True:
        ok, img = cap.read()
        if not ok: break
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        yield ImageFrame(ts=ts, width=img.shape[1], height=img.shape[0], frame=img)
