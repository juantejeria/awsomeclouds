from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    idx: int
    rgb: np.ndarray  # HxWx3 uint8


def iter_video_frames(path: str, stride: int = 10, max_frames: int | None = 150) -> Iterator[VideoFrame]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")

    try:
        idx = 0
        emitted = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield VideoFrame(idx=idx, rgb=frame_rgb)
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break

            idx += 1
    finally:
        cap.release()


