"""Video I/O helpers: probing, frame sampling, scene cuts, sharpness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _open(path: Path | str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def _fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and np.isfinite(fps) and fps > 0 else 30.0


def probe_video(path: Path | str) -> VideoInfo:
    cap = _open(path)
    try:
        return VideoInfo(
            fps=_fps(cap),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()


def iter_sampled_frames(
    path: Path | str,
    sample_fps: float,
    start: float = 0.0,
    end: Optional[float] = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time_seconds, frame) roughly `sample_fps` times per second.

    Frames are counted sequentially, so indices match `iter_frames_at`.
    """
    cap = _open(path)
    try:
        fps = _fps(cap)
        step = max(1, int(round(fps / sample_fps)))
        idx = -1
        while True:
            if not cap.grab():
                break
            idx += 1
            t = idx / fps
            if end is not None and t > end:
                break
            if t < start or idx % step:
                continue
            ok, frame = cap.retrieve()
            if ok:
                yield idx, t, frame
    finally:
        cap.release()


def iter_frames_at(path: Path | str, wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, full-resolution frame) for the requested indices, in order."""
    if not wanted:
        return
    remaining = set(wanted)
    last = max(remaining)
    cap = _open(path)
    try:
        idx = -1
        while remaining:
            if not cap.grab():
                break
            idx += 1
            if idx in remaining:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx, frame
                remaining.discard(idx)
            if idx >= last:
                break
    finally:
        cap.release()


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


class CutDetector:
    """Flags a scene cut when the colour histogram changes abruptly between two frames."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._prev: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cut = False
        if self._prev is not None:
            corr = cv2.compareHist(self._prev, hist, cv2.HISTCMP_CORREL)
            cut = bool(corr < self.threshold)
        self._prev = hist
        return cut


def person_sharpness(frame: np.ndarray, kp: Optional[np.ndarray], min_vis: float = 0.5) -> float:
    """Laplacian variance of the area around the athlete (whole frame if no keypoints).

    The crop is resized to a fixed height so scores compare fairly between frames.
    """
    crop = frame
    if kp is not None:
        h, w = frame.shape[:2]
        visible = np.asarray(kp)[:, 2] >= min_vis
        if visible.sum() >= 4:
            pts = np.asarray(kp)[visible, :2]
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
            x0, x1 = int(max(0, x0 - mx)), int(min(w, x1 + mx))
            y0, y1 = int(max(0, y0 - my)), int(min(h, y1 + my))
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = 256.0 / gray.shape[0]
    gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
