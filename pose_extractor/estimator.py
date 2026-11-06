"""Pose estimation backends.

Any object with `detect(frame_bgr, timestamp_ms)` and `close()` works as a backend. The default one
wraps MediaPipe's PoseLandmarker (Tasks API), which tracks a single person across frames.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

VARIANTS = ("lite", "full", "heavy")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)


class PoseEstimator(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        """Return a (33, 3) array of x, y (pixels) and visibility, or None if nobody is found.

        Optionally, a backend can set a `last_mask` attribute after each call: the person
        silhouette (H x W, same size as the frame, values above 0.5 are the body), or None.
        It is used to measure the stomach vacuum in abs and thighs. Without it, that pose is
        chosen by sharpness and position in the hold only.
        """
        ...

    def close(self) -> None: ...


def cache_dir() -> Path:
    override = os.environ.get("POSE_EXTRACTOR_CACHE")
    return Path(override) if override else Path.home() / ".cache" / "pose_extractor"


def ensure_model(model_path: Optional[str] = None, variant: str = "full") -> Path:
    """Return the path to a pose_landmarker .task file, downloading it once if needed."""
    if model_path:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant {variant!r}, choose from {VARIANTS}")

    target = cache_dir() / f"pose_landmarker_{variant}.task"
    if target.is_file():
        return target

    url = MODEL_URL.format(variant=variant)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out)
        part.replace(target)
    except Exception as exc:  # network errors, proxies, disk problems...
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the pose model ({exc}).\n"
            f"Download it manually from:\n  {url}\n"
            f"and pass it with --model /path/to/pose_landmarker_{variant}.task"
        ) from exc
    return target


class MediaPipeEstimator:
    """MediaPipe PoseLandmarker in VIDEO mode (single person, tracked between frames)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "full",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        segmentation: bool = True,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = ensure_model(model_path, variant)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=segmentation,
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1
        self._segmentation = segmentation
        self.last_mask: Optional[np.ndarray] = None

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(timestamp_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing timestamps
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        self.last_mask = self._read_mask(result, frame_bgr.shape[:2])
        if not result.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array(
            [
                [lm.x * w, lm.y * h, lm.visibility if lm.visibility is not None else 0.0]
                for lm in result.pose_landmarks[0]
            ],
            dtype=np.float32,
        )

    def _read_mask(self, result, frame_shape) -> Optional[np.ndarray]:
        """The person silhouette of the first pose as a (H, W) float array, or None."""
        if not self._segmentation or not getattr(result, "segmentation_masks", None):
            return None
        try:
            mask = np.array(result.segmentation_masks[0].numpy_view(), dtype=np.float32)  # copy: the view is reused
            mask = np.squeeze(mask)
            if mask.shape != tuple(frame_shape):
                mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_LINEAR)
            return mask
        except Exception:  # a missing mask only disables the vacuum measurement
            return None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
