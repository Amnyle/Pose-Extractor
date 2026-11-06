"""Pose geometry: normalisation and distances.

Poses use the 33-landmark BlazePose layout. Only 13 joints matter for telling poses apart:
nose, shoulders, elbows, wrists, hips, knees and ankles.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

JOINTS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
_TORSO_IDS = (11, 12, 23, 24)  # shoulders and hips
MIN_TORSO_PX = 20.0


def normalize_pose(kp: np.ndarray, min_vis: float = 0.5) -> Optional[np.ndarray]:
    """Turn raw keypoints into a position- and scale-free pose.

    kp: (33, 3) array of x, y (pixels) and visibility.
    Returns a (13, 2) array centred on the hip midpoint and divided by the torso length
    (shoulder midpoint to hip midpoint). Joints below `min_vis` become NaN.
    Returns None if the torso itself is not reliably visible.
    """
    kp = np.asarray(kp, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 29 or kp.shape[1] < 3:
        return None
    if np.any(kp[list(_TORSO_IDS), 2] < min_vis):
        return None
    shoulders = kp[[11, 12], :2].mean(axis=0)
    hips = kp[[23, 24], :2].mean(axis=0)
    torso = float(np.linalg.norm(shoulders - hips))
    if not np.isfinite(torso) or torso < MIN_TORSO_PX:
        return None
    joints = kp[list(JOINTS)]
    pose = (joints[:, :2] - hips) / torso
    pose[joints[:, 2] < min_vis] = np.nan
    return pose


def pose_distance(a: np.ndarray, b: np.ndarray, top_k: int = 4, min_common: int = 6) -> float:
    """Distance between two normalised poses, in torso lengths.

    Averages the `top_k` largest per-joint distances instead of all joints. A pose change
    often moves only the arms, and a plain mean over every joint would dilute that signal.
    Returns NaN if the poses share fewer than `min_common` visible joints.
    """
    per_joint = np.linalg.norm(a - b, axis=1)
    per_joint = per_joint[~np.isnan(per_joint)]
    if per_joint.size < min_common:
        return float("nan")
    k = min(top_k, per_joint.size)
    return float(np.sort(per_joint)[-k:].mean())


def completeness(kp: np.ndarray, width: int, height: int, min_vis: float = 0.5) -> float:
    """Share of the key joints that are confidently detected and inside the frame."""
    joints = np.asarray(kp, dtype=float)[list(JOINTS)]
    ok = (
        (joints[:, 2] >= min_vis)
        & (joints[:, 0] >= 0)
        & (joints[:, 0] < width)
        & (joints[:, 1] >= 0)
        & (joints[:, 1] < height)
    )
    return float(ok.mean())
