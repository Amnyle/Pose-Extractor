"""Find the moments where the athlete holds a pose.

Bodybuilding poses are static, so a "pose moment" is just a stretch of video where the body
barely moves. Motion is measured on poses normalised by torso length, so it does not depend on
camera distance or on the athlete walking across the stage.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .geometry import pose_distance
from .models import Hold


def pose_speeds(
    poses: Sequence[Optional[np.ndarray]],
    times: Sequence[float],
    cuts: Sequence[bool],
    top_k: int = 3,
) -> np.ndarray:
    """Speed (torso lengths / second) between each sample and the previous one.

    NaN where it cannot be measured: first sample, missing pose, scene cut.
    """
    n = len(poses)
    speeds = np.full(n, np.nan)
    for i in range(1, n):
        if cuts[i] or poses[i] is None or poses[i - 1] is None:
            continue
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        d = pose_distance(poses[i - 1], poses[i], top_k=top_k)
        if not np.isnan(d):
            speeds[i] = d / dt
    return speeds


def _median3(values: np.ndarray) -> np.ndarray:
    """Median filter (window 3) that removes one-sample spikes and keeps NaNs as NaNs."""
    out = values.copy()
    for i in range(1, len(values) - 1):
        if np.isnan(values[i]):
            continue
        window = values[i - 1 : i + 2]
        out[i] = np.median(window[~np.isnan(window)])
    return out


def find_holds(
    speeds: np.ndarray,
    times: Sequence[float],
    motion_thresh: float,
    min_duration: float,
) -> list[Hold]:
    """Runs of consecutive samples that stay below `motion_thresh` for at least `min_duration`."""
    smooth = _median3(np.asarray(speeds, dtype=float))
    low = np.nan_to_num(smooth, nan=np.inf) < motion_thresh
    holds: list[Hold] = []
    n = len(smooth)
    i = 1
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1]:
            j += 1
        # speed[i] compares sample i with sample i-1, so the hold starts one sample earlier
        start, end = i - 1, j
        if times[end] - times[start] >= min_duration:
            holds.append(Hold(start, end))
        i = j + 1
    return holds


def split_on_drift(
    holds: Sequence[Hold],
    poses: Sequence[Optional[np.ndarray]],
    times: Sequence[float],
    drift_thresh: float,
    min_duration: float,
) -> list[Hold]:
    """Cut holds where the pose has slowly wandered away from where the hold began.

    Frame-to-frame speed alone would accept a very slow transition between two poses as one
    long hold. Comparing against the first pose of the hold catches that.
    """
    out: list[Hold] = []

    def emit(start: int, end: int) -> None:
        if times[end] - times[start] >= min_duration:
            out.append(Hold(start, end))

    for hold in holds:
        seg_start = hold.start
        ref = poses[seg_start]
        for k in range(hold.start + 1, hold.end + 1):
            d = pose_distance(ref, poses[k]) if ref is not None and poses[k] is not None else np.nan
            if np.isnan(d) or d > drift_thresh:
                emit(seg_start, k - 1)
                seg_start, ref = k, poses[k]
        emit(seg_start, hold.end)
    return out
