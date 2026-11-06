"""Recognise the six classic bodybuilding poses from body keypoints.

Everything here works on ratios of the athlete's own torso length, so it does not depend on camera
distance. The thresholds were calibrated on real MediaPipe keypoints of a home posing video
(1080x1920, athlete roughly 3 m from the camera); see tests/data/user_frames.json.

Features, all measured per frame (y grows downward, hips and shoulders are the torso reference):
  raise      height of the lowest visible wrist above the hip midpoint
             (hanging arms: about 0, hands on hips: 0.3-0.4, hands at the head: 1.4)
  elbow_lat  how far the elbows stand out sideways from the shoulder midline
             (hanging arms: 0.3-0.6, flared lats and double biceps: 0.7-0.9)
  shoulder_w shoulder width as seen by the camera (about 0.55-0.7 facing it, below 0.55 turned sideways)
  facing     +1 if the athlete's left shoulder is on the right of the image (facing the camera), -1 if
             it is on the left (back to the camera)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import MIN_TORSO_PX

FRONT_DOUBLE_BICEPS = "front_double_biceps"
BACK_DOUBLE_BICEPS = "back_double_biceps"
FRONT_LAT_SPREAD = "front_lat_spread"
BACK_LAT_SPREAD = "back_lat_spread"
ABS_AND_THIGHS = "abs_and_thighs"
SIDE_CHEST = "side_chest"
OTHER = "other"  # an intentional pose (arms raised) that matches none of the six

CLASSIC_POSES = (
    FRONT_DOUBLE_BICEPS,
    BACK_DOUBLE_BICEPS,
    FRONT_LAT_SPREAD,
    BACK_LAT_SPREAD,
    ABS_AND_THIGHS,
    SIDE_CHEST,
)

DEFAULT_ARM_RAISE_MIN = 0.15      # below this the arms are hanging: relaxed or in transition
HIGH_RAISE = 1.0                  # wrists above the shoulders: double biceps or abs and thighs
DOUBLE_BICEPS_MIN_ELBOW_LAT = 0.70
SIDE_MAX_SHOULDER_W = 0.55
SIDE_MAX_ELBOW_LAT = 0.55
LAT_MIN_ELBOW_LAT = 0.60

TORSO_MIN_VIS = 0.5               # shoulders and hips must be clearly visible
ARM_MIN_VIS = 0.3                 # wrists and elbows may be partly hidden (hands on hips, behind back)


@dataclass(frozen=True)
class ArmFeatures:
    raise_: float
    elbow_lat: float
    shoulder_w: float
    facing: int


def arm_features(kp: np.ndarray, arm_min_vis: float = ARM_MIN_VIS) -> Optional[ArmFeatures]:
    """Compute the features above from a (33, 3) keypoint array (x, y in pixels, visibility)."""
    kp = np.asarray(kp, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 29 or kp.shape[1] < 3:
        return None
    if np.any(kp[[11, 12, 23, 24], 2] < TORSO_MIN_VIS):
        return None
    shoulders = kp[[11, 12], :2].mean(axis=0)
    hips = kp[[23, 24], :2].mean(axis=0)
    torso = float(np.linalg.norm(shoulders - hips))
    if not np.isfinite(torso) or torso < MIN_TORSO_PX:
        return None

    wrists = [i for i in (15, 16) if kp[i, 2] >= arm_min_vis]
    elbows = [i for i in (13, 14) if kp[i, 2] >= arm_min_vis]
    if not wrists or not elbows:
        return None

    raise_ = min((hips[1] - kp[i, 1]) / torso for i in wrists)
    elbow_lat = float(np.mean([abs(kp[i, 0] - shoulders[0]) / torso for i in elbows]))
    dx = (kp[11, 0] - kp[12, 0]) / torso
    return ArmFeatures(float(raise_), elbow_lat, abs(float(dx)), 1 if dx >= 0 else -1)


def classify(features: Optional[ArmFeatures], arm_raise_min: float = DEFAULT_ARM_RAISE_MIN) -> Optional[str]:
    """Name the pose, OTHER for an intentional but unrecognised one, or None if the arms hang."""
    f = features
    if f is None or f.raise_ < arm_raise_min:
        return None
    side = "front" if f.facing > 0 else "back"

    if f.raise_ >= HIGH_RAISE:
        if f.elbow_lat >= DOUBLE_BICEPS_MIN_ELBOW_LAT:
            return f"{side}_double_biceps"
        return ABS_AND_THIGHS if side == "front" else OTHER

    if f.shoulder_w < SIDE_MAX_SHOULDER_W and f.elbow_lat < SIDE_MAX_ELBOW_LAT:
        return SIDE_CHEST
    if f.elbow_lat >= LAT_MIN_ELBOW_LAT:
        return f"{side}_lat_spread"
    return OTHER


def strength(features: ArmFeatures) -> float:
    """How fully the pose is expressed: elbows pushed out and wrists raised. Used to pick the peak frame."""
    return features.elbow_lat + 0.25 * min(max(features.raise_, 0.0), 1.5)
