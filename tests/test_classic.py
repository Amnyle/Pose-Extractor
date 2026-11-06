import json
from pathlib import Path

import numpy as np

from pose_extractor.classic import (
    ABS_AND_THIGHS,
    BACK_DOUBLE_BICEPS,
    BACK_LAT_SPREAD,
    FRONT_DOUBLE_BICEPS,
    FRONT_LAT_SPREAD,
    OTHER,
    SIDE_CHEST,
    ArmFeatures,
    arm_features,
    classify,
    strength,
)

# Real MediaPipe keypoints of the 14 frames the first version of the tool extracted from a home
# posing video: 6 genuine classic poses and 8 relaxed / transitional frames.
FRAMES = {k: np.array(v) for k, v in json.loads((Path(__file__).parent / "data" / "user_frames.json").read_text()).items()}

POSING = {
    "pose_01": ABS_AND_THIGHS,
    "pose_04": FRONT_DOUBLE_BICEPS,
    "pose_08": SIDE_CHEST,
    "pose_12": BACK_DOUBLE_BICEPS,
    "pose_13": BACK_LAT_SPREAD,
    "pose_14": FRONT_LAT_SPREAD,
}


def test_real_frames_get_the_right_name_and_relaxed_ones_are_rejected():
    assert len(FRAMES) == 14
    for name, kp in FRAMES.items():
        assert classify(arm_features(kp)) == POSING.get(name), name


def test_real_frames_are_not_borderline_for_the_arm_raise_threshold():
    raises = {name: arm_features(kp).raise_ for name, kp in FRAMES.items()}
    posing = [raises[n] for n in POSING]
    relaxed = [r for n, r in raises.items() if n not in POSING]
    assert min(posing) > 0.2 and max(relaxed) < 0.1


def test_facing_comes_from_which_shoulder_is_on_the_right_of_the_image():
    assert arm_features(FRAMES["pose_04"]).facing == 1     # facing the camera
    assert arm_features(FRAMES["pose_12"]).facing == -1    # back to the camera


def test_double_biceps_without_wide_elbows_from_the_back_is_not_named():
    f = ArmFeatures(raise_=1.4, elbow_lat=0.5, shoulder_w=0.65, facing=-1)
    assert classify(f) == OTHER


def test_hanging_arms_are_never_a_pose():
    assert classify(ArmFeatures(0.05, 0.9, 0.7, 1)) is None
    assert classify(None) is None


def test_arm_raise_threshold_is_adjustable():
    f = ArmFeatures(raise_=0.10, elbow_lat=0.8, shoulder_w=0.7, facing=1)
    assert classify(f) is None
    assert classify(f, arm_raise_min=0.05) == FRONT_LAT_SPREAD


def test_features_survive_one_hidden_wrist_but_not_a_hidden_torso():
    kp = FRAMES["pose_04"].copy()
    kp[15, 2] = 0.0  # left wrist invisible: the right one is used
    assert classify(arm_features(kp)) == FRONT_DOUBLE_BICEPS
    kp[16, 2] = 0.0  # both wrists invisible: cannot judge
    assert arm_features(kp) is None

    kp = FRAMES["pose_04"].copy()
    kp[23, 2] = 0.1  # hip invisible: no torso reference
    assert arm_features(kp) is None


def test_strength_grows_with_flared_elbows_and_raised_wrists():
    base = ArmFeatures(0.3, 0.6, 0.7, 1)
    assert strength(ArmFeatures(0.3, 0.9, 0.7, 1)) > strength(base)
    assert strength(ArmFeatures(1.4, 0.6, 0.7, 1)) > strength(base)
    assert strength(ArmFeatures(-1.0, 0.6, 0.7, 1)) == strength(ArmFeatures(0.0, 0.6, 0.7, 1))
