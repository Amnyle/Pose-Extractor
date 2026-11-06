import numpy as np

from pose_extractor.geometry import completeness, normalize_pose, pose_distance
from tests.synthetic import POSE_A, POSE_B, to_keypoints


def test_normalize_is_translation_and_scale_invariant():
    base = normalize_pose(to_keypoints(POSE_A, center=(200, 200)))
    moved = normalize_pose(to_keypoints(POSE_A, center=(400, 300)))
    scaled_kp = to_keypoints(POSE_A * 1.7, center=(300, 250))
    scaled = normalize_pose(scaled_kp)
    assert np.allclose(base, moved, atol=1e-5)
    assert np.allclose(base, scaled, atol=1e-5)


def test_normalize_centres_on_hips_and_uses_torso_length():
    pose = normalize_pose(to_keypoints(POSE_A))
    # joints order: nose, shoulders (L, R), ... hips (L, R) at index 7, 8
    hip_mid = pose[[7, 8]].mean(axis=0)
    shoulder_mid = pose[[1, 2]].mean(axis=0)
    assert np.allclose(hip_mid, 0, atol=1e-6)
    assert np.isclose(np.linalg.norm(shoulder_mid - hip_mid), 1.0, atol=1e-6)


def test_low_visibility_joint_becomes_nan_and_bad_torso_is_rejected():
    kp = to_keypoints(POSE_A)
    kp[15, 2] = 0.1  # left wrist hidden
    pose = normalize_pose(kp)
    assert np.isnan(pose[5]).all() and not np.isnan(pose[[0, 1, 2]]).any()

    kp = to_keypoints(POSE_A)
    kp[23, 2] = 0.1  # left hip hidden -> no torso reference
    assert normalize_pose(kp) is None

    tiny = to_keypoints(POSE_A * 0.05)
    assert normalize_pose(tiny) is None


def test_distance_separates_different_poses_and_ignores_jitter():
    rng = np.random.default_rng(1)
    a1 = normalize_pose(to_keypoints(POSE_A, jitter=0.5, rng=rng))
    a2 = normalize_pose(to_keypoints(POSE_A, jitter=0.5, rng=rng))
    b = normalize_pose(to_keypoints(POSE_B))
    assert pose_distance(a1, a2) < 0.05
    assert pose_distance(a1, b) > 0.8


def test_distance_nan_when_too_few_common_joints():
    a = normalize_pose(to_keypoints(POSE_A))
    b = a.copy()
    b[5:] = np.nan
    assert np.isnan(pose_distance(a, b))


def test_completeness_counts_joints_outside_frame():
    kp = to_keypoints(POSE_A, center=(320, 250))
    assert completeness(kp, 640, 480) == 1.0
    kp[27:29, 1] = 600  # ankles below the frame
    assert np.isclose(completeness(kp, 640, 480), 11 / 13)
