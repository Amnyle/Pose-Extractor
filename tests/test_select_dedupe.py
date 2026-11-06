import numpy as np

from pose_extractor.dedupe import group_holds
from pose_extractor.geometry import normalize_pose
from pose_extractor.models import Hold, Sample
from pose_extractor.select import pick_best_frame
from tests.synthetic import POSE_A, POSE_B, POSE_C, to_keypoints


def _sample(i, sharpness, completeness=1.0, pose=True):
    return Sample(i, i / 10, np.zeros((13, 2)) if pose else None, completeness, sharpness, False)


def test_pick_best_prefers_sharp_central_frames():
    samples = [_sample(i, s) for i, s in enumerate([50, 60, 100, 95, 60, 50])]
    idx, score = pick_best_frame(samples, Hold(0, 5))
    assert idx in (2, 3) and score > 0


def test_pick_best_avoids_blurry_frames_even_in_the_middle():
    samples = [_sample(i, s) for i, s in enumerate([100, 100, 5, 5, 100, 100])]
    idx, _ = pick_best_frame(samples, Hold(0, 5))
    assert idx in (1, 4)


def test_pick_best_penalises_incomplete_bodies():
    samples = [_sample(0, 100, 0.6), _sample(1, 100, 1.0), _sample(2, 100, 0.6)]
    idx, _ = pick_best_frame(samples, Hold(0, 2))
    assert idx == 1


def test_pick_best_skips_samples_without_pose():
    samples = [_sample(0, 100, pose=False), _sample(1, 10)]
    assert pick_best_frame(samples, Hold(0, 1))[0] == 1
    assert pick_best_frame([_sample(0, 100, pose=False)], Hold(0, 0)) == (-1, 0.0)


def test_group_holds_merges_repeated_poses_and_keeps_the_best_frame():
    rng = np.random.default_rng(3)
    p = lambda rel: normalize_pose(to_keypoints(rel, jitter=0.5, rng=rng))
    poses = [p(POSE_A), p(POSE_B), p(POSE_C), p(POSE_B), p(POSE_A)]
    scores = [0.5, 0.4, 0.9, 0.8, 0.3]
    groups = group_holds(poses, scores, thresh=0.25)
    assert groups == [(0, [0, 4]), (3, [1, 3]), (2, [2])]


def test_group_holds_threshold_controls_merging():
    a = normalize_pose(to_keypoints(POSE_A))
    c = normalize_pose(to_keypoints(POSE_C))
    assert len(group_holds([a, c], [1, 1], thresh=0.25)) == 2
    assert len(group_holds([a, c], [1, 1], thresh=5.0)) == 1
