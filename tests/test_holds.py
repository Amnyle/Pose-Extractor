import numpy as np

from pose_extractor.geometry import normalize_pose, pose_distance
from pose_extractor.holds import find_holds, pose_speeds, split_on_drift
from tests.synthetic import POSE_A, POSE_B, pose_at, to_keypoints

FPS = 10
THRESH, MIN_HOLD = 0.5, 0.8


def _build(rel_poses, jitter=0.5):
    rng = np.random.default_rng(0)
    poses = [normalize_pose(to_keypoints(p, jitter=jitter, rng=rng)) for p in rel_poses]
    times = [i / FPS for i in range(len(poses))]
    return poses, times


def _timeline(duration=8.0):
    return _build([pose_at(i / FPS) for i in range(int(duration * FPS))])


def test_finds_the_four_holds_of_the_timeline():
    poses, times = _timeline()
    cuts = [False] * len(poses)
    holds = find_holds(pose_speeds(poses, times, cuts), times, THRESH, MIN_HOLD)
    spans = [(round(times[h.start], 1), round(times[h.end], 1)) for h in holds]
    assert spans == [(0.0, 1.5), (2.1, 4.0), (4.6, 6.0), (6.6, 7.9)]


def test_scene_cut_splits_a_hold():
    poses, times = _timeline()
    cuts = [False] * len(poses)
    cuts[31] = True  # t = 3.1, inside the second hold
    holds = find_holds(pose_speeds(poses, times, cuts), times, THRESH, MIN_HOLD)
    spans = [(round(times[h.start], 1), round(times[h.end], 1)) for h in holds]
    assert (2.1, 3.0) in spans and (3.1, 4.0) in spans
    assert len(holds) == 5


def test_short_pauses_are_ignored():
    # 0.3 s pause in pose A, a fast move to B, then a real hold of B
    rel = [POSE_A] * 4 + [(1 - a) * POSE_A + a * POSE_B for a in np.linspace(0.25, 1, 4)] + [POSE_B] * 12
    poses, times = _build(rel)
    holds = find_holds(pose_speeds(poses, times, [False] * len(poses)), times, THRESH, MIN_HOLD)
    assert len(holds) == 1
    assert times[holds[0].end] - times[holds[0].start] >= MIN_HOLD
    assert pose_distance(poses[holds[0].start], normalize_pose(to_keypoints(POSE_B))) < 0.1


def test_missing_pose_breaks_a_hold():
    poses, times = _timeline()
    for i in range(33, 37):  # nobody detected for 0.4 s in the middle of the second hold
        poses[i] = None
    holds = find_holds(pose_speeds(poses, times, [False] * len(poses)), times, THRESH, MIN_HOLD)
    spans = [(round(times[h.start], 1), round(times[h.end], 1)) for h in holds]
    assert (2.1, 3.2) in spans                       # the part before the gap is kept
    assert not any(3.3 <= s <= 4.0 for s, _ in spans)  # the part after (0.3 s) is too short


def test_slow_drift_is_split_even_though_frame_to_frame_speed_is_low():
    # A -> B spread over 6 s: never fast enough to count as motion
    rel = [(1 - a) * POSE_A + a * POSE_B for a in np.linspace(0, 1, 60)]
    poses, times = _build(rel, jitter=0.0)
    speeds = pose_speeds(poses, times, [False] * len(poses))
    holds = find_holds(speeds, times, THRESH, MIN_HOLD)
    assert len(holds) == 1  # one long "hold" if we only look at speed

    pieces = split_on_drift(holds, poses, times, drift_thresh=0.3, min_duration=MIN_HOLD)
    assert len(pieces) >= 3
    for piece in pieces:
        assert pose_distance(poses[piece.start], poses[piece.end]) <= 0.3 + 1e-9
