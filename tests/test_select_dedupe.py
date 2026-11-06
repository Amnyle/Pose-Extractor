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


# ---- moment-of-peak selection ----------------------------------------------------------------

from pose_extractor.classic import ABS_AND_THIGHS, BACK_LAT_SPREAD, FRONT_LAT_SPREAD, ArmFeatures


def _s(i, sharp=100.0, waist=None, head=None, legs_dy=0.0, strength=1.0):
    return Sample(i, i / 10, np.zeros((13, 2)), 1.0, sharp, False,
                  ArmFeatures(1.4, 0.6, 0.6, 1, legs_dy), ABS_AND_THIGHS, strength, waist, head)


def test_abs_and_thighs_picks_the_narrowest_waist_even_when_a_wider_frame_is_sharper():
    waists = [0.50, 0.50, 0.49, 0.44, 0.43, 0.44, 0.49, 0.50, 0.50]
    sharp = [100, 100, 100, 70, 70, 70, 100, 100, 100]
    samples = [_s(i, sharp[i], waists[i]) for i in range(9)]
    idx, _ = pick_best_frame(samples, Hold(0, 8, ABS_AND_THIGHS))
    assert idx in (3, 4, 5)


def test_abs_and_thighs_prefers_level_feet_over_a_slightly_better_vacuum():
    waists = [0.50, 0.42, 0.42, 0.50, 0.50, 0.44, 0.44, 0.50]
    legs = [0.0, 0.3, 0.3, 0.0, 0.0, 0.02, 0.02, 0.0]   # the deepest vacuum has a foot in front
    samples = [_s(i, 100, waists[i], legs_dy=legs[i]) for i in range(8)]
    idx, _ = pick_best_frame(samples, Hold(0, 7, ABS_AND_THIGHS))
    assert idx in (5, 6)


def test_abs_and_thighs_falls_back_to_sharpness_when_the_waist_does_not_change():
    waists = [0.500, 0.505, 0.498, 0.502, 0.499, 0.501]   # all inside the noise floor
    sharp = [60, 60, 60, 100, 60, 60]
    samples = [_s(i, sharp[i], waists[i]) for i in range(6)]
    assert pick_best_frame(samples, Hold(0, 5, ABS_AND_THIGHS))[0] == 3
    no_mask = [_s(i, sharp[i], None) for i in range(6)]
    assert pick_best_frame(no_mask, Hold(0, 5, ABS_AND_THIGHS))[0] == 3


def test_front_lat_spread_prefers_the_straightest_head_in_the_hold():
    drops = [0.55, 0.52, 0.50, 0.30, 0.28, 0.45, 0.50, 0.55]   # looks down, then straight, then down
    samples = []
    for i, d in enumerate(drops):
        s = _s(i, 100.0, head=d)
        s.label = FRONT_LAT_SPREAD
        samples.append(s)
    idx, _ = pick_best_frame(samples, Hold(0, 7, FRONT_LAT_SPREAD))
    assert idx in (3, 4)


def test_other_poses_do_not_use_waist_or_head():
    samples = [_s(i, 60.0 if i != 2 else 100.0, waist=0.3 + 0.1 * (i == 2), head=0.9) for i in range(5)]
    for s in samples:
        s.label = "front_double_biceps"
    assert pick_best_frame(samples, Hold(0, 4, "front_double_biceps"))[0] == 2   # sharpest frame wins


# ---- timing fixes: lat spread, abs and thighs across takes ------------------------------------

def _lat(i, elbow, sharp=100.0, head=None, label=FRONT_LAT_SPREAD, speed=None, raise_=0.3):
    return Sample(i, i / 6, np.zeros((13, 2)), 1.0, sharp, False,
                  ArmFeatures(raise_, elbow, 0.7, 1), label, elbow + 0.25 * raise_, None, head, speed=speed)


def test_front_lat_spread_is_taken_at_the_widest_spread_not_the_middle_of_the_hold():
    # the lats open progressively and are fully spread late in the hold, then released
    elbows = [0.62, 0.64, 0.66, 0.68, 0.70, 0.73, 0.78, 0.81, 0.82, 0.80, 0.70]
    sharp = [100, 100, 100, 100, 100, 95, 92, 92, 92, 95, 100]
    samples = [_lat(i, e, sharp[i]) for i, e in enumerate(elbows)]
    idx, _ = pick_best_frame(samples, Hold(0, 10, FRONT_LAT_SPREAD))
    assert idx in (7, 8, 9)


def test_front_lat_spread_head_tilt_does_not_override_a_clearly_wider_spread():
    elbows = [0.62, 0.63, 0.64, 0.65, 0.78, 0.80, 0.80, 0.70]
    heads = [0.20, 0.20, 0.20, 0.20, 0.45, 0.45, 0.45, 0.30]   # looks down at the lats at the peak
    samples = [_lat(i, e, head=heads[i]) for i, e in enumerate(elbows)]
    idx, _ = pick_best_frame(samples, Hold(0, 7, FRONT_LAT_SPREAD))
    assert idx in (4, 5, 6)


def test_front_lat_spread_ignores_the_hands_sliding_on_the_waist():
    elbows = [0.70, 0.70, 0.78, 0.78, 0.70, 0.70, 0.70]
    raises = [0.30, 0.30, 0.30, 0.30, 0.90, 0.90, 0.30]   # hands slide up, elbows do not open
    samples = [_lat(i, e, raise_=raises[i]) for i, e in enumerate(elbows)]
    assert pick_best_frame(samples, Hold(0, 6, FRONT_LAT_SPREAD))[0] in (2, 3)


def test_a_moving_frame_is_not_chosen_when_the_pose_is_otherwise_equal():
    speeds = [0.05, 0.05, 0.05, 0.45, 0.05, 0.05, 0.05]
    samples = [_lat(i, 0.75, speed=v) for i, v in enumerate(speeds)]
    assert pick_best_frame(samples, Hold(0, 6, FRONT_LAT_SPREAD))[0] != 3


def test_abs_waist_is_not_fooled_by_the_torso_looking_longer():
    # frames 2-3: the athlete stands taller (torso 230 px), the waist does not change (100 px);
    # frames 5-6: a real vacuum, the waist drops to 90 px with the usual 200 px torso.
    torso = [200, 200, 230, 230, 200, 200, 200, 200, 200]
    waist_px = [100, 100, 100, 100, 100, 90, 90, 100, 100]
    samples = []
    for i in range(9):
        s = _s(i, 100.0, waist_px[i] / torso[i])
        s.torso_px = float(torso[i])
        samples.append(s)
    assert pick_best_frame(samples, Hold(0, 8, ABS_AND_THIGHS))[0] in (5, 6)


def test_abs_take_with_the_real_vacuum_beats_a_sharper_take_without_one():
    # take 1 (frames 0-7): no vacuum, only noise; take 2 (frames 10-17): a real vacuum
    waists = [0.50, 0.52, 0.50, 0.53, 0.51, 0.50, 0.52, 0.50, None, None,
              0.50, 0.50, 0.46, 0.41, 0.40, 0.46, 0.50, 0.50]
    sharp = [100] * 8 + [0, 0] + [85] * 8
    samples = [_s(i, sharp[i], waists[i]) for i in range(18)]
    take1, take2 = Hold(0, 7, ABS_AND_THIGHS), Hold(10, 17, ABS_AND_THIGHS)
    pool = list(range(0, 8)) + list(range(10, 18))
    i1, score1 = pick_best_frame(samples, take1, ref_sharp=100, pool=pool)
    i2, score2 = pick_best_frame(samples, take2, ref_sharp=100, pool=pool)
    assert i2 in (13, 14) and score2 > score1


def test_back_lat_spread_keeps_the_original_scoring():
    # strength (elbows + wrists) / 1.3, sharpness and position, nothing else
    elbows = [0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.70]
    raises = [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
    sharp = [100, 100, 100, 100, 70, 70, 100]
    speeds = [0.05, 0.05, 0.05, 0.4, 0.05, 0.05, 0.05]
    samples = [_lat(i, e, sharp[i], raise_=raises[i], label=BACK_LAT_SPREAD, speed=speeds[i]) for i, e in enumerate(elbows)]
    hold = Hold(0, 6, BACK_LAT_SPREAD)
    idx, score = pick_best_frame(samples, hold, ref_sharp=100, ref_strength=1.3, pool=range(7))

    def old(i):
        return 0.5 * min(samples[i].strength / 1.3, 1) + 0.35 * min(sharp[i] / 100, 1) + 0.15 * max(0, 1 - abs(i - 3) / 3)

    expected = max(range(7), key=old)
    assert idx == expected and abs(score - old(expected)) < 1e-9
