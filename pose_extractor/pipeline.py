"""End-to-end analysis: video -> samples -> labelled holds -> best frames -> one photo per pose."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .classic import (
    ABS_AND_THIGHS,
    BACK_LAT_SPREAD,
    CLASSIC_POSES,
    FRONT_LAT_SPREAD,
    OTHER,
    arm_features,
    classify,
    head_drop,
    strength,
    waist_ratio,
)
from .dedupe import group_holds
from .estimator import PoseEstimator
from .geometry import completeness, normalize_pose
from .holds import find_holds, pose_speeds, split_by_label, split_on_drift
from .models import Config, ExtractedPose, Hold, Sample
from .select import pick_best_frame
from .video import CutDetector, VideoInfo, iter_sampled_frames, person_sharpness, prefetch, probe_video, resize_max_side

Progress = Callable[[int, int], None]

# A fully expressed pose has elbows ~0.9 torso lengths out and wrists raised: strength about 1.3.
# Only the back lat spread still uses this fixed reference (its original scoring).
FULL_STRENGTH = 1.3


def analyze_video(
    video_path: Path | str,
    estimator: PoseEstimator,
    cfg: Config,
    progress: Optional[Progress] = None,
) -> tuple[VideoInfo, list[Sample]]:
    """Pass 1: sample frames, detect the pose, name it, measure sharpness, flag scene cuts."""
    info = probe_video(video_path)
    detector = CutDetector(cfg.cut_threshold)
    stop = min(info.duration, cfg.end) if cfg.end is not None else info.duration
    expected = max(1, int((stop - cfg.start) * cfg.sample_fps))

    def frames():
        # decoding, downscaling and cut detection run in a background thread (see `prefetch`)
        for idx, t, frame in iter_sampled_frames(video_path, cfg.sample_fps, cfg.start, cfg.end):
            small = resize_max_side(frame, cfg.analysis_max_side)
            yield idx, t, small, detector.update(small)

    samples: list[Sample] = []
    for idx, t, small, cut in prefetch(frames()):
        kp = estimator.detect(small, int(round(t * 1000)))
        if kp is None:
            samples.append(Sample(idx, t, None, 0.0, person_sharpness(small, None), cut))
        else:
            h, w = small.shape[:2]
            pose = normalize_pose(kp, cfg.min_visibility)
            feats = arm_features(kp) if pose is not None else None
            if cfg.classic:
                label = classify(feats, cfg.arm_raise_min)
            else:
                label = OTHER if pose is not None else None
            mask = getattr(estimator, "last_mask", None)
            samples.append(
                Sample(
                    index=idx,
                    time=t,
                    pose=pose,
                    completeness=completeness(kp, w, h, cfg.min_visibility),
                    sharpness=person_sharpness(small, kp, cfg.min_visibility),
                    cut=cut,
                    features=feats,
                    label=label,
                    strength=strength(feats) if feats is not None else 0.0,
                    # moment-of-peak cues, measured only where the pose needs them
                    waist=waist_ratio(mask, kp) if label == ABS_AND_THIGHS and mask is not None else None,
                    head_drop=head_drop(kp) if label == FRONT_LAT_SPREAD else None,
                    torso_px=_torso_px(kp),
                )
            )
        if progress:
            progress(len(samples), expected)
    return info, samples


def extract_poses(
    video_path: Path | str,
    estimator: PoseEstimator,
    cfg: Optional[Config] = None,
    progress: Optional[Progress] = None,
) -> tuple[VideoInfo, list[Sample], list[Hold], list[ExtractedPose]]:
    """Run the whole analysis. Images are not written here, see `output.save_results`.

    Returns the video info, every sample, every labelled hold (including rejected ones with
    label None, for the report), and the final poses in chronological order.
    """
    cfg = cfg or Config()
    info, samples = analyze_video(video_path, estimator, cfg, progress)

    poses = [s.pose for s in samples]
    times = [s.time for s in samples]
    cuts = [s.cut for s in samples]
    labels = [s.label for s in samples]

    speeds = pose_speeds(poses, times, cuts)
    for i, s in enumerate(samples):
        s.speed = _frame_speed(speeds, i)
    holds = find_holds(speeds, times, cfg.motion_thresh, cfg.min_hold)
    holds = split_on_drift(holds, poses, times, cfg.drift_thresh, cfg.min_hold)
    holds = split_by_label(holds, labels, times, cfg.min_hold)

    candidates = [h for h in holds if h.label is not None and (cfg.keep_other or h.label != OTHER)]

    # Fixed reference values keep the scores of different holds comparable.
    sharp_values = [s.sharpness for s in samples if s.pose is not None]
    ref_sharp = float(np.percentile(sharp_values, 90)) if sharp_values else None

    # Every frame of every take of a pose: the peak cues are ranked on this common scale.
    pools: dict[str, list[int]] = {}
    for hold in candidates:
        if hold.label not in (OTHER, BACK_LAT_SPREAD):
            pools.setdefault(hold.label, []).extend(range(hold.start, hold.end + 1))

    picks = []
    for hold in candidates:
        idx, score = pick_best_frame(
            samples, hold, ref_sharp=ref_sharp, ref_strength=FULL_STRENGTH,
            pool=pools.get(hold.label), motion_thresh=cfg.motion_thresh,
        )
        if idx >= 0:
            picks.append((hold, idx, score))
    if not picks:
        return info, samples, holds, []

    descriptors = [samples[i].pose for _, i, _ in picks]
    scores = [score for _, _, score in picks]

    groups: list[tuple[int, list[int]]] = []
    if cfg.dedupe:
        named: dict[str, list[int]] = {}
        others: list[int] = []
        for k, (hold, _, _) in enumerate(picks):
            if hold.label == OTHER:
                others.append(k)
            else:
                named.setdefault(hold.label, []).append(k)
        for members in named.values():  # one photo per named pose: its best take
            groups.append((max(members, key=lambda k: scores[k]), members))
        if others:  # unnamed poses: merge the ones that look alike
            for rep, members in group_holds([descriptors[k] for k in others], [scores[k] for k in others], cfg.dedupe_thresh):
                groups.append((others[rep], [others[m] for m in members]))
    else:
        groups = [(k, [k]) for k in range(len(picks))]

    results: list[ExtractedPose] = []
    for rep, members in sorted(groups, key=lambda g: g[1][0]):
        hold, sample_idx, score = picks[rep]
        best = samples[sample_idx]
        results.append(
            ExtractedPose(
                frame_index=best.index,
                time=best.time,
                hold_start=samples[hold.start].time,
                hold_end=samples[hold.end].time,
                score=float(score),
                label=hold.label,
                appearances=len(members),
                times=[samples[picks[m][0].start].time for m in sorted(members)],
                alternates=[(samples[i].index, samples[i].time) for i in _alternates(samples, hold, sample_idx, cfg.alternates)],
            )
        )
    return info, samples, holds, results


def _torso_px(kp: np.ndarray) -> Optional[float]:
    kp = np.asarray(kp, dtype=float)
    torso = float(np.linalg.norm(kp[[11, 12], :2].mean(axis=0) - kp[[23, 24], :2].mean(axis=0)))
    return torso if np.isfinite(torso) and torso > 0 else None


def _frame_speed(speeds: np.ndarray, i: int) -> Optional[float]:
    """Movement around sample i: the larger of the speeds just before and just after it."""
    around = [v for v in speeds[i : i + 2] if not np.isnan(v)]
    return float(max(around)) if around else None


def _alternates(samples: list[Sample], hold: Hold, best: int, count: int) -> list[int]:
    """Sample indices spread evenly over the hold, other than the chosen one and with a pose."""
    if count <= 0:
        return []
    spread = np.linspace(hold.start, hold.end, count + 2)[1:-1]
    chosen: list[int] = []
    for position in spread:
        i = int(round(position))
        if i != best and i not in chosen and samples[i].pose is not None:
            chosen.append(i)
    return chosen


def missing_poses(results: list[ExtractedPose]) -> list[str]:
    """Classic poses that were not found."""
    found = {r.label for r in results}
    return [name for name in CLASSIC_POSES if name not in found]
