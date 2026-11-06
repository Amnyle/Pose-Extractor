"""End-to-end analysis: video -> samples -> holds -> best frames -> unique poses."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .dedupe import group_holds
from .estimator import PoseEstimator
from .geometry import completeness, normalize_pose
from .holds import find_holds, pose_speeds, split_on_drift
from .models import Config, ExtractedPose, Hold, Sample
from .select import pick_best_frame
from .video import CutDetector, VideoInfo, iter_sampled_frames, person_sharpness, probe_video, resize_max_side

Progress = Callable[[int, int], None]


def analyze_video(
    video_path: Path | str,
    estimator: PoseEstimator,
    cfg: Config,
    progress: Optional[Progress] = None,
) -> tuple[VideoInfo, list[Sample]]:
    """Pass 1: sample frames, detect the pose, measure sharpness, flag scene cuts."""
    info = probe_video(video_path)
    detector = CutDetector(cfg.cut_threshold)
    stop = min(info.duration, cfg.end) if cfg.end is not None else info.duration
    expected = max(1, int((stop - cfg.start) * cfg.sample_fps))

    samples: list[Sample] = []
    for idx, t, frame in iter_sampled_frames(video_path, cfg.sample_fps, cfg.start, cfg.end):
        small = resize_max_side(frame, cfg.analysis_max_side)
        cut = detector.update(small)
        kp = estimator.detect(small, int(round(t * 1000)))
        if kp is None:
            samples.append(Sample(idx, t, None, 0.0, person_sharpness(small, None), cut))
        else:
            h, w = small.shape[:2]
            samples.append(
                Sample(
                    index=idx,
                    time=t,
                    pose=normalize_pose(kp, cfg.min_visibility),
                    completeness=completeness(kp, w, h, cfg.min_visibility),
                    sharpness=person_sharpness(small, kp, cfg.min_visibility),
                    cut=cut,
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
    """Run the whole analysis. Images are not written here, see `output.save_results`."""
    cfg = cfg or Config()
    info, samples = analyze_video(video_path, estimator, cfg, progress)

    poses = [s.pose for s in samples]
    times = [s.time for s in samples]
    cuts = [s.cut for s in samples]

    speeds = pose_speeds(poses, times, cuts)
    holds = find_holds(speeds, times, cfg.motion_thresh, cfg.min_hold)
    holds = split_on_drift(holds, poses, times, cfg.drift_thresh, cfg.min_hold)

    picks = [(h, *pick_best_frame(samples, h)) for h in holds]
    picks = [(h, i, score) for h, i, score in picks if i >= 0]
    holds = [h for h, _, _ in picks]
    if not picks:
        return info, samples, holds, []

    descriptors = [samples[i].pose for _, i, _ in picks]
    scores = [score for _, _, score in picks]
    if cfg.dedupe:
        groups = group_holds(descriptors, scores, cfg.dedupe_thresh)
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
                appearances=len(members),
                times=[samples[picks[m][0].start].time for m in members],
            )
        )
    return info, samples, holds, results
