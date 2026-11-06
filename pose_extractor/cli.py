"""Command line interface: pose-extractor VIDEO -o OUTPUT_DIR"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from .estimator import VARIANTS, MediaPipeEstimator, PoseEstimator
from .models import Config
from .output import make_contact_sheet, save_results, write_report
from .pipeline import extract_poses


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    p = argparse.ArgumentParser(
        prog="pose-extractor",
        description="Extract one photo per distinct pose held in a bodybuilding video.",
    )
    p.add_argument("video", help="input video file")
    p.add_argument("-o", "--output", default="poses_out", help="output folder (default: %(default)s)")

    g = p.add_argument_group("sampling")
    g.add_argument("--sample-fps", type=float, default=defaults.sample_fps, help="frames analysed per second (default: %(default)s)")
    g.add_argument("--start", type=float, default=defaults.start, help="start time in seconds")
    g.add_argument("--end", type=float, default=None, help="end time in seconds")

    g = p.add_argument_group("pose detection")
    g.add_argument("--model", default=None, help="path to a pose_landmarker .task file (default: download once)")
    g.add_argument("--model-variant", choices=VARIANTS, default="full", help="MediaPipe model size (default: %(default)s)")

    g = p.add_argument_group("tuning")
    g.add_argument("--motion-thresh", type=float, default=defaults.motion_thresh, help="speed (torso lengths/s) below which the athlete counts as still (default: %(default)s)")
    g.add_argument("--min-hold", type=float, default=defaults.min_hold, help="minimum seconds a pose must be held (default: %(default)s)")
    g.add_argument("--drift-thresh", type=float, default=defaults.drift_thresh, help="max slow drift inside one hold, in torso lengths (default: %(default)s)")
    g.add_argument("--dedupe-thresh", type=float, default=defaults.dedupe_thresh, help="poses closer than this are merged, in torso lengths (default: %(default)s)")
    g.add_argument("--no-dedupe", action="store_true", help="keep every hold, even repeated poses")
    g.add_argument("--cut-threshold", type=float, default=defaults.cut_threshold, help="histogram correlation below which a scene cut is assumed (default: %(default)s)")

    p.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality of saved photos (default: %(default)s)")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    return p


def _progress_printer() -> Callable[[int, int], None]:
    last = {"pct": -1}

    def show(done: int, total: int) -> None:
        pct = min(100, int(100 * done / max(1, total)))
        if pct != last["pct"]:
            last["pct"] = pct
            print(f"\rAnalysing frames: {pct:3d}%", end="", file=sys.stderr, flush=True)

    return show


def main(
    argv: Optional[Sequence[str]] = None,
    estimator_factory: Optional[Callable[[], PoseEstimator]] = None,
) -> int:
    args = build_parser().parse_args(argv)

    video = Path(args.video)
    if not video.is_file():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 2
    if args.sample_fps <= 0:
        print("error: --sample-fps must be positive", file=sys.stderr)
        return 2

    cfg = Config(
        sample_fps=args.sample_fps,
        start=args.start,
        end=args.end,
        cut_threshold=args.cut_threshold,
        motion_thresh=args.motion_thresh,
        min_hold=args.min_hold,
        drift_thresh=args.drift_thresh,
        dedupe=not args.no_dedupe,
        dedupe_thresh=args.dedupe_thresh,
    )

    factory = estimator_factory or (lambda: MediaPipeEstimator(args.model, args.model_variant))
    try:
        estimator = factory()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        info, samples, holds, results = extract_poses(
            video, estimator, cfg, progress=None if args.quiet else _progress_printer()
        )
    finally:
        estimator.close()
    if not args.quiet:
        print(file=sys.stderr)

    with_pose = sum(s.pose is not None for s in samples)
    if not results:
        print(
            f"No pose found ({with_pose}/{len(samples)} frames had a usable body). "
            "Check that the athlete's torso is visible, try --model-variant heavy, "
            "or lower --min-hold / raise --motion-thresh.",
            file=sys.stderr,
        )
        return 1

    save_results(video, args.output, results, args.jpg_quality)
    sheet = make_contact_sheet(args.output, results)
    report = write_report(args.output, video, info, cfg, samples, holds, results)

    print(f"{len(results)} poses saved to {Path(args.output) / 'poses'}")
    print(f"({len(holds)} holds found, {with_pose}/{len(samples)} frames with a usable pose)")
    if sheet:
        print(f"Contact sheet: {sheet}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
