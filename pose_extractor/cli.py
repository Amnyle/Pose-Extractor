"""Command line interface: pose-extractor VIDEO -o OUTPUT_DIR"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from .estimator import VARIANTS, MediaPipeEstimator, PoseEstimator
from .models import Config
from .output import make_contact_sheet, save_results, write_report
from .pipeline import extract_poses, missing_poses


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    p = argparse.ArgumentParser(
        prog="pose-extractor",
        description="Extract one photo of each classic bodybuilding pose held in a video.",
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

    g = p.add_argument_group("which poses")
    g.add_argument("--generic", action="store_true", help="do not look for the six classic poses: keep every held pose, whatever it is")
    g.add_argument("--arm-raise-min", type=float, default=defaults.arm_raise_min, help="how high the wrists must be above the hips, in torso lengths, to count as posing instead of standing relaxed (default: %(default)s)")
    g.add_argument("--no-other", action="store_true", help="drop intentional poses that match none of the six classic ones")
    g.add_argument("--alternates", type=int, default=defaults.alternates, help="extra frames saved per pose, spread over its hold, so you can pick the timing by eye; 0 turns them off (default: %(default)s)")

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
        classic=not args.generic,
        arm_raise_min=args.arm_raise_min,
        keep_other=not args.no_other,
        alternates=max(0, args.alternates),
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
    missing = missing_poses(results) if cfg.classic else []
    if not results:
        print(
            f"No pose found ({with_pose}/{len(samples)} frames had a usable body). "
            "Check that the athlete's torso is visible and try --model-variant heavy. "
            "If the athlete is clearly visible, the arm rules may be rejecting the poses: "
            "lower --arm-raise-min, or use --generic.",
            file=sys.stderr,
        )
        # still write a report so the rejected holds can be inspected
        Path(args.output).mkdir(parents=True, exist_ok=True)
        report = write_report(args.output, video, info, cfg, samples, holds, results, missing)
        print(f"Report: {report}", file=sys.stderr)
        return 1

    save_results(video, args.output, results, args.jpg_quality)
    sheet = make_contact_sheet(args.output, results)
    report = write_report(args.output, video, info, cfg, samples, holds, results, missing)

    print(f"{len(results)} photos saved to {Path(args.output) / 'poses'}:")
    for r in results:
        print(f"  {r.time:7.1f}s  {r.label}" + (f"  (seen {r.appearances}x)" if r.appearances > 1 else ""))
    if any(r.alternates for r in results):
        print(f"Other moments of each pose: {Path(args.output) / 'alternates'}")
    if missing:
        print(f"Classic poses not found: {', '.join(missing)}")
        print("  See the 'holds' list in report.json to find out why a hold was rejected.")
    if sheet:
        print(f"Contact sheet: {sheet}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
