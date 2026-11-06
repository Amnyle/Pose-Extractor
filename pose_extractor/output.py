"""Write the photos, a contact sheet and a JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .models import Config, ExtractedPose, Hold, Sample
from .video import VideoInfo, iter_frames_at


def save_results(
    video_path: Path | str,
    out_dir: Path | str,
    results: Sequence[ExtractedPose],
    jpg_quality: int = 95,
) -> None:
    """Pass 2: re-read the video and save each chosen frame at full resolution."""
    out = Path(out_dir)
    poses_dir = out / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    wanted: dict[int, list[Path]] = {}
    for k, result in enumerate(results, 1):
        name = f"{k:02d}_{result.label}_{result.time:07.2f}s.jpg"
        result.path = f"poses/{name}"
        wanted.setdefault(result.frame_index, []).append(poses_dir / name)
        result.alternate_paths = []
        for frame_index, t in result.alternates:
            alt = out / "alternates" / f"{k:02d}_{result.label}" / f"{t:07.2f}s.jpg"
            alt.parent.mkdir(parents=True, exist_ok=True)
            result.alternate_paths.append(str(alt.relative_to(out)))
            wanted.setdefault(frame_index, []).append(alt)

    for idx, frame in iter_frames_at(video_path, set(wanted)):
        for target in wanted[idx]:
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]):
                raise IOError(f"Could not write {target}")

    missing = [str(p) for paths in wanted.values() for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"Some frames could not be read back from the video: {missing}")


def make_contact_sheet(out_dir: Path | str, results: Sequence[ExtractedPose], thumb_h: int = 480, cols: int = 3):
    """One image with every extracted pose, for a quick visual check. Returns its path or None."""
    out = Path(out_dir)
    banner = 40
    tiles = []
    for k, result in enumerate(results, 1):
        img = cv2.imread(str(out / result.path)) if result.path else None
        if img is None:
            continue
        w = max(1, int(img.shape[1] * thumb_h / img.shape[0]))
        tile = cv2.resize(img, (w, thumb_h), interpolation=cv2.INTER_AREA)
        line1 = f"#{k} {result.label.replace('_', ' ')}"
        line2 = f"{result.time:.1f}s" + (f"  (seen {result.appearances}x)" if result.appearances > 1 else "")
        cv2.rectangle(tile, (0, 0), (w, banner), (0, 0, 0), -1)
        cv2.putText(tile, line1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, line2, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None

    cell_w = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = [
            cv2.copyMakeBorder(t, 0, 0, 0, cell_w - t.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for t in tiles[i : i + cols]
        ]
        while len(row) < cols:
            row.append(np.zeros((thumb_h, cell_w, 3), np.uint8))
        rows.append(np.hstack(row))

    path = out / "contact_sheet.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _hold_report(hold: Hold, samples: Sequence[Sample]) -> dict:
    """Time span, label and median arm features of a hold, so rejected holds can be understood."""
    chunk = [s for s in samples[hold.start : hold.end + 1] if s.features is not None]
    info = {
        "start": round(samples[hold.start].time, 2),
        "end": round(samples[hold.end].time, 2),
        "label": hold.label,
    }
    if chunk:
        info["arm_raise"] = round(float(np.median([s.features.raise_ for s in chunk])), 2)
        info["elbow_lat"] = round(float(np.median([s.features.elbow_lat for s in chunk])), 2)
        info["shoulder_w"] = round(float(np.median([s.features.shoulder_w for s in chunk])), 2)
        info["facing"] = "front" if np.median([s.features.facing for s in chunk]) > 0 else "back"
    return info


def write_report(
    out_dir: Path | str,
    video_path: Path | str,
    info: VideoInfo,
    cfg: Config,
    samples: Sequence[Sample],
    holds: Sequence[Hold],
    results: Sequence[ExtractedPose],
    missing: Sequence[str] = (),
) -> Path:
    report = {
        "video": str(video_path),
        "video_info": {
            "fps": info.fps,
            "frames": info.frame_count,
            "width": info.width,
            "height": info.height,
            "duration_s": round(info.duration, 2),
        },
        "config": asdict(cfg),
        "stats": {
            "frames_analysed": len(samples),
            "frames_with_usable_pose": sum(s.pose is not None for s in samples),
            "scene_cuts": sum(s.cut for s in samples),
            "holds_found": len(holds),
            "holds_rejected_as_relaxed": sum(h.label is None for h in holds),
            "photos": len(results),
        },
        "missing_poses": list(missing),
        "poses": [asdict(r) for r in results],
        "holds": [_hold_report(h, samples) for h in holds],
    }
    path = Path(out_dir) / "report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
