import json

import cv2
import numpy as np
import pytest

from pose_extractor.cli import main
from pose_extractor.estimator import cache_dir, ensure_model
from pose_extractor.models import Config
from pose_extractor.output import make_contact_sheet, save_results
from pose_extractor.pipeline import extract_poses
from pose_extractor.video import CutDetector, iter_frames_at, iter_sampled_frames, probe_video
from tests.synthetic import CUT_TIME, NobodyEstimator, TimelineEstimator, render_frame, write_video


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "posing.avi"
    write_video(path, fps=20)
    return path


def test_video_helpers(video):
    info = probe_video(video)
    assert (info.width, info.height, round(info.fps)) == (640, 480, 20)
    frames = list(iter_sampled_frames(video, sample_fps=10))
    assert [f[0] for f in frames[:3]] == [0, 2, 4]
    assert np.isclose(frames[1][1], 0.1)
    picked = dict(iter_frames_at(video, {0, 10, 61}))
    assert sorted(picked) == [0, 10, 61]


def test_cut_detector_fires_only_at_the_background_change():
    detector = CutDetector(0.5)
    flags = [detector.update(render_frame(t)) for t in np.arange(2.8, 3.5, 0.1)]
    times = [round(t, 1) for t, f in zip(np.arange(2.8, 3.5, 0.1), flags) if f]
    assert times == [round(CUT_TIME, 1)]


def test_end_to_end_with_dedupe(video, tmp_path):
    cfg = Config(sample_fps=10)
    info, samples, holds, results = extract_poses(video, TimelineEstimator(), cfg)

    assert sum(s.cut for s in samples) == 1
    assert len(holds) == 5            # A, B (split by the cut) x2, C, B again
    assert len(results) == 3          # A, B, C
    assert [r.appearances for r in results] == [1, 3, 1]
    assert 0.0 <= results[0].time <= 1.5
    assert 4.6 <= results[2].time <= 6.0

    save_results(video, tmp_path, results)
    sheet = make_contact_sheet(tmp_path, results)
    for r in results:
        img = cv2.imread(str(tmp_path / r.path))
        assert img is not None and img.shape == (480, 640, 3)
    assert sheet.is_file()


def test_end_to_end_without_dedupe_keeps_every_hold(video):
    _, _, holds, results = extract_poses(video, TimelineEstimator(), Config(sample_fps=10, dedupe=False))
    assert len(results) == len(holds) == 5


def test_nobody_in_the_video_gives_no_poses(video):
    _, samples, holds, results = extract_poses(video, NobodyEstimator(), Config(sample_fps=10))
    assert holds == [] and results == []
    assert all(s.pose is None for s in samples)


def test_cli_writes_photos_sheet_and_report(video, tmp_path, capsys):
    estimator = TimelineEstimator()
    out = tmp_path / "out"
    code = main([str(video), "-o", str(out), "--sample-fps", "10", "-q"], estimator_factory=lambda: estimator)
    assert code == 0 and estimator.closed
    assert len(list((out / "poses").glob("pose_*.jpg"))) == 3
    assert (out / "contact_sheet.jpg").is_file()
    report = json.loads((out / "report.json").read_text())
    assert report["stats"]["unique_poses"] == 3 and report["stats"]["scene_cuts"] == 1
    assert "3 poses saved" in capsys.readouterr().out


def test_cli_reports_errors(video, tmp_path, capsys):
    assert main([str(tmp_path / "missing.mp4")]) == 2
    code = main([str(video), "-o", str(tmp_path / "o"), "--sample-fps", "10", "-q"], estimator_factory=NobodyEstimator)
    assert code == 1
    assert "No pose found" in capsys.readouterr().err


def test_ensure_model_uses_explicit_path_and_cache(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        ensure_model(str(tmp_path / "nope.task"))

    model = tmp_path / "custom.task"
    model.write_bytes(b"x")
    assert ensure_model(str(model)) == model

    monkeypatch.setenv("POSE_EXTRACTOR_CACHE", str(tmp_path / "cache"))
    cached = cache_dir() / "pose_landmarker_lite.task"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x")
    assert ensure_model(None, "lite") == cached  # no download when cached

    with pytest.raises(ValueError):
        ensure_model(None, "giant")
