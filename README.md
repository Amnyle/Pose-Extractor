# pose-extractor

Give it a bodybuilding video, get one photo per distinct pose the athlete holds.

```
pose-extractor posing_routine.mp4 -o results
```

```
results/
  poses/pose_01_0001.80s.jpg     one full-resolution photo per distinct pose
  poses/pose_02_0004.20s.jpg
  contact_sheet.jpg              all photos in one image, for a quick check
  report.json                    timestamps, repeats, stats and the settings used
```

## Install

```
pip install -r requirements.txt        # or: pip install -e .
```

The first run downloads MediaPipe's pose model to `~/.cache/pose_extractor`.
If you are offline, download a `pose_landmarker_*.task` file yourself and pass `--model PATH`.

## How it works

1. **Sample**: read the video at `--sample-fps` (default 6 frames/s), not every frame.
2. **Pose**: MediaPipe PoseLandmarker gives 33 body keypoints per frame. Poses are normalised
   (centred on the hips, divided by torso length) so camera distance and position do not matter.
3. **Holds**: poses are static, so a pose moment is a stretch where the keypoints barely move
   (`--motion-thresh`) for at least `--min-hold` seconds. Scene cuts always end a hold, and slow
   drift away from the hold's first pose also ends it (`--drift-thresh`).
4. **Best frame**: inside each hold, pick the frame with the best mix of sharpness (on the athlete's
   crop), a fully visible body, and distance from the edges of the hold (those touch movement).
5. **Dedupe**: holds whose poses are closer than `--dedupe-thresh` are merged into one photo, so a
   pose shown twice, or split by a camera cut, appears once. `report.json` lists every repeat.
   Use `--no-dedupe` to keep all of them.
6. **Save**: the video is read a second time to save the chosen frames at full resolution.

## Tuning

Distances are in torso lengths. If the result is off, open `contact_sheet.jpg` and `report.json`:

| Symptom | Try |
|---|---|
| Missing poses, or transitions counted as poses | lower or raise `--motion-thresh` (0.3 to 0.8) |
| Short poses missing | lower `--min-hold` (e.g. 0.5) |
| Two different poses merged into one photo | lower `--dedupe-thresh` (e.g. 0.15) |
| Same pose appears several times | raise `--dedupe-thresh` (e.g. 0.4) |
| Poor detection, athlete small in frame | `--model-variant heavy` |
| Long video, slow | lower `--sample-fps`, or use `--start` / `--end` |

## Limitations

- **One person.** MediaPipe's pose model tracks a single person. On a stage with several
  competitors or judges in view, it can jump between people. Crop the video to the athlete first,
  or swap in a multi-person backend (see below).
- **Cut detection uses colour histograms.** Two shots with very similar backgrounds may not be
  detected as a cut. Pose motion still separates most of those cases.
- **Poses are not named.** Two poses are only "same" or "different". Naming them (front double
  biceps, side chest...) would need a classifier trained on labelled frames.
- **Thresholds are untuned.** Defaults come from reasoning about torso-normalised motion and were
  tested on synthetic stick-figure footage only. Expect to adjust them on real competition video.

## Using another pose model

Any object with these two methods works. Pass it to `extract_poses(video, estimator, Config())`:

```
detect(frame_bgr, timestamp_ms) -> (33, 3) array of x, y (pixels), visibility, or None
close()
```

Keypoints follow the BlazePose order (11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips,
25/26 knees, 27/28 ankles, 0 nose). A YOLO-pose wrapper with person tracking is the natural
upgrade for multi-person footage (its 17 COCO keypoints would need mapping onto these indices).

## Layout

```
pose_extractor/
  cli.py         command line
  pipeline.py    analyze_video, extract_poses
  video.py       frame sampling, scene cuts, sharpness
  estimator.py   MediaPipe backend + model download
  geometry.py    pose normalisation and distance
  holds.py       hold detection
  select.py      best frame per hold
  dedupe.py      merging repeated poses
  output.py      photos, contact sheet, report
tests/           pytest suite using a synthetic stick-figure video
```

Run the tests with `pytest`. They need no pose model.
