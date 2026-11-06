# pose-extractor

Give it a bodybuilding posing video, get one photo of each of the six classic poses:
front double biceps, back double biceps, front lat spread, back lat spread, abs and thighs, side chest.

```
python -m pose_extractor posing_routine.mp4 -o results
```

```
results/
  poses/01_abs_and_thighs_0001.50s.jpg        one full-resolution photo per pose, named
  poses/02_front_double_biceps_0011.70s.jpg
  ...
  contact_sheet.jpg                           all photos in one image, for a quick check
  report.json                                 what was found, what was rejected and why
```

The tool also prints which of the six poses it did not find.

## Install (Linux)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads MediaPipe's pose model to `~/.cache/pose_extractor`.
If you are offline, download a `pose_landmarker_*.task` file yourself and pass `--model PATH`.
On a headless machine, `sudo apt install libgl1` may be needed for OpenCV.

## How it works

1. **Sample**: read the video at `--sample-fps` (default 6 frames/s).
2. **Pose**: MediaPipe PoseLandmarker gives 33 body keypoints per frame.
3. **Name every frame**: a few ratios, all relative to the athlete's own torso length, tell the
   six poses apart (table below). Arms hanging means "relaxed or in transition" and the frame is
   dropped. This is what removes the standing-around and turning-around frames.
4. **Holds**: a pose must also be held still (`--min-hold`, default 0.8 s). Camera cuts and slow
   drift end a hold, and a hold is cut wherever the pose name changes.
5. **Best frame**: inside each hold, pick the right moment of the pose (see below). By default
   that is the frame where the pose is most fully expressed (elbows pushed out, wrists up), sharp,
   showing the whole body, and away from the hold's edges.
6. **One photo per pose**: if a pose was held several times, only its best take is kept.
   Poses that are clearly intentional but match none of the six are saved as `other` and merged
   when they look alike.

| Pose | Wrists above hips | Elbows out | Other |
|---|---|---|---|
| Front / back double biceps | above the shoulders (≥ 1.0) | wide (≥ 0.70) | facing camera / back to camera |
| Abs and thighs | above the shoulders (≥ 1.0) | narrow (< 0.70) | facing camera |
| Front / back lat spread | hands near the waist (0.15 to 1.0) | wide (≥ 0.60) | facing camera / back to camera |
| Side chest | 0.15 to 1.0 | narrow (< 0.55) | shoulders turned (apparent width < 0.55) |

Facing camera or not comes from which shoulder is on the right of the image. Face keypoints
cannot tell front from back, because MediaPipe reports a confident face even from behind.

## The right moment inside a pose

- **Abs and thighs**: the stomach vacuum. MediaPipe also gives the outline of the body; the tool
  measures the width of the waist on it in every frame of the hold and takes the narrowest one, with
  both feet level (a foot in front of the other lowers the score). The waist is compared with the
  other frames of the same hold, so no absolute numbers are needed. If the waist hardly changes
  (under 2% of the torso length, which is the measurement noise), it is ignored and the frame is
  chosen by sharpness and position in the hold instead.
- **Front lat spread**: head straight instead of looking down. The nose is compared with the ears
  in every frame of the hold and the frames with the head most upright are preferred. This is a
  preference, not a rule: the body model's face points are noisy.
- **Alternates**: for every pose, `--alternates N` (default 3) also saves N frames spread over the
  hold in `alternates/`, so the timing can be corrected by eye. `--alternates 0` turns them off.

## When a pose is missing or wrong

Open `report.json`. The `holds` list shows every hold with its label and the measured values
(`arm_raise`, `elbow_lat`, `shoulder_w`, `facing`). A hold with `"label": null` was rejected as
relaxed: compare its `arm_raise` with `--arm-raise-min`.

| Symptom | Try |
|---|---|
| A real pose is rejected, `arm_raise` is just under 0.15 | `--arm-raise-min 0.1` |
| Relaxed frames still appear | `--arm-raise-min 0.25` |
| Pose is short and missing | `--min-hold 0.5` |
| Jittery video breaks holds | `--motion-thresh 0.8` |
| Poor detection, athlete small in frame | `--model-variant heavy` |
| Your own poses, not the classic six | `--generic` keeps every held pose |
| Long video, slow | lower `--sample-fps`, or use `--start` / `--end` |

Other flags: `--no-other` (drop unnamed poses), `--no-dedupe` (keep every take), `--dedupe-thresh`.

## Limitations

- **Calibrated on one setup.** The thresholds come from real MediaPipe keypoints of one home
  video (1080x1920, athlete facing a fixed camera). The tests check that the six poses of that
  session are named correctly and the eight relaxed frames are rejected. Other angles, lenses or
  body types may need `--arm-raise-min` or a look at `report.json`.
- **The peak of a pose is estimated, not seen.** "Most fully expressed" means widest elbows and
  highest wrists, not the moment the muscles are tensed hardest.
- **The vacuum detector has not seen a real vacuum.** It was checked on synthetic silhouettes and
  on real frames without a vacuum (two of them gave waist widths 1.8% apart, which is the noise
  floor). If the abs and thighs photo is still at the wrong moment, look in `alternates/`.
- **Head position is a weak signal.** In the test frames, looking down and looking straight gave
  overlapping numbers, so frames are only ranked against each other inside a hold.
- **Side chest keeps one photo.** If you show both sides, only the best one is kept.
- **One person.** MediaPipe tracks a single person. Crop other people out of the frame.
- **Only the six poses are named.** Others (most muscular, side triceps) come out as `other`.

## Using another pose model

Any object with these two methods works. Pass it to `extract_poses(video, estimator, Config())`:

```
detect(frame_bgr, timestamp_ms) -> (33, 3) array of x, y (pixels), visibility, or None
close()
```

Keypoints follow the BlazePose order (11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips,
25/26 knees, 27/28 ankles, 0 nose).

## Layout

```
pose_extractor/
  cli.py         command line
  pipeline.py    analyze_video, extract_poses
  classic.py     the six-pose rules
  video.py       frame sampling, scene cuts, sharpness
  estimator.py   MediaPipe backend (keypoints + body outline) + model download
  geometry.py    pose normalisation and distance
  holds.py       hold detection, splitting by pose name
  select.py      best frame per hold (vacuum, head, sharpness)
  dedupe.py      merging look-alike unnamed poses
  output.py      photos, contact sheet, report
tests/           pytest suite: synthetic video + real keypoints from the calibration session
```

Run the tests with `pytest`. They need no pose model.
