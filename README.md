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

## Phone app

`app/` is the same tool as an app for your phone: import a video, tick which of the six poses are in
it, and get the best photo of each. When the tool is not sure (another take or moment is nearly as
good), the pose is marked **Check** and the other good moments are shown underneath, to pick with a
tap. A pose you ticked but that was not recognised gets **Pick one**: the held moments that look
most like it, or none. **Save photos** saves the chosen photos at full video resolution (share sheet
on the phone, downloads on a computer).

Everything runs on the phone (MediaPipe in the browser); the video is never uploaded. After the
first use it also works offline.

**Install it**: it is published on GitHub Pages by `.github/workflows/pages.yml` on every push to
`main`. Once, in the repository on GitHub: Settings > Pages > Source: **GitHub Actions**. The app is
then at `https://<owner>.github.io/<repo>/`. Open that link on the phone and choose
**Add to Home Screen** (Safari: Share button; Chrome: menu or the install prompt). It opens full
screen like any other app.

**Run it locally**:

```
app/scripts/fetch-vendor.sh                 # MediaPipe runtime + pose model into app/vendor (once)
cd app && python -m http.server 8000        # then open http://localhost:8000
node --test app/tests/*.test.mjs            # the JavaScript gives the same results as the Python tool
```

The app's logic (`app/js/classic.js`, `holds.js`, `select.js`, `pipeline.js`) is a line-by-line port of
the Python modules. `app/tests/golden.json` holds three posing routines run through the Python tool;
regenerate it with `python app/tests/make_golden.py` after changing the Python logic, and the
JavaScript tests show where the port must follow.

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

Every frame of a hold gets a score from a few terms, each ranked from 0 to 1. A term that hardly
changes (inside the measurement noise) is ignored, so it cannot decide the photo by accident.

- **Expression**: how fully the pose is expressed (elbows out, wrists up). Ranked against the other
  frames of the same pose, so a slightly wider spread really wins.
- **Sharpness**, **stillness** (the peak of a pose is held, not passed through) and **position**
  (away from the edges of the hold, which touch the movement in and out of the pose).

Two poses have their own definition of the peak:

- **Abs and thighs**: the stomach vacuum. MediaPipe also gives the outline of the body; the tool
  measures the width of the waist in every frame and takes the narrowest one, with both feet level
  (a foot in front of the other lowers the score). The waist is measured in pixels and divided by
  the median torso length of the pose, not the torso of its own frame: crunching or standing taller
  changes how long the torso looks, and would otherwise fake a wider or narrower waist.
  Weights: 50% vacuum, 20% sharpness, 15% stillness, 15% position.
- **Front lat spread**: the widest elbows, which is when the lats are fully open. Only the
  elbows count: the hands sliding on the waist do not. For the front lat spread a straight head is a
  small tie-breaker (10%): the face points are noisy and athletes often look down at their lats.
  Weights (front): 50% spread, 20% sharpness, 10% each for head, stillness and position.

- **Back lat spread** keeps the original scoring, which already gave the right moment:
  50% strength (elbows out, wrists up, against a fixed full-pose value), 35% sharpness, 15% position.

When a pose is held several times, every take is ranked on one common scale (all frames of all
takes of that pose), so the take with the real vacuum or the widest spread is the one kept (back lat spread excepted).

- **Alternates**: for every pose, `--alternates N` (default 3) also saves N frames spread over the
  hold in `alternates/`, so the timing can be corrected by eye. `--alternates 0` turns them off.
- **Timeline**: in `report.json`, each pose has a `timeline` with what was measured on every frame
  of its hold (`elbow_lat`, `waist_px`, `head_drop`, `speed`, `sharpness`) and which frame was
  `chosen`. If the timing is still off, this shows which measurement misled the choice.

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
app/             the phone app (installable web app, same logic in JavaScript)
```

Run the tests with `pytest`. They need no pose model.
