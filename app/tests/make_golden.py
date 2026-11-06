"""Write app/tests/golden.json: posing routines run through the Python tool, to check the app's port.

Routines are built from the real keypoints in tests/data/user_frames.json: each pose is held for a
few seconds with jitter, lats that open progressively, a stomach vacuum (on a person mask), a foot
moved forward, head movement, transitions through a relaxed stance, and one scene cut.

    python app/tests/make_golden.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pose_extractor.classic import OTHER  # noqa: E402
from pose_extractor.models import Config  # noqa: E402
from pose_extractor.pipeline import FULL_STRENGTH, build_sample, select_poses  # noqa: E402
from pose_extractor.select import pick_best_frame  # noqa: E402

FRAMES = {k: np.array(v) for k, v in json.loads((ROOT / "tests/data/user_frames.json").read_text()).items()}
W, H, FPS = 1080, 1920, 6

# (frame, seconds held, effects)
ROUTINE = [
    ("pose_02", 1.5, {}),
    ("pose_01", 3.5, {"vacuum": (1.8, 2.8), "foot": (0.3, 1.2)}),
    ("pose_03", 1.5, {}),
    ("pose_04", 2.5, {"spread": 0.12}),
    ("pose_05", 1.5, {"cut": 0.7}),
    ("pose_14", 3.5, {"spread": 0.18, "head": (1.5, 2.5)}),
    ("pose_09", 1.2, {}),
    ("pose_08", 2.2, {}),
    ("pose_10", 1.5, {}),
    ("pose_12", 2.5, {"spread": 0.1}),
    ("pose_13", 3.0, {"spread": 0.15}),
    ("pose_11", 1.5, {}),
    ("pose_01", 2.2, {}),
    ("pose_02", 1.2, {}),
    ("pose_14", 2.2, {"spread": 0.08}),
    ("pose_03", 1.5, {}),
]


def torso(kp):
    return float(np.linalg.norm(kp[[11, 12], :2].mean(0) - kp[[23, 24], :2].mean(0)))


def apply(kp, t, dur, fx):
    kp = kp.copy()
    tl = torso(kp)
    if "spread" in fx:   # elbows open progressively, peak at 70% of the hold
        u = t / dur
        k = fx["spread"] * (u / 0.7 if u < 0.7 else max(0.0, 1 - (u - 0.7) / 0.3) * 0.6 + 0.4)
        mid_x = kp[[11, 12], 0].mean()
        for e in (13, 14):
            kp[e, 0] += np.sign(kp[e, 0] - mid_x) * k * tl
    if "foot" in fx and fx["foot"][0] <= t <= fx["foot"][1]:
        kp[28, 1] += 0.3 * tl
    if "head" in fx:
        a, b = fx["head"]
        if not a <= t <= b:
            kp[0, 1] += 0.25 * np.linalg.norm(kp[7, :2] - kp[8, :2])
    return kp


def mask_rect(kp, vacuum):
    hips = kp[[23, 24], :2].mean(0)
    tl = torso(kp)
    half = (0.20 if vacuum else 0.25) * tl
    return [int(hips[0] - half), int(hips[0] + half), int(hips[1] - 0.8 * tl), int(hips[1])]


def build(seed):
    rng = np.random.default_rng(seed)
    items = []
    t0 = 0.0
    prev = None
    for name, dur, fx in ROUTINE:
        target = FRAMES[name]
        if prev is not None:   # 0.5 s transition
            for k in range(1, 4):
                u = k / 4
                kp = prev * (1 - u) + target * u
                items.append((t0, kp, None, False))
                t0 += 1 / FPS
        steps = int(dur * FPS)
        for k in range(steps):
            t = k / FPS
            kp = apply(target, t, dur, fx)
            vac = "vacuum" in fx and fx["vacuum"][0] <= t <= fx["vacuum"][1]
            rect = mask_rect(kp, vac) if name == "pose_01" else None
            items.append((t0, kp, rect, "cut" in fx and abs(t - fx["cut"]) < 1e-6))
            t0 += 1 / FPS
        prev = apply(target, (steps - 1) / FPS, dur, fx)

    samples_in = []
    for idx, (t, kp, rect, cut) in enumerate(items):
        kp = kp.copy()
        kp[:, :2] += rng.normal(0, 1.2, (33, 2))
        kp = np.round(kp, 2)
        sharp = round(float(rng.uniform(60, 140) * (0.5 if rng.random() < 0.1 else 1.0)), 2)
        samples_in.append({"index": idx, "time": round(t, 4), "kp": kp.tolist(), "width": W, "height": H,
                           "sharpness": sharp, "cut": bool(cut), "maskRect": rect})
    return samples_in


def make_mask(rect):
    m = np.zeros((H, W), np.float32)
    x0, x1, y0, y1 = rect
    m[y0:y1, x0:x1] = 1.0
    return m


def run(samples_in):
    cfg = Config()
    samples = [
        build_sample(s["index"], s["time"], np.array(s["kp"]), s["width"], s["height"], s["sharpness"], s["cut"],
                     cfg, make_mask(s["maskRect"]) if s["maskRect"] else None)
        for s in samples_in
    ]
    holds, results = select_poses(samples, cfg)
    sharp = [s.sharpness for s in samples if s.pose is not None]
    ref_sharp = float(np.percentile(sharp, 90))
    pools = {}
    for h in holds:
        if h.label is not None and h.label not in (OTHER, "back_lat_spread"):
            pools.setdefault(h.label, []).extend(range(h.start, h.end + 1))
    picks = [list(pick_best_frame(samples, h, ref_sharp=ref_sharp, ref_strength=FULL_STRENGTH,
                                  pool=pools.get(h.label), motion_thresh=cfg.motion_thresh))
             if h.label is not None else None for h in holds]
    return {
        "samples": [{"label": s.label, "strength": s.strength, "speed": s.speed, "waist": s.waist,
                     "headDrop": s.head_drop} for s in samples],
        "holds": [[h.start, h.end, h.label] for h in holds],
        "picks": picks,
        "results": [{"label": r.label, "time": r.time, "score": r.score} for r in results if r.label != OTHER],
    }


if __name__ == "__main__":
    out = []
    for seed in (1, 2, 3):
        samples_in = build(seed)
        out.append({"seed": seed, "input": samples_in, "expected": run(samples_in)})
        exp = out[-1]["expected"]
        print(seed, len(samples_in), "samples,", len(exp["holds"]), "holds:",
              [r["label"] + "@" + str(round(r["time"], 1)) for r in exp["results"]])
    (Path(__file__).parent / "golden.json").write_text(json.dumps(out))
