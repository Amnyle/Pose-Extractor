"""Synthetic athlete used by the tests: stick-figure poses, a timeline, a fake estimator, a video."""

from __future__ import annotations

import cv2
import numpy as np

JOINT_IDS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
TORSO = 100.0  # px, shoulder midpoint to hip midpoint


def _body(left_elbow, left_wrist, right_elbow, right_wrist) -> np.ndarray:
    pts = {
        0: (0, -140),
        11: (-40, -100), 12: (40, -100),
        13: left_elbow, 14: right_elbow,
        15: left_wrist, 16: right_wrist,
        23: (-25, 0), 24: (25, 0),
        25: (-30, 100), 26: (30, 100),
        27: (-30, 200), 28: (30, 200),
    }
    return np.array([pts[j] for j in JOINT_IDS], dtype=float)


POSE_A = _body((-55, -50), (-60, -5), (55, -50), (60, -5))        # arms down
POSE_B = _body((-90, -110), (-90, -170), (90, -110), (90, -170))  # double biceps
POSE_C = _body((-20, -70), (20, -60), (50, -70), (10, -60))       # arms across the torso

# (start, end, pose): the athlete holds each pose, and moves linearly in between
HOLDS = [(0.0, 1.5, POSE_A), (2.1, 4.0, POSE_B), (4.6, 6.0, POSE_C), (6.6, 8.0, POSE_B)]
DURATION = 8.0
CUT_TIME = 3.1  # background colour changes here (inside the first B hold)


def pose_at(t: float) -> np.ndarray:
    if t <= HOLDS[0][0]:
        return HOLDS[0][2]
    for (s0, e0, p0), (s1, _e1, p1) in zip(HOLDS, HOLDS[1:]):
        if s0 <= t <= e0:
            return p0
        if e0 < t < s1:
            a = (t - e0) / (s1 - e0)
            return (1 - a) * p0 + a * p1
    return HOLDS[-1][2]


def to_keypoints(rel: np.ndarray, center=(320.0, 250.0), jitter: float = 0.0, rng=None) -> np.ndarray:
    kp = np.zeros((33, 3), dtype=np.float32)
    pts = rel + np.asarray(center)
    if jitter and rng is not None:
        pts = pts + rng.normal(0, jitter, pts.shape)
    kp[list(JOINT_IDS), :2] = pts
    kp[list(JOINT_IDS), 2] = 1.0
    return kp


class TimelineEstimator:
    """Pretends to be a pose model: looks up the athlete's pose from the timestamp."""

    def __init__(self, jitter: float = 0.5, seed: int = 0):
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.closed = False

    def detect(self, frame_bgr, timestamp_ms):
        return to_keypoints(pose_at(timestamp_ms / 1000.0), jitter=self.jitter, rng=self.rng)

    def close(self):
        self.closed = True


class NobodyEstimator:
    def detect(self, frame_bgr, timestamp_ms):
        return None

    def close(self):
        pass


EDGES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
         (23, 25), (25, 27), (24, 26), (26, 28), (0, 11)]


def render_frame(t: float) -> np.ndarray:
    background = (0, 0, 0) if t < CUT_TIME else (150, 40, 0)
    frame = np.full((480, 640, 3), background, np.uint8)
    kp = to_keypoints(pose_at(t))
    for a, b in EDGES:
        cv2.line(frame, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), (255, 255, 255), 6)
    return frame


def write_video(path, fps: int = 20) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    assert writer.isOpened(), "OpenCV cannot write MJPG videos here"
    for i in range(int(fps * DURATION)):
        writer.write(render_frame(i / fps))
    writer.release()
