"""Plain data containers shared by the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Config:
    """All tunable parameters. Distances are in torso lengths, speeds in torso lengths / second."""

    sample_fps: float = 6.0          # how many frames per second of video are analysed
    analysis_max_side: int = 960     # frames are downscaled to this size for analysis only
    start: float = 0.0               # analyse from this time (s)
    end: Optional[float] = None      # ... until this time (s); None = end of video
    cut_threshold: float = 0.5       # histogram correlation below this = scene cut
    motion_thresh: float = 0.5       # below this speed the athlete counts as "still"
    min_hold: float = 0.8            # a pose must be held at least this long (s)
    drift_thresh: float = 0.3        # max slow drift allowed inside one hold
    dedupe: bool = True              # merge repeated poses into one photo
    dedupe_thresh: float = 0.25      # poses closer than this are "the same pose"
    min_visibility: float = 0.5      # keypoints below this confidence are ignored


@dataclass
class Sample:
    """What we know about one analysed frame."""

    index: int                       # frame index in the video
    time: float                      # seconds
    pose: Optional[np.ndarray]       # normalised (J, 2) pose, or None if nobody usable was found
    completeness: float              # share of key joints that are visible and inside the frame
    sharpness: float                 # Laplacian variance of the person crop
    cut: bool                        # True if a scene cut happened just before this frame


@dataclass
class Hold:
    """A run of consecutive samples (inclusive) where the athlete stays in one pose."""

    start: int
    end: int


@dataclass
class ExtractedPose:
    """One photo in the final result."""

    frame_index: int
    time: float
    hold_start: float
    hold_end: float
    score: float
    appearances: int = 1                                  # how many holds were merged into this pose
    times: list[float] = field(default_factory=list)      # start time of each merged hold
    path: Optional[str] = None                            # relative path of the saved image
