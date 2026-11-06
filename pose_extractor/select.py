"""Choose the best photo inside a hold."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .classic import ABS_AND_THIGHS, BACK_LAT_SPREAD, FRONT_LAT_SPREAD, LEGS_LEVEL_MAX
from .models import Hold, Sample

MIN_WAIST_RANGE = 0.02      # waist changes smaller than this (in torso lengths) are measurement noise
MIN_HEAD_RANGE = 0.12       # same for the head-tilt ratio
MIN_STRENGTH_RANGE = 0.04   # same for how far the elbows are out (torso lengths)
LAT_SPREADS = (FRONT_LAT_SPREAD, BACK_LAT_SPREAD)

# Weights of each term, per pose. They add up to 1.
#   expression: how fully the pose is expressed (elbows out, wrists up; for lat spreads: elbows out)
#   vacuum:     narrowest waist (abs and thighs)
#   head:       head straight rather than looking down (front lat spread, a weak tie-breaker)
#   sharp:      sharpness of the athlete's crop
#   still:      the body is not moving at that moment (the peak of a pose is held, not passed through)
#   center:     away from the edges of the hold, which touch the movement into and out of the pose
WEIGHTS = {
    ABS_AND_THIGHS: dict(vacuum=0.50, expression=0.0, head=0.0, sharp=0.20, still=0.15, center=0.15),
    FRONT_LAT_SPREAD: dict(vacuum=0.0, expression=0.50, head=0.10, sharp=0.20, still=0.10, center=0.10),
    BACK_LAT_SPREAD: dict(vacuum=0.0, expression=0.55, head=0.0, sharp=0.20, still=0.15, center=0.10),
    None: dict(vacuum=0.0, expression=0.40, head=0.0, sharp=0.35, still=0.10, center=0.15),
}


def _median3(values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Median filter, window 3, that leaves missing values missing."""
    out = list(values)
    for i in range(1, len(values) - 1):
        window = sorted(v for v in values[i - 1 : i + 2] if v is not None)
        if values[i] is not None and window:
            out[i] = window[len(window) // 2]
    return out


def _runs(indices: Sequence[int]) -> list[list[int]]:
    """Split sorted sample indices into runs of consecutive ones (one run per hold)."""
    runs: list[list[int]] = []
    for i in sorted(set(indices)):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _smoothed(indices: Sequence[int], value: Callable[[int], Optional[float]]) -> dict[int, Optional[float]]:
    """value(i) for every index, median-filtered inside each run of consecutive indices."""
    out: dict[int, Optional[float]] = {}
    for run in _runs(indices):
        out.update(zip(run, _median3([value(i) for i in run])))
    return out


def _ranker(pool_values: Sequence[Optional[float]], lower_is_better: bool, min_range: float) -> Callable[[Optional[float]], float]:
    """Map a value to 0 (worst) .. 1 (best) relative to the values seen in the pool.

    Everything is neutral (0.5) when too few values exist or they barely differ: that means the
    quantity was not observed to change, so it must not decide the choice. The 5th and 95th
    percentiles are used as the ends of the scale, so one outlier frame cannot squash the rest.
    """
    present = np.array([v for v in pool_values if v is not None], dtype=float)
    if present.size < 3:
        return lambda v: 0.5
    lo, hi = np.percentile(present, [5, 95])
    if hi - lo < min_range:
        return lambda v: 0.5

    def rank(v: Optional[float]) -> float:
        if v is None:
            return 0.5
        t = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
        return 1.0 - t if lower_is_better else t

    return rank


def _legs_factor(sample: Sample) -> float:
    """1 when both feet are level, falling towards 0.2 as one foot moves ahead of the other."""
    if sample.features is None:
        return 1.0
    excess = sample.features.legs_dy - LEGS_LEVEL_MAX
    return 1.0 if excess <= 0 else max(0.2, 1.0 - excess / 0.2)


def _expression(sample: Sample, label: Optional[str]) -> Optional[float]:
    """How fully the pose is expressed. For lat spreads only the width of the elbows matters:
    the hands slide on the waist while the lats open, and that must not count."""
    if sample.features is None:
        return None
    if label in LAT_SPREADS:
        return sample.features.elbow_lat
    return sample.strength


def _waists(samples: Sequence[Sample], indices: Sequence[int]) -> dict[int, Optional[float]]:
    """Waist widths of the given frames, all in the same unit.

    The waist is stored in torso lengths of its own frame, but crunching the abs shortens the
    torso as the camera sees it, which would make the waist look wider exactly at the peak.
    Inside one pose the camera distance does not change, so the waists are brought back to
    pixels and divided by the median torso length of all the frames instead.
    """
    torsos = [samples[i].torso_px for i in indices if samples[i].torso_px]
    ref = float(np.median(torsos)) if torsos else None

    def value(i: int) -> Optional[float]:
        s = samples[i]
        if s.waist is None:
            return None
        if ref and s.torso_px:
            return s.waist * s.torso_px / ref
        return s.waist

    return _smoothed(indices, value)


def pick_best_frame(
    samples: Sequence[Sample],
    hold: Hold,
    w_strength: Optional[float] = None,
    w_sharp: Optional[float] = None,
    w_center: Optional[float] = None,
    ref_sharp: Optional[float] = None,
    ref_strength: Optional[float] = None,  # kept for compatibility; expression is now ranked, see `pool`
    pool: Optional[Sequence[int]] = None,
    motion_thresh: float = 0.5,
) -> tuple[int, float]:
    """Return (sample_index, score) of the best frame in the hold, or (-1, 0.0) if none is usable.

    Every frame gets  completeness * (weighted sum of terms), all terms between 0 and 1 (see
    WEIGHTS for the terms and their weight per pose).

    Expression, waist and head are ranked against the frames of `pool`: pass every frame of every
    hold of the same pose, and a pose held twice is judged on one scale, so the take with the real
    vacuum or the widest spread wins instead of each take's own best frame scoring full marks.
    Without a pool the hold itself is used. A quantity that hardly changes is ignored.

    Abs and thighs also prefers both feet level (a foot in front of the other lowers the score).

    `ref_sharp` is the sharpness that counts as full marks. Pass a value taken from the whole
    video to make scores comparable between holds; without it the sharpest frame of the hold is used.
    """
    indices = list(range(hold.start, hold.end + 1))
    pool = sorted(set(pool) | set(indices)) if pool is not None else indices
    label = hold.label if hold.label in WEIGHTS else None
    w = dict(WEIGHTS[label])
    if w_strength is not None or w_sharp is not None or w_center is not None:
        w.update(expression=w_strength or 0.0, sharp=w_sharp or 0.0, center=w_center or 0.0, still=0.0)

    max_sharp = ref_sharp or max((samples[i].sharpness for i in indices), default=0.0) or 1.0
    mid = (hold.start + hold.end) / 2
    half = max((hold.end - hold.start) / 2, 1e-9)

    expr = _smoothed(pool, lambda i: _expression(samples[i], label))
    rank_expr = _ranker(list(expr.values()), False, MIN_STRENGTH_RANGE)
    waist = _waists(samples, pool) if w["vacuum"] else {}
    rank_waist = _ranker(list(waist.values()), True, MIN_WAIST_RANGE)
    head = _smoothed(pool, lambda i: samples[i].head_drop) if w["head"] else {}
    rank_head = _ranker(list(head.values()), True, MIN_HEAD_RANGE)

    best_idx, best_score = -1, 0.0
    for i in indices:
        sample = samples[i]
        if sample.pose is None:
            continue
        terms = dict(
            expression=rank_expr(expr.get(i)),
            vacuum=rank_waist(waist.get(i)),
            head=rank_head(head.get(i)),
            sharp=min(sample.sharpness / max_sharp, 1.0),
            still=0.5 if sample.speed is None else max(0.0, 1.0 - sample.speed / motion_thresh),
            center=max(0.0, 1.0 - abs(i - mid) / half),
        )
        score = sample.completeness * sum(w[k] * terms[k] for k in w)
        if label == ABS_AND_THIGHS:
            score *= _legs_factor(sample)
        if best_idx < 0 or score > best_score:
            best_idx, best_score = i, score
    return best_idx, best_score
