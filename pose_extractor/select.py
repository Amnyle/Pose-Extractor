"""Choose the best photo inside a hold."""

from __future__ import annotations

from typing import Optional, Sequence

from .classic import ABS_AND_THIGHS, FRONT_LAT_SPREAD, LEGS_LEVEL_MAX
from .models import Hold, Sample

MIN_WAIST_RANGE = 0.02   # waist changes smaller than this (in torso lengths) are measurement noise
MIN_HEAD_RANGE = 0.12    # same for the head-tilt ratio


def _median3(values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Median filter, window 3, that leaves missing values missing."""
    out = list(values)
    for i in range(1, len(values) - 1):
        window = sorted(v for v in values[i - 1 : i + 2] if v is not None)
        if values[i] is not None and window:
            out[i] = window[len(window) // 2]
    return out


def _relative(values: Sequence[Optional[float]], lower_is_better: bool, min_range: float) -> list[float]:
    """Rank values inside one hold from 0 (worst) to 1 (best).

    Everything is neutral (0.5) when too few values exist or they barely differ: that means the
    quantity was not observed to change, so it must not decide the choice.
    """
    present = [v for v in values if v is not None]
    if len(present) < 3:
        return [0.5] * len(values)
    lo, hi = min(present), max(present)
    if hi - lo < min_range:
        return [0.5] * len(values)
    out = []
    for v in values:
        if v is None:
            out.append(0.5)
        else:
            t = (v - lo) / (hi - lo)
            out.append(1.0 - t if lower_is_better else t)
    return out


def _legs_factor(sample: Sample) -> float:
    """1 when both feet are level, falling towards 0.2 as one foot moves ahead of the other."""
    if sample.features is None:
        return 1.0
    excess = sample.features.legs_dy - LEGS_LEVEL_MAX
    return 1.0 if excess <= 0 else max(0.2, 1.0 - excess / 0.2)


def pick_best_frame(
    samples: Sequence[Sample],
    hold: Hold,
    w_strength: float = 0.5,
    w_sharp: float = 0.35,
    w_center: float = 0.15,
    ref_sharp: Optional[float] = None,
    ref_strength: Optional[float] = None,
) -> tuple[int, float]:
    """Return (sample_index, score) of the best frame in the hold, or (-1, 0.0) if none is usable.

    Every frame gets  completeness * (weighted sum of terms), all terms between 0 and 1:
      - completeness: how many key joints are visible and inside the frame
      - strength: how fully the pose is expressed (elbows out, wrists up)
      - sharpness: sharpness of the athlete's crop, capped at 1
      - centrality: 1 in the middle of the hold, 0 at its edges (edges are next to movement)

    Two poses have their own definition of "the right moment":
      - abs and thighs: the stomach vacuum, measured as the narrowest waist of the hold, with both
        feet level (a foot in front of the other lowers the score).  0.55 vacuum, 0.25 sharpness,
        0.20 centrality.
      - front lat spread: the straightest head of the hold instead of one looking down.
        0.35 strength, 0.35 head, 0.20 sharpness, 0.10 centrality.
    These are ranked inside the hold, so they work without knowing the athlete's absolute numbers.
    If they do not change inside the hold, they have no effect.

    `ref_sharp` and `ref_strength` are the values that count as "full marks". Pass fixed values
    (e.g. taken from the whole video) to make scores comparable between different holds. Without
    them, the best frame of this hold is used as the reference, which only ranks frames within it.
    """
    indices = list(range(hold.start, hold.end + 1))
    max_sharp = ref_sharp or max((samples[i].sharpness for i in indices), default=0.0) or 1.0
    max_strength = ref_strength or max((samples[i].strength for i in indices), default=0.0) or 1.0
    mid = (hold.start + hold.end) / 2
    half = max((hold.end - hold.start) / 2, 1e-9)

    vacuum = head = None
    if hold.label == ABS_AND_THIGHS:
        vacuum = _relative(_median3([samples[i].waist for i in indices]), True, MIN_WAIST_RANGE)
    elif hold.label == FRONT_LAT_SPREAD:
        head = _relative(_median3([samples[i].head_drop for i in indices]), True, MIN_HEAD_RANGE)

    best_idx, best_score = -1, 0.0
    for pos, i in enumerate(indices):
        sample = samples[i]
        if sample.pose is None:
            continue
        centrality = max(0.0, 1.0 - abs(i - mid) / half)
        sharp = min(sample.sharpness / max_sharp, 1.0)
        strength = min(sample.strength / max_strength, 1.0)
        if vacuum is not None:
            score = sample.completeness * _legs_factor(sample) * (0.55 * vacuum[pos] + 0.25 * sharp + 0.20 * centrality)
        elif head is not None:
            score = sample.completeness * (0.35 * strength + 0.35 * head[pos] + 0.20 * sharp + 0.10 * centrality)
        else:
            score = sample.completeness * (w_strength * strength + w_sharp * sharp + w_center * centrality)
        if best_idx < 0 or score > best_score:
            best_idx, best_score = i, score
    return best_idx, best_score
