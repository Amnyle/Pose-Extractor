"""Choose the best photo inside a hold."""

from __future__ import annotations

from typing import Optional, Sequence

from .models import Hold, Sample


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

    score = completeness * (w_strength * strength_term
                            + w_sharp * sharpness_term
                            + w_center * centrality)
      - completeness: how many key joints are visible and inside the frame
      - strength_term: how fully the pose is expressed (elbows out, wrists up). This picks the peak
        of a slowly building pose such as a lat spread. Zero for samples without pose features.
      - sharpness_term: sharpness of the athlete's crop, capped at 1
      - centrality: 1 in the middle of the hold, 0 at its edges (edges are next to movement)

    `ref_sharp` and `ref_strength` are the values that count as "full marks". Pass fixed values
    (e.g. taken from the whole video) to make scores comparable between different holds. Without
    them, the best frame of this hold is used as the reference, which only ranks frames within it.
    """
    indices = range(hold.start, hold.end + 1)
    max_sharp = ref_sharp or max((samples[i].sharpness for i in indices), default=0.0) or 1.0
    max_strength = ref_strength or max((samples[i].strength for i in indices), default=0.0) or 1.0
    mid = (hold.start + hold.end) / 2
    half = max((hold.end - hold.start) / 2, 1e-9)

    best_idx, best_score = -1, 0.0
    for i in indices:
        sample = samples[i]
        if sample.pose is None:
            continue
        centrality = max(0.0, 1.0 - abs(i - mid) / half)
        score = sample.completeness * (
            w_strength * min(sample.strength / max_strength, 1.0)
            + w_sharp * min(sample.sharpness / max_sharp, 1.0)
            + w_center * centrality
        )
        if best_idx < 0 or score > best_score:
            best_idx, best_score = i, score
    return best_idx, best_score
