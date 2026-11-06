"""Choose the best photo inside a hold."""

from __future__ import annotations

from typing import Sequence

from .models import Hold, Sample


def pick_best_frame(
    samples: Sequence[Sample],
    hold: Hold,
    w_sharp: float = 0.7,
    w_center: float = 0.3,
) -> tuple[int, float]:
    """Return (sample_index, score) of the best frame in the hold, or (-1, 0.0) if none is usable.

    score = completeness * (w_sharp * relative_sharpness + w_center * centrality)
      - completeness: how many key joints are visible and inside the frame
      - relative_sharpness: sharpness divided by the sharpest frame of the hold
      - centrality: 1 in the middle of the hold, 0 at its edges (edges are next to movement)
    """
    indices = range(hold.start, hold.end + 1)
    max_sharp = max((samples[i].sharpness for i in indices), default=0.0) or 1.0
    mid = (hold.start + hold.end) / 2
    half = max((hold.end - hold.start) / 2, 1e-9)

    best_idx, best_score = -1, 0.0
    for i in indices:
        sample = samples[i]
        if sample.pose is None:
            continue
        centrality = max(0.0, 1.0 - abs(i - mid) / half)
        score = sample.completeness * (w_sharp * sample.sharpness / max_sharp + w_center * centrality)
        if best_idx < 0 or score > best_score:
            best_idx, best_score = i, score
    return best_idx, best_score
