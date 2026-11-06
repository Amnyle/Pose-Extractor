"""Merge repeated poses (the same pose shown twice, or a hold split by a camera cut)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .geometry import pose_distance


def group_holds(
    poses: Sequence[np.ndarray],
    scores: Sequence[float],
    thresh: float,
) -> list[tuple[int, list[int]]]:
    """Greedy leader clustering of holds, in chronological order.

    poses: the normalised pose of each hold (taken at its best frame).
    scores: quality of that best frame.
    Returns [(representative, members)] where the representative is the best-scoring member
    and `members` lists every hold index in the group, ascending.
    """
    reps: list[int] = []
    members: list[list[int]] = []
    for i, pose in enumerate(poses):
        nearest: Optional[int] = None
        nearest_d = np.inf
        for g, rep in enumerate(reps):
            d = pose_distance(poses[rep], pose)
            if not np.isnan(d) and d < nearest_d:
                nearest, nearest_d = g, d
        if nearest is not None and nearest_d <= thresh:
            members[nearest].append(i)
            if scores[i] > scores[reps[nearest]]:
                reps[nearest] = i
        else:
            reps.append(i)
            members.append([i])
    return list(zip(reps, members))
