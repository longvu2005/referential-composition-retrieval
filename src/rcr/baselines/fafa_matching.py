"""CPR Benchmark Partial SetMatch: max-sum one-to-one, mean over components.

Includes the exact vectorized K=1/K=2 and N=1/N=2 shortcuts. Missing gallery
persons contribute the fixed background score; real pairs are not optional.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def partial_setmatch(matrix, unmatched_score: float = -1.0) -> float:
    s = np.asarray(matrix, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < 1 or not np.isfinite(s).all():
        raise ValueError("Partial SetMatch requires finite [K,N] scores, K >= 1")
    if not np.isfinite(unmatched_score):
        raise ValueError("Unmatched score must be finite")
    k, n = s.shape
    if n == 0:
        return float(unmatched_score)
    rows, cols = linear_sum_assignment(-s)
    return float((s[rows, cols].sum() + (k - len(rows)) * unmatched_score) / k)


def _two_best(a, b):
    """For each image, best a[i] + b[j] with i != j."""
    ia, ib = np.argmax(a, axis=1), np.argmax(b, axis=1)
    r = np.arange(len(a))
    values = a[r, ia] + b[r, ib]
    same = ia == ib
    if np.any(same):
        aa, bb = a[same].copy(), b[same].copy()
        rr = np.arange(len(aa))
        aa[rr, ia[same]] = -np.inf
        bb[rr, ib[same]] = -np.inf
        values[same] = np.maximum(
            a[same, ia[same]] + bb.max(axis=1), aa.max(axis=1) + b[same, ib[same]]
        )
    return values


def aggregate_people(pair_scores, offsets, unmatched_score: float = -1.0):
    """Return one score per gallery image without per-image Hungarian for K<=2."""
    s = np.asarray(pair_scores, dtype=np.float64)
    offsets = np.asarray(offsets)
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or not np.issubdtype(offsets.dtype, np.integer)
        or np.any(np.diff(offsets) < 0)
        or s.ndim != 2
        or s.shape[0] < 1
        or s.shape[1] != offsets[-1]
        or not np.isfinite(s).all()
        or not np.isfinite(unmatched_score)
    ):
        raise ValueError("Invalid component-person scores or offsets")
    counts = np.diff(offsets)
    width = max(2, int(counts.max(initial=0)))
    ids = offsets[:-1, None] + np.arange(width)[None, :]
    ids[np.arange(width)[None, :] >= counts[:, None]] = -1
    k = len(s)
    result = np.full(len(counts), unmatched_score, dtype=np.float64)
    one = counts == 1
    two = counts == 2
    many = counts >= 3
    if np.any(one):
        result[one] = (s[:, ids[one, 0]].max(axis=0) + (k - 1) * unmatched_score) / k
    if k == 1:
        active = counts >= 2
        if np.any(active):
            chosen = ids[active]
            vals = s[0, np.maximum(chosen, 0)]
            vals[chosen < 0] = -np.inf
            result[active] = vals.max(axis=1)
    elif k == 2:
        active = counts >= 2
        if np.any(active):
            chosen = ids[active]
            vals = s[:, np.maximum(chosen, 0)].transpose(1, 0, 2)
            vals[np.broadcast_to((chosen < 0)[:, None, :], vals.shape)] = -np.inf
            result[active] = _two_best(vals[:, 0], vals[:, 1]) / 2
    else:
        if np.any(two):
            vals = s[:, ids[two, :2]].transpose(1, 2, 0)
            result[two] = (
                _two_best(vals[:, 0], vals[:, 1]) + (k - 2) * unmatched_score
            ) / k
        for gi in np.flatnonzero(many):
            result[gi] = partial_setmatch(
                s[:, offsets[gi] : offsets[gi + 1]], unmatched_score
            )
    return result.astype(np.float32)
