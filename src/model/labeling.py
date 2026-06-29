"""Supervised labels for short-term price-movement classification.

For each point we look `horizon` ticks into the future and classify the
forward return into one of three classes:
    0 = DOWN  (forward return < -threshold)
    1 = FLAT  (|forward return| <= threshold)
    2 = UP    (forward return >  +threshold)

The threshold is expressed in basis points (1 bp = 0.01%).
"""
from __future__ import annotations

import numpy as np

DOWN, FLAT, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "DOWN", FLAT: "FLAT", UP: "UP"}


def forward_return_labels(prices: np.ndarray, horizon: int, threshold_bps: float) -> np.ndarray:
    """Return an int label array aligned to `prices` (last `horizon` entries
    are unlabeled and returned as -1)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    threshold = threshold_bps / 10_000.0
    labels = np.full(prices.shape[0], -1, dtype=np.int64)
    future = prices[horizon:]
    current = prices[:-horizon]
    fwd_ret = (future - current) / current
    cls = np.full(fwd_ret.shape[0], FLAT, dtype=np.int64)
    cls[fwd_ret > threshold] = UP
    cls[fwd_ret < -threshold] = DOWN
    labels[: fwd_ret.shape[0]] = cls
    return labels
