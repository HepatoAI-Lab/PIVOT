from __future__ import annotations

import numpy as np


def binary_auc(labels: list[float], scores: list[float]) -> float:
    """Small dependency-light AUROC implementation for logging."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + ties)
