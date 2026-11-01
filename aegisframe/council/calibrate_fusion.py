"""Per-agent temperature calibration: one scalar T that scales an agent's logits so its stated
confidence matches how often it is actually right. Fitted on held-out data by minimising log-loss.
"""

from __future__ import annotations

import numpy as np


def _nll(t: float, logits: np.ndarray, y: np.ndarray) -> float:
    z = logits / max(t, 1e-3)
    z = np.clip(z, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid: int = 61) -> float:
    """Temperature in [0.1, 10] that minimises binary log-loss. A golden-section-free grid, then refine.

    T > 1 softens overconfident logits; T < 1 sharpens underconfident ones.
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if logits.size == 0 or len(np.unique(y)) < 2:
        return 1.0
    ts = np.geomspace(0.1, 10.0, grid)
    losses = [_nll(t, logits, y) for t in ts]
    best = ts[int(np.argmin(losses))]
    fine = np.linspace(best * 0.7, best * 1.4, 41)
    losses = [_nll(t, logits, y) for t in fine]
    return float(fine[int(np.argmin(losses))])
