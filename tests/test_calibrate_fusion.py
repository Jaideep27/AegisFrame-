"""Temperature calibration for agent messages: scale logits so predicted confidence matches reality."""

from __future__ import annotations

import numpy as np

from aegisframe.council.calibrate_fusion import fit_temperature


def test_overconfident_scores_get_a_temperature_above_one():
    """Logits that are too sharp for the actual accuracy should be softened (T > 1)."""
    rng = np.random.default_rng(0)
    # logits are large (+/-5) but the truth follows a much gentler sigmoid(logit/5),
    # so the raw logits are 5x too confident and want softening (T ~ 5).
    logits = rng.choice([-5.0, 5.0], size=4000)
    y = (rng.random(4000) < 1.0 / (1.0 + np.exp(-logits / 5.0))).astype(int)
    t = fit_temperature(logits, y)
    assert t > 2.0


def test_a_perfectly_calibrated_signal_stays_near_one():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    logits = np.log(p / (1 - p))
    t = fit_temperature(logits, y)
    assert 0.7 < t < 1.4


def test_temperature_is_positive_even_with_degenerate_input():
    assert fit_temperature(np.zeros(10), np.ones(10, dtype=int)) > 0
    assert fit_temperature(np.array([1.0, 2.0]), np.array([1, 1])) > 0
