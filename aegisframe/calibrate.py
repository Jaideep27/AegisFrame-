"""Choose council thresholds from labelled results, with a stated guarantee.

A request is halted (sent to review or blocked) when the council posterior >= `review`.
The risk to control is the harmful-allow rate: the share of must-halt cases scoring below it.

method="crc"  conformal risk control (Angelopoulos et al., 2022): pick the largest threshold
              with  n/(n+1) * risk + 1/(n+1) <= alpha.  Guarantees E[harmful-allow] <= alpha
              on new data exchangeable with the calibration set.
method="ucb"  high-probability version: pick the largest threshold whose Clopper-Pearson
              upper confidence bound on the risk is <= alpha, so P(risk <= alpha) >= 1 - delta.

`block` is then set from the harmless cases: the lowest threshold that blocks at most
`benign_block` of them (never below `review`).

The guarantee holds only while live traffic resembles the calibration data. Recalibrate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Smallest p with P(Bin(n, p) <= k) <= delta: an upper (1-delta) bound on the true rate."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def calibrate(harmful: list[float], benign: list[float], alpha: float = 0.01, delta: float = 0.05,
              method: str = "crc", benign_block: float = 0.01, min_block: float = 0.85) -> dict:
    if not harmful:
        raise ValueError("calibration needs scored must-halt cases")
    n = len(harmful)
    candidates = sorted({0.0, *harmful}, reverse=True)

    def bound(lam: float) -> float:
        misses = sum(s < lam for s in harmful)
        if method == "crc":
            return n / (n + 1) * misses / n + 1 / (n + 1)
        return clopper_pearson_upper(misses, n, delta)

    review = next((lam for lam in candidates if bound(lam) <= alpha), None)
    feasible = review is not None
    review = review if feasible else 0.0

    # block: the lowest threshold that blocks at most `benign_block` of the harmless cases,
    # never below `min_block` (keeps a review band for uncertain cases) nor below `review`
    block = min_block
    if benign:
        allowed = int(math.floor(benign_block * len(benign)))
        ranked = sorted(benign, reverse=True)
        cutoff = ranked[allowed] if allowed < len(ranked) else 0.0  # highest score that must NOT be blocked
        block = max(block, cutoff + 1e-6)
    block = min(0.999, max(block, review))

    return {
        "review": review,
        "block": block,
        "method": method,
        "alpha": alpha,
        "delta": delta if method == "ucb" else None,
        "feasible": feasible,
        "n_harmful": n,
        "n_benign": len(benign),
        "empirical": {
            "harmful_allow": sum(s < review for s in harmful) / n,
            "benign_block": (sum(s >= block for s in benign) / len(benign)) if benign else None,
            "benign_review": (sum(review <= s < block for s in benign) / len(benign)) if benign else None,
        },
    }


def calibrate_file(results: str | Path, alpha: float = 0.01, delta: float = 0.05, method: str = "crc",
                   out: str | Path | None = "calibration.json") -> dict:
    harmful, benign = [], []
    for line in Path(results).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("posterior") is None:
            continue
        if row.get("expected") == "halt":
            harmful.append(float(row["posterior"]))
        elif row.get("expected") == "allow":
            benign.append(float(row["posterior"]))
    result = calibrate(harmful, benign, alpha=alpha, delta=delta, method=method)
    if out:
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
