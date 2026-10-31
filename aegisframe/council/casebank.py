"""A two-sided case-bank agent: how close is a request to known misuse vs known legitimate edits?

It holds two banks of unit embeddings - harmful and benign - in the space supervised-contrastive training
shaped. For a query it compares the k-nearest-neighbour densities in each bank and emits a BLOCK-vs-ALLOW
log-likelihood-ratio message:

    m_BLOCK = logsumexp_j in H (sigma(e, e_j) / tau)  -  logsumexp_j in N (sigma(e, e_j) / tau)
              - log(k_H / k_N)

with sigma the cosine similarity (dot product of unit vectors). The bank says nothing about the grey
REVIEW class, so m_REVIEW is 0. A query far from every stored case is marked uncertain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fusion import DECISION, Message

_B = DECISION.index("BLOCK") - 1


def _logsumexp(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


class CaseBank:
    def __init__(self, harmful: np.ndarray, benign: np.ndarray, tau: float = 0.1, k: int = 10):
        self.harmful = np.asarray(harmful, dtype=float).reshape(-1, harmful.shape[-1] if harmful.size else 1)
        self.benign = np.asarray(benign, dtype=float).reshape(-1, benign.shape[-1] if benign.size else 1)
        self.tau = float(tau)
        self.k = int(k)

    def _topk_sims(self, bank: np.ndarray, q: np.ndarray) -> np.ndarray:
        if bank.size == 0:
            return np.empty(0)
        sims = bank @ q
        k = min(self.k, sims.shape[0])
        return np.sort(sims)[-k:]

    def message(self, agent: str, embedding: np.ndarray) -> Message:
        q = np.asarray(embedding, dtype=float).reshape(-1)
        if self.harmful.size == 0 or self.benign.size == 0:
            return Message(agent=agent, m=np.zeros(2), one_sided=False, u=1.0)
        q = q / (np.linalg.norm(q) or 1.0)
        sh, sn = self._topk_sims(self.harmful, q), self._topk_sims(self.benign, q)
        m_block = (_logsumexp(sh / self.tau) - _logsumexp(sn / self.tau)
                   - np.log(len(sh) / len(sn)))
        m = np.zeros(2)
        m[_B] = m_block
        # uncertainty: 1 when the nearest case in either bank is orthogonal or worse, 0 when identical
        nearest = max(float(sh.max()), float(sn.max()))
        u = float(np.clip(1.0 - nearest, 0.0, 1.0))
        return Message(agent=agent, m=m, one_sided=False, u=u)

    def save(self, path: str | Path) -> None:
        np.savez(path, harmful=self.harmful, benign=self.benign,
                 tau=np.array(self.tau), k=np.array(self.k))

    @classmethod
    def load(cls, path: str | Path) -> CaseBank:
        d = np.load(path)
        return cls(d["harmful"], d["benign"], float(d["tau"]), int(d["k"]))
