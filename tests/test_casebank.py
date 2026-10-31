"""The case bank: how close is this request to known misuse vs known legitimate edits?

It turns the two nearest-neighbour densities, in the contrastive embedding space, into a BLOCK-vs-ALLOW
LLR message. No model here - the embeddings are given directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegisframe.council.casebank import CaseBank
from aegisframe.council.fusion import DECISION

B = DECISION.index("BLOCK") - 1  # index of the BLOCK component in a message vector


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


HARMFUL = np.stack([unit(1, 0), unit(0.9, 0.1), unit(0.95, -0.05)])
BENIGN = np.stack([unit(0, 1), unit(0.1, 0.9), unit(-0.05, 0.95)])


def test_a_request_like_known_misuse_sends_a_positive_block_message():
    bank = CaseBank(HARMFUL, BENIGN, tau=0.1, k=2)
    m = bank.message("casebank", unit(1, 0))
    assert m.m[B] > 0


def test_a_request_like_a_known_legitimate_edit_sends_a_negative_block_message():
    bank = CaseBank(HARMFUL, BENIGN, tau=0.1, k=2)
    m = bank.message("casebank", unit(0, 1))
    assert m.m[B] < 0


def test_a_request_equidistant_from_both_banks_is_near_zero():
    bank = CaseBank(HARMFUL, BENIGN, tau=0.3, k=3)
    m = bank.message("casebank", unit(1, 1))
    assert abs(m.m[B]) < 0.5


def test_the_message_says_nothing_about_review():
    """The bank separates harmful from legitimate; it has no opinion on the grey REVIEW class."""
    bank = CaseBank(HARMFUL, BENIGN, tau=0.1, k=2)
    m = bank.message("casebank", unit(1, 0))
    assert m.m[DECISION.index("REVIEW") - 1] == 0.0


def test_a_far_away_request_is_marked_uncertain():
    """A query far from every stored case gets high uncertainty, so fusion can discount it."""
    bank = CaseBank(HARMFUL, BENIGN, tau=0.1, k=2)
    near = bank.message("casebank", unit(1, 0))
    far = bank.message("casebank", unit(-1, -1))
    assert far.u > near.u


def test_an_empty_bank_is_neutral():
    bank = CaseBank(np.empty((0, 2)), np.empty((0, 2)), tau=0.1, k=2)
    m = bank.message("casebank", unit(1, 0))
    assert np.allclose(m.m, 0.0)
    assert m.u == pytest.approx(1.0)


def test_k_is_clamped_to_the_bank_size():
    bank = CaseBank(HARMFUL, BENIGN, tau=0.1, k=50)
    m = bank.message("casebank", unit(1, 0))
    assert np.isfinite(m.m[B])


def test_the_message_is_two_sided():
    bank = CaseBank(HARMFUL, BENIGN, tau=0.1, k=2)
    assert bank.message("casebank", unit(1, 0)).one_sided is False


def test_a_bank_round_trips_through_a_file(tmp_path):
    bank = CaseBank(HARMFUL, BENIGN, tau=0.2, k=3)
    path = tmp_path / "bank.npz"
    bank.save(path)
    again = CaseBank.load(path)
    a = bank.message("casebank", unit(1, 0)).m
    b = again.message("casebank", unit(1, 0)).m
    assert np.allclose(a, b)
    assert again.tau == 0.2 and again.k == 3
