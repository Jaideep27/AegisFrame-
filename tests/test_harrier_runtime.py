"""The v0.7 Harrier runtime: input building and the two-pass decision rule.

These are the pieces that must behave identically in training and in the shipped package, so they
live in aegisframe and the trainer imports them. No model is loaded here.
"""

from __future__ import annotations

import pytest

from aegisframe.local import harrier


class WordTokenizer:
    """One id per word, so budgets are readable in tests. ids are 1-based; 0 is unused."""

    def encode(self, text: str) -> list[int]:
        return [i + 1 for i, _ in enumerate(text.split())]


# ---------------------------------------------------------------- the decision rule
def test_the_raw_text_pass_can_raise_the_decision():
    assert harrier.final_decision("ALLOW", "BLOCK") == "BLOCK"
    assert harrier.final_decision("ALLOW", "REVIEW") == "REVIEW"
    assert harrier.final_decision("REVIEW", "BLOCK") == "BLOCK"


def test_the_raw_text_pass_can_never_lower_the_decision():
    assert harrier.final_decision("BLOCK", "ALLOW") == "BLOCK"
    assert harrier.final_decision("BLOCK", "REVIEW") == "BLOCK"
    assert harrier.final_decision("REVIEW", "ALLOW") == "REVIEW"


def test_an_unknown_decision_label_is_rejected_rather_than_silently_allowed():
    with pytest.raises(ValueError):
        harrier.final_decision("ALLOW", "probably fine")


# ---------------------------------------------------------------- input building
def test_a_long_message_keeps_its_end():
    """v0.6 truncated at 112 tokens from the left, which dropped the edit at the end of a long prompt."""
    tok = WordTokenizer()
    message = " ".join(f"w{i}" for i in range(400))
    ids = harrier.build_ids(tok, "", message, max_len=64, bos=101, eos=102)
    assert len(ids) <= 64
    # the last word of the message survives: its id is the 400th token of the message encoding
    assert ids[-1] == 102
    assert ids[-2] == 400


def test_the_record_context_is_cut_before_the_message():
    tok = WordTokenizer()
    context = " ".join(f"c{i}" for i in range(200))
    message = " ".join(f"m{i}" for i in range(20))
    ids = harrier.build_ids(tok, context, message, max_len=64, bos=101, eos=102)
    assert len(ids) <= 64
    # all 20 message words are present; the 200-word context lost words instead
    assert ids[-21:-1] == list(range(1, 21))


def test_the_message_still_leaves_room_for_the_record():
    """The message is capped so a huge prompt cannot squeeze the record out entirely."""
    tok = WordTokenizer()
    context = " ".join(f"c{i}" for i in range(50))
    message = " ".join(f"m{i}" for i in range(400))
    ids = harrier.build_ids(tok, context, message, max_len=64, bos=101, eos=102)
    body = ids[1 + len(tok.encode(harrier.query_prefix())):-1]
    assert len(body) == 64 - 2 - len(tok.encode(harrier.query_prefix()))
    assert len(body) > harrier.CONTEXT_RESERVE  # the message did not take the whole budget


def test_a_short_pair_keeps_everything_between_bos_and_eos():
    tok = WordTokenizer()
    ids = harrier.build_ids(tok, "a b", "c d e", max_len=64, bos=101, eos=102)
    assert ids[0] == 101
    assert ids[-1] == 102
    prefix = len(tok.encode(harrier.query_prefix()))
    assert len(ids) == 1 + prefix + 2 + 3 + 1


def test_the_instruction_prefix_states_the_task():
    """Harrier requires the Instruct/Query form; a wrong prefix silently costs accuracy."""
    prefix = harrier.query_prefix()
    assert prefix.startswith("Instruct: ")
    assert prefix.endswith("\nQuery: ")
    assert "falsified record" in prefix


# ---------------------------------------------------------------- training and serving agree
def test_the_package_builds_the_same_ids_as_the_trainer():
    """Train/serve skew in the tokenisation would quietly break every number we have measured."""
    train_v2 = pytest.importorskip("training.train_v2")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(train_v2.ENCODER)

    context = "Record:\nPayslip\nNet Pay: 42,000\n"
    message = "Message: please change Net Pay to 92,000"
    trainer_ids = train_v2.Encoder(tok, 384).ids(context, message)

    class HFWrapper:
        def encode(self, text: str) -> list[int]:
            return tok(text, add_special_tokens=False)["input_ids"]

    package_ids = harrier.build_ids(HFWrapper(), context, message, max_len=384,
                                    bos=tok.bos_token_id, eos=tok.eos_token_id)
    assert package_ids == trainer_ids


def test_the_trainer_uses_the_packages_decision_rule():
    """One definition of story monotonicity, not two that can drift apart."""
    train_v2 = pytest.importorskip("training.train_v2")
    assert train_v2.final_decision is harrier.final_decision
