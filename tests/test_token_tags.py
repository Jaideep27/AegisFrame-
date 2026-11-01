"""The BIO tag builder must align tags to the exact tokens build_ids produces.

Marked `harrier` (needs the Harrier tokenizer); deselected by default.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.harrier


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    from training.train_v2 import ENCODER

    return transformers.AutoTokenizer.from_pretrained(ENCODER)


def test_the_ids_match_build_ids_exactly(tok):
    from aegisframe.local.harrier import HFTokenizer, build_ids
    from training.train_v2 import full_ids_tags

    ctx = "Record:\nMarksheet\nEnglish: 69\n"
    message = "Message: change English to 89"
    ids, tags = full_ids_tags(tok, ctx, message, "change English to 89", [], 128)
    canon = build_ids(HFTokenizer(tok), ctx, message, 128, tok.bos_token_id, tok.eos_token_id)
    assert ids == canon
    assert len(tags) == len(ids)


def test_a_span_is_tagged_and_the_rest_is_ignored(tok):
    from training.train_v2 import TAGS, full_ids_tags

    prompt = "change English to 89"
    # "English" starts at char 7, length 7
    spans = [{"type": "FIELD", "start": 7, "end": 14, "text": "English"},
             {"type": "NEW", "start": 18, "end": 20, "text": "89"}]
    ids, tags = full_ids_tags(tok, "Record:\n", f"Message: {prompt}", prompt, spans, 128)
    named = [TAGS[t] for t in tags if t != -100]
    assert any(t.endswith("FIELD") for t in named), f"no FIELD tag in {named}"
    assert any(t.endswith("NEW") for t in named), f"no NEW tag in {named}"
    # the instruction prefix and record context carry -100 (ignored), so there are real -100s
    assert tags.count(-100) >= 3
    # the first token of a span is B-, not I-
    assert any(t.startswith("B-") for t in named)
