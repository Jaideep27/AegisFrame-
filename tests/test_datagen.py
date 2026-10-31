from collections import Counter

import pytest

from aegisframe.local.features import LABELS
from reframebench.datagen import intents as I
from reframebench.datagen.builder import generate
from reframebench.datagen.doclib import DOCS

TRAIN = generate("train", 1500, seed=3)
HELD = generate("heldout", 600, seed=4)


def test_spans_point_at_the_right_text():
    for e in TRAIN + HELD:
        for a, b, tag in e.spans:
            assert 0 <= a < b <= len(e.prompt), e.id
            assert e.prompt[a:b].strip(), (e.id, tag)


def test_labels_use_the_shared_vocabulary():
    for e in TRAIN + HELD:
        lab = e.labels
        assert lab["intent"] in LABELS["intent"]
        assert lab["decision"] in LABELS["decision"] and lab["decision_final"] in LABELS["decision"]
        for head in ("field_class", "pattern", "misuse", "benign_need"):
            assert lab[head] in LABELS[head], (head, lab[head])
        assert set(lab["twist"]) <= set(LABELS["twist"])
        assert {t for _, _, t in e.spans} <= {"FIELD", "OLD", "NEW", "MARK", "OBJ", "STORY"}


def test_the_story_never_reaches_the_canonical_instruction():
    for e in TRAIN:
        for a, b, tag in e.spans:
            if tag == "STORY":
                assert e.prompt[a:b] not in e.canonical


def test_a_story_can_only_make_the_final_label_stricter():
    order = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
    for e in TRAIN + HELD:
        assert order[e.labels["decision_final"]] >= order[e.labels["decision"]]


def test_heldout_uses_only_heldout_record_types_and_unseen_twists():
    held_docs = {d.key for d in DOCS if d.split == "heldout"}
    train_docs = {d.key for d in DOCS if d.split == "train"}
    assert {e.scene["doc"] for e in HELD if e.scene.get("doc")} <= held_docs
    assert {e.scene["doc"] for e in TRAIN if e.scene.get("doc")} <= train_docs
    held_twists = Counter(t for e in HELD for t in e.labels["twist"])
    assert held_twists["roleplay"] > 0 and held_twists["correction_claim"] > 0
    assert not any(t in I.HELDOUT_STORY_KINDS for e in TRAIN for t in e.labels["twist"])


def test_class_balance_is_reasonable():
    c = Counter(e.labels["decision_final"] for e in TRAIN)
    for d in ("ALLOW", "REVIEW", "BLOCK"):
        assert c[d] / len(TRAIN) > 0.08, c


def test_generation_is_deterministic():
    a = [e.prompt for e in generate("train", 50, seed=9)]
    b = [e.prompt for e in generate("train", 50, seed=9)]
    assert a == b


@pytest.mark.parametrize("intent", ["question", "cosmetic", "redaction", "fictional", "photo_visual", "marked_sample"])
def test_harmless_intents_are_allowed(intent):
    for e in TRAIN:
        if e.labels["intent"] == intent:
            assert e.labels["decision"] == "ALLOW", e.prompt


def test_every_example_has_a_reason():
    assert all(e.reason.split(":")[0] == e.labels["decision_final"] for e in TRAIN)


def test_a_story_escalates_every_review_except_vague_edits():
    for e in TRAIN + HELD:
        has_story = any(t == "STORY" for _, _, t in e.spans)
        if has_story and e.labels["decision"] == "REVIEW":
            expected = "REVIEW" if e.labels["intent"] == "unclear" else "BLOCK"
            assert e.labels["decision_final"] == expected, e.prompt
