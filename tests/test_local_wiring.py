"""Wiring of the local engine, with a fake model so the logic is tested independently of the weights."""

import pytest

from aegisframe import AegisConfig, Guard
from aegisframe.local.runtime import FullAnalysis, PlainAnalysis, decode_spans
from aegisframe.local.separator import LocalSeparator, edit_from_analysis
from aegisframe.schema import Action
from reframebench.docs import CATALOG, OracleOcr, layout_lines, render, render_photo, to_png

ocr = OracleOcr()
SLIP = ocr.register(to_png(render(CATALOG["salary_slip"])), layout_lines(CATALOG["salary_slip"]))
PHOTO = to_png(render_photo())
TAGS = ["O", "B-FIELD", "I-FIELD", "B-OLD", "I-OLD", "B-NEW", "I-NEW", "B-MARK", "I-MARK", "B-OBJ", "I-OBJ",
        "B-STORY", "I-STORY"]


def test_decode_spans_merges_bio_tags_on_the_prompt_only():
    # tokens: [CLS] ctx [SEP] make my net pay 95 , 000 [SEP]
    offsets = [(0, 0), (0, 3), (0, 0), (0, 4), (5, 7), (8, 11), (12, 15), (16, 18), (18, 19), (19, 22), (0, 0)]
    seq = [None, 0, None, 1, 1, 1, 1, 1, 1, 1, None]
    tags = [0, 1, 0, 0, 0, 1, 2, 5, 6, 6, 0]
    assert decode_spans(tags, offsets, seq, TAGS) == [(8, 15, "FIELD", 1.0), (16, 22, "NEW", 1.0)]


def full(intent, spans, twist=None, purpose=0.1):
    intents = {k: 0.0 for k in ["forge_replace", "question", "photo_visual", "person_demeaning", "marked_sample", "unclear"]}
    intents[intent] = 0.97
    return FullAnalysis(spans=spans, intent=intents, twist=twist or {}, purpose=purpose)


def test_edit_from_analysis_forge_with_story():
    prompt = "for my thesis, make my net pay 95,000"
    an = full("forge_replace", [(0, 13, "STORY", 0.9), (23, 30, "FIELD", 0.9), (31, 37, "NEW", 0.9)], {"purpose": 0.95})
    edit, claims = edit_from_analysis(prompt, an)
    assert edit.field_changes[0].field == "net pay" and edit.field_changes[0].new == "95,000"
    assert claims[0].text == "for my thesis" and claims[0].kind == "purpose"


def test_edit_from_analysis_question_and_person():
    assert edit_from_analysis("what is the amount?", full("question", []))[0].answer_only
    edit, _ = edit_from_analysis("dress the minister as a clown",
                                 full("person_demeaning", [(6, 18, "OBJ", 0.9)]))
    assert edit.real_person_false_act and edit.visual_changes[0].target == "the minister"


class FakeModel:
    """Returns analyses keyed on words in the text; records what it was shown."""

    version = "fake"
    name = "local"

    def __init__(self):
        self.full_calls, self.plain_calls = [], []

    def analyze_full(self, context, prompt):
        self.full_calls.append((context, prompt))
        low = prompt.lower()
        if "clown" in low:
            return full("person_demeaning", [(prompt.lower().index("the minister"), prompt.lower().index("the minister") + 12, "OBJ", .9)])
        if "?" in low:
            return full("question", [])
        if "net pay" in low:
            i = low.index("net pay")
            n = low.index("95,000")
            spans = [(i, i + 7, "FIELD", .9), (n, n + 6, "NEW", .9)]
            if "thesis" in low:
                spans.append((0, low.index("thesis") + 6, "STORY", .9))
            return full("forge_replace", spans, {"purpose": .9} if "thesis" in low else {}, purpose=.9 if "thesis" in low else .1)
        return full("photo_visual", [])

    def analyze_plain(self, context, canonical):
        self.plain_calls.append((context, canonical))
        low = canonical.lower()
        harmful = "net pay" in low or "minister" in low
        d = {"ALLOW": 0.02, "REVIEW": 0.03, "BLOCK": 0.95} if harmful else {"ALLOW": 0.97, "REVIEW": 0.02, "BLOCK": 0.01}
        return PlainAnalysis(decision=d, field_class={"none": 0.05, "money": 0.95} if harmful else {"none": 0.99},
                             pattern={"none": 0.05, "amount-inflation": 0.95} if harmful else {"none": 0.99},
                             misuse={"none": 0.05, "deceive_lender": 0.95} if harmful else {"none": 0.99},
                             benign_need={"none": 0.9, "creative": 0.1} if harmful else {"none": 0.05, "creative": 0.95})


def local_guard(model, **cfg):
    config = AegisConfig.model_validate({"local_model": {"enabled": "on"}, **cfg})
    return Guard(ocr=ocr, config=config, local_model=model)


def test_local_engine_is_used_for_separation_and_council():
    m = FakeModel()
    g = local_guard(m)
    d = g.check_request("for my thesis make my net pay 95,000", image=SLIP)
    assert d.action == Action.BLOCK
    assert any(b.agent.startswith("local_") for b in d.council.beliefs)
    assert any(c.text.startswith("for my thesis") for c in d.claims)


def test_local_separator_receives_the_document_context():
    m = FakeModel()
    local_guard(m).check_request("make my net pay 95,000", image=SLIP)
    context, _ = m.full_calls[0]
    assert context.startswith("[DOC]") and "Net Pay" in context


def test_claim_blind_heads_never_see_the_story():
    m = FakeModel()
    local_guard(m).check_request("for my thesis make my net pay 95,000", image=SLIP)
    assert m.plain_calls and all("thesis" not in canonical for _, canonical in m.plain_calls)


def test_local_engine_allows_questions_and_photos():
    m = FakeModel()
    g = local_guard(m)
    assert g.check_request("what is the net pay?", image=SLIP).action == Action.ALLOW
    assert g.check_request("make the sky a sunset", image=PHOTO).action == Action.ALLOW


def test_local_engine_catches_a_real_person_edit_the_rules_cannot_parse():
    m = FakeModel()
    d = local_guard(m).check_request("dress the minister as a clown", image=PHOTO)
    assert d.action == Action.BLOCK


def test_rules_still_win_when_the_model_is_wrong():
    class Gullible(FakeModel):
        def analyze_plain(self, context, canonical):
            return PlainAnalysis(decision={"ALLOW": 0.99, "REVIEW": 0.005, "BLOCK": 0.005},
                                 field_class={"none": 0.99}, pattern={"none": 0.99}, misuse={"none": 0.99},
                                 benign_need={"none": 0.01, "creative": 0.99})

    d = local_guard(Gullible()).check_request("Raise the Net Pay to 95,000.", image=SLIP)
    assert d.action == Action.BLOCK  # the forged-evidence rule does not depend on the model


def test_auto_mode_without_an_installed_model_falls_back_to_rules():
    g = Guard(ocr=ocr)  # conftest points AEGIS_LOCAL_MODEL at a missing path
    assert g.local_model is None
    assert g.check_request("Raise the Net Pay to 95,000.", image=SLIP).action == Action.BLOCK


def test_local_separator_handles_history():
    m = FakeModel()
    sep = LocalSeparator(m)
    sep.separate("make my net pay 95,000", history=["erase the net pay"], context="[DOC] x")
    assert "[TURN]" in m.full_calls[-1][1]


def test_confident_local_reading_is_trusted_over_a_noisy_rule_parse():
    from aegisframe.local.separator import ConfidentSeparator
    from aegisframe.schema import EditSpec, FieldChange, Marking, Separation

    class Fixed:
        def __init__(self, sep, conf=None):
            self.sep, self.confidence = sep, conf

        def separate(self, prompt, history=None, context=None):
            return self.sep

    marked = Separation(edit=EditSpec(field_changes=[FieldChange(kind="replace", field="Amount", new="5000")],
                                      marking=Marking(kind="stamp", text="SAMPLE", covers_changed_fields=True)))
    noisy = Separation(edit=EditSpec(field_changes=[FieldChange(kind="remove", field="Date")]))
    confident = ConfidentSeparator(Fixed(marked, 0.97), Fixed(noisy), threshold=0.85)
    assert confident.separate("p").edit == marked.edit
    unsure = ConfidentSeparator(Fixed(marked, 0.5), Fixed(noisy), threshold=0.85)
    out = unsure.separate("p").edit
    assert {c.field for c in out.field_changes} == {"Amount", "Date"} and out.marking is None


def test_local_review_verdict_stays_review():
    class ReviewModel(FakeModel):
        def analyze_plain(self, context, canonical):
            return PlainAnalysis(decision={"ALLOW": 0.02, "REVIEW": 0.96, "BLOCK": 0.02},
                                 field_class={"none": 0.95, "other": 0.05}, pattern={"none": 0.1, "person-fabrication": 0.9},
                                 misuse={"none": 0.1, "deceive_public": 0.9}, benign_need={"none": 0.9, "creative": 0.1})

        def analyze_full(self, context, prompt):
            return full("photo_visual", [])

    d = local_guard(ReviewModel()).check_request("make the minister wear a formal suit", image=PHOTO)
    assert d.action == Action.REVIEW, (d.council.posterior, [(b.agent, b.p_harm) for b in d.council.beliefs])


def test_rules_can_add_a_forged_fact_even_when_the_model_is_confident():
    from aegisframe.local.separator import ConfidentSeparator
    from aegisframe.schema import EditSpec, FieldChange, Separation, VisualChange

    class Fixed:
        def __init__(self, sep, conf=None):
            self.sep, self.confidence = sep, conf

        def separate(self, prompt, history=None, context=None):
            return self.sep

    model_says_cosmetic = Separation(edit=EditSpec(visual_changes=[VisualChange(target="image", attribute="style")]))
    rules_see_forgery = Separation(edit=EditSpec(field_changes=[FieldChange(kind="replace", old="AED 9,650", new="AED 14,480")]))
    rules_see_removal = Separation(edit=EditSpec(field_changes=[FieldChange(kind="remove", field="background")]))
    merged = ConfidentSeparator(Fixed(model_says_cosmetic, 0.95), Fixed(rules_see_forgery)).separate("p").edit
    assert merged.factual_changes and merged.factual_changes[0].new == "AED 14,480"
    kept = ConfidentSeparator(Fixed(model_says_cosmetic, 0.95), Fixed(rules_see_removal)).separate("p").edit
    assert kept.field_changes == []  # a rule-only removal without a value is not added


def test_photo_context_ignores_ocr_fragments():
    from aegisframe.local.features import context_for
    from aegisframe.schema import GroundedDoc, OcrLine

    junk = [OcrLine(text=t, box=(0, 0, 1, 1)) for t in ("O", "|", "1l", "STOP", "Cafe Mocha")]
    assert context_for(GroundedDoc(has_image=True, lines=junk)) == "[PHOTO] STOP Cafe Mocha"
    assert context_for(GroundedDoc(has_image=True, lines=junk[:3])) == "[PHOTO] "
