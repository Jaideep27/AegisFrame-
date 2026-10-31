from collections import Counter

from aegisframe.schema import Action
from reframebench.cases import FRAMINGS, SCENARIOS, build_cases
from reframebench.metrics import CaseResult, compute_metrics


def test_benchmark_shape():
    cases = build_cases()
    assert len(SCENARIOS) == 20
    assert len(FRAMINGS) == 8
    assert len(cases) == 20 * 16
    per_scenario = Counter(c.scenario for c in cases)
    assert set(per_scenario.values()) == {16}
    kinds = Counter(c.kind for c in cases)
    assert kinds == {"plain": 20, "framed": 160, "multi_turn": 40, "retry": 20, "control": 80}


def test_every_case_has_an_expected_label_and_an_image_when_needed():
    for c in build_cases():
        assert c.expected in {"halt", "allow", "unlock"}
        if c.control != "fictional":
            assert c.image_ref is not None, c.id


def test_framed_cases_inherit_the_plain_label():
    cases = build_cases()
    plain = {c.scenario: c.expected for c in cases if c.kind == "plain"}
    for c in cases:
        if c.kind in {"framed", "multi_turn", "retry"}:
            assert c.expected == plain[c.scenario]


def test_case_ids_are_unique():
    ids = [c.id for c in build_cases()]
    assert len(ids) == len(set(ids))


def r(scenario, kind, expected, action, framing=None, control=None, split="dev"):
    return CaseResult(case_id=f"{scenario}-{kind}-{framing or control}", scenario=scenario, kind=kind,
                      framing=framing, control=control, split=split, expected=expected, action=action)


def test_metrics_on_known_results():
    results = [
        # scenario a: plain halted, one framing flips to allow, one holds
        r("a", "plain", "halt", Action.BLOCK),
        r("a", "framed", "halt", Action.ALLOW, framing="research"),
        r("a", "framed", "halt", Action.BLOCK, framing="legal"),
        # scenario b: plain allowed already (system failure), framed allowed: not a flip
        r("b", "plain", "halt", Action.ALLOW),
        r("b", "framed", "halt", Action.ALLOW, framing="research"),
        # controls
        r("a", "control", "allow", Action.ALLOW, control="harmless"),
        r("a", "control", "allow", Action.BLOCK, control="marked"),
        r("a", "control", "unlock", Action.REVIEW, control="verified_role"),
    ]
    m = compute_metrics(results)
    assert m["justification_flip_rate"] == 0.5  # 1 of the 2 framed cases whose plain was halted
    assert m["harmful_allow_rate"] == 3 / 5
    assert m["benign_block_rate"] == 0.5
    assert m["unlock_fidelity"] == 1.0
    assert m["monotonicity_violations"] == 1
