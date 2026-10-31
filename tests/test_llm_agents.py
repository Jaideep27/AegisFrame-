import pytest

from aegisframe import AegisConfig, Guard
from aegisframe.config import ConfigError, CouncilConfig
from aegisframe.council import AgentContext, Belief, build_council
from aegisframe.council.llm_agents import AdvocateAgent, LLMPurposeAgent, PlainRequestAgent, RedTeamAgent
from aegisframe.providers.base import ProviderError
from aegisframe.providers.scripted import ScriptedProvider
from aegisframe.schema import Action, Claim, EditSpec, FieldChange, GroundedDoc, Severity
from aegisframe.policy import attributes

DOC = GroundedDoc(has_image=True, is_document=True, is_evidence=True, real_source=True, doc_type="payslip",
                  fields={"Net Pay": "Rs. 52,000"})
EDIT = EditSpec(field_changes=[FieldChange(kind="replace", field="Net Pay", new="Rs. 95,000")])


def ctx(claims_for_ablation=()):
    return AgentContext(edit=EDIT, doc=DOC, attrs=attributes(DOC, EDIT), resolved=[], severity=Severity.HIGH,
                        canonical="On this real payslip, change 'Net Pay' to 'Rs. 95,000'.", prior=0.1,
                        image=b"img", claims_for_ablation=tuple(claims_for_ablation))


@pytest.mark.parametrize("cls", [PlainRequestAgent, RedTeamAgent, AdvocateAgent])
def test_llm_agents_return_beliefs_from_json(cls):
    provider = ScriptedProvider([{"p_harm": 0.9, "confidence": 0.7, "reason_code": "forged-payslip"}])
    belief = cls(provider=provider).assess(ctx())
    assert belief == Belief(agent=cls.name, p_harm=0.9, confidence=0.7, reason_code="forged-payslip")


@pytest.mark.parametrize("cls", [PlainRequestAgent, RedTeamAgent, AdvocateAgent])
def test_llm_agents_see_only_the_canonical_instruction(cls):
    provider = ScriptedProvider([{"p_harm": 0.5}])
    cls(provider=provider).assess(ctx())
    system, user, image = provider.calls[0]
    assert "Net Pay" in user and "research" not in user.lower()
    assert image == b"img"


def test_ablation_puts_the_story_into_the_prompt():
    provider = ScriptedProvider([{"p_harm": 0.5}])
    PlainRequestAgent(provider=provider).assess(ctx([Claim(text="for my research paper", kind="purpose")]))
    assert "for my research paper" in provider.calls[0][1]


def test_llm_agent_reply_is_validated_and_clamped():
    belief = RedTeamAgent(provider=ScriptedProvider([{"p_harm": 3, "confidence": -1}])).assess(ctx())
    assert belief.p_harm == 1.0 and belief.confidence == 0.0
    with pytest.raises(ProviderError):
        RedTeamAgent(provider=ScriptedProvider([{"verdict": "bad"}])).assess(ctx())


def test_llm_agent_updates_after_reading_peer_reason_codes():
    provider = ScriptedProvider([{"p_harm": 0.2, "reason_code": "looks-benign"},
                                 {"p_harm": 0.6, "reason_code": "concede-forgery"}])
    agent = AdvocateAgent(provider=provider)
    own = agent.assess(ctx())
    peers = [Belief(agent="fraud_patterns", p_harm=0.92, confidence=0.8, reason_code="amount-inflation")]
    updated = agent.update(ctx(), own, peers, openness=0.3)
    assert updated.p_harm == 0.6 and updated.round == 1
    assert "amount-inflation" in provider.calls[1][1]


def test_llm_purpose_agent_reads_claims_and_cannot_go_below_the_prior():
    provider = ScriptedProvider([{"p_harm": 0.01, "confidence": 1.0, "reason_code": "seems-legit"}])
    belief = LLMPurposeAgent(provider=provider).assess_claims(ctx(), [Claim(text="for research", kind="purpose")])
    assert belief.p_harm >= 0.1
    assert "for research" in provider.calls[0][1]


def test_council_config_wires_llm_agents_to_named_providers():
    provider = ScriptedProvider(lambda s, u, i: {"p_harm": 0.95, "confidence": 0.9, "reason_code": "x"})
    cfg = CouncilConfig(agents=[{"name": "plain_request", "weight": 1.5, "provider": "p1"}],
                        purpose={"provider": "p1"})
    council = build_council(cfg, providers={"p1": provider})
    assert council.deliberate(ctx()).action == Action.BLOCK


def test_llm_agent_without_provider_is_a_config_error():
    with pytest.raises(ConfigError):
        build_council(CouncilConfig(agents=[{"name": "red_team"}]), providers={})


def test_guard_with_llm_council_end_to_end():
    from reframebench.docs import CATALOG, OracleOcr, layout_lines, render, to_png

    ocr = OracleOcr()
    slip = ocr.register(to_png(render(CATALOG["salary_slip"])), layout_lines(CATALOG["salary_slip"]))
    provider = ScriptedProvider(lambda s, u, i: {"p_harm": 0.9, "confidence": 0.8, "reason_code": "forgery"})
    cfg = AegisConfig.model_validate({"council": {"agents": [{"name": "plain_request", "provider": "p"}]}})
    guard = Guard(ocr=ocr, config=cfg, providers={"p": provider})
    assert guard.check_request("Raise the Net Pay to 95,000.", image=slip).action == Action.BLOCK
