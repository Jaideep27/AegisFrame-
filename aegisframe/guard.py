"""The Guard: check_request() before generation, check_output() after.

    separate -> ground -> floor -> decide -> (generate) -> verify

Justification monotonicity holds by construction: claims never reach the floor;
they can only escalate (reframing signal), and only a verified identity from the
company's auth layer can lower a decision.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audit import AuditLog, decision_record
from .config import AegisConfig
from .errors import Blocked, OutputRejected, ReviewRequired
from .floor import Judge, compute_floor
from .ground import OcrEngine, RapidOcrEngine, ground
from .identity import VerifiedIdentity, apply_unlock
from .memory import SessionMemory, image_key
from .policy import Policy, attributes
from .schema import Action, Decision, OcrLine, Separation
from .separate import AgreementSeparator, LLMSeparator, RuleSeparator, Separator
from .verify import VerifyResult, verify_output


@dataclass
class GuardedResult:
    image: bytes
    decision: Decision
    verification: VerifyResult | None


class NullOcr:
    """grounding.ocr = none: every image is treated as a photo (records are not recognised)."""

    def read(self, image: bytes) -> list[OcrLine]:
        return []


class Guard:
    def __init__(
        self,
        policy: Policy | None = None,
        separator: Separator | None = None,
        judges: list[Judge] | tuple[Judge, ...] = (),
        ocr: OcrEngine | None = None,
        audit: AuditLog | None = None,
        config: AegisConfig | None = None,
        providers: dict | None = None,
        council=None,
        reviews=None,
        sessions=None,
        local_model=None,
    ):
        self.config = config or AegisConfig.load()  # defaults + AEGIS__* environment overrides
        self.local_model = local_model if local_model is not None else _load_local(self.config)
        providers = dict(providers or {})
        if self.local_model is not None:
            providers.setdefault("local", self.local_model)
        if council is None and self.config.council.enabled:
            from .council import build_council

            council = build_council(_council_config(self.config, self.local_model), providers)
        self.council = council
        self.policy = policy or (
            Policy.from_file(self.config.policy_file) if self.config.policy_file else Policy.default()
        )
        if separator is None:
            if self.local_model is not None:
                from .local.separator import ConfidentSeparator, LocalSeparator

                separator = ConfidentSeparator(LocalSeparator(self.local_model), RuleSeparator(),
                                               threshold=self.config.local_model.trust_threshold)
            else:
                separator = RuleSeparator()
        self.separator = separator
        self.judges = list(judges)
        self.ocr = ocr or (NullOcr() if self.config.grounding.ocr == "none" else RapidOcrEngine())
        self.audit = audit or _build_audit(self.config)
        self.sessions = sessions or _build_sessions(self.config)
        self.reviews = reviews or _build_reviews(self.config)
        self._callbacks: list[Callable[[Decision], None]] = []

    # ---- construction -------------------------------------------------------------

    @classmethod
    def from_config(cls, config: AegisConfig | str | Path | None = None, **overrides) -> Guard:
        """Build a guard from a config object or YAML path. Keyword overrides win (useful in tests)."""
        cfg = config if isinstance(config, AegisConfig) else AegisConfig.load(config)
        from .providers.factory import build_providers

        providers = build_providers(cfg)
        if "separator" not in overrides:
            sep = _build_separator(cfg, providers)
            if sep is not None:
                overrides["separator"] = sep
        return cls(config=cfg, providers=providers, **overrides)

    @classmethod
    def from_env(cls, policy_file: str | None = None, **kw) -> Guard:
        """Wire up whichever model providers have keys in the environment; fall back to offline rules."""
        import os

        from .providers.gemini import GeminiProvider
        from .providers.openai_compat import OpenAICompatProvider

        providers = []
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            providers.append(GeminiProvider())
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("AEGIS_OPENAI_BASE_URL"):
            providers.append(OpenAICompatProvider())
        separators: list[Separator] = [LLMSeparator(p) for p in providers] + [RuleSeparator()]
        separator = AgreementSeparator(*separators) if len(separators) > 1 else separators[0]
        policy = Policy.from_file(policy_file) if policy_file else None
        from .floor import LLMJudge

        return cls(policy=policy, separator=separator, judges=[LLMJudge(p) for p in providers], **kw)

    def on_decision(self, callback: Callable[[Decision], None]) -> Callable[[Decision], None]:
        """Register a callback (metrics, alerts, review webhooks). Usable as a decorator."""
        self._callbacks.append(callback)
        return callback

    def _memory(self, session_id: str) -> SessionMemory:
        return self.sessions.get(session_id)

    # ---- request check ------------------------------------------------------------

    def check_request(
        self,
        prompt: str,
        image: bytes | None = None,
        history: list[str] | None = None,
        verified_identity: VerifiedIdentity | None = None,
        session_id: str = "default",
        approval_token: str | None = None,
    ) -> Decision:
        request_id = "req_" + uuid.uuid4().hex[:10]
        try:
            decision = self._decide(prompt, image, history, verified_identity, session_id, request_id,
                                    approval_token)
            if decision.action == Action.REVIEW and self.reviews is not None:
                record = decision_record(decision, prompt, self.config.privacy.store_claims)
                item = self.reviews.enqueue(decision, claims=record["claims"])
                decision = decision.model_copy(update={"review_id": item.id})
        except Exception as exc:
            if self.config.decision.fail_mode == "open":
                decision = Decision(action=Action.ALLOW, rule="guard-error-fail-open", request_id=request_id,
                                    generator_prompt=prompt,
                                    reasons=[f"guard error, failing OPEN by configuration: {type(exc).__name__}: {exc}"])
            else:  # a guard that errors must not let the request through
                decision = Decision(action=Action.BLOCK, rule="guard-error", request_id=request_id,
                                    reasons=[f"guard error, failing closed: {type(exc).__name__}: {exc}"])
        self.audit.write(decision_record(decision, prompt, self.config.privacy.store_claims))
        for callback in self._callbacks:
            callback(decision)
        return decision

    def _decide(self, prompt, image, history, identity, session_id, request_id, approval_token=None) -> Decision:
        doc = ground(image, self.ocr)
        from .local.features import context_for
        from .separate import _call

        separation: Separation = _call(self.separator, prompt, history, context_for(doc))
        memory = self._memory(session_id)
        key = image_key(image) if image is not None else None

        edit = memory.cumulative(key, separation.edit) if key else separation.edit
        floor = compute_floor(edit, doc, self.policy, self.judges, image)
        rules_action = floor.action
        reasons = list(floor.reasons)

        # the council may raise the rules' floor, never lower it; its deliberation is claim-blind
        # and the story enters only through the one-sided purpose term
        verdict = None
        plain_action = floor.action
        if self.council is not None:
            verdict = self.council.deliberate(self._agent_context(edit, doc, floor, image, memory, key, separation),
                                              separation.claims)
            plain_action = max(floor.action, self.council._action(verdict.deliberation_posterior))
            if verdict.vetoed:
                plain_action = Action.BLOCK
            if verdict.action > floor.action:
                reasons.append(f"council posterior {verdict.posterior:.2f} raised the decision to {verdict.action.name}")
                floor = floor.model_copy(update={"action": verdict.action, "rule": floor.rule or "council"})

        action, unlock_reason = apply_unlock(floor, identity, self.policy)
        if unlock_reason:
            reasons.append(unlock_reason)

        # a story = a recognised justification; unclassified leftover text (often an edit the
        # separator could not parse) is kept in the ledger but does not count as one
        reframing = plain_action != Action.ALLOW and any(c.kind != "other" for c in separation.claims)
        retry = bool(key) and plain_action != Action.ALLOW and memory.halted_before(key)
        if not unlock_reason and action == Action.REVIEW and self.policy.escalates(floor.rule):
            if reframing and self.policy.reframing_escalates:
                action = Action.BLOCK
                reasons.append("reframing signal: a justification was attached to an edit that is not allowed in plain form")
            elif retry:
                action = Action.BLOCK
                reasons.append("retry after an earlier halt on the same image")

        # a human-approved review token unlocks exactly the approved edit, once; never a hard block
        approved = False
        if approval_token and self.reviews is not None and action != Action.ALLOW and rules_action <= Action.REVIEW:
            if self.reviews.redeem(approval_token, floor.canonical_instruction, key):
                action, approved = Action.ALLOW, True
                reasons.append("approved by a human reviewer")

        if key:
            memory.record(key, separation.edit, action)

        generator_prompt = ""
        if action == Action.ALLOW:
            # on the record path the image model only ever sees the canonical instruction
            generator_prompt = floor.canonical_instruction if (doc.is_document or edit.from_scratch) else prompt

        return Decision(
            action=action,
            rule=floor.rule,
            severity=floor.severity,
            reasons=reasons,
            canonical_instruction=floor.canonical_instruction,
            generator_prompt=generator_prompt,
            edit=edit,
            claims=separation.claims,
            floor=floor,
            council=verdict,
            signals={
                "reframing": reframing,
                "retry_after_halt": retry,
                "verified_unlock": bool(unlock_reason),
                "separator_disagreement": not separation.agreed,
                "human_approved": approved,
            },
            image_key=key,
            request_id=request_id,
        )

    def _agent_context(self, edit, doc, floor, image, memory, key, separation):
        from .council import AgentContext

        return AgentContext(
            edit=edit, doc=doc, attrs=attributes(doc, edit), resolved=floor.resolved, severity=floor.severity,
            canonical=floor.canonical_instruction, prior=self.council.prior, image=image,
            halted_before=memory.halted_before(key) if key else False,
            prior_turns=memory.turns(key) if key else 0,
            claims_for_ablation=() if self.config.decision.claim_blind else tuple(separation.claims),
        )

    # ---- output check -------------------------------------------------------------

    def check_output(self, input_image: bytes | None, output_image: bytes, request_decision: Decision) -> VerifyResult:
        try:
            return verify_output(input_image, output_image, request_decision, self.ocr)
        except Exception as exc:
            return VerifyResult(action=Action.BLOCK, reasons=[f"verify error, failing closed: {exc}"])

    # ---- protected calls -----------------------------------------------------------

    def run(
        self,
        prompt: str,
        image: bytes | None,
        generator,
        identity: VerifiedIdentity | None = None,
        session_id: str = "default",
        approval_token: str | None = None,
        **generator_kwargs,
    ) -> GuardedResult:
        """check -> generate (with the safe prompt) -> verify. Raises Blocked / ReviewRequired / OutputRejected."""
        decision = self.check_request(prompt, image=image, verified_identity=identity, session_id=session_id,
                                      approval_token=approval_token)
        decision = self.enforce(decision)
        send = decision.generator_prompt or decision.canonical_instruction or prompt
        call = getattr(generator, "generate", generator)
        output = call(send, image, **generator_kwargs)
        verification = None
        if self.config.verify.enabled:
            verification = self.check_output(image, output, decision)
            if verification.action != Action.ALLOW:
                raise OutputRejected(verification, decision)
        return GuardedResult(image=output, decision=decision, verification=verification)

    def enforce(self, decision: Decision) -> Decision:
        """Raise Blocked / ReviewRequired per config, or return the decision to generate with."""
        if decision.action == Action.BLOCK:
            raise Blocked(decision)
        if decision.action == Action.REVIEW:
            mode = self.config.review.on_review
            if mode == "block":
                raise Blocked(decision)
            if mode == "raise":
                raise ReviewRequired(decision, review_id=decision.review_id)
            # "allow": generate anyway (testing only); verify against the edit as if approved
            decision = decision.model_copy(update={"action": Action.ALLOW})
        return decision

    def protect(self, fn: Callable) -> Callable:
        """Decorator for any `fn(prompt, image=None, **kw) -> bytes` image generator.

        The wrapped function accepts extra keyword arguments:
        aegis_identity (VerifiedIdentity) and aegis_session (str).
        """

        @functools.wraps(fn)
        def wrapper(prompt: str, image: bytes | None = None, *, aegis_identity=None, aegis_session="default",
                    aegis_approval=None, **kw):
            return self.run(prompt, image, fn, identity=aegis_identity, session_id=aegis_session,
                            approval_token=aegis_approval, **kw).image

        return wrapper

    # ---- async ---------------------------------------------------------------------

    async def acheck_request(self, prompt: str, image: bytes | None = None, **kw) -> Decision:
        return await asyncio.to_thread(self.check_request, prompt, image, **kw)

    async def acheck_output(self, input_image: bytes | None, output_image: bytes, request_decision: Decision) -> VerifyResult:
        return await asyncio.to_thread(self.check_output, input_image, output_image, request_decision)


def _load_local(cfg: AegisConfig):
    mode = cfg.local_model.enabled
    if mode == "off":
        return None
    from .local.runtime import LocalModel

    if mode == "auto" and not LocalModel.available(cfg.local_model.path):
        return None
    return LocalModel.load(cfg.local_model.path, quantized=cfg.local_model.quantized, threads=cfg.local_model.threads)


def _council_config(cfg: AegisConfig, local_model):
    """With a local model and the default agent list, add the local agents and the local purpose agent."""
    from .config import AgentConfig, CouncilConfig

    cc = cfg.council
    if local_model is None:
        return cc
    if cc.agents == CouncilConfig().agents:
        from .local.agents import LOCAL_AGENTS

        cc = cc.model_copy(update={"agents": [*cc.agents, *(AgentConfig(**a) for a in LOCAL_AGENTS)]})
    if cc.purpose.provider is None:
        cc = cc.model_copy(update={"purpose": cc.purpose.model_copy(update={"provider": "local"})})
    return cc


def _build_separator(cfg: AegisConfig, providers: dict) -> Separator | None:
    if cfg.separator.mode == "rules":
        return None  # the Guard picks: local model + rules if a local model is available, else rules
    llm = [LLMSeparator(providers[name]) for name in cfg.separator.providers]
    if not llm:
        from .config import ConfigError

        raise ConfigError(f"separator.mode={cfg.separator.mode} needs separator.providers")
    if cfg.separator.mode == "llm":
        return llm[0] if len(llm) == 1 else AgreementSeparator(*llm)
    return AgreementSeparator(*llm, RuleSeparator())


def _build_audit(cfg: AegisConfig) -> AuditLog:
    if cfg.audit.sink == "sqlite":
        from .stores import SqliteAuditLog

        return SqliteAuditLog(cfg.audit.path or "aegis_audit.db")
    return AuditLog(cfg.audit.path if cfg.audit.sink == "jsonl" else None)


def _build_sessions(cfg: AegisConfig):
    from .stores import InMemorySessionStore, SqliteSessionStore

    if cfg.memory.backend == "sqlite":
        return SqliteSessionStore(cfg.memory.path or "aegis_sessions.db")
    return InMemorySessionStore()


def _build_reviews(cfg: AegisConfig):
    if not cfg.review.queue_path:
        return None
    from .review import ReviewQueue

    hook = None
    if cfg.review.webhook_url:
        url = cfg.review.webhook_url

        def hook(item):
            import dataclasses

            import httpx

            try:
                httpx.post(url, json=dataclasses.asdict(item), timeout=5)
            except httpx.HTTPError:
                pass  # a webhook failure must not break the request path

    return ReviewQueue(cfg.review.queue_path, on_enqueue=hook, secret_env=cfg.review.secret_env)
