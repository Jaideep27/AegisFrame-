"""The v0.7 blackboard: every agent reads its own view, writes an LLR message, and the pool fuses them.

The agents, and why each is one-sided or two-sided:

  plain_judge        two-sided   the story-blind decision head - the backbone reading
  sentinel           one-sided   the raw-message decision head; may only raise (Theorem 1)
  value_diff         one-sided   deterministic change detector; finding a change never lowers risk
  output_authenticity one-sided  raises when the result would pass as genuine (distinguishable = "no")
  purpose_auditor    one-sided   raises when a justifying story is present at all
  casebank           two-sided   nearness to known misuse vs known legitimate edits (needs embeddings)
  history            one-sided   raises when this looks like a retry after a refusal

Only two-sided agents can argue for ALLOW; every one-sided agent can only push toward REVIEW/BLOCK, so no
story and no single heuristic can talk the pool into allowing a request (Theorems 1 and 2 in fusion.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..local.harrier import HarrierAnalysis
from ..local.labels import HARMFUL_INTENTS
from .casebank import CaseBank
from .fusion import DECISION, Fused, Message, fuse, message_from_probs

_RISK = {d: i for i, d in enumerate(DECISION)}


def _stricter(a: str, b: str) -> str:
    return a if _RISK[a] >= _RISK[b] else b
from .valuediff import ValueDiff, detect_value_changes

#: default class prior - roughly the training mix (lots of BLOCK and ALLOW, little REVIEW)
DEFAULT_PRIOR = {"ALLOW": 0.45, "REVIEW": 0.10, "BLOCK": 0.45}

#: modest fixed LLR strengths for the one-sided heuristic raises (calibrated later by train_fusion)
VALUE_DIFF_RAISE = 2.5
INTENT_RAISE = 3.0        # the intent head reads a forging intent
MARKING_RAISE = 6.0       # remove_marking: the plan's rule that this can never end in ALLOW
AUTH_RAISE = 2.0
PURPOSE_RAISE = 0.8
HISTORY_RAISE = 1.5

_REFUSAL = re.compile(r"\b(can'?t|cannot|won'?t|unable|not able|i refuse|i'?m sorry)\b", re.I)
_B = DECISION.index("BLOCK") - 1


def _raise(agent: str, strength: float, note: str) -> Message:
    m = np.zeros(2)
    m[_B] = strength
    msg = Message(agent=agent, m=m, one_sided=True)
    msg.note = note  # type: ignore[attr-defined]
    return msg


@dataclass
class Verdict:
    decision: str
    p: dict[str, float]
    analysis: HarrierAnalysis
    value_changes: list[ValueDiff]
    messages: list[Message]
    fused: Fused
    trace: list[dict] = field(default_factory=list)


class Blackboard:
    def __init__(self, model, casebank: CaseBank | None = None, prior: dict[str, float] | None = None,
                 weights: dict[str, float] | None = None, f: int = 1):
        self.model = model
        self.casebank = casebank
        self.prior = prior or DEFAULT_PRIOR
        self.weights = weights or {}
        self.f = f

    def assess(self, context: str, prompt: str, plain: str | None = None,
               history: list[str] | None = None) -> Verdict:
        a = self.model.analyze(context, prompt, plain)
        changes = detect_value_changes(context, prompt)
        messages: list[Message] = []
        notes: dict[str, str] = {}

        plain_judge = message_from_probs("plain_judge", a.plain, self.prior)
        messages.append(plain_judge)
        notes["plain_judge"] = f"story-blind decision {max(a.plain, key=a.plain.get)}"

        sentinel = message_from_probs("sentinel", a.sentinel, self.prior, one_sided=True)
        messages.append(sentinel)
        notes["sentinel"] = f"raw-text decision {max(a.sentinel, key=a.sentinel.get)}"

        top_intent = max(a.intent, key=a.intent.get)
        if top_intent in HARMFUL_INTENTS:
            strength = MARKING_RAISE if top_intent == "remove_marking" else INTENT_RAISE
            messages.append(_raise("intent_criticality", strength * a.intent[top_intent],
                                   f"intent reads as {top_intent}"))
        else:
            messages.append(_raise("intent_criticality", 0.0, f"intent {top_intent}"))

        if changes:
            messages.append(_raise("value_diff", VALUE_DIFF_RAISE,
                                   f"{len(changes)} record value(s) would change"))
        else:
            messages.append(_raise("value_diff", 0.0, "no record value changes"))

        d = a.distinguishable
        if max(d, key=d.get) == "no":
            messages.append(_raise("output_authenticity", AUTH_RAISE * d["no"], "would pass as genuine"))
        else:
            messages.append(_raise("output_authenticity", 0.0, "marked or no image"))

        pur = a.purpose
        if max(pur, key=pur.get) != "none":
            messages.append(_raise("purpose_auditor", PURPOSE_RAISE,
                                   f"stated story: {max(pur, key=pur.get)}"))
        else:
            messages.append(_raise("purpose_auditor", 0.0, "no story"))

        if self.casebank is not None and a.embedding is not None:
            cb = self.casebank.message("casebank", a.embedding)
            messages.append(cb)
            notes["casebank"] = f"m_block {cb.m[_B]:+.2f}, u {cb.u:.2f}"

        if history and any(_REFUSAL.search(h) for h in history):
            messages.append(_raise("history", HISTORY_RAISE, "retry after a refusal"))

        fused = fuse(messages, self.prior, weights=self.weights, f=self.f)
        decision = fused.decision
        # the plan's hard rule: a detected marking removal can never end in ALLOW
        if top_intent == "remove_marking" and a.intent[top_intent] > 0.5:
            decision = _stricter(decision, "REVIEW")
        trace = []
        for msg in messages:
            trace.append({"agent": msg.agent, "m": msg.m.round(3).tolist(), "one_sided": msg.one_sided,
                          "triggered": bool(np.abs(msg.m).sum() > 1e-9),
                          "note": notes.get(msg.agent) or getattr(msg, "note", "")})
        return Verdict(decision=decision, p=fused.p, analysis=a, value_changes=changes,
                       messages=messages, fused=fused, trace=trace)
