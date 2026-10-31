"""Topic-free policy rules over risk attributes, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from .schema import Action, EditSpec, GroundedDoc, Severity, strictest

ATTRIBUTES = {
    "has_image", "is_document", "is_evidence", "real_source", "factual_change",
    "passes_as_genuine", "real_person_false_act", "from_scratch", "edit_understood",
}
ACTIONS = {"allow": Action.ALLOW, "review": Action.REVIEW, "block": Action.BLOCK}
SEVERITIES = {s.name.lower(): s for s in Severity}


class PolicyError(ValueError):
    pass


def attributes(doc: GroundedDoc, edit: EditSpec) -> dict[str, bool]:
    """The fingerprint: computed from the image and the EditSpec only."""
    real = doc.real_source and not (edit.from_scratch and not doc.has_image)
    return {
        "has_image": doc.has_image,
        "is_document": doc.is_document,
        "is_evidence": doc.is_evidence,
        "real_source": real,
        "factual_change": bool(edit.factual_changes),
        "passes_as_genuine": edit.passes_as_genuine,
        "real_person_false_act": edit.real_person_false_act,
        "from_scratch": edit.from_scratch,
        # an edit the separator could not parse into any concrete operation
        "edit_understood": bool(
            edit.field_changes or edit.visual_changes or edit.marking or edit.from_scratch
            or edit.real_person_false_act or edit.answer_only
        ),
    }


@dataclass(frozen=True)
class Rule:
    id: str
    when: dict[str, bool]
    action: Action | None
    by_severity: dict[Severity, Action]
    escalate_on_reframing: bool = True

    def matches(self, attrs: dict[str, bool]) -> bool:
        return all(attrs.get(k) == v for k, v in self.when.items())

    def action_for(self, severity: Severity) -> Action:
        if self.action is not None:
            return self.action
        # unlisted severities fall to the next listed one above, else block (fail closed)
        for s in sorted(Severity):
            if s >= severity and s in self.by_severity:
                return self.by_severity[s]
        return Action.BLOCK


@dataclass(frozen=True)
class Unlock:
    role: str
    rule: str
    max_severity: Severity
    action: Action


class Policy:
    def __init__(self, raw: dict):
        self.raw = raw
        self.rules = [self._rule(r) for r in raw.get("rules", [])]
        self.overrides = [
            (o["match"].get("doc_type_contains", "").lower(), self._action(o["action"]))
            for o in raw.get("overrides") or []
        ]
        self.unlocks = [
            Unlock(u["role"], u["rule"], self._severity(u["max_severity"]), self._action(u.get("action", "review")))
            for u in raw.get("verified_roles") or []
        ]
        claims = raw.get("claims") or {}
        self.reframing_escalates = bool(claims.get("reframing_escalates_review_to_block", True))
        judges = raw.get("judges") or {}
        self.refuse_threshold = float(judges.get("refuse_threshold", 0.5))
        self.review_threshold = float(judges.get("review_threshold", 0.3))

    @classmethod
    def from_dict(cls, raw: dict) -> Policy:
        return cls(raw)

    @classmethod
    def from_file(cls, path: str | Path) -> Policy:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> Policy:
        text = resources.files("aegisframe.policies").joinpath("default.yaml").read_text(encoding="utf-8")
        return cls(yaml.safe_load(text))

    @staticmethod
    def _action(name: str) -> Action:
        try:
            return ACTIONS[name.lower()]
        except KeyError:
            raise PolicyError(f"unknown action {name!r}") from None

    @staticmethod
    def _severity(name: str) -> Severity:
        try:
            return SEVERITIES[name.lower()]
        except KeyError:
            raise PolicyError(f"unknown severity {name!r}") from None

    def _rule(self, r: dict) -> Rule:
        when = r.get("when") or {}
        unknown = set(when) - ATTRIBUTES
        if unknown:
            raise PolicyError(f"rule {r.get('id')!r} uses unknown attributes {sorted(unknown)}")
        if "action" not in r and "action_by_severity" not in r:
            raise PolicyError(f"rule {r.get('id')!r} needs action or action_by_severity")
        return Rule(
            id=r["id"],
            when={k: bool(v) for k, v in when.items()},
            action=self._action(r["action"]) if "action" in r else None,
            by_severity={self._severity(k): self._action(v) for k, v in (r.get("action_by_severity") or {}).items()},
            escalate_on_reframing=bool(r.get("escalate_on_reframing", True)),
        )

    def escalates(self, rule_id: str | None) -> bool:
        """Does a story (or a retry) attached to this rule's halt escalate review to block?"""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule.escalate_on_reframing
        return True

    def evaluate(self, attrs: dict[str, bool], severity: Severity, doc_type: str) -> tuple[Action, str | None, list[str]]:
        action, rule_id, reasons = Action.ALLOW, None, []
        for rule in self.rules:
            if rule.matches(attrs):
                a = rule.action_for(severity)
                reasons.append(f"rule {rule.id} matched at severity {severity.name.lower()} -> {a.name}")
                if a > action or rule_id is None:
                    action, rule_id = strictest(action, a), rule.id
        if attrs.get("factual_change"):
            dt = doc_type.lower()
            for needle, a in self.overrides:
                if needle and needle in dt and a > action:
                    reasons.append(f"company override for {needle!r} -> {a.name}")
                    action = a
        return action, rule_id, reasons

    def unlock_for(self, role: str, rule: str) -> Unlock | None:
        for u in self.unlocks:
            if u.role == role and u.rule == rule:
                return u
        return None
