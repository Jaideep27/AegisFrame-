"""Decision records. Hashes, attributes, claims and reasons — never raw prompts or images."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Literal

from .schema import Claim, Decision


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _claim(c: Claim, mode: str) -> dict:
    if mode == "text":
        return c.model_dump()
    return {"kind": c.kind, "sha256": _sha(c.text)}


def decision_record(decision: Decision, prompt: str, store_claims: Literal["hash", "text", "none"] = "hash") -> dict:
    floor = decision.floor
    return {
        "request_id": decision.request_id,
        "decision": decision.action.name,
        "rule": decision.rule,
        "severity": decision.severity.name.lower(),
        "prompt_sha256": _sha(prompt),
        "image_key": decision.image_key,
        "canonical_instruction": decision.canonical_instruction,
        "edit": decision.edit.model_dump(exclude_defaults=True),
        "claims": [] if store_claims == "none" else [_claim(c, store_claims) for c in decision.claims],
        "signals": decision.signals,
        "council": decision.council.model_dump(exclude={"beliefs"}) | {
            "beliefs": [b.model_dump() for b in decision.council.beliefs]
        } if getattr(decision, "council", None) else None,
        "votes": [v.model_dump() for v in floor.votes] if floor else [],
        "reasons": decision.reasons,
    }


class AuditLog:
    """Keeps the most recent records in memory (bounded, so a long-running server does not grow
    without limit); pass a path to also append every record to a JSON-lines file."""

    def __init__(self, path: str | Path | None = None, max_records: int = 1000):
        self.path = Path(path) if path else None
        self.records: deque[dict] = deque(maxlen=max_records)

    def write(self, record: dict) -> None:
        self.records.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
