"""Human review queue with signed, single-use approval tokens.

REVIEW decisions are queued. A reviewer approves (CLI `aegisframe review approve <id>`,
or your own UI calling ReviewQueue.approve). The token returned is bound to that exact
canonical edit on that exact image, can be used once, and is HMAC-signed so it cannot be
forged without the secret. Resubmitting the request with the token yields ALLOW.

The secret comes from AEGIS_REVIEW_SECRET; without it a random secret is generated and kept
in the queue database (fine for a single deployment; set the env var for anything shared).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .memory import SessionMemory
from .schema import Decision


@dataclass
class ReviewItem:
    id: str
    request_id: str
    created: float
    status: str  # pending | approved | rejected
    canonical_instruction: str
    image_key: str | None
    rule: str | None
    severity: str
    reasons: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    reviewer: str | None = None
    note: str | None = None


class ReviewQueue:
    def __init__(self, path: str | Path = ":memory:", secret: str | bytes | None = None,
                 on_enqueue: Callable[[ReviewItem], None] | None = None, secret_env: str = "AEGIS_REVIEW_SECRET"):
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self.on_enqueue = on_enqueue
        with self._lock:
            self._conn.execute(
                "create table if not exists reviews (id text primary key, request_id text, created real, status text,"
                " canonical text, image_key text, rule text, severity text, reasons text, claims text,"
                " reviewer text, note text, token_used integer default 0)"
            )
            self._conn.execute("create table if not exists meta (k text primary key, v text)")
        self._secret = self._load_secret(secret if secret is not None else os.environ.get(secret_env))

    def _load_secret(self, given) -> bytes:
        if given:
            return given.encode() if isinstance(given, str) else given
        with self._lock:
            row = self._conn.execute("select v from meta where k = 'secret'").fetchone()
            if row is None:
                value = secrets.token_hex(32)
                self._conn.execute("insert into meta values ('secret', ?)", (value,))
                return value.encode()
        return row[0].encode()

    def _sign(self, review_id: str, canonical: str, image_key: str | None) -> str:
        msg = f"{review_id}|{canonical}|{image_key or ''}".encode("utf-8")
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    @staticmethod
    def _item(row) -> ReviewItem:
        return ReviewItem(id=row[0], request_id=row[1], created=row[2], status=row[3], canonical_instruction=row[4],
                          image_key=row[5], rule=row[6], severity=row[7], reasons=json.loads(row[8] or "[]"),
                          claims=json.loads(row[9] or "[]"), reviewer=row[10], note=row[11])

    def enqueue(self, decision: Decision, claims: list[dict] | None = None) -> ReviewItem:
        item = ReviewItem(id="rev_" + uuid.uuid4().hex[:12], request_id=decision.request_id, created=time.time(),
                          status="pending", canonical_instruction=decision.canonical_instruction,
                          image_key=decision.image_key, rule=decision.rule, severity=decision.severity.name.lower(),
                          reasons=list(decision.reasons), claims=claims or [])
        with self._lock:
            self._conn.execute(
                "insert into reviews (id, request_id, created, status, canonical, image_key, rule, severity, reasons,"
                " claims) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item.id, item.request_id, item.created, item.status, item.canonical_instruction, item.image_key,
                 item.rule, item.severity, json.dumps(item.reasons), json.dumps(item.claims)),
            )
        if self.on_enqueue:
            self.on_enqueue(item)
        return item

    def get(self, review_id: str) -> ReviewItem | None:
        with self._lock:
            row = self._conn.execute("select * from reviews where id = ?", (review_id,)).fetchone()
        return self._item(row) if row else None

    def pending(self) -> list[ReviewItem]:
        with self._lock:
            rows = self._conn.execute("select * from reviews where status = 'pending' order by created").fetchall()
        return [self._item(r) for r in rows]

    def approve(self, review_id: str, reviewer: str, note: str | None = None) -> str:
        item = self.get(review_id)
        if item is None or item.status != "pending":
            raise ValueError(f"review {review_id} is not pending")
        with self._lock:
            self._conn.execute("update reviews set status = 'approved', reviewer = ?, note = ? where id = ?",
                               (reviewer, note, review_id))
        return f"{review_id}.{self._sign(review_id, item.canonical_instruction, item.image_key)}"

    def reject(self, review_id: str, reviewer: str, note: str | None = None) -> None:
        item = self.get(review_id)
        if item is None or item.status != "pending":
            raise ValueError(f"review {review_id} is not pending")
        with self._lock:
            self._conn.execute("update reviews set status = 'rejected', reviewer = ?, note = ? where id = ?",
                               (reviewer, note, review_id))

    def redeem(self, token: str, canonical: str, image_key: str | None) -> bool:
        """True exactly once, for the approved edit on the approved image."""
        review_id, _, sig = token.partition(".")
        item = self.get(review_id)
        if item is None or item.status != "approved":
            return False
        if not hmac.compare_digest(sig, self._sign(review_id, item.canonical_instruction, item.image_key)):
            return False
        if canonical != item.canonical_instruction:
            return False
        if (item.image_key is None) != (image_key is None):
            return False
        if item.image_key and not SessionMemory.same_image(item.image_key, image_key):
            return False
        with self._lock:
            cur = self._conn.execute("update reviews set token_used = 1 where id = ? and token_used = 0", (review_id,))
        return cur.rowcount == 1
