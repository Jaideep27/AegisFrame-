"""Persistence: session memory and audit sinks. SQLite (stdlib) so it works across processes
and restarts without extra infrastructure; swap in your own by implementing the same methods."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .audit import AuditLog
from .memory import SessionMemory, _Entry
from .schema import Action, EditSpec


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.execute("pragma journal_mode=wal")
    return conn


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        return self._sessions.setdefault(session_id, SessionMemory())


class SqliteSessionMemory(SessionMemory):
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock, session_id: str):
        self._conn, self._lock, self._sid = conn, lock, session_id

    @property
    def entries(self) -> list[_Entry]:  # type: ignore[override]
        with self._lock:
            rows = self._conn.execute(
                "select key, edit, action from session_edits where session_id = ? order by rowid", (self._sid,)
            ).fetchall()
        return [_Entry(k, EditSpec.model_validate_json(e), Action(a)) for k, e, a in rows]

    def record(self, key: str, edit: EditSpec, action: Action) -> None:
        with self._lock:
            self._conn.execute(
                "insert into session_edits (session_id, key, edit, action, ts) values (?, ?, ?, ?, ?)",
                (self._sid, key, edit.model_dump_json(), int(action), time.time()),
            )


class SqliteSessionStore:
    def __init__(self, path: str | Path):
        self._conn = _connect(path)
        self._lock = threading.Lock()
        self._conn.execute(
            "create table if not exists session_edits "
            "(session_id text, key text, edit text, action integer, ts real)"
        )
        self._conn.execute("create index if not exists session_edits_sid on session_edits (session_id)")

    def get(self, session_id: str) -> SqliteSessionMemory:
        return SqliteSessionMemory(self._conn, self._lock, session_id)


class SqliteAuditLog(AuditLog):
    def __init__(self, path: str | Path):
        super().__init__(None)
        self._conn = _connect(path)
        self._lock = threading.Lock()
        self._conn.execute("create table if not exists audit (ts real, request_id text, decision text, record text)")

    def write(self, record: dict) -> None:
        super().write(record)
        with self._lock:
            self._conn.execute("insert into audit values (?, ?, ?, ?)",
                               (time.time(), record.get("request_id"), record.get("decision"),
                                json.dumps(record, ensure_ascii=False)))
