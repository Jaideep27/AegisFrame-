import json
import sqlite3

import pytest

from aegisframe import AegisConfig, Guard, ReviewRequired
from aegisframe.review import ReviewQueue
from aegisframe.schema import Action
from reframebench.docs import CATALOG, OracleOcr, layout_lines, render, render_photo, to_png

ocr = OracleOcr()
PASS = ocr.register(to_png(render(CATALOG["boarding_pass"])), layout_lines(CATALOG["boarding_pass"]))
HALL = ocr.register(to_png(render(CATALOG["hall_ticket"])), layout_lines(CATALOG["hall_ticket"]))
PHOTO = to_png(render_photo())


def guard(tmp_path, **extra):
    cfg = {"review": {"queue_path": str(tmp_path / "reviews.db")},
           "memory": {"backend": "sqlite", "path": str(tmp_path / "memory.db")}, **extra}
    return Guard(ocr=ocr, config=AegisConfig.model_validate(cfg))


# ---- persistent session memory --------------------------------------------------------


def test_session_memory_survives_a_restart(tmp_path):
    assert guard(tmp_path).check_request("Change the seat to 2A", image=PASS, session_id="u1").action == Action.REVIEW
    restarted = guard(tmp_path)
    d = restarted.check_request("Change the seat to 3C", image=PASS, session_id="u1")
    assert d.action == Action.BLOCK and d.signals["retry_after_halt"]


def test_sessions_are_isolated(tmp_path):
    guard(tmp_path).check_request("Change the seat to 2A", image=PASS, session_id="u1")
    d = guard(tmp_path).check_request("Change the seat to 3C", image=PASS, session_id="u2")
    assert d.action == Action.REVIEW


# ---- review queue + approval tokens -----------------------------------------------------


def test_review_decisions_are_queued(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS)
    assert d.review_id
    items = g.reviews.pending()
    assert [i.id for i in items] == [d.review_id]
    assert items[0].canonical_instruction == d.canonical_instruction


def test_approved_review_allows_exactly_that_edit_once(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS, session_id="u1")
    token = g.reviews.approve(d.review_id, reviewer="ops-lead")

    ok = g.check_request("Change the seat to 2A", image=PASS, session_id="u1", approval_token=token)
    assert ok.action == Action.ALLOW and ok.signals["human_approved"]

    again = g.check_request("Change the seat to 2A", image=PASS, session_id="u1", approval_token=token)
    assert again.action != Action.ALLOW  # tokens are single use


def test_token_does_not_transfer_to_a_different_edit_or_image(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS)
    token = g.reviews.approve(d.review_id, reviewer="ops")
    assert g.check_request("Change the seat to 9F", image=PASS, approval_token=token).action != Action.ALLOW
    assert g.check_request("Change the seat to 2A", image=HALL, approval_token=token).action != Action.ALLOW


def test_tampered_token_is_rejected(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS)
    token = g.reviews.approve(d.review_id, reviewer="ops")
    forged = token[:-4] + ("0000" if not token.endswith("0000") else "1111")
    assert g.check_request("Change the seat to 2A", image=PASS, approval_token=forged).action != Action.ALLOW


def test_token_cannot_unlock_a_hard_block(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS)
    token = g.reviews.approve(d.review_id, reviewer="ops")
    # a forged name change is a high-severity block: no review token can unlock it
    assert g.check_request("Change the candidate name to Kavya R", image=HALL, approval_token=token).action == Action.BLOCK


def test_rejected_review_issues_no_token(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS)
    g.reviews.reject(d.review_id, reviewer="ops", note="not our passenger")
    assert g.reviews.get(d.review_id).status == "rejected"
    with pytest.raises(ValueError):
        g.reviews.approve(d.review_id, reviewer="ops")


def test_approval_works_across_processes(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS)
    token = ReviewQueue(tmp_path / "reviews.db").approve(d.review_id, reviewer="ops")  # e.g. the CLI
    assert guard(tmp_path).check_request("Change the seat to 2A", image=PASS, approval_token=token).action == Action.ALLOW


def test_review_required_exception_carries_the_review_id(tmp_path):
    g = guard(tmp_path)
    with pytest.raises(ReviewRequired) as exc:
        g.protect(lambda p, image=None: image)("Change the seat to 2A", PASS)
    assert exc.value.review_id == g.reviews.pending()[0].id


def test_protect_accepts_an_approval_token(tmp_path):
    g = guard(tmp_path)
    d = g.check_request("Change the seat to 2A", image=PASS)
    token = g.reviews.approve(d.review_id, reviewer="ops")
    from aegisframe.config import AegisConfig as _C  # noqa: F401
    out = g.protect(lambda p, image=None: image)("Change the seat to 2A", PASS, aegis_approval=token)
    assert out == PASS or out  # verification passes for an unchanged image


def test_enqueue_hook_is_called(tmp_path):
    seen = []
    q = ReviewQueue(tmp_path / "r.db", on_enqueue=seen.append)
    g = Guard(ocr=ocr, config=AegisConfig.model_validate({}), reviews=q)
    g.check_request("Change the seat to 2A", image=PASS)
    assert len(seen) == 1 and seen[0].status == "pending"


# ---- audit sink -----------------------------------------------------------------------


def test_sqlite_audit_sink(tmp_path):
    path = tmp_path / "audit.db"
    g = Guard(ocr=ocr, config=AegisConfig.model_validate({"audit": {"sink": "sqlite", "path": str(path)}}))
    g.check_request("Change my outfit from blue to red", image=PHOTO)
    rows = sqlite3.connect(path).execute("select record from audit").fetchall()
    assert json.loads(rows[0][0])["decision"] == "ALLOW"
