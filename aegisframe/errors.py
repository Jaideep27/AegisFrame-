"""Exceptions raised by the protected-call APIs (protect, run, client wrappers, proxy)."""

from __future__ import annotations

from .schema import Decision


class AegisError(Exception):
    """Base class for everything AegisFrame raises on purpose."""


class Blocked(AegisError):
    def __init__(self, decision: Decision):
        self.decision = decision
        super().__init__(f"[{decision.rule or 'blocked'}] {decision.message_for_user}")


class ReviewRequired(AegisError):
    def __init__(self, decision: Decision, review_id: str | None = None):
        self.decision = decision
        self.review_id = review_id
        super().__init__(decision.message_for_user + (f" (review id {review_id})" if review_id else ""))


class OutputRejected(AegisError):
    """The image model produced something other than the approved edit."""

    def __init__(self, verification, decision: Decision):
        self.verification = verification
        self.decision = decision
        super().__init__("; ".join(verification.reasons) or "output did not match the approved edit")


class GuardUnavailable(AegisError):
    """A dependency (OCR engine, provider) is missing or misconfigured."""
