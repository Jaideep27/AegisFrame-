"""AegisFrame: justification-monotone guardrail for image editing.

A good story should not unlock a forgery.
"""

from .config import AegisConfig
from .errors import AegisError, Blocked, GuardUnavailable, OutputRejected, ReviewRequired
from .guard import Guard, GuardedResult
from .harrier_guard import HarrierGuard, HarrierVerdict
from .identity import VerifiedIdentity
from .policy import Policy
from .schema import Action, Decision, EditSpec, Severity
from .verify import VerifyResult

__all__ = [
    "HarrierGuard", "HarrierVerdict",
    "Action", "AegisConfig", "AegisError", "Blocked", "Decision", "EditSpec", "Guard", "GuardUnavailable",
    "GuardedResult", "OutputRejected", "Policy", "ReviewRequired", "Severity", "VerifiedIdentity", "VerifyResult",
]
__version__ = "0.6.0"
