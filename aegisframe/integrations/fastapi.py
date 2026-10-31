"""FastAPI helpers: turn AegisFrame exceptions into OpenAI-style JSON errors.

    app = FastAPI()
    install_exception_handlers(app)

    edit = guard.protect(my_model)          # raises Blocked / ReviewRequired / OutputRejected

    @app.post("/edit")
    def endpoint(...): return edit(prompt, image)
"""

from __future__ import annotations

from ..errors import Blocked, GuardUnavailable, OutputRejected, ReviewRequired
from ..generators.base import GeneratorError


def error_body(kind: str, message: str, decision=None, **extra) -> dict:
    body = {"type": kind, "message": message}
    if decision is not None:
        body.update({"code": decision.rule, "request_id": decision.request_id, "reasons": decision.reasons})
    body.update({k: v for k, v in extra.items() if v is not None})
    return {"error": body}


def install_exception_handlers(app) -> None:
    from fastapi.responses import JSONResponse

    @app.exception_handler(Blocked)
    async def _blocked(request, exc: Blocked):
        return JSONResponse(error_body("aegis_blocked", exc.decision.message_for_user, exc.decision), status_code=403)

    @app.exception_handler(ReviewRequired)
    async def _review(request, exc: ReviewRequired):
        return JSONResponse(error_body("aegis_review_required", exc.decision.message_for_user, exc.decision,
                                       review_id=exc.review_id), status_code=403)

    @app.exception_handler(OutputRejected)
    async def _rejected(request, exc: OutputRejected):
        return JSONResponse(error_body("aegis_output_rejected", str(exc), exc.decision), status_code=422)

    @app.exception_handler(GeneratorError)
    async def _upstream(request, exc: GeneratorError):
        return JSONResponse(error_body("upstream_error", str(exc)), status_code=502)

    @app.exception_handler(GuardUnavailable)
    async def _unavailable(request, exc: GuardUnavailable):
        return JSONResponse(error_body("aegis_unavailable", str(exc)), status_code=503)
