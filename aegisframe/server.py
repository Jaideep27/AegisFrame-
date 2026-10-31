"""OpenAI-compatible guarding proxy. Teams point their OpenAI client's base_url here.

    aegisframe serve --config aegis.yaml --port 8080
    client = OpenAI(base_url="http://localhost:8080/v1")      # nothing else changes

POST /v1/images/edits         multipart: image, prompt, model, ... (forwarded)
POST /v1/images/generations   JSON: prompt, model, ...           (forwarded)
GET  /healthz

The caller's Authorization header is forwarded to the upstream image API. Optional headers:
  X-Aegis-Session   session id (retry / multi-turn tracking)
  X-Aegis-Approval  approval token from a human review
  X-Aegis-Identity  JSON {"subject", "roles", "issuer"}: honoured ONLY when
                    server.trust_identity_header is true, i.e. behind your own auth gateway.
"""


import base64
import contextlib
import json
import time

import httpx

from .generators import OpenAIImagesGenerator
from .guard import Guard
from .identity import VerifiedIdentity

_RESERVED = {"image", "prompt", "model", "response_format", "n"}


def create_app(guard: "Guard | None" = None, generator_factory=None):
    from fastapi import FastAPI, Request
    from starlette.concurrency import run_in_threadpool

    from .integrations.fastapi import install_exception_handlers

    guard = guard or Guard()
    http_client = httpx.Client(timeout=180.0)  # one pooled client for all upstream calls

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        http_client.close()

    app = FastAPI(title="AegisFrame proxy", version="0.5.0", lifespan=lifespan)
    app.state.http_client = http_client
    install_exception_handlers(app)

    def default_factory(request: Request, model: str | None):
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else None
        return OpenAIImagesGenerator(api_key=key, model=model or "gpt-image-1", base_url=guard.config.server.upstream,
                                     client=app.state.http_client)

    factory = generator_factory or default_factory

    def identity_from(request: Request) -> VerifiedIdentity | None:
        raw = request.headers.get("x-aegis-identity")
        if not raw or not guard.config.server.trust_identity_header:
            return None
        return VerifiedIdentity(**json.loads(raw))

    def reply(image: bytes) -> dict:
        return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(image).decode()}]}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/images/edits")
    async def edits(request: Request):
        form = await request.form()
        upload = form.get("image") or form.get("image[]")
        png = await upload.read() if upload is not None else None
        prompt = str(form.get("prompt", ""))
        model = form.get("model")
        extra = {k: v for k, v in form.items() if k not in _RESERVED and not k.startswith("image")}
        result = await run_in_threadpool(
            guard.run, prompt, png, factory(request, model), identity_from(request),
            request.headers.get("x-aegis-session", "default"), request.headers.get("x-aegis-approval"), **extra,
        )
        return reply(result.image)

    @app.post("/v1/images/generations")
    async def generations(request: Request):
        body = await request.json()
        prompt = str(body.get("prompt", ""))
        extra = {k: v for k, v in body.items() if k not in _RESERVED}
        result = await run_in_threadpool(
            guard.run, prompt, None, factory(request, body.get("model")), identity_from(request),
            request.headers.get("x-aegis-session", "default"), request.headers.get("x-aegis-approval"), **extra,
        )
        return reply(result.image)

    return app
