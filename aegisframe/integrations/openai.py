"""Drop-in guard for the official OpenAI Python client (and AzureOpenAI).

    from openai import OpenAI
    from aegisframe import Guard
    from aegisframe.integrations.openai import protect_openai

    client = protect_openai(OpenAI(), Guard())
    resp = client.images.edit(model="gpt-image-1", image=open("slip.png", "rb"), prompt="...",
                              aegis_identity=user_identity, aegis_session=user_id)

images.edit / images.generate / images.create_variation are guarded: the request is checked
before the call (Blocked / ReviewRequired raised, the API is never called), the model receives
the safe prompt, and the returned image is verified (OutputRejected). Everything else on the
client passes through unchanged. aegis_* keyword arguments are never sent to OpenAI.
"""

from __future__ import annotations

import base64

import httpx

from ..errors import OutputRejected
from ..generators.base import as_png
from ..schema import Action
from ..verify import VerifyResult


def protect_openai(client, guard):
    return _GuardedClient(client, guard)


class _GuardedClient:
    def __init__(self, client, guard):
        self._client = client
        self._guard = guard
        self.images = _GuardedImages(client.images, guard)

    def __getattr__(self, name):
        return getattr(self._client, name)


def _image_bytes(resp) -> bytes | None:
    data = getattr(resp, "data", None) or []
    if not data:
        return None
    item = data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        return httpx.get(url, timeout=60).content
    return None


class _GuardedImages:
    def __init__(self, images, guard):
        self._images = images
        self._guard = guard

    def __getattr__(self, name):
        return getattr(self._images, name)

    def _check(self, prompt, png, identity, session, approval=None):
        decision = self._guard.check_request(prompt, image=png, verified_identity=identity, session_id=session,
                                             approval_token=approval)
        return self._guard.enforce(decision)

    def _verify(self, png, resp, decision):
        if not self._guard.config.verify.enabled:
            return
        out = _image_bytes(resp)
        if out is None:
            raise OutputRejected(VerifyResult(action=Action.BLOCK, reasons=["no image in the reply to verify"]), decision)
        result = self._guard.check_output(png, as_png(out), decision)
        if result.action != Action.ALLOW:
            raise OutputRejected(result, decision)

    def edit(self, *, image, prompt: str, aegis_identity=None, aegis_session: str = "default", aegis_approval=None,
             **kwargs):
        png = as_png(image)
        decision = self._check(prompt, png, aegis_identity, aegis_session, aegis_approval)
        resp = self._images.edit(image=("image.png", png, "image/png"),
                                 prompt=decision.generator_prompt or prompt, **kwargs)
        self._verify(png, resp, decision)
        return resp

    def generate(self, *, prompt: str, aegis_identity=None, aegis_session: str = "default", aegis_approval=None,
                 **kwargs):
        decision = self._check(prompt, None, aegis_identity, aegis_session, aegis_approval)
        resp = self._images.generate(prompt=decision.generator_prompt or prompt, **kwargs)
        self._verify(None, resp, decision)
        return resp

    def create_variation(self, *, image, aegis_identity=None, aegis_session: str = "default", **kwargs):
        # a variation of a record is an unparsed edit of that record: the policy decides (review by default)
        png = as_png(image)
        decision = self._check("Create a variation of this image.", png, aegis_identity, aegis_session)
        resp = self._images.create_variation(image=("image.png", png, "image/png"), **kwargs)
        self._verify(png, resp, decision)
        return resp
