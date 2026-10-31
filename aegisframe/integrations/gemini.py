"""Drop-in guard for the google-genai client (Gemini image models).

    from google import genai
    from aegisframe.integrations.gemini import protect_gemini

    client = protect_gemini(genai.Client(), Guard())
    resp = client.models.generate_content(model="gemini-2.5-flash-image",
                                          contents=["Change my outfit to red", pil_image])

Text parts are joined into the prompt and checked with the first image. The model receives
the safe prompt plus the original non-text parts; any image in the reply is verified.
"""

from __future__ import annotations

from ..errors import GuardUnavailable, OutputRejected
from ..generators.base import as_png
from ..schema import Action


def protect_gemini(client, guard):
    return _GuardedClient(client, guard)


class _GuardedClient:
    def __init__(self, client, guard):
        self._client = client
        self.models = _GuardedModels(client.models, guard)

    def __getattr__(self, name):
        return getattr(self._client, name)


def _is_text(item) -> bool:
    if isinstance(item, str):
        return True
    # PIL images also have a `.text` attribute (a dict of PNG metadata): only a str counts
    return isinstance(getattr(item, "text", None), str) and getattr(item, "inline_data", None) is None


def _text(item) -> str:
    return item if isinstance(item, str) else item.text


def _reply_image(resp) -> bytes | None:
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return as_png(inline.data)
    return None


class _GuardedModels:
    def __init__(self, models, guard):
        self._models = models
        self._guard = guard

    def __getattr__(self, name):
        return getattr(self._models, name)

    def generate_content(self, *, model, contents, config=None, aegis_identity=None, aegis_session: str = "default",
                         aegis_approval=None, **kwargs):
        items = contents if isinstance(contents, list) else [contents]
        texts = [_text(x) for x in items if _is_text(x)]
        others = [x for x in items if not _is_text(x)]
        png = as_png(others[0]) if others else None
        prompt = " ".join(texts)

        decision = self._guard.check_request(prompt, image=png, verified_identity=aegis_identity,
                                             session_id=aegis_session, approval_token=aegis_approval)
        decision = self._guard.enforce(decision)
        send = decision.generator_prompt or prompt
        resp = self._models.generate_content(model=model, contents=[send, *others], config=config, **kwargs)

        out = _reply_image(resp)
        if out is not None and self._guard.config.verify.enabled:
            result = self._guard.check_output(png, out, decision)
            if result.action != Action.ALLOW:
                raise OutputRejected(result, decision)
        return resp

    def generate_content_stream(self, *args, **kwargs):
        raise GuardUnavailable("streaming is not guarded; use generate_content for image edits")
