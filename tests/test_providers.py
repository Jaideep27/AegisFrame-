import base64
import json

import httpx
import pytest

from aegisframe.providers.base import ProviderError, parse_json_reply
from aegisframe.providers.gemini import GeminiProvider
from aegisframe.providers.openai_compat import OpenAICompatProvider


def test_parse_json_reply_strips_code_fences():
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_reply_rejects_prose():
    with pytest.raises(ProviderError):
        parse_json_reply("I can't help with that.")


def gemini_transport(captured):
    def handler(request: httpx.Request):
        captured.append(request)
        body = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_gemini_sends_system_user_and_image_and_parses_json():
    captured = []
    p = GeminiProvider(api_key="k", model="m", client=httpx.Client(transport=gemini_transport(captured)))
    assert p.generate_json("SYS", "USER", image=b"\x89PNG") == {"ok": True}
    req = captured[0]
    assert req.url.path.endswith("/models/m:generateContent")
    assert req.headers["x-goog-api-key"] == "k"
    assert "key=" not in str(req.url)  # keys stay out of URLs and logs
    body = json.loads(req.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
    parts = body["contents"][0]["parts"]
    assert parts[0]["text"] == "USER"
    assert parts[1]["inline_data"]["data"] == base64.b64encode(b"\x89PNG").decode()


def test_gemini_http_error_is_provider_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(429, json={"error": "rate"}))
    p = GeminiProvider(api_key="k", model="m", client=httpx.Client(transport=transport))
    with pytest.raises(ProviderError):
        p.generate_json("s", "u")


def test_gemini_blocked_reply_is_provider_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    p = GeminiProvider(api_key="k", model="m", client=httpx.Client(transport=transport))
    with pytest.raises(ProviderError):
        p.generate_json("s", "u")


def test_openai_compat_request_shape():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"x": 2}'}}]})

    p = OpenAICompatProvider(
        base_url="https://api.example.com/v1", api_key="k", model="m",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert p.generate_json("SYS", "USER", image=b"img") == {"x": 2}
    req = captured[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    content = body["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "USER"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["temperature"] == 0


def test_openai_compat_network_error_is_provider_error():
    def handler(request):
        raise httpx.ConnectError("down")

    p = OpenAICompatProvider(base_url="http://x/v1", api_key=None, model="m",
                             client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError):
        p.generate_json("s", "u")


def test_retrying_provider_retries_then_succeeds():
    from aegisframe.providers.retry import RetryingProvider

    attempts = []

    class Flaky:
        name = "flaky"

        def generate_json(self, system, user, image=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise ProviderError("429")
            return {"ok": True}

    slept = []
    p = RetryingProvider(Flaky(), attempts=3, base_delay=0.1, sleep=slept.append)
    assert p.generate_json("s", "u") == {"ok": True}
    assert len(attempts) == 3 and len(slept) == 2 and slept[1] > slept[0]


def test_retrying_provider_gives_up():
    from aegisframe.providers.retry import RetryingProvider

    class Down:
        name = "down"

        def generate_json(self, system, user, image=None):
            raise ProviderError("503")

    with pytest.raises(ProviderError):
        RetryingProvider(Down(), attempts=2, sleep=lambda s: None).generate_json("s", "u")
