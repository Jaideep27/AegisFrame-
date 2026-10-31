import base64
import io
import json

import httpx
import pytest
from PIL import Image

from aegisframe.generators import (
    CallableGenerator, DiffusersGenerator, GeminiImageGenerator, GeneratorError, HTTPGenerator,
    OpenAIImagesGenerator, as_png,
)
from reframebench.docs import render_photo, to_png

PNG = to_png(render_photo())
B64 = base64.b64encode(PNG).decode()


def test_as_png_accepts_common_inputs(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(PNG)
    img = Image.open(io.BytesIO(PNG))
    for value in (PNG, io.BytesIO(PNG), str(path), path, img, ("x.png", PNG, "image/png"), [PNG]):
        out = as_png(value)
        assert out[:8] == b"\x89PNG\r\n\x1a\n"
    assert as_png(None) is None


def test_file_like_input_is_rewound_after_reading():
    f = io.BytesIO(PNG)
    as_png(f)
    assert f.tell() == 0


def test_callable_generator():
    assert CallableGenerator(lambda p, i=None: PNG).generate("x", None) == PNG


def openai_handler(captured):
    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": B64}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_edit_uses_multipart_and_decodes_b64():
    captured = []
    g = OpenAIImagesGenerator(api_key="k", model="gpt-image-1", client=openai_handler(captured))
    assert g.generate("make it red", PNG) == PNG
    req = captured[0]
    assert req.url.path.endswith("/images/edits")
    assert req.headers["authorization"] == "Bearer k"
    assert b"make it red" in req.content and b"gpt-image-1" in req.content


def test_openai_generate_without_image_uses_json():
    captured = []
    g = OpenAIImagesGenerator(api_key="k", client=openai_handler(captured))
    assert g.generate("a fictional receipt labelled SAMPLE", None) == PNG
    assert captured[0].url.path.endswith("/images/generations")
    assert json.loads(captured[0].content)["prompt"] == "a fictional receipt labelled SAMPLE"


def test_openai_error_is_generator_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"error": "safety"})))
    with pytest.raises(GeneratorError):
        OpenAIImagesGenerator(api_key="k", client=client).generate("x", PNG)


def test_gemini_image_generator_parses_inline_data():
    captured = []

    def handler(request):
        captured.append(request)
        parts = [{"text": "here you go"}, {"inlineData": {"mimeType": "image/png", "data": B64}}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}}]})

    g = GeminiImageGenerator(api_key="k", model="m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert g.generate("make it red", PNG) == PNG
    body = json.loads(captured[0].content)
    assert body["contents"][0]["parts"][0]["text"] == "make it red"
    assert "IMAGE" in body["generationConfig"]["responseModalities"]


def test_gemini_text_only_reply_is_generator_error():
    reply = {"candidates": [{"content": {"parts": [{"text": "I can't do that"}]}}]}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))
    with pytest.raises(GeneratorError):
        GeminiImageGenerator(api_key="k", model="m", client=client).generate("x", PNG)


def test_http_generator_round_trip():
    def handler(request):
        body = json.loads(request.content)
        assert body["instruction"] == "x" and base64.b64decode(body["input"]) == PNG
        return httpx.Response(200, json={"result": {"png": B64}})

    g = HTTPGenerator("https://gen.internal/edit", prompt_field="instruction", image_field="input",
                      response_path="result.png", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert g.generate("x", PNG) == PNG


def test_diffusers_generator_duck_types_a_pipeline():
    seen = {}

    class FakePipe:
        def __call__(self, prompt, image=None, **kw):
            seen["prompt"], seen["image"] = prompt, image
            return type("Out", (), {"images": [Image.open(io.BytesIO(PNG))]})()

    out = DiffusersGenerator(FakePipe(), num_inference_steps=4).generate("make it red", PNG)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    assert seen["prompt"] == "make it red" and seen["image"].size == Image.open(io.BytesIO(PNG)).size
