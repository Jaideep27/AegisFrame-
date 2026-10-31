# Integrations

All integrations share one flow: **check → generate with the safe prompt → verify the output**.
Each raises:

| Exception | Meaning | Proxy status |
|---|---|---|
| `Blocked` | not allowed, whatever the reason given; the model was never called | 403 `aegis_blocked` |
| `ReviewRequired` | needs a human; `.review_id` is set when a queue is configured | 403 `aegis_review_required` |
| `OutputRejected` | the model produced more than (or other than) the approved edit | 422 `aegis_output_rejected` |
| `GuardUnavailable` | a dependency is missing or misused | 503 |

Extra keyword arguments, accepted everywhere and never forwarded to the vendor:
- `aegis_identity`: a `VerifiedIdentity(subject, roles, issuer)` built by **your** auth layer.
- `aegis_session`: a stable id per user or conversation, used for retries and step-by-step edits.
- `aegis_approval`: a one-time token from a human review.

## OpenAI client

```python
from openai import OpenAI
from aegisframe import Guard
from aegisframe.integrations.openai import protect_openai

client = protect_openai(OpenAI(), Guard.from_config("aegis.yaml"))
resp = client.images.edit(model="gpt-image-1", image=open("in.png", "rb"), prompt=prompt,
                          aegis_identity=identity, aegis_session=user_id)
```

`images.edit`, `images.generate` and `images.create_variation` are guarded. Everything else passes
through. When several images are passed, the first one is checked and sent.

## Gemini (google-genai)

```python
from google import genai
from aegisframe.integrations.gemini import protect_gemini

client = protect_gemini(genai.Client(), guard)
resp = client.models.generate_content(model="gemini-2.5-flash-image", contents=[prompt, pil_image])
```

Text parts form the prompt, and the first non-text part is the image. Streaming is refused, because
an image can't be verified mid-stream.

## Any model

```python
@guard.protect
def edit(prompt, image=None): ...                    # your function

from aegisframe.generators import DiffusersGenerator, HTTPGenerator, OpenAIImagesGenerator, GeminiImageGenerator
result = guard.run(prompt, image_bytes, DiffusersGenerator(pipe))     # local open-source pipeline
result = guard.run(prompt, image_bytes, HTTPGenerator("https://gen.internal/edit",
                                                      prompt_field="instruction", image_field="input",
                                                      response_path="result.png"))
result.image, result.decision, result.verification
```

Images may be bytes, file objects, paths, PIL images, upload tuples or Gemini parts.

## Step by step

```python
decision = guard.check_request(prompt, image=png, verified_identity=identity, session_id=user_id)
if decision.action.name == "ALLOW":
    out = model(decision.generator_prompt, png)       # never the raw prompt on the record path
    check = guard.check_output(png, out, decision)
```

Async: `await guard.acheck_request(...)` and `await guard.acheck_output(...)`. Hooks:
`guard.on_decision(callback)` for metrics, alerts and SIEM export.

## Proxy (no code changes)

```bash
pip install "aegisframe[server]"
aegisframe serve --config aegis.yaml --port 8080
```

```python
client = OpenAI(base_url="http://aegis.internal:8080/v1")   # nothing else changes
```

The caller's `Authorization` header is forwarded upstream (`server.upstream`). Optional headers:
- `X-Aegis-Session`
- `X-Aegis-Approval`
- `X-Aegis-Identity`: JSON. It is honoured **only** with `server.trust_identity_header: true`, which
  you should set only behind your own auth gateway, because otherwise anyone could claim a role.

## Your own FastAPI app

```python
from aegisframe.integrations.fastapi import install_exception_handlers
install_exception_handlers(app)    # Blocked -> 403 JSON, OutputRejected -> 422 JSON, ...
```

## Human review

```yaml
review: {queue_path: reviews.db, webhook_url: https://hooks.example/aegis}
```

1. REVIEW decisions are queued, and the webhook receives each item.
2. A reviewer runs `aegisframe review approve <id>` (or calls `ReviewQueue.approve`) and gets a token.
3. The user resubmits with `aegis_approval=<token>`. The token is HMAC-signed, valid once, and bound
   to that exact edit on that exact image. It can never unlock a hard rule block.
