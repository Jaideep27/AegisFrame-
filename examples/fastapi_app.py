"""Your own FastAPI endpoint with AegisFrame. Needs `pip install 'aegisframe[server]'`.

    uvicorn examples.fastapi_app:app --reload

Blocked -> 403 {"error": {"type": "aegis_blocked", ...}}, review -> 403 aegis_review_required,
output mismatch -> 422 aegis_output_rejected. Or skip this file entirely and run the
OpenAI-compatible proxy:  aegisframe serve --config aegis.yaml
"""

import base64

from fastapi import FastAPI, File, Form, UploadFile

from aegisframe import Guard
from aegisframe.generators import OpenAIImagesGenerator
from aegisframe.integrations.fastapi import install_exception_handlers

guard = Guard()
model = OpenAIImagesGenerator(model="gpt-image-1")
app = FastAPI()
install_exception_handlers(app)


@app.post("/edit")
async def edit(prompt: str = Form(...), image: UploadFile = File(...), user_id: str = Form("anonymous")):
    result = guard.run(prompt, await image.read(), model, session_id=user_id)
    return {"image": base64.b64encode(result.image).decode(), "rule": result.decision.rule}
