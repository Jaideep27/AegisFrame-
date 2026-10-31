"""The image-generator contract and input normalisation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol, runtime_checkable

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class GeneratorError(RuntimeError):
    """The image model failed or returned no image."""


@runtime_checkable
class ImageGenerator(Protocol):
    def generate(self, prompt: str, image: bytes | None, **kwargs) -> bytes: ...


def _to_png(data: bytes) -> bytes:
    if data[:8] == PNG_SIGNATURE:
        return data
    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).save(buf, format="PNG")
    return buf.getvalue()


def as_png(value) -> bytes | None:
    """Accept what developers actually pass: bytes, file objects, paths, PIL images,
    upload tuples ("name", bytes, "mime"), Gemini parts, or a list of any of these (first is used)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _to_png(bytes(value))
    if isinstance(value, (str, Path)):
        return _to_png(Path(value).read_bytes())
    if isinstance(value, tuple) and len(value) in (2, 3, 4) and isinstance(value[0], (str, type(None))):
        return as_png(value[1])
    if isinstance(value, (list, tuple)):
        return as_png(value[0]) if value else None
    if hasattr(value, "read"):
        pos = value.tell() if hasattr(value, "tell") else None
        data = value.read()
        if pos is not None and hasattr(value, "seek"):
            value.seek(pos)
        return _to_png(data)
    if hasattr(value, "save") and hasattr(value, "size"):  # PIL.Image
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        return buf.getvalue()
    inline = getattr(value, "inline_data", None)
    if inline is None and isinstance(value, dict):
        inline = value.get("inline_data") or value.get("inlineData")
    if inline is not None:
        data = inline.get("data") if isinstance(inline, dict) else getattr(inline, "data", None)
        if isinstance(data, str):
            import base64

            data = base64.b64decode(data)
        return as_png(data)
    raise TypeError(f"cannot read an image from {type(value).__name__}")
