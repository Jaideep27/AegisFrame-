"""Draw a doclib record (fields and tables) as an image, optionally degraded like a phone photo.

Used for the OCR bake-off and to give part of the training data contexts read by the real OCR engine.
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from aegisframe.ocr.layout import Layout

W = 1000


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw(lay: Layout, marking: str | None = None) -> Image.Image:
    rows = len(lay.fields) + sum(len(t.rows) + 2 for t in lay.tables) + len(lay.text)
    h = 220 + rows * 44 + 60
    img = Image.new("RGB", (W, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 80), fill=(31, 58, 96))
    d.text((40, 22), lay.issuer or "", fill="white", font=_font(30))
    d.text((40, 105), lay.title, fill=(20, 20, 20), font=_font(32))
    y = 170
    for k, v in lay.fields:
        d.text((40, y), k, fill=(90, 90, 90), font=_font(24))
        d.text((460, y), v, fill=(10, 10, 10), font=_font(24))
        y += 44
    for t in lay.tables:
        y += 14
        n = len(t.header)
        cw = (W - 80) // n
        d.rectangle((40, y - 6, W - 40, y + 32), fill=(232, 236, 242))
        for c, hd in enumerate(t.header):
            d.text((50 + c * cw, y), hd, fill=(20, 20, 20), font=_font(22))
        y += 44
        for row in t.rows:
            for c, cell in enumerate(row):
                d.text((50 + c * cw, y), cell, fill=(10, 10, 10), font=_font(22))
            d.line((40, y + 34, W - 40, y + 34), fill=(220, 220, 220))
            y += 44
    for line in lay.text:
        d.text((40, y + 10), line, fill=(120, 120, 120), font=_font(20))
        y += 40
    if marking:
        stamp = Image.new("RGBA", (W, 160), (0, 0, 0, 0))
        ImageDraw.Draw(stamp).text((W // 2 - 250, 20), marking, fill=(200, 0, 0, 110), font=_font(110))
        img.paste(stamp.rotate(25, expand=False), (0, h // 2 - 80), stamp.rotate(25, expand=False))
    return img


def degrade(img: Image.Image, rng: random.Random, level: str = "photo") -> Image.Image:
    """scan: slight blur and JPEG; photo: plus rotation, uneven light and stronger compression."""
    img = img.rotate(rng.uniform(-3, 3) if level == "photo" else rng.uniform(-0.8, 0.8), expand=True,
                     fillcolor=(235, 235, 230))
    if level == "photo":
        shade = Image.linear_gradient("L").resize(img.size).point(lambda p: 200 + p // 5)
        img = Image.composite(img, Image.new("RGB", img.size, (170, 165, 150)), shade)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1 if level == "photo" else 0.6)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(35, 60) if level == "photo" else rng.randint(65, 85))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
