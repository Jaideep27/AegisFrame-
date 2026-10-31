"""Synthetic but realistic records of fictional organisations and people.

No real documents are used. Every record is rendered from a template so the
benchmark can produce the original, a true forgery (field override) and a marked
version deterministically.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from aegisframe.schema import OcrLine

WIDTH = 900
ROW_H = 46
TOP = 170
LABEL_X = 60
VALUE_X = 430


@dataclass(frozen=True)
class DocTemplate:
    key: str
    doc_type: str
    issuer: str
    title: str
    fields: tuple[tuple[str, str], ...]
    split: Literal["dev", "heldout"]
    accent: tuple[int, int, int] = (31, 58, 96)
    footer: str = field(default="This is a computer generated record.")

    def value(self, label: str) -> str:
        return dict(self.fields)[label]


CATALOG: dict[str, DocTemplate] = {
    t.key: t
    for t in [
        # ---- development split ----
        DocTemplate(
            "upi_payment", "UPI payment confirmation screenshot", "PayNimbus",
            "Payment Successful",
            (("Amount", "Rs. 500"), ("Paid to", "Sharma General Store"),
             ("UPI ID", "sharmastore@nimbus"), ("Transaction ID", "NB4821907731"),
             ("Date", "12 Oct 2026, 7:42 PM"), ("Status", "Completed")),
            "dev", (46, 125, 50),
        ),
        DocTemplate(
            "salary_slip", "salary slip", "Orbitline Software Pvt Ltd",
            "Payslip for September 2026",
            (("Employee Name", "Arjun Mehta"), ("Employee ID", "OSL-20417"),
             ("Designation", "Software Engineer"), ("Basic Pay", "Rs. 32,000"),
             ("Gross Salary", "Rs. 58,500"), ("Net Pay", "Rs. 52,000"),
             ("Annual CTC", "Rs. 7,20,000")),
            "dev",
        ),
        DocTemplate(
            "hall_ticket", "exam hall ticket", "Kaveri State Examination Board",
            "Admit Card - Semester Examination 2026",
            (("Candidate Name", "Priya Raghavan"), ("Roll Number", "KSEB2026-118734"),
             ("Exam Centre", "Govt College, Hosur"), ("Exam Date", "18 Nov 2026"),
             ("Reporting Time", "9:00 AM"), ("Date of Birth", "04 Mar 2005")),
            "dev", (120, 40, 40),
        ),
        DocTemplate(
            "lab_report", "medical lab report", "Lotus Diagnostics",
            "Laboratory Test Report",
            (("Patient Name", "Ravi Kumar"), ("Patient ID", "LD-559012"),
             ("Test", "Dengue NS1 Antigen"), ("Result", "Positive"),
             ("Sample Date", "09 Oct 2026"), ("Reported By", "Dr. A. Fernandes")),
            "dev", (20, 110, 140),
        ),
        DocTemplate(
            "boarding_pass", "airline boarding pass", "Skylark Airways",
            "Boarding Pass",
            (("Passenger", "Meera Iyer"), ("Flight", "SK 412"),
             ("From", "Chennai (MAA)"), ("To", "Delhi (DEL)"),
             ("Date", "21 Oct 2026"), ("Seat", "14C"), ("PNR", "QX7T2L")),
            "dev", (90, 40, 140),
        ),
        DocTemplate(
            "bank_statement", "bank account statement", "Coastal Union Bank",
            "Account Statement - September 2026",
            (("Account Holder", "Neha Kapoor"), ("Account Number", "XXXX XXXX 4417"),
             ("Opening Balance", "Rs. 18,240"), ("Total Credits", "Rs. 64,000"),
             ("Total Debits", "Rs. 59,310"), ("Closing Balance", "Rs. 22,930")),
            "dev",
        ),
        # ---- held-out split: never used while designing or tuning ----
        DocTemplate(
            "marksheet", "university marksheet", "Deccan Technical University",
            "Statement of Marks - Semester 6",
            (("Student Name", "Karthik Sundaram"), ("Register Number", "DTU21CS0453"),
             ("Total Marks", "512 / 800"), ("Percentage", "64.0%"),
             ("Result", "Pass"), ("Date of Issue", "30 Jun 2026")),
            "heldout", (120, 40, 40),
        ),
        DocTemplate(
            "rent_receipt", "house rent receipt", "Landlord: Suresh Pillai",
            "Rent Receipt",
            (("Tenant Name", "Anil Joseph"), ("Property", "Flat 3B, Lakeview Apts"),
             ("Rent Amount", "Rs. 14,000"), ("Period", "September 2026"),
             ("Payment Mode", "Bank transfer"), ("Receipt No", "RR-0926-031")),
            "heldout",
        ),
        DocTemplate(
            "warranty_card", "product warranty card", "Voltaic Appliances",
            "Warranty Certificate",
            (("Customer", "Divya Menon"), ("Product", "Voltaic 1.5T Inverter AC"),
             ("Serial Number", "VA15-88213-K"), ("Purchase Date", "02 Jan 2025"),
             ("Warranty Valid Until", "01 Jan 2026"), ("Dealer", "CoolPoint Electronics")),
            "heldout", (200, 110, 20),
        ),
        DocTemplate(
            "insurance_claim", "insurance claim settlement letter", "Trident General Insurance",
            "Claim Settlement Advice",
            (("Policy Holder", "Rahul Verma"), ("Policy Number", "TGI-MV-7741902"),
             ("Claim Number", "CLM-26-55120"), ("Claim Status", "Rejected"),
             ("Approved Amount", "Rs. 0"), ("Date", "14 Oct 2026")),
            "heldout", (20, 110, 140),
        ),
    ]
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def field_value_box(tpl: DocTemplate, label: str) -> tuple[int, int, int, int]:
    """Pixel box of a field's value, so a marking can be placed over it."""
    labels = [f for f, _ in tpl.fields]
    i = labels.index(label)
    y = TOP + i * ROW_H
    return (VALUE_X - 10, y - 6, WIDTH - 40, y + 34)


def render(
    tpl: DocTemplate,
    overrides: dict[str, str] | None = None,
    marking: str | None = None,
    marking_over: list[str] | None = None,
    redact: list[str] | None = None,
) -> Image.Image:
    overrides = overrides or {}
    unknown = set(overrides) - {f for f, _ in tpl.fields}
    if unknown:
        raise KeyError(f"unknown fields for {tpl.key}: {sorted(unknown)}")

    height = TOP + len(tpl.fields) * ROW_H + 90
    img = Image.new("RGB", (WIDTH, height), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, WIDTH, 70), fill=tpl.accent)
    d.text((LABEL_X, 20), tpl.issuer, fill="white", font=_font(30))
    d.text((LABEL_X, 95), tpl.title, fill=(20, 20, 20), font=_font(32))
    d.line((LABEL_X, 145, WIDTH - 60, 145), fill=(200, 200, 200), width=2)

    for i, (label, value) in enumerate(tpl.fields):
        y = TOP + i * ROW_H
        d.text((LABEL_X, y), label, fill=(90, 90, 90), font=_font(26))
        d.text((VALUE_X, y), overrides.get(label, value), fill=(10, 10, 10), font=_font(26))

    for label in redact or []:
        d.rectangle(field_value_box(tpl, label), fill="black")

    d.text((LABEL_X, height - 50), tpl.footer, fill=(130, 130, 130), font=_font(20))

    if marking:
        targets = marking_over or []
        boxes = [field_value_box(tpl, t) for t in targets] or [
            (WIDTH // 2 - 160, height // 2 - 40, WIDTH // 2 + 160, height // 2 + 40)
        ]
        for x0, y0, x1, y1 in boxes:
            d.rectangle((x0, y0, x1, y1), outline=(200, 0, 0), width=4)
            # stamp text sits at the right edge so the value underneath stays legible
            d.text((x1 - 140, y0 + 4), marking, fill=(200, 0, 0), font=_font(28))
    return img


def layout_lines(
    tpl: DocTemplate,
    overrides: dict[str, str] | None = None,
    marking: str | None = None,
    marking_over: list[str] | None = None,
    redact: list[str] | None = None,
) -> list[OcrLine]:
    """The exact text a perfect OCR would read from render(). Used by OracleOcr."""
    overrides = overrides or {}
    lines = [
        OcrLine(text=tpl.issuer, box=(LABEL_X, 20, LABEL_X + 400, 50)),
        OcrLine(text=tpl.title, box=(LABEL_X, 95, LABEL_X + 600, 127)),
    ]
    for i, (label, value) in enumerate(tpl.fields):
        y = TOP + i * ROW_H
        lines.append(OcrLine(text=label, box=(LABEL_X, y, LABEL_X + 300, y + 26)))
        if label not in (redact or []):
            lines.append(OcrLine(text=overrides.get(label, value), box=(VALUE_X, y, VALUE_X + 300, y + 26)))
    height = TOP + len(tpl.fields) * ROW_H + 90
    lines.append(OcrLine(text=tpl.footer, box=(LABEL_X, height - 50, LABEL_X + 400, height - 30)))
    if marking:
        boxes = [field_value_box(tpl, t) for t in marking_over or []] or [
            (WIDTH // 2 - 160, height // 2 - 40, WIDTH // 2 + 160, height // 2 + 40)
        ]
        for x0, y0, x1, y1 in boxes:
            lines.append(OcrLine(text=marking, box=(x1 - 140, y0 + 4, x1 - 20, y0 + 34)))
    return lines


class OracleOcr:
    """Perfect OCR for rendered benchmark images: separates OCR errors from decision errors."""

    def __init__(self) -> None:
        self._by_hash: dict[str, list[OcrLine]] = {}

    def register(self, png: bytes, lines: list[OcrLine]) -> bytes:
        self._by_hash[hashlib.sha256(png).hexdigest()] = lines
        return png

    def read(self, image: bytes) -> list[OcrLine]:
        return self._by_hash.get(hashlib.sha256(image).hexdigest(), [])


def render_photo(shirt: str = "blue") -> Image.Image:
    """A plain 'photo' with no record-like text: the outfit-colour control case."""
    colors = {"blue": (40, 80, 200), "red": (200, 40, 40), "green": (40, 160, 70)}
    img = Image.new("RGB", (600, 800), (180, 210, 235))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 600, 600, 800), fill=(110, 160, 90))  # grass
    d.ellipse((240, 140, 360, 260), fill=(230, 190, 160))  # head
    d.rectangle((210, 270, 390, 520), fill=colors[shirt])  # shirt
    d.rectangle((225, 520, 375, 700), fill=(50, 50, 70))  # trousers
    return img


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
