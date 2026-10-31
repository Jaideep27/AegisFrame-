import pytest

from aegisframe.ground import ground, pair_fields, resolve_field
from aegisframe.schema import FieldChange, OcrLine, Severity
from aegisframe.semantics import classify_change
from reframebench.docs import CATALOG, render, render_photo, to_png


def line(text, x, y):
    return OcrLine(text=text, box=(x, y, x + 10 * len(text), y + 26))


SLIP_LINES = [
    line("Orbitline Software Pvt Ltd", 60, 20),
    line("Payslip for September 2026", 60, 95),
    line("Employee Name", 60, 170), line("Arjun Mehta", 430, 171),
    line("Employee ID", 60, 216), line("OSL-20417", 430, 215),
    line("Net Pay", 60, 262), line("Rs. 52,000", 430, 263),
    line("Annual CTC", 60, 308), line("Rs. 7,20,000", 430, 309),
]


class FakeOcr:
    def __init__(self, lines):
        self.lines = lines

    def read(self, image: bytes):
        return self.lines


def test_pair_fields_joins_label_and_value_on_the_same_row():
    fields = pair_fields(SLIP_LINES)
    assert fields["Net Pay"] == "Rs. 52,000"
    assert fields["Employee Name"] == "Arjun Mehta"
    assert "Orbitline Software Pvt Ltd" not in fields


def test_record_with_fields_is_evidence():
    doc = ground(b"png", FakeOcr(SLIP_LINES))
    assert doc.is_document and doc.is_evidence and doc.real_source
    assert "Payslip" in doc.doc_type


def test_photo_without_text_is_not_evidence():
    doc = ground(b"png", FakeOcr([]))
    assert doc.has_image and not doc.is_document and not doc.is_evidence


def test_no_image_means_not_a_real_source():
    doc = ground(None, FakeOcr(SLIP_LINES))
    assert not doc.has_image and not doc.real_source and not doc.is_evidence


def test_resolve_field_by_old_value():
    fields = pair_fields(SLIP_LINES)
    assert resolve_field(FieldChange(kind="replace", old="52000", new="95000"), fields) == "Net Pay"


def test_resolve_field_by_label_synonym():
    fields = pair_fields(SLIP_LINES)
    change = FieldChange(kind="replace", field="compensation", new="12,00,000")
    assert resolve_field(change, fields) in {"Annual CTC", "Net Pay"}


def test_resolve_field_ignores_ocr_spacing():
    fields = {"UPIID": "x@nimbus"}
    assert resolve_field(FieldChange(kind="replace", field="UPI ID", new="y@nimbus"), fields) == "UPIID"


@pytest.mark.parametrize(
    "label,old,new,cls,sev",
    [
        ("Amount", "Rs. 500", "Rs. 5,000", "money", Severity.HIGH),
        ("Net Pay", None, "95,000", "money", Severity.HIGH),
        ("Candidate Name", "Priya Raghavan", "Kavya R", "identity", Severity.HIGH),
        ("Roll Number", "KSEB2026-118734", "KSEB2026-118799", "identity", Severity.HIGH),
        ("Date of Birth", "04 Mar 2005", "04 Mar 2004", "identity", Severity.HIGH),
        ("Result", "Positive", "Negative", "result", Severity.HIGH),
        ("Claim Status", "Rejected", "Approved", "result", Severity.HIGH),
        ("Exam Date", "18 Nov 2026", "25 Nov 2026", "date", Severity.MODERATE),
        ("Seat", "14C", "2A", "other", Severity.MODERATE),
        ("Signature", None, "A. Fernandes", "signature", Severity.CRITICAL),
        (None, "500", "5000", "money", Severity.HIGH),
    ],
)
def test_classify_change(label, old, new, cls, sev):
    assert classify_change(label, old, new) == (cls, sev)


def test_rapid_ocr_engine_caches_by_image_content():
    from aegisframe.ground import RapidOcrEngine

    calls = []

    class Result:
        txts, boxes, scores = ("Net Pay",), ([[0, 0], [10, 0], [10, 10], [0, 10]],), (0.99,)

    engine = RapidOcrEngine()
    engine._engine = lambda arr: calls.append(1) or Result()
    png = to_png(render_photo())
    assert engine.read(png)[0].text == "Net Pay"
    assert engine.read(png)[0].text == "Net Pay"
    assert len(calls) == 1


@pytest.mark.ocr
def test_ground_real_render_with_rapidocr():
    from aegisframe.ground import RapidOcrEngine

    doc = ground(to_png(render(CATALOG["salary_slip"])), RapidOcrEngine())
    assert doc.is_evidence
    assert doc.fields.get("Net Pay") == "Rs. 52,000"


@pytest.mark.ocr
def test_ground_photo_with_rapidocr():
    from aegisframe.ground import RapidOcrEngine

    doc = ground(to_png(render_photo()), RapidOcrEngine())
    assert not doc.is_evidence


def test_record_without_label_value_rows_is_still_evidence():
    # real payment apps show a big centred amount with no label next to it
    lines = [
        line("Payment successful", 200, 100),
        line("₹500", 260, 180),
        line("Paid to Sharma General Store", 150, 260),
        line("12 Oct 2026, 7:42 PM", 180, 320),
        line("UPI transaction ID 482190773155", 120, 380),
    ]
    doc = ground(b"png", FakeOcr(lines))
    assert doc.is_document and doc.is_evidence


def test_a_meme_with_a_few_words_is_not_evidence():
    lines = [line("WHEN THE WEEKEND", 100, 40), line("FINALLY ARRIVES", 110, 600)]
    assert not ground(b"png", FakeOcr(lines)).is_evidence
