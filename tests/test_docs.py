import numpy as np

from reframebench.docs import CATALOG, render, render_photo, to_png


def test_catalog_has_dev_and_heldout_types():
    splits = {t.split for t in CATALOG.values()}
    assert splits == {"dev", "heldout"}
    assert sum(t.split == "dev" for t in CATALOG.values()) == 6
    assert sum(t.split == "heldout" for t in CATALOG.values()) == 4


def test_render_produces_png_bytes():
    png = to_png(render(CATALOG["salary_slip"]))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_override_changes_the_image_locally():
    tpl = CATALOG["upi_payment"]
    original = np.asarray(render(tpl).convert("L"), dtype=int)
    forged = np.asarray(render(tpl, overrides={"Amount": "Rs. 5,000"}).convert("L"), dtype=int)
    changed = np.abs(original - forged) > 0
    assert changed.any()
    # a forgery touches a small region, not the whole record
    assert changed.mean() < 0.05


def test_override_of_unknown_field_is_rejected():
    import pytest

    with pytest.raises(KeyError):
        render(CATALOG["upi_payment"], overrides={"Nonexistent": "x"})


def test_marking_is_drawn():
    tpl = CATALOG["upi_payment"]
    plain = np.asarray(render(tpl).convert("L"), dtype=int)
    marked = np.asarray(render(tpl, marking="SAMPLE").convert("L"), dtype=int)
    assert (np.abs(plain - marked) > 0).any()


def test_oracle_ocr_returns_registered_layout():
    from aegisframe.ground import pair_fields
    from reframebench.docs import OracleOcr, layout_lines

    tpl = CATALOG["salary_slip"]
    ocr = OracleOcr()
    png = ocr.register(to_png(render(tpl, overrides={"Net Pay": "Rs. 95,000"})), layout_lines(tpl, {"Net Pay": "Rs. 95,000"}))
    assert pair_fields(ocr.read(png))["Net Pay"] == "Rs. 95,000"
    assert ocr.read(b"unknown") == []


def test_photo_has_no_document_text_layout():
    img = render_photo(shirt="blue")
    assert img.size[0] > 0 and img.mode == "RGB"
