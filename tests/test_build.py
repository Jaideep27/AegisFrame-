import csv
import json

from reframebench.build import export


def test_export_writes_images_cases_and_a_schema_free_annotation_sheet(tmp_path):
    export(tmp_path)
    cases = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 320
    assert (tmp_path / "images" / "salary_slip.png").read_bytes()[:4] == b"\x89PNG"

    with (tmp_path / "annotation_sheet.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # annotators label only plain requests and controls, and never see our attribute schema
    assert {r["kind"] for r in rows} == {"plain", "control"}
    assert len(rows) == 20 + 80
    assert "expected" not in rows[0] and "is_evidence" not in rows[0]
    assert set(rows[0]) >= {"case_id", "image", "request", "your_label (allow/review/block)"}
