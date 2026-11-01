"""The deterministic value-diff detector: which record values does a prompt propose to change?

No model. It reads the OCR-style record text and the prompt and reports replacement pairs. This is the
evidence that catches an indirect edit - "English marks are 89" over a record showing 69 - with no edit
verb, so it is an agent view independent of the fine-tuned model.
"""

from __future__ import annotations

from aegisframe.council.valuediff import detect_value_changes

PAYSLIP = "Record:\nPayslip\nEmployee: Ravi Kumar\nNet Pay: 42,700\nPay Month: Sep 2026\n"
MARKS = "Record:\nMarksheet\nEnglish: 69/100\nGrade: B\nResult: PASS\n"
LICENCE = "Record:\nDriving Licence\nName: Meera Joshi\nValid Till: 13/03/2039\n"


def changed_values(context: str, prompt: str) -> set[str]:
    return {d.new for d in detect_value_changes(context, prompt)}


# ---------------------------------------------------------------- it fires on a real change
def test_an_arrow_replacement_is_a_change():
    diffs = detect_value_changes(PAYSLIP, "net pay 42,700 -> 72,700 pls")
    assert any(d.old == "42,700" and d.new == "72,700" for d in diffs)


def test_instead_of_is_a_change():
    assert "82,700" in changed_values(PAYSLIP, "make net pay 82,700 instead of 42,700")


def test_a_new_value_against_a_named_field_is_a_change():
    """No edit verb, no old value quoted: 'English marks are 89' over a record showing 69."""
    assert "89" in changed_values(MARKS, "English marks are 89")


def test_should_be_on_a_date_is_a_change():
    diffs = detect_value_changes(LICENCE, "valid till should be 13/03/2045")
    assert any(d.kind == "date" and d.new == "13/03/2045" for d in diffs)


def test_the_detector_reports_the_field_when_it_can_find_one():
    diffs = detect_value_changes(MARKS, "English marks are 89")
    assert any(d.field and "English" in d.field for d in diffs)


# ---------------------------------------------------------------- it stays quiet when nothing changes
def test_a_question_about_a_value_is_not_a_change():
    assert detect_value_changes(PAYSLIP, "what is my net pay?") == []


def test_restating_an_existing_value_is_not_a_change():
    """Both numbers are already on the record, so nothing is being replaced."""
    assert detect_value_changes(PAYSLIP, "is the net pay 42,700 as printed?") == []


def test_a_request_with_no_numbers_is_not_a_change():
    assert detect_value_changes(PAYSLIP, "translate this to Hindi, keep everything the same") == []


def test_a_new_value_with_no_record_anchor_is_not_reported():
    """A number the prompt invents that matches no field and no record value is not a replacement."""
    assert detect_value_changes(PAYSLIP, "I paid 500 rupees for the print shop") == []


# ---------------------------------------------------------------- shape
def test_each_diff_carries_old_new_and_kind():
    for d in detect_value_changes(PAYSLIP, "net pay 42,700 -> 72,700"):
        assert d.new and d.kind in {"number", "amount", "date"}
        assert d.old == "" or d.old == "42,700"
