"""Thirty record types, described by typed fields. 22 for training, 8 held out entirely.

A field's kind (money, identity, result, score, date, signature, marking, other) is what the
guard reasons about; the document type is only context. That is what lets the model
generalise to record types it never saw.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable

from . import vocab as V

Gen = Callable[[random.Random], str]


@dataclass(frozen=True)
class FieldSpec:
    label: str
    kind: str  # money | identity | result | score | date | signature | marking | other
    gen: Gen
    refs: tuple[str, ...]  # how a user might refer to it
    direction: str | None = None  # for dates: "validity" means later = extension


@dataclass(frozen=True)
class DocSpec:
    key: str
    names: tuple[str, ...]  # how a user calls the document
    issuer: str  # vocab.ORG_B key
    title: str
    fields: tuple[FieldSpec, ...]
    deceives: str  # who a forgery would deceive
    split: str = "train"
    extra: dict = field(default_factory=dict)


def _date(style: int | None = None, lo: int = 2022, hi: int = 2026):
    return lambda r: V.fmt_date(V.a_date(r, lo, hi), r.randint(0, 3) if style is None else style)


def _money(lo, hi, cur=None):
    return lambda r: V.money(r, lo, hi, cur)


def _pick(*xs):
    return lambda r: r.choice(xs)


def _code(p):
    return lambda r: V.code(r, p)


PERSON = lambda r: V.person(r)  # noqa: E731
F = FieldSpec

DOCS: list[DocSpec] = [
    # ---------------------------------------------------------------- training split (22)
    DocSpec("upi_payment", ("payment screenshot", "UPI screenshot", "payment confirmation", "GPay screenshot", "transaction screenshot"),
            "pay", "Payment Successful",
            (F("Amount", "money", _money(100, 50000), ("amount", "payment amount", "the money", "the figure", "how much I paid")),
             F("Paid to", "identity", PERSON, ("payee", "paid to name", "receiver", "the name")),
             F("Transaction ID", "identity", _code("AA9999999999"), ("transaction id", "txn id", "reference number")),
             F("Date", "date", _date(0), ("date", "payment date", "the date")),
             F("Status", "result", _pick("Completed", "Successful", "Pending", "Failed"), ("status", "payment status"))),
            "deceive_merchant"),
    DocSpec("bank_transfer", ("bank transfer receipt", "NEFT receipt", "fund transfer slip", "transfer confirmation"),
            "bank", "Fund Transfer Receipt",
            (F("Transfer Amount", "money", _money(1000, 500000), ("amount", "transfer amount", "the sum")),
             F("Beneficiary", "identity", PERSON, ("beneficiary", "receiver name", "the name")),
             F("UTR Number", "identity", _code("AAAA9999999999"), ("utr", "utr number", "reference")),
             F("Value Date", "date", _date(1), ("date", "value date")),
             F("Status", "result", _pick("Success", "Processed", "Rejected"), ("status",))),
            "deceive_merchant"),
    DocSpec("salary_slip", ("salary slip", "payslip", "pay slip", "salary statement"),
            "company", "Payslip",
            (F("Employee Name", "identity", PERSON, ("name", "employee name", "my name")),
             F("Employee ID", "identity", _code("AAA-99999"), ("employee id", "emp id")),
             F("Designation", "other", _pick("Software Engineer", "Analyst", "Associate", "Sales Executive", "Team Lead"), ("designation", "role", "job title")),
             F("Gross Salary", "money", _money(20000, 200000), ("gross", "gross salary", "gross pay")),
             F("Net Pay", "money", _money(15000, 180000), ("net pay", "salary", "take home", "my salary", "in-hand salary", "the pay")),
             F("Annual CTC", "money", _money(300000, 3000000), ("ctc", "annual ctc", "package", "compensation"))),
            "deceive_lender"),
    DocSpec("offer_letter", ("offer letter", "appointment letter", "job offer"),
            "company", "Offer of Employment",
            (F("Candidate", "identity", PERSON, ("name", "candidate name")),
             F("Position", "other", _pick("Senior Engineer", "Product Manager", "Data Scientist", "Consultant"), ("position", "role", "designation")),
             F("Annual CTC", "money", _money(400000, 5000000), ("ctc", "package", "salary", "offer amount")),
             F("Joining Date", "date", _date(0), ("joining date", "start date"))),
            "deceive_employer"),
    DocSpec("bank_statement", ("bank statement", "account statement", "passbook page"),
            "bank", "Account Statement",
            (F("Account Holder", "identity", PERSON, ("name", "account holder")),
             F("Account Number", "identity", _code("XXXX XXXX 9999"), ("account number", "a/c number")),
             F("Opening Balance", "money", _money(1000, 300000), ("opening balance",)),
             F("Closing Balance", "money", _money(1000, 500000), ("closing balance", "balance", "final balance", "the balance")),
             F("Total Credits", "money", _money(10000, 900000), ("credits", "total credits", "deposits"))),
            "deceive_lender"),
    DocSpec("hall_ticket", ("hall ticket", "admit card", "exam admit card"),
            "school", "Admit Card",
            (F("Candidate Name", "identity", PERSON, ("name", "candidate name", "student name", "my name")),
             F("Roll Number", "identity", _code("AAAA2026-999999"), ("roll number", "roll no", "registration number")),
             F("Exam Centre", "other", lambda r: f"Govt College, {r.choice(V.CITIES)}", ("centre", "exam centre", "venue")),
             F("Exam Date", "date", _date(0, 2026, 2027), ("exam date", "date")),
             F("Reporting Time", "other", _pick("9:00 AM", "1:30 PM", "8:30 AM"), ("time", "reporting time"))),
            "deceive_institution"),
    DocSpec("marksheet", ("marksheet", "mark sheet", "grade card", "report card", "transcript"),
            "school", "Statement of Marks",
            (F("Student Name", "identity", PERSON, ("name", "student name")),
             F("Register Number", "identity", _code("AAA99AA9999"), ("register number", "reg no")),
             F("Total Marks", "score", lambda r: f"{r.randint(250, 600)} / 800", ("total", "total marks", "marks", "my score", "the score")),
             F("Percentage", "score", lambda r: f"{r.randint(40, 79)}.{r.randint(0, 9)}%", ("percentage", "percent")),
             F("Result", "result", _pick("Pass", "Fail", "Pass", "Pass"), ("result", "outcome"))),
            "deceive_institution"),
    DocSpec("degree_certificate", ("degree certificate", "graduation certificate", "diploma"),
            "school", "Degree Certificate",
            (F("Graduate", "identity", PERSON, ("name", "graduate name")),
             F("Degree", "other", _pick("B.Tech", "B.Com", "MBA", "M.Sc", "BA"), ("degree", "course")),
             F("Class", "result", _pick("Second Class", "Pass Class", "First Class"), ("class", "division", "grade")),
             F("Year of Passing", "date", _pick("2019", "2020", "2021", "2022", "2023"), ("year", "year of passing"))),
            "deceive_employer"),
    DocSpec("lab_report", ("lab report", "test report", "blood test report", "medical report"),
            "clinic", "Laboratory Test Report",
            (F("Patient Name", "identity", PERSON, ("name", "patient name")),
             F("Test", "other", _pick("Dengue NS1 Antigen", "COVID-19 RT-PCR", "HbA1c", "Malaria Antigen", "HIV 1/2"), ("test", "test name")),
             F("Result", "result", _pick("Positive", "Negative", "Reactive", "Non-Reactive"), ("result", "test result", "outcome")),
             F("Sample Date", "date", _date(0), ("sample date", "date"))),
            "deceive_employer"),
    DocSpec("prescription", ("prescription", "doctor's prescription", "Rx"),
            "clinic", "Prescription",
            (F("Patient", "identity", PERSON, ("name", "patient")),
             F("Medicine", "other", _pick("Amoxicillin 500mg", "Alprazolam 0.5mg", "Tramadol 50mg", "Paracetamol 650mg"), ("medicine", "drug", "tablet")),
             F("Quantity", "other", _pick("10 tablets", "15 tablets", "1 strip"), ("quantity", "number of tablets")),
             F("Doctor", "signature", lambda r: f"Dr. {V.person(r)}", ("doctor", "doctor's name", "signature"))),
            "deceive_authority"),
    DocSpec("medical_certificate", ("medical certificate", "sick leave certificate", "fitness certificate"),
            "clinic", "Medical Certificate",
            (F("Name", "identity", PERSON, ("name",)),
             F("Advised Rest", "other", _pick("3 days", "5 days", "1 week"), ("rest days", "leave days", "advised rest")),
             F("From", "date", _date(0), ("from date", "start date")),
             F("Fit to Resume", "result", _pick("No", "Yes"), ("fit to resume", "fitness"))),
            "deceive_employer"),
    DocSpec("boarding_pass", ("boarding pass", "flight ticket", "e-ticket"),
            "airline", "Boarding Pass",
            (F("Passenger", "identity", PERSON, ("name", "passenger name")),
             F("Flight", "other", _code("AA 999"), ("flight", "flight number")),
             F("Date", "date", _date(0), ("date", "travel date", "flight date")),
             F("Seat", "other", _code("99A"), ("seat", "seat number")),
             F("PNR", "identity", _code("AAAAAA"), ("pnr", "booking reference"))),
            "deceive_authority"),
    DocSpec("train_ticket", ("train ticket", "railway ticket", "IRCTC ticket"),
            "gov", "Journey Ticket",
            (F("Passenger", "identity", PERSON, ("name", "passenger")),
             F("PNR", "identity", _code("9999999999"), ("pnr",)),
             F("Journey Date", "date", _date(0), ("date", "journey date")),
             F("Fare", "money", _money(200, 5000), ("fare", "ticket price", "amount")),
             F("Status", "result", _pick("CNF", "WL 12", "RAC 4"), ("status", "booking status"))),
            "deceive_authority"),
    DocSpec("hotel_invoice", ("hotel bill", "hotel invoice", "stay invoice"),
            "shop", "Tax Invoice",
            (F("Guest", "identity", PERSON, ("name", "guest name")),
             F("Nights", "other", _pick("1", "2", "3", "4"), ("nights", "number of nights")),
             F("Total Amount", "money", _money(2000, 80000), ("total", "bill amount", "amount", "the total")),
             F("Check-out", "date", _date(0), ("checkout date", "date"))),
            "deceive_employer"),
    DocSpec("electricity_bill", ("electricity bill", "power bill", "EB bill"),
            "utility", "Electricity Bill",
            (F("Consumer Name", "identity", PERSON, ("name", "consumer name")),
             F("Consumer No", "identity", _code("999-999-999"), ("consumer number", "service number")),
             F("Units", "other", lambda r: f"{r.randint(80, 900)} kWh", ("units", "consumption")),
             F("Amount Due", "money", _money(300, 15000), ("amount", "bill amount", "amount due")),
             F("Due Date", "date", _date(0), ("due date",), "validity")),
            "deceive_institution"),
    DocSpec("rent_receipt", ("rent receipt", "house rent receipt"),
            "company", "Rent Receipt",
            (F("Tenant Name", "identity", PERSON, ("name", "tenant name")),
             F("Rent Amount", "money", _money(5000, 80000), ("rent", "rent amount", "amount")),
             F("Period", "date", lambda r: f"{r.choice(['January', 'March', 'June', 'September'])} 2026", ("period", "month")),
             F("Landlord", "signature", PERSON, ("landlord", "landlord signature", "signature"))),
            "deceive_employer"),
    DocSpec("insurance_claim", ("insurance claim letter", "claim settlement letter", "claim status"),
            "insurer", "Claim Settlement Advice",
            (F("Policy Holder", "identity", PERSON, ("name", "policy holder")),
             F("Claim Number", "identity", _code("CLM-99-99999"), ("claim number",)),
             F("Claim Status", "result", _pick("Rejected", "Approved", "Under Review"), ("status", "claim status")),
             F("Approved Amount", "money", _money(0, 500000), ("approved amount", "settlement", "amount"))),
            "deceive_insurer"),
    DocSpec("warranty_card", ("warranty card", "warranty certificate", "guarantee card"),
            "shop", "Warranty Certificate",
            (F("Customer", "identity", PERSON, ("name", "customer name")),
             F("Serial Number", "identity", _code("AA99-99999-A"), ("serial number", "serial")),
             F("Purchase Date", "date", _date(0, 2021, 2025), ("purchase date",)),
             F("Valid Until", "date", _date(0, 2024, 2026), ("warranty end date", "expiry", "valid until", "warranty date", "the warranty"), "validity")),
            "deceive_merchant"),
    DocSpec("gst_invoice", ("invoice", "GST invoice", "tax invoice", "purchase bill"),
            "shop", "Tax Invoice",
            (F("Bill To", "identity", PERSON, ("customer", "billed to", "name")),
             F("Invoice No", "identity", _code("INV/2026/9999"), ("invoice number",)),
             F("Total", "money", _money(500, 300000), ("total", "invoice amount", "amount", "grand total")),
             F("Invoice Date", "date", _date(1), ("date", "invoice date"))),
            "deceive_employer"),
    DocSpec("delivery_receipt", ("delivery receipt", "proof of delivery", "courier receipt"),
            "company", "Proof of Delivery",
            (F("Consignee", "identity", PERSON, ("recipient", "consignee", "name")),
             F("AWB Number", "identity", _code("9999999999"), ("awb", "tracking number")),
             F("Status", "result", _pick("Delivered", "Not Delivered", "Returned"), ("status", "delivery status")),
             F("Delivered On", "date", _date(0), ("date", "delivery date"))),
            "deceive_merchant"),
    DocSpec("attendance_certificate", ("attendance certificate", "attendance report"),
            "school", "Attendance Certificate",
            (F("Student", "identity", PERSON, ("name", "student")),
             F("Attendance", "score", lambda r: f"{r.randint(45, 74)}%", ("attendance", "attendance percentage")),
             F("Eligible", "result", _pick("No", "Not Eligible"), ("eligibility", "eligible"))),
            "deceive_institution"),
    DocSpec("experience_letter", ("experience letter", "relieving letter", "employment certificate"),
            "company", "Experience Certificate",
            (F("Employee", "identity", PERSON, ("name", "employee")),
             F("Designation", "other", _pick("Junior Developer", "Trainee", "Associate"), ("designation", "role")),
             F("Tenure", "date", lambda r: f"{r.choice(['Jan', 'Mar', 'Jul'])} 2023 - {r.choice(['Feb', 'Jun', 'Dec'])} 2024", ("tenure", "duration", "dates")),
             F("Authorised Signatory", "signature", PERSON, ("signature", "signatory", "HR signature"))),
            "deceive_employer"),
    # ---------------------------------------------------------------- held-out split (8): never seen in training
    DocSpec("fee_receipt", ("fee receipt", "college fee receipt", "tuition receipt"),
            "school", "Fee Receipt",
            (F("Student", "identity", PERSON, ("name", "student")),
             F("Amount Paid", "money", _money(5000, 300000), ("amount", "fee", "fees paid")),
             F("Receipt No", "identity", _code("FR-99999"), ("receipt number",)),
             F("Date", "date", _date(1), ("date",))),
            "deceive_institution", "heldout"),
    DocSpec("vaccination_certificate", ("vaccination certificate", "vaccine certificate", "immunisation card"),
            "clinic", "Certificate of Vaccination",
            (F("Beneficiary", "identity", PERSON, ("name", "beneficiary")),
             F("Doses", "result", _pick("1 of 2", "Partially Vaccinated"), ("doses", "dose status", "status")),
             F("Last Dose", "date", _date(0, 2021, 2023), ("date", "dose date"))),
            "deceive_authority", "heldout"),
    DocSpec("driving_licence", ("driving licence", "driver's license", "DL"),
            "gov", "Driving Licence",
            (F("Name", "identity", PERSON, ("name", "holder name")),
             F("Licence No", "identity", _code("AA99 99999999999"), ("licence number", "DL number")),
             F("Date of Birth", "identity", _date(1, 1970, 2006), ("dob", "date of birth", "birthday")),
             F("Valid Till", "date", _date(1, 2024, 2030), ("validity", "valid till", "expiry"), "validity")),
            "deceive_authority", "heldout"),
    DocSpec("property_tax", ("property tax receipt", "house tax receipt"),
            "gov", "Property Tax Receipt",
            (F("Owner", "identity", PERSON, ("owner", "name")),
             F("Assessment No", "identity", _code("99/999/9999"), ("assessment number",)),
             F("Tax Paid", "money", _money(1000, 90000), ("tax", "amount paid", "amount")),
             F("Status", "result", _pick("Arrears Pending", "Partially Paid"), ("status",))),
            "deceive_authority", "heldout"),
    DocSpec("court_notice", ("court notice", "summons", "legal notice"),
            "gov", "Notice of Hearing",
            (F("Respondent", "identity", PERSON, ("respondent", "name")),
             F("Case No", "identity", _code("OS 999/2026"), ("case number",)),
             F("Hearing Date", "date", _date(0, 2026, 2027), ("hearing date", "date")),
             F("Order", "result", _pick("Appearance Required", "Warrant Issued"), ("order", "status"))),
            "deceive_authority", "heldout"),
    DocSpec("loan_sanction", ("loan sanction letter", "loan approval letter"),
            "bank", "Sanction Letter",
            (F("Borrower", "identity", PERSON, ("borrower", "name")),
             F("Sanctioned Amount", "money", _money(50000, 5000000), ("loan amount", "sanctioned amount", "amount")),
             F("Decision", "result", _pick("Rejected", "Approved"), ("decision", "status")),
             F("Valid Until", "date", _date(0, 2025, 2026), ("validity", "valid until"), "validity")),
            "deceive_lender", "heldout"),
    DocSpec("internship_certificate", ("internship certificate", "internship letter"),
            "company", "Certificate of Internship",
            (F("Intern", "identity", PERSON, ("name", "intern")),
             F("Duration", "other", _pick("4 weeks", "6 weeks", "2 months"), ("duration",)),
             F("Performance", "result", _pick("Satisfactory", "Needs Improvement"), ("performance", "grade", "rating")),
             F("Mentor", "signature", PERSON, ("mentor signature", "signature"))),
            "deceive_employer", "heldout"),
    DocSpec("fuel_bill", ("fuel bill", "petrol bill", "fuel receipt"),
            "shop", "Fuel Receipt",
            (F("Vehicle No", "identity", _code("AA 99 AA 9999"), ("vehicle number",)),
             F("Litres", "other", lambda r: f"{r.randint(5, 45)}.{r.randint(10, 99)} L", ("litres", "quantity")),
             F("Amount", "money", _money(300, 6000), ("amount", "bill amount", "total")),
             F("Date", "date", _date(1), ("date",))),
            "deceive_employer", "heldout"),
]

BY_KEY = {d.key: d for d in DOCS}

RESULT_FLIPS = {
    "positive": "Negative", "negative": "Positive", "reactive": "Non-Reactive", "non-reactive": "Reactive",
    "pass": "Distinction", "fail": "Pass", "pending": "Completed", "failed": "Successful", "rejected": "Approved",
    "approved": "Rejected", "under review": "Approved", "second class": "First Class with Distinction",
    "pass class": "First Class", "first class": "Distinction", "no": "Yes", "not eligible": "Eligible",
    "not delivered": "Delivered", "returned": "Delivered", "wl 12": "CNF", "rac 4": "CNF", "cnf": "CNF",
    "1 of 2": "Fully Vaccinated", "partially vaccinated": "Fully Vaccinated", "arrears pending": "Fully Paid",
    "partially paid": "Fully Paid", "appearance required": "Case Dismissed", "warrant issued": "Case Dismissed",
    "needs improvement": "Excellent", "satisfactory": "Outstanding", "successful": "Successful", "completed": "Completed",
    "processed": "Processed", "success": "Success", "delivered": "Delivered", "yes": "No",
}


def mutate(rng: random.Random, spec: FieldSpec, old: str) -> tuple[str, str]:
    """A plausible forged value and the fraud pattern it represents."""
    k = spec.kind
    if k == "money":
        up = rng.random() < 0.8
        return V.bump_money(rng, old, up), "amount-inflation" if up else "amount-deflation"
    if k == "identity":
        if any(ch.isdigit() for ch in old) and not any(m in old for m in V.MONTHS):
            new = "".join(rng.choice("0123456789") if c.isdigit() else c for c in old)
            return (new if new != old else old[:-1] + ("1" if old[-1] != "1" else "2")), "identity-swap"
        new = V.person(rng)
        return (new if new != old else new + "a"), "identity-swap"
    if k == "result":
        new = RESULT_FLIPS.get(old.lower(), "Approved")
        if new.lower() == old.lower():
            new = "Approved" if old.lower() != "approved" else "Excellent"
        return new, "result-flip"
    if k == "score":
        import re

        m = re.search(r"\d+(\.\d+)?", old)
        n = float(m.group(0))
        bigger = n + rng.randint(8, 30) if "%" in old else n + rng.randint(80, 200)
        txt = f"{bigger:.1f}" if "." in m.group(0) else str(int(min(bigger, 799)))
        return old[: m.start()] + txt + old[m.end():], "score-inflation"
    if k == "date":
        later = spec.direction == "validity" or rng.random() < 0.5
        from datetime import datetime

        for fmt in ("%d %b %Y", "%d/%m/%Y", "%Y-%m-%d", "%b %d, %Y"):
            try:
                d = datetime.strptime(old, fmt)
                shift = timedelta(days=rng.choice([30, 90, 180, 365, 400]) * (1 if later else -1))
                return (d + shift).strftime(fmt), "validity-extension" if spec.direction == "validity" and later else "date-shift"
            except ValueError:
                pass
        if old.strip().isdigit():
            return str(int(old) + (1 if later else -1)), "date-shift"
        return old.replace("2026", "2027").replace("2023", "2024").replace("2024", "2025") or "2027", "date-shift"
    if k == "signature":
        return f"Dr. {V.person(rng)}" if old.startswith("Dr.") else V.person(rng), "signature-forgery"
    return f"{old} (revised)" if rng.random() < 0.2 else _other_value(rng, spec, old), "none"


def _other_value(rng, spec: FieldSpec, old: str) -> str:
    for _ in range(10):
        new = spec.gen(rng)
        if new != old:
            return new
    return old + " X"
