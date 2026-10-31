"""~47 universal record families, each instantiated for any locale with internally consistent facts.

A family describes what kind of facts a record holds and who relies on it; the locale decides currency,
number and date formats, names and (for some families) label language. Totals add up (payslip
components, invoice tax, bank running balance, GPA from grade points), so a forged change is a
change to a coherent record, as in real life.

Families marked ood=True are never used for training: they form the unseen-record-type test set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from aegisframe.ocr.layout import Layout, Table

from . import regions as R
from .regions import Locale


@dataclass
class Fact:
    key: str                 # canonical English label
    label: str               # label as printed on this record
    value: str
    cls: str = "none"        # aegisframe.local.labels.FACT_CLASS
    kind: str = "text"       # money | date | person | id | text | grade | number | status | address
    raw: Any = None          # number / date behind the printed value
    critical: bool = False   # a forger would want to change it
    alt: str | None = None   # the value a forger would want (status / result flips)
    validity: bool = False   # a date where later means "still valid"
    scale: float | None = None  # best possible value of a score (10, 4.0, 100, 9.0); 1.0 for German grades


@dataclass
class DocTable:
    header: list[str]
    rows: list[list[Fact]]


@dataclass
class Doc:
    family: str
    category: str
    locale: str
    title: str
    issuer: str
    holder: str
    names: tuple[str, ...]
    reliance: tuple[str, ...]
    facts: list[Fact] = field(default_factory=list)
    tables: list[DocTable] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def all_facts(self) -> list[Fact]:
        return self.facts + [c for t in self.tables for row in t.rows for c in row]

    def critical(self) -> list[Fact]:
        return [f for f in self.all_facts() if f.critical]

    def layout(self, overrides: dict[int, str] | None = None) -> Layout:
        """overrides: id(fact) -> new printed value (used to render a forged or marked version)."""
        ov = overrides or {}
        val = lambda f: ov.get(id(f), f.value)  # noqa: E731
        return Layout(
            title=self.title, issuer=self.issuer,
            fields=[(f.label, val(f)) for f in self.facts],
            tables=[Table(t.header, [[val(c) for c in row] for row in t.rows]) for t in self.tables],
            text=list(self.notes),
        )


class B:
    """Small builder so each family reads like the record it describes."""

    def __init__(self, rng: random.Random, loc: Locale):
        self.rng, self.loc = rng, loc
        self.facts: list[Fact] = []
        self.cur = rng.choice(loc.currency)

    def lab(self, key: str) -> str:
        return R.label(key, self.loc.doc_lang)

    def add(self, f: Fact, row: bool = False) -> Fact:
        if not row:
            self.facts.append(f)
        return f

    def money(self, key: str, x: float, critical: bool = True, row: bool = False, cls: str = "monetary",
              exact: bool = False) -> Fact:
        """exact=True for totals derived from already-rounded parts, so the record adds up."""
        x = round(x, 2) if exact else R.nice(x, self.loc)
        return self.add(Fact(key, self.lab(key), R.fmt_money(x, self.loc, self.cur), cls, "money", x, critical), row)

    def salary(self, lo: float, hi: float) -> float:
        return R.nice(self.loc.salary * self.rng.uniform(lo, hi), self.loc)

    def date(self, key: str, d: date, critical: bool = False, validity: bool = False, row: bool = False,
             cls: str = "validity_date") -> Fact:
        return self.add(Fact(key, self.lab(key), R.fmt_date(d, self.loc), cls, "date", d, critical, None, validity),
                        row)

    def person(self, key: str, name: str | None = None, critical: bool = True) -> Fact:
        name = name or R.person(self.rng, self.loc)
        return self.add(Fact(key, self.lab(key), name, "identity", "person", name, critical))

    def ident(self, key: str, pattern: str, critical: bool = False) -> Fact:
        return self.add(Fact(key, self.lab(key), R.code(self.rng, pattern), "identity", "id", None, critical))

    def text(self, key: str, value: str, cls: str = "none", critical: bool = False, alt: str | None = None,
             kind: str = "text", row: bool = False, raw: Any = None) -> Fact:
        return self.add(Fact(key, self.lab(key), value, cls, kind, raw, critical, alt), row)

    def status(self, key: str, bad: str, good: str, cls: str, p_bad: float = 0.5) -> Fact:
        """A status a forger would flip. Records start in the unfavourable state with probability p_bad."""
        if self.rng.random() < p_bad:
            return self.text(key, bad, cls, True, good, "status")
        return self.text(key, good, cls, False, None, "status")

    def number(self, key: str, n: float, cls: str = "quantity_other", critical: bool = True, fmt: str = "{:,}",
               row: bool = False, scale: float | None = None) -> Fact:
        return self.add(Fact(key, self.lab(key), fmt.format(n), cls, "number", n, critical, scale=scale), row)

    def address(self, key: str = "Address", critical: bool = True) -> Fact:
        a = f"{self.rng.randint(2, 480)} {self.rng.choice(['Park', 'Lake', 'Station', 'Market', 'Hill', 'Garden'])} " \
            f"{self.rng.choice(['Road', 'Street', 'Avenue', 'Lane'])}, {self.rng.choice(self.loc.cities)}"
        return self.add(Fact(key, self.lab(key), a, "identity", "address", a, critical))

    def when(self, lo: int = 2024, hi: int = 2026) -> date:
        return R.a_date(self.rng, lo, hi)


@dataclass(frozen=True)
class Family:
    key: str
    category: str
    names: tuple[str, ...]          # how users refer to it (teacher localises)
    reliance: tuple[str, ...]       # who would be deceived by a forgery
    build: Callable[[B], tuple[str, str, str, list[DocTable], list[str]]]  # -> title, issuer, holder, tables, notes
    ood: bool = False


FAMILIES: dict[str, Family] = {}


def family(key: str, category: str, names: tuple[str, ...], reliance: tuple[str, ...], ood: bool = False):
    def deco(fn):
        FAMILIES[key] = Family(key, category, names, reliance, fn, ood)
        return fn
    return deco


VAT = {"US": 7.0, "UK": 20.0, "DE": 19.0, "FR": 20.0, "ES": 21.0, "IN": 18.0, "ID": 11.0, "PH": 12.0, "SG": 9.0,
       "CN": 13.0, "AE": 5.0, "SA": 15.0, "NG": 7.5, "KE": 16.0, "ZA": 15.0, "BR": 17.0, "MX": 16.0}
TAX_ID = {"IN": "AAAAA9999A", "US": "999-99-9999", "UK": "AA 99 99 99 A", "BR": "999.999.999-99",
          "MX": "AAAA999999AAA", "DE": "99 999 999 999", "FR": "9 99 99 99 999 999"}
ACCOUNT = {"DE": "DE99 9999 9999 9999 9999 99", "FR": "FR99 9999 9999 9999 9999 9999 999",
           "ES": "ES99 9999 9999 9999 9999 9999", "UK": "99-99-99 99999999"}
PERIOD_FMT = lambda b, d: f"{R.MONTHS.get(b.loc.doc_lang, R.MONTHS['en'])[d.month - 1]} {d.year}"  # noqa: E731


def gov(b: B, office: str) -> str:
    return f"{b.rng.choice(b.loc.cities)} {office}"


def acct(b: B) -> str:
    return R.code(b.rng, ACCOUNT.get(b.loc.code, "XXXX XXXX 9999"))


# ------------------------------------------------------------------ employment and income
@family("payslip", "employment", ("payslip", "salary slip", "pay stub"), ("lender_landlord", "government"))
def _payslip(b: B):
    org = R.org(b.rng, b.loc, "company")
    name = b.person("Employee Name").value
    b.ident("Employee ID", b.rng.choice(["AAA-99999", "E99999", "9999999"]))
    b.text("Designation", b.rng.choice(["Software Engineer", "Accountant", "Sales Executive", "Operations Lead",
                                        "Nurse", "Teacher", "Data Analyst", "Store Manager"]))
    d = b.when()
    b.text("Pay Period", PERIOD_FMT(b, d), "validity_date")
    gross = b.salary(0.6, 2.5)
    parts = [("Basic Salary", 0.5), ("Housing Allowance", 0.2), ("Transport Allowance", 0.08), ("Bonus", 0.0)]
    rows, left = [], gross
    for i, (k, share) in enumerate(parts):
        amt = left if i == len(parts) - 1 else R.nice(gross * share, b.loc)
        left -= amt
        rows.append([b.text("Component", k, row=True), b.money(k, amt, critical=k != "Transport Allowance", row=True, exact=True)])
    ded = R.nice(gross * b.rng.uniform(0.08, 0.3), b.loc)
    b.money("Gross Pay", gross)
    b.money("Total Deductions", ded, critical=False)
    b.money("Net Pay", gross - ded, exact=True)
    if b.loc.code == "IN":
        b.money("Annual CTC", R.nice(gross * 12 * b.rng.uniform(1.05, 1.2), b.loc))
    return f"Payslip - {PERIOD_FMT(b, d)}", org, name, [DocTable(["Earnings", "Amount"], rows)], \
        ["This is a system generated payslip."]


@family("tax_wage_statement", "employment", ("annual tax wage statement", "employer tax certificate",
                                              "year-end wage summary"), ("lender_landlord", "government"))
def _wage(b: B):
    org = R.org(b.rng, b.loc, "company")
    name = b.person("Employee").value
    if b.loc.code in TAX_ID:
        b.ident("Tax ID", TAX_ID[b.loc.code])
    y = b.rng.randint(2023, 2025)
    b.text("Tax Year", f"{y}-{(y + 1) % 100:02d}" if b.loc.code in ("IN", "UK") else str(y), "validity_date")
    wages = b.salary(8, 30)
    b.money("Total Wages", wages)
    b.money("Tax Withheld", wages * b.rng.uniform(0.05, 0.25), critical=False)
    b.money("Social Security Contributions", wages * 0.06, critical=False)
    return "Statement of Annual Earnings and Tax Withheld", org, name, [], []


@family("tax_return_ack", "employment", ("tax return acknowledgement", "income tax return receipt",
                                          "tax assessment notice"), ("lender_landlord", "government"))
def _itr(b: B):
    name = b.person("Taxpayer Name").value
    b.ident("Tax ID", TAX_ID.get(b.loc.code, "999999999"))
    y = b.rng.randint(2023, 2025)
    b.text("Assessment Year", str(y), "validity_date")
    inc = b.salary(8, 40)
    b.money("Total Income", inc)
    b.money("Tax Payable", inc * b.rng.uniform(0.05, 0.2), critical=False)
    b.status("Status", "Under Processing", "Processed", "credential_result", 0.3)
    b.date("Filing Date", date(y, b.rng.randint(6, 12), b.rng.randint(1, 28)))
    return "Acknowledgement of Tax Return", gov(b, "Revenue Office"), name, [], []


@family("offer_letter", "employment", ("offer letter", "job offer", "appointment letter"),
        ("lender_landlord", "employer", "government"))
def _offer(b: B):
    org = R.org(b.rng, b.loc, "company")
    name = b.person("Candidate").value
    b.text("Position", b.rng.choice(["Senior Analyst", "Product Manager", "Civil Engineer", "Registered Nurse",
                                     "Backend Developer", "Branch Manager"]), "credential_result", True,
           kind="text")
    b.money("Annual Salary", b.salary(10, 40))
    b.date("Joining Date", b.when(2025, 2027), critical=True)
    b.text("Authorised Signatory", "[signed] HR Director", "authorisation", True, kind="signature")
    return "Letter of Offer", org, name, [], ["We are pleased to offer you the position on the terms below."]


@family("experience_letter", "employment", ("experience letter", "relieving letter", "reference letter",
                                             "employment certificate"), ("employer", "government"))
def _exp(b: B):
    org = R.org(b.rng, b.loc, "company")
    name = b.person("Employee Name").value
    b.text("Designation", b.rng.choice(["Junior Developer", "Accounts Assistant", "Team Lead", "Lab Technician"]),
           "credential_result", True)
    start = b.when(2016, 2022)
    b.date("Employed From", start, critical=True, cls="validity_date")
    b.date("Employed Until", start + timedelta(days=b.rng.randint(200, 1500)), critical=True)
    b.money("Last Drawn Monthly Salary", b.salary(0.6, 2))
    b.text("Signed By", "[signed] Head of Human Resources", "authorisation", True, kind="signature")
    return "Experience Certificate", org, name, [], ["To whom it may concern."]


@family("employment_verification", "employment", ("employment verification letter", "proof of employment",
                                                   "salary certificate"), ("lender_landlord", "government"))
def _ev(b: B):
    org = R.org(b.rng, b.loc, "company")
    name = b.person("Employee").value
    b.status("Employment Status", "Terminated", "Active, Full-time", "credential_result", 0.25)
    b.date("Start Date", b.when(2015, 2024))
    b.money("Annual Gross Salary", b.salary(10, 35))
    b.text("Verified By", "[signed] Payroll Officer", "authorisation", True, kind="signature")
    return "Employment and Salary Verification", org, name, [], []


# ------------------------------------------------------------------ banking and payments
@family("bank_statement", "banking", ("bank statement", "account statement", "passbook page"),
        ("lender_landlord", "government"))
def _bank(b: B):
    org = R.org(b.rng, b.loc, "bank")
    name = b.person("Account Holder").value
    b.text("Account Number", acct(b), "identity")
    d0 = b.when()
    b.text("Statement Period", PERIOD_FMT(b, d0), "validity_date")
    bal = b.salary(0.2, 4)
    b.money("Opening Balance", bal)
    rows = []
    for i in range(b.rng.randint(4, 8)):
        d = d0.replace(day=min(28, 1 + i * 4 + b.rng.randint(0, 3)))
        credit = b.rng.random() < 0.35
        amt = b.salary(0.6, 1.5) if credit and i == 0 else b.salary(0.01, 0.4)
        bal = round(bal + (amt if credit else -amt), 2)
        if credit:
            desc = ("SALARY " + R.org(b.rng, b.loc, "company").split()[0].upper()) if i == 0 else \
                b.rng.choice(["REFUND", "CASH DEPOSIT", "INTEREST CREDIT", "TRANSFER FROM " + R.person(b.rng, b.loc)])
        else:
            desc = b.rng.choice(["POS GROCERY", "ATM WITHDRAWAL", "ONLINE TRANSFER", "UTILITY PAYMENT",
                                 "CARD PURCHASE", "RENT", "MOBILE TOPUP", "LOAN EMI"])
        rows.append([b.date("Date", d, row=True), b.text("Description", desc, row=True),
                     b.money("Amount", amt if credit else -amt, critical=credit, row=True),
                     b.money("Balance", bal, critical=False, row=True, exact=True)])
    b.money("Closing Balance", bal, exact=True)
    head = [b.lab("Date"), b.lab("Description"), b.lab("Amount"), b.lab("Balance")]
    return f"Account Statement - {PERIOD_FMT(b, d0)}", org, name, [DocTable(head, rows)], []


@family("payment_screenshot", "banking", ("payment screenshot", "payment confirmation", "transfer success screen",
                                           "wallet receipt"), ("merchant_counterparty", "platform_public"))
def _pay(b: B):
    w = R.org(b.rng, b.loc, "wallet").rsplit(" ", 1)[0] if b.loc.code not in ("IN",) else \
        b.rng.choice(R.ORG_PARTS) + " Pay"
    st = b.status("Status", "Payment Failed", "Payment Successful", "credential_result", 0.3)
    b.money("Amount", b.salary(0.005, 1.2))
    b.text("Paid To", R.org(b.rng, b.loc, "shop") if b.rng.random() < 0.6 else R.person(b.rng, b.loc),
           "identity", True)
    b.ident("Transaction ID", b.rng.choice(["999999999999", "AA9999999999", "T99999999999999"]), True)
    d = b.when()
    b.text("Date & Time", f"{R.fmt_date(d, b.loc)}, {b.rng.randint(1, 12)}:{b.rng.randint(0, 59):02d} "
                          f"{b.rng.choice(['AM', 'PM'])}", "validity_date", True, kind="date", raw=d)
    payer = R.person(b.rng, b.loc)
    b.text("From", payer, "identity")
    return st.value, w, payer, [], []


@family("bank_transfer_receipt", "banking", ("bank transfer receipt", "wire confirmation", "remittance advice"),
        ("merchant_counterparty", "lender_landlord"))
def _wire(b: B):
    org = R.org(b.rng, b.loc, "bank")
    payer = b.person("Sender").value
    b.person("Beneficiary")
    b.text("Beneficiary Account", acct(b), "identity", True)
    b.money("Amount", b.salary(0.3, 6))
    b.ident("Reference", "AAA99999999")
    b.date("Value Date", b.when(), critical=True)
    b.status("Status", "Pending", "Completed", "credential_result", 0.35)
    return "Funds Transfer Confirmation", org, payer, [], []


@family("cheque", "banking", ("cheque", "check", "bank cheque"), ("merchant_counterparty", "lender_landlord"),
        ood=True)
def _cheque(b: B):
    org = R.org(b.rng, b.loc, "bank")
    b.person("Pay")
    b.money("Amount", b.salary(0.1, 3))
    b.date("Date", b.when(), critical=True)
    b.ident("Cheque No", "999999")
    holder = R.person(b.rng, b.loc)
    b.text("Account Holder Signature", "[blank]", "authorisation", True, "[signed] " + holder, kind="signature")
    return "Cheque", org, holder, [], []


@family("credit_report", "banking", ("credit report", "credit score report", "credit bureau report"),
        ("lender_landlord",))
def _credit(b: B):
    name = b.person("Name").value
    b.date("Report Date", b.when())
    score = b.rng.randint(480, 690)
    b.number("Credit Score", score, "credential_result", scale=850)
    b.number("Accounts in Default", b.rng.randint(1, 4), "credential_result")
    b.money("Total Outstanding", b.salary(2, 20))
    return "Consumer Credit Report", f"{b.rng.choice(R.ORG_PARTS)} Credit Bureau", name, [], []


@family("loan_sanction_letter", "banking", ("loan sanction letter", "loan approval letter", "mortgage approval"),
        ("government", "lender_landlord"))
def _loan(b: B):
    org = R.org(b.rng, b.loc, "bank")
    name = b.person("Applicant").value
    b.money("Sanctioned Amount", b.salary(10, 80))
    b.text("Interest Rate", f"{b.rng.uniform(6, 14):.2f}% p.a.", "monetary")
    b.text("Tenure", f"{b.rng.choice([60, 120, 180, 240])} months")
    b.status("Status", "Rejected", "Sanctioned", "credential_result", 0.4)
    b.date("Valid Until", b.when(2025, 2027), True, True)
    return "Loan Sanction Letter", org, name, [], []


@family("balance_certificate", "banking", ("bank balance certificate", "proof of funds letter",
                                            "account balance confirmation"), ("government", "education"))
def _bal(b: B):
    org = R.org(b.rng, b.loc, "bank")
    name = b.person("Account Holder").value
    b.text("Account Number", acct(b), "identity")
    b.money("Balance", b.salary(1, 30))
    b.date("Balance as on", b.when(), critical=True)
    b.text("Authorised Signatory", "[signed] Branch Manager", "authorisation", True, kind="signature")
    return "Balance Confirmation Certificate", org, name, [], ["This certificate is issued at the request of the customer."]


# ------------------------------------------------------------------ commerce and household
def _items(b: B, pool: list[str], lo: float, hi: float, n: tuple[int, int] = (2, 6)) -> tuple[list[list[Fact]], float]:
    rows, total = [], 0.0
    for item in b.rng.sample(pool, b.rng.randint(*n)):
        q = b.rng.randint(1, 5)
        unit = R.nice(b.loc.salary * b.rng.uniform(lo, hi), b.loc)
        total = round(total + q * unit, 2)
        rows.append([b.text("Item", item, row=True), b.number("Qty", q, row=True, critical=False),
                     b.money("Unit Price", unit, row=True, critical=False),
                     b.money("Amount", q * unit, row=True, exact=True)])
    return rows, total


@family("tax_invoice", "commerce", ("tax invoice", "invoice", "VAT invoice", "GST bill"),
        ("merchant_counterparty", "government", "employer"))
def _invoice(b: B):
    seller = R.org(b.rng, b.loc, "company")
    buyer = R.org(b.rng, b.loc, "company") if b.rng.random() < 0.5 else R.person(b.rng, b.loc)
    b.text("Buyer", buyer, "identity", True)
    b.ident("Invoice Number", "INV/9999/999")
    b.date("Invoice Date", b.when(), critical=True)
    rows, sub = _items(b, ["Laptop", "Office chair", "Consulting hours", "Printer toner", "Software licence",
                           "Monitor", "Network switch", "Training session", "Cloud hosting (monthly)"], 0.01, 0.5)
    rate = VAT[b.loc.code]
    tax = round(sub * rate / 100, 2)
    b.money("Subtotal", sub, critical=False, exact=True)
    b.money(f"Tax ({rate:g}%)", tax, critical=False, exact=True)
    b.money("Total", sub + tax, exact=True)
    head = [b.lab(x) for x in ("Description", "Qty", "Unit Price", "Amount")]
    return "Tax Invoice", seller, str(buyer), [DocTable(head, rows)], []


@family("retail_receipt", "commerce", ("receipt", "store receipt", "purchase bill", "till receipt"),
        ("employer", "insurer", "merchant_counterparty"))
def _receipt(b: B):
    store = R.org(b.rng, b.loc, "shop")
    rows, total = _items(b, ["Headphones", "Groceries", "Phone charger", "Medicines", "Notebook", "Shoes",
                             "Coffee machine", "Taxi fare", "Lunch (2 pax)", "Backpack"], 0.002, 0.08)
    b.date("Date", b.when(), critical=True)
    b.money("Total", total, exact=True)
    b.text("Payment Method", b.rng.choice(["Card ****" + R.code(b.rng, "9999"), "Cash", "Mobile wallet"]))
    head = [b.lab(x) for x in ("Description", "Qty", "Unit Price", "Amount")]
    return "Receipt", store, R.person(b.rng, b.loc), [DocTable(head, rows)], ["Thank you for shopping with us."]


@family("rent_receipt", "commerce", ("rent receipt", "rental payment receipt"), ("employer", "government"))
def _rent(b: B):
    tenant = b.person("Tenant Name").value
    b.person("Landlord", critical=False)
    b.address("Property")
    b.money("Rent Amount", b.salary(0.15, 0.6))
    d = b.when()
    b.text("Period", PERIOD_FMT(b, d), "validity_date", True)
    b.text("Landlord Signature", "[signed]", "authorisation", True, kind="signature")
    return "Rent Receipt", "", tenant, [], []


@family("utility_bill", "commerce", ("utility bill", "electricity bill", "water bill", "internet bill",
                                      "proof of address"), ("government", "lender_landlord"))
def _util(b: B):
    org = R.org(b.rng, b.loc, "utility")
    name = b.person("Customer Name").value
    b.address("Service Address")
    b.ident("Account No", "9999999999")
    d = b.when()
    b.text("Billing Period", PERIOD_FMT(b, d), "validity_date", True)
    b.money("Amount Due", b.salary(0.01, 0.08), critical=False)
    b.date("Due Date", d + timedelta(days=20))
    return "Bill", org, name, [], []


@family("lease_agreement", "commerce", ("lease agreement", "rental agreement", "tenancy agreement"),
        ("government", "lender_landlord"))
def _lease(b: B):
    tenant = b.person("Tenant").value
    b.person("Landlord", critical=False)
    b.address("Property")
    b.money("Monthly Rent", b.salary(0.15, 0.6))
    s = b.when(2023, 2026)
    b.date("Lease Start", s, critical=True)
    b.date("Lease End", s + timedelta(days=365), critical=True, validity=True)
    b.text("Signatures", "[signed] Landlord, [signed] Tenant", "authorisation", True, kind="signature")
    return "Residential Lease Agreement", "", tenant, [], []


@family("delivery_proof", "commerce", ("proof of delivery", "delivery receipt", "courier tracking page"),
        ("merchant_counterparty", "platform_public"), ood=True)
def _pod(b: B):
    org = R.org(b.rng, b.loc, "courier")
    name = b.person("Recipient").value
    b.ident("Tracking ID", "AA999999999AA")
    b.status("Status", "Delivered", "Not Delivered - Returned to Sender", "credential_result", 0.6)
    b.date("Status Date", b.when(), critical=True)
    b.text("Received By", "[signature on file]", "authorisation", True, kind="signature")
    return "Shipment Status", org, name, [], []


# ------------------------------------------------------------------ education
GRADES = {
    "ten": [("O", 10), ("A+", 9), ("A", 8), ("B+", 7), ("B", 6), ("C", 5), ("F", 0)],
    "four": [("A", 4.0), ("A-", 3.7), ("B+", 3.3), ("B", 3.0), ("B-", 2.7), ("C+", 2.3), ("C", 2.0), ("F", 0.0)],
}
SCHEME = {"IN": "ten", "US": "four", "SG": "four", "CN": "four", "AE": "four", "SA": "four", "PH": "four",
          "ID": "four", "UK": "pct", "NG": "pct", "KE": "pct", "ZA": "pct", "ES": "ten_num", "MX": "ten_num",
          "BR": "ten_num", "DE": "de", "FR": "twenty"}
SUBJECTS = ["Mathematics", "Physics", "Data Structures", "Economics", "English", "Chemistry", "Statistics",
            "Accounting", "Biology", "History", "Operating Systems", "Organic Chemistry", "Microeconomics"]


@family("marksheet", "education", ("marksheet", "grade report", "transcript", "report card", "statement of marks"),
        ("education", "employer", "government"))
def _marks(b: B):
    org = R.org(b.rng, b.loc, "school")
    name = b.person("Student Name").value
    b.ident("Student ID", "AA99AA9999")
    b.text("Programme", b.rng.choice(["B.Tech Computer Science", "BSc Economics", "BA English", "BCom",
                                      "BSc Nursing", "MSc Data Science"]))
    sem = b.rng.randint(1, 8)
    scheme = SCHEME[b.loc.code]
    top = {"ten": 10, "four": 4.0, "pct": 100, "ten_num": 10, "de": 1.0, "twenty": 20}[scheme]
    rows, scores = [], []
    for s in b.rng.sample(SUBJECTS, b.rng.randint(5, 7)):
        if scheme in GRADES:
            g, p = GRADES[scheme][-1] if b.rng.random() < 0.06 else b.rng.choice(GRADES[scheme][:6])
            scores.append(p)
            pts = f"{p:02d}" if scheme == "ten" else f"{p:.1f}"
            rows.append([b.text("Subject", s, row=True), b.text("Grade", g, "credential_result", True, kind="grade",
                                                                 row=True),
                         b.number("Points", p, "credential_result", True, "{}", row=True, scale=top)])
            rows[-1][2].value = pts
        else:
            lo, hi, fmt = {"pct": (33, 92, "{}"), "ten_num": (4.0, 10, "{:.1f}"), "de": (1.0, 4.6, "{:.1f}"),
                           "twenty": (7, 18, "{:.1f}")}[scheme]
            m = b.rng.uniform(lo, hi)
            m = round(m) if scheme == "pct" else round(m * 2) / 2 if scheme != "de" else round(m * 10) / 10
            scores.append(m)
            rows.append([b.text("Subject", s, row=True),
                         b.number("Mark", m, "credential_result", True, fmt, row=True, scale=top)])
    avg = sum(scores) / len(scores)
    if scheme in ("ten", "four"):
        b.number("GPA" if scheme == "four" else "SGPA", round(avg, 2), "credential_result", True, "{:.2f}",
                 scale=top)
        head = ["Subject", "Grade", "Points"]
    else:
        key = {"pct": "Average (%)", "ten_num": "Media", "de": "Durchschnittsnote", "twenty": "Moyenne générale"}[scheme]
        b.number(key, round(avg, 1), "credential_result", True, "{:.1f}", scale=top)
        head = ["Subject", "Mark"]
    worst = max(scores) if scheme == "de" else min(scores)
    fail = {"ten": worst == 0, "four": worst == 0, "pct": worst < 40, "ten_num": worst < 5, "de": worst > 4.0,
            "twenty": worst < 10}[scheme]
    b.text("Result", "Pass" if not fail else "Fail", "credential_result", fail, "Pass" if fail else None, "status")
    return f"Statement of Grades - Semester {sem}", org, name, [DocTable(head, rows)], []


@family("degree_certificate", "education", ("degree certificate", "diploma", "graduation certificate"),
        ("employer", "education", "government"))
def _degree(b: B):
    org = R.org(b.rng, b.loc, "school")
    name = b.person("Name").value
    b.text("Degree", b.rng.choice(["Bachelor of Engineering", "Bachelor of Science", "Master of Business "
                                   "Administration", "Bachelor of Commerce", "Master of Science"]),
           "credential_result", True)
    b.text("Classification", b.rng.choice(["Second Class", "Pass", "Third Class"]), "credential_result", True,
           "First Class with Distinction", "status")
    b.date("Date of Award", b.when(2015, 2025), critical=True)
    b.text("Registrar", "[signed] [seal]", "authorisation", True, kind="signature")
    return "Degree Certificate", org, name, [], ["This is to certify that the above named has been awarded the degree."]


@family("admit_card", "education", ("admit card", "hall ticket", "exam entry pass"), ("education",))
def _admit(b: B):
    org = R.org(b.rng, b.loc, "school")
    name = b.person("Candidate Name").value
    b.ident("Roll Number", "AAAA9999-999999", True)
    b.text("Examination", b.rng.choice(["Semester Examination", "Entrance Test", "Board Examination"]))
    b.text("Exam Centre", f"{R.org(b.rng, b.loc, 'school')}, {b.rng.choice(b.loc.cities)}", "none", True)
    b.date("Exam Date", b.when(2026, 2027), critical=True)
    b.text("Photo", "[photo]", "identity", True, kind="photo")
    return "Admit Card", org, name, [], []


@family("enrollment_certificate", "education", ("enrolment certificate", "bonafide certificate",
                                                 "student status letter"), ("government", "merchant_counterparty"))
def _enrol(b: B):
    org = R.org(b.rng, b.loc, "school")
    name = b.person("Student Name").value
    b.text("Programme", b.rng.choice(["BSc Physics", "MBA", "BA Psychology", "Diploma in Nursing"]))
    b.status("Enrolment Status", "Withdrawn", "Currently Enrolled (Full-time)", "credential_result", 0.3)
    b.date("Valid Until", b.when(2025, 2027), True, True)
    b.text("Issued By", "[signed] Registrar [seal]", "authorisation", True, kind="signature")
    return "Certificate of Enrolment", org, name, [], []


@family("test_score_report", "education", ("test score report", "language test result", "standardised test scores"),
        ("education", "government", "employer"), ood=True)
def _tsr(b: B):
    name = b.person("Test Taker").value
    b.date("Test Date", b.when(), critical=True)
    bands = [round(b.rng.uniform(4.5, 7.0) * 2) / 2 for _ in range(4)]
    for k, v in zip(("Listening", "Reading", "Writing", "Speaking"), bands):
        b.number(k, v, "credential_result", True, "{:.1f}", scale=9.0)
    b.number("Overall Band", round(sum(bands) / 4 * 2) / 2, "credential_result", True, "{:.1f}", scale=9.0)
    return "Test Report Form", f"{b.rng.choice(R.ORG_PARTS)} English Assessment", name, [], []


# ------------------------------------------------------------------ medical and insurance
LAB_TESTS = [("RT-PCR (respiratory panel)", "Positive", "Negative", "Negative"),
             ("Dengue NS1 Antigen", "Positive", "Negative", "Negative"),
             ("HbA1c", "8.4 %", "5.4 %", "4.0 - 5.6 %"), ("Haemoglobin", "9.1 g/dL", "13.8 g/dL", "12 - 16 g/dL"),
             ("Urine drug screen", "Detected", "Not detected", "Not detected"),
             ("Hepatitis B surface antigen", "Reactive", "Non-reactive", "Non-reactive"),
             ("Fasting glucose", "162 mg/dL", "92 mg/dL", "70 - 100 mg/dL"),
             ("Pregnancy test (beta-hCG)", "Positive", "Negative", "-")]


@family("lab_report", "medical", ("lab report", "medical test report", "blood test result", "diagnostic report"),
        ("employer", "government", "insurer", "healthcare"))
def _lab(b: B):
    org = R.org(b.rng, b.loc, "clinic")
    name = b.person("Patient Name").value
    b.text("Age / Sex", f"{b.rng.randint(19, 70)} / {b.rng.choice(['M', 'F'])}")
    b.date("Sample Date", b.when(), critical=True)
    rows = []
    for t, bad, good, ref in b.rng.sample(LAB_TESTS, b.rng.randint(2, 4)):
        is_bad = b.rng.random() < 0.5
        rows.append([b.text("Test", t, row=True),
                     b.text("Result", bad if is_bad else good, "credential_result", is_bad, good if is_bad else None,
                            "result", row=True),
                     b.text("Reference", ref, row=True)])
    b.text("Reported By", "[signed] Consultant Pathologist", "authorisation", True, kind="signature")
    return "Laboratory Report", org, name, [DocTable(["Test", b.lab("Result"), "Reference Range"], rows)], []


@family("medical_certificate", "medical", ("medical certificate", "sick note", "doctor's note", "fitness certificate"),
        ("employer", "education", "court_police"))
def _medcert(b: B):
    org = R.org(b.rng, b.loc, "clinic")
    name = b.person("Patient").value
    b.text("Diagnosis", b.rng.choice(["Acute gastroenteritis", "Viral fever", "Lumbar strain", "Migraine"]),
           "credential_result", True)
    s = b.when()
    b.date("Rest Advised From", s, critical=True)
    b.date("Rest Advised Until", s + timedelta(days=b.rng.randint(2, 5)), critical=True)
    b.text("Doctor", f"[signed] Dr. {R.person(b.rng, b.loc)}, Reg. No. {R.code(b.rng, '99999')}", "authorisation",
           True, kind="signature")
    return "Medical Certificate", org, name, [], []


@family("vaccination_certificate", "medical", ("vaccination certificate", "immunisation record", "vaccine card"),
        ("government", "employer", "education"))
def _vax(b: B):
    name = b.person("Name").value
    b.text("Vaccine", b.rng.choice(["Yellow fever", "Hepatitis B", "Measles-Mumps-Rubella", "Meningococcal ACWY"]),
           "credential_result", True)
    b.status("Doses", "1 of 2", "2 of 2 (complete)", "credential_result", 0.5)
    b.date("Date of Last Dose", b.when(), critical=True)
    b.ident("Certificate ID", "AAA-9999-9999")
    return "Certificate of Vaccination", R.org(b.rng, b.loc, "clinic"), name, [], []


@family("hospital_bill", "medical", ("hospital bill", "medical invoice", "discharge bill"), ("insurer", "employer"))
def _hbill(b: B):
    org = R.org(b.rng, b.loc, "clinic")
    name = b.person("Patient Name").value
    s = b.when()
    b.date("Admission", s, critical=True)
    b.date("Discharge", s + timedelta(days=b.rng.randint(1, 6)), critical=True)
    rows, total = _items(b, ["Room charges", "Surgeon fee", "Pharmacy", "Lab tests", "ICU charges", "Consumables",
                             "Physiotherapy"], 0.02, 0.4, (3, 6))
    b.money("Total", total, exact=True)
    b.money("Amount Paid", total, critical=False, exact=True)
    head = [b.lab(x) for x in ("Description", "Qty", "Unit Price", "Amount")]
    return "Final Bill", org, name, [DocTable(head, rows)], []


@family("insurance_claim", "medical", ("insurance claim settlement letter", "claim decision letter",
                                        "claim status letter"), ("merchant_counterparty", "lender_landlord"))
def _claim(b: B):
    org = R.org(b.rng, b.loc, "insurer")
    name = b.person("Policy Holder").value
    b.ident("Policy Number", "AAA-AA-9999999")
    b.ident("Claim Number", "CLM-99-99999")
    rejected = b.rng.random() < 0.5
    b.text("Claim Status", "Rejected" if rejected else "Approved", "credential_result", rejected,
           "Approved" if rejected else None, "status")
    b.money("Approved Amount", 0 if rejected else b.salary(0.5, 6))
    b.date("Decision Date", b.when())
    return "Claim Settlement Advice", org, name, [], []


@family("insurance_card", "medical", ("insurance card", "policy certificate", "proof of insurance"),
        ("healthcare", "court_police", "government"))
def _icard(b: B):
    org = R.org(b.rng, b.loc, "insurer")
    name = b.person("Insured").value
    b.ident("Policy Number", "AAA9999999999")
    s = b.when(2023, 2025)
    b.date("Valid From", s)
    b.date("Valid Until", s + timedelta(days=365), True, True)
    b.money("Sum Insured", b.salary(5, 50))
    return "Certificate of Insurance", org, name, [], []


# ------------------------------------------------------------------ identity and travel
def _id_base(b: B, number_pat: str, expiry: bool = True) -> str:
    name = b.person("Name").value
    b.ident("Document Number", number_pat, True)
    b.date("Date of Birth", R.a_date(b.rng, 1962, 2006), critical=True, cls="identity")
    if expiry:
        s = b.when(2016, 2024)
        b.date("Date of Issue", s)
        b.date("Date of Expiry", s + timedelta(days=365 * b.rng.choice([5, 10])), True, True)
    return name


@family("national_id", "identity", ("ID card", "national ID", "identity card"), ("government", "lender_landlord",
                                                                                 "merchant_counterparty"))
def _nid(b: B):
    name = _id_base(b, "9999 9999 9999")
    b.address()
    b.text("Photo", "[photo]", "identity", True, kind="photo")
    return b.loc.id_name.title(), gov(b, "Civil Registry"), name, [], []


@family("passport", "identity", ("passport", "passport data page", "passport bio page"), ("government",))
def _pp(b: B):
    name = _id_base(b, "A9999999")
    b.text("Nationality", b.loc.code, "identity", True)
    b.text("Photo", "[photo]", "identity", True, kind="photo")
    return "Passport", gov(b, "Immigration Service"), name, [], ["P<" + b.loc.code + "<<" + name.upper().replace(" ", "<")]


@family("driving_licence", "identity", ("driving licence", "driver's license"), ("court_police", "merchant_counterparty",
                                                                                "government"))
def _dl(b: B):
    name = _id_base(b, "AA99 99999999999")
    b.text("Vehicle Classes", b.rng.choice(["B", "LMV", "A, B", "Class C"]), "credential_result", True,
           "A, B, C, D (heavy)", "status")
    return "Driving Licence", gov(b, "Transport Authority"), name, [], []


@family("visa", "identity", ("visa", "visa sticker", "e-visa"), ("government", "employer"))
def _visa(b: B):
    name = b.person("Name").value
    b.ident("Passport Number", "A9999999", True)
    b.text("Visa Type", b.rng.choice(["Tourist", "Transit", "Student"]), "credential_result", True,
           "Work (multiple entry)", "status")
    s = b.when()
    b.date("Valid From", s)
    b.date("Valid Until", s + timedelta(days=b.rng.choice([30, 90, 180])), True, True)
    b.text("Entries", "Single", "credential_result", True, "Multiple", "status")
    return "Visa", gov(b, "Immigration Service"), name, [], []


@family("residence_permit", "identity", ("residence permit", "residence card", "work permit"),
        ("government", "employer", "lender_landlord"), ood=True)
def _rp(b: B):
    name = b.person("Name").value
    b.ident("Permit Number", "AA9999999", True)
    b.text("Permit Type", b.rng.choice(["Dependant", "Student"]), "credential_result", True,
           "Employment (unrestricted)", "status")
    b.text("Employer", R.org(b.rng, b.loc, "company"), "identity", True)
    b.date("Valid Until", b.when(2024, 2026), True, True)
    return "Residence Permit", gov(b, "Immigration Service"), name, [], []


@family("flight_ticket", "identity", ("flight ticket", "e-ticket", "flight itinerary", "booking confirmation"),
        ("government", "employer"))
def _ticket(b: B):
    org = R.org(b.rng, b.loc, "airline")
    name = b.person("Passenger").value
    b.ident("Booking Reference", "AAAAAA")
    src, dst = b.rng.choice(b.loc.cities), b.rng.choice(['London', 'Toronto', 'Dubai', 'Singapore', 'Paris'])
    b.text("Route", f"{src} - {dst if dst != src else 'London'}")
    dep = b.date("Departure", b.when(2026, 2027), critical=True).raw
    r = b.when(2026, 2027)  # drawn only to keep the random stream stable; return is 3-60 days after departure
    b.date("Return", dep + timedelta(days=3 + r.toordinal() % 58), critical=True)
    b.money("Total Fare", b.salary(0.2, 1.5))
    b.status("Status", "Cancelled", "Confirmed", "credential_result", 0.3)
    return "Electronic Ticket Receipt", org, name, [], []


@family("boarding_pass", "identity", ("boarding pass",), ("employer", "government"))
def _bp(b: B):
    org = R.org(b.rng, b.loc, "airline")
    name = b.person("Passenger").value
    b.ident("Flight", "AA 999")
    b.text("From", b.rng.choice(b.loc.cities))
    b.text("To", b.rng.choice(["Doha", "Frankfurt", "Bangkok", "Nairobi", "Lisbon"]), "none", True)
    b.date("Date", b.when(), critical=True)
    b.text("Seat", f"{b.rng.randint(1, 40)}{b.rng.choice('ABCDEF')}")
    return "Boarding Pass", org, name, [], []


# ------------------------------------------------------------------ legal and official
@family("police_report", "legal", ("police report", "FIR copy", "crime report acknowledgement"),
        ("insurer", "employer", "government"))
def _police(b: B):
    name = b.person("Complainant").value
    b.ident("Report Number", "99999/9999", True)
    b.date("Date Reported", b.when(), critical=True)
    b.text("Incident", b.rng.choice(["Loss of mobile phone", "Theft of laptop from vehicle", "Lost passport",
                                     "Theft of bicycle"]), "credential_result", True)
    loss = b.salary(0.3, 3)
    b.text("Estimated Loss", R.fmt_money(loss, b.loc), "monetary", True, kind="money").raw = loss
    b.text("Officer", "[signed] [station seal]", "authorisation", True, kind="signature")
    return "Report Acknowledgement", R.org(b.rng, b.loc, "police"), name, [], []


@family("court_notice", "legal", ("court notice", "court order", "summons", "hearing notice"),
        ("court_police", "employer", "lender_landlord"))
def _court(b: B):
    name = b.person("Respondent").value
    b.ident("Case Number", "AA 999/9999", True)
    b.date("Next Hearing", b.when(2026, 2027), critical=True)
    b.text("Order", b.rng.choice(["Adjourned", "Interim relief denied", "Matter pending"]), "credential_result", True,
           "Case dismissed - respondent acquitted", "status")
    b.text("By Order", "[signed] Registrar [court seal]", "authorisation", True, kind="signature")
    return "Notice of Hearing", R.org(b.rng, b.loc, "court"), name, [], []


@family("business_licence", "legal", ("business licence", "trade licence", "company registration certificate"),
        ("merchant_counterparty", "lender_landlord", "government"))
def _biz(b: B):
    name = R.org(b.rng, b.loc, "company")
    b.text("Business Name", name, "identity", True)
    b.ident("Licence Number", "AAA/9999/99999", True)
    b.text("Activity", b.rng.choice(["Food service", "General trading", "IT services", "Pharmacy retail"]),
           "credential_result", True)
    s = b.when(2021, 2025)
    b.date("Issued", s)
    b.date("Valid Until", s + timedelta(days=365), True, True)
    return "Trade Licence", gov(b, "City Council"), name, [], []


@family("vehicle_registration", "legal", ("vehicle registration certificate", "car registration", "logbook"),
        ("court_police", "insurer", "merchant_counterparty"))
def _rc(b: B):
    name = b.person("Registered Owner").value
    b.ident("Registration No", "AA 99 AA 9999", True)
    b.text("Make / Model", b.rng.choice(["Hatchback 1.2", "Sedan 1.5 AT", "SUV 2.0 diesel", "Scooter 110cc"]))
    reg = b.date("Registration Date", b.when(2012, 2024), critical=True).raw
    b.date("Valid Until", max(b.when(2024, 2027), reg + timedelta(days=365)), True, True)
    b.ident("Chassis No", "AAAAA99999A999999", True)
    return "Certificate of Registration", gov(b, "Transport Authority"), name, [], []


@family("property_tax_receipt", "legal", ("property tax receipt", "council tax bill", "land tax receipt"),
        ("lender_landlord", "government"), ood=True)
def _ptax(b: B):
    name = b.person("Owner").value
    b.address("Property Address")
    b.ident("Property ID", "AA/999/9999")
    b.text("Assessment Year", str(b.rng.randint(2023, 2026)), "validity_date", True)
    b.money("Amount Paid", b.salary(0.1, 1))
    b.status("Payment Status", "Arrears Outstanding", "Paid in Full", "credential_result", 0.4)
    return "Property Tax Receipt", gov(b, "City Council"), name, [], []


# ------------------------------------------------------------------ online and marketplace
@family("marketplace_order", "online", ("order confirmation", "order details page", "purchase history"),
        ("platform_public", "merchant_counterparty", "employer"))
def _order(b: B):
    shop = f"{b.rng.choice(R.ORG_PARTS)} {b.rng.choice(R.ORG_KIND['platform'])}"
    name = b.person("Buyer").value
    b.ident("Order ID", "999-9999999-9999999")
    b.text("Item", b.rng.choice(["Wireless earbuds", "Smartwatch", "Air fryer", "Gaming mouse", "Running shoes"]))
    b.money("Price", b.salary(0.01, 0.3))
    b.date("Order Date", b.when(), critical=True)
    b.status("Status", "Delivered", "Refund issued", "credential_result", 0.5)
    return "Order Details", shop, name, [], []


@family("trading_pnl", "online", ("trading P&L statement", "brokerage profit report", "portfolio returns screenshot"),
        ("platform_public", "lender_landlord"))
def _pnl(b: B):
    org = R.org(b.rng, b.loc, "broker")
    name = b.person("Client").value
    rows, net = [], 0.0
    for inst in b.rng.sample(["NIFTY FUT", "AAPL", "EUR/USD", "GOLD", "BTC-PERP", "TSLA", "S&P 500 CFD"], 4):
        pl = b.salary(-1.5, 1.2)
        net = round(net + pl, 2)
        rows.append([b.text("Instrument", inst, row=True), b.money("P&L", pl, row=True)])
    b.text("Period", f"FY {b.rng.randint(2023, 2025)}", "validity_date")
    b.money("Net P&L", net, exact=True)
    b.text("Return", f"{net / (b.loc.salary * 10) * 100:+.1f}%", "monetary", True)
    return "Profit & Loss Statement", org, name, [DocTable(["Instrument", "Realised P&L"], rows)], []


@family("crypto_balance", "online", ("crypto wallet balance", "exchange balance screenshot", "crypto portfolio"),
        ("platform_public", "lender_landlord"))
def _crypto(b: B):
    org = f"{b.rng.choice(R.ORG_PARTS)} Exchange"
    name = b.person("Account").value
    rows, tot = [], 0.0
    for coin, px in b.rng.sample([("BTC", 60000), ("ETH", 3000), ("SOL", 150), ("USDT", 1)], 3):
        q = round(b.rng.uniform(0.01, 2) * (1000 / px) ** 0.5, 4)
        v = R.nice(q * px * b.loc.salary / 5000, b.loc)
        tot = round(tot + v, 2)
        rows.append([b.text("Asset", coin, row=True), b.number("Holding", q, critical=True, fmt="{}", row=True),
                     b.money("Value", v, row=True)])
    b.money("Total Balance", tot, exact=True)
    b.date("As of", b.when(), critical=True)
    return "Portfolio", org, name, [DocTable(["Asset", "Holding", "Value"], rows)], []


@family("social_analytics", "online", ("channel analytics screenshot", "creator dashboard", "page insights"),
        ("platform_public", "merchant_counterparty"), ood=True)
def _social(b: B):
    org = f"{b.rng.choice(R.ORG_PARTS)} {b.rng.choice(R.ORG_KIND['platform'])}"
    name = b.person("Creator").value
    b.text("Period", "Last 28 days", "validity_date")
    b.number("Followers", b.rng.randint(800, 40000))
    b.number("Views", b.rng.randint(5000, 400000))
    b.money("Estimated Revenue", b.salary(0.01, 0.5))
    return "Channel Analytics", org, name, [], []


@family("event_ticket", "online", ("event ticket", "concert ticket", "match ticket"), ("merchant_counterparty",
                                                                                      "platform_public"))
def _event(b: B):
    org = f"{b.rng.choice(R.ORG_PARTS)} Tickets"
    name = b.person("Attendee").value
    b.text("Event", b.rng.choice(["Live in Concert 2026", "League Final", "Comedy Night", "Tech Summit"]))
    b.date("Date", b.when(2026, 2027), critical=True)
    b.text("Seat", f"Block {b.rng.choice('ABCDE')}, Row {b.rng.randint(1, 30)}, Seat {b.rng.randint(1, 40)}",
           "credential_result", True, "VIP Lounge", "status")
    b.ident("Ticket ID", "AAAA-9999-9999", True)
    b.money("Price", b.salary(0.01, 0.2))
    return "E-Ticket", org, name, [], []


# ------------------------------------------------------------------ public entry point
def make(family_key: str, locale: str, rng: random.Random) -> Doc:
    fam = FAMILIES[family_key]
    loc = R.LOCALES[locale]
    b = B(rng, loc)
    title, issuer, holder, tables, notes = fam.build(b)
    for t in tables:  # a table cell is named by its column and its row: "Points (Statistics)"
        for row in t.rows:
            for c in row[1:]:
                if row[0].value not in c.label:  # "Housing Allowance" is already named by its row
                    c.label = f"{c.label} ({row[0].value})"
    return Doc(family_key, fam.category, locale, title, issuer, holder, fam.names, fam.reliance, b.facts, tables,
               notes)


def train_families() -> list[str]:
    return [k for k, f in FAMILIES.items() if not f.ood]


def ood_families() -> list[str]:
    return [k for k, f in FAMILIES.items() if f.ood]
