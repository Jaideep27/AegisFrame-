"""What a request asks for, decided before any text is written: the edit plan.

Each harmful plan has a benign twin on the same record that differs only in the edit (blur the
field instead of changing it, add a SAMPLE watermark instead of removing one, ...). The teacher LLM
then writes both requests with the same story, so the model must learn the edit, not the topic.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from . import regions as R
from .doclib import Doc, Fact
from .regions import Locale

MARKS = {"en": ["SAMPLE", "SPECIMEN", "VOID", "COPY", "NOT VALID"], "de": ["MUSTER", "ENTWERTET"],
         "fr": ["SPÉCIMEN", "ANNULÉ"], "es": ["MUESTRA", "ANULADO"], "pt": ["AMOSTRA", "CANCELADO"],
         "id": ["CONTOH", "BATAL"]}
REMOVABLE = ["watermark reading '{m}'", "'{m}' stamp", "diagonal '{m}' text", "'{m}' overlay"]

PATTERN_BY_FAMILY = {
    "payslip": "income_inflation", "tax_wage_statement": "income_inflation", "tax_return_ack": "income_inflation",
    "offer_letter": "income_inflation", "experience_letter": "income_inflation",
    "employment_verification": "income_inflation", "bank_statement": "balance_inflation",
    "balance_certificate": "balance_inflation", "crypto_balance": "balance_inflation",
    "trading_pnl": "balance_inflation", "loan_sanction_letter": "balance_inflation",
    "payment_screenshot": "payment_proof", "bank_transfer_receipt": "payment_proof", "cheque": "payment_proof",
    "retail_receipt": "expense_padding", "hospital_bill": "expense_padding", "rent_receipt": "expense_padding",
    "police_report": "expense_padding", "insurance_claim": "expense_padding", "tax_invoice": "expense_padding",
    "social_analytics": "balance_inflation", "marketplace_order": "payment_proof",
}
DEFLATE = {"tax_invoice": 0.35, "credit_report": 1.0, "utility_bill": 1.0, "property_tax_receipt": 0.5}


@dataclass
class Change:
    field: str
    old: str
    new: str
    cls: str
    pattern: str


@dataclass
class EditPlan:
    intent: str
    describe: str                     # what the user wants, in plain English, for the teacher
    changes: list[Change] = field(default_factory=list)
    has_image: bool = True
    real_record: bool = True
    distinguishable: str = "no"
    marking: str | None = None        # marking text involved (added, removed or negated)
    marking_mode: str | None = None   # add | remove | negate
    context_marking: str | None = None  # a marking already printed on the uploaded record
    fact_class: str = "none"
    fraud_pattern: str = "none"
    legitimate_need: str = "none"
    target: str = ""                  # the field / area a benign edit touches
    context_overrides: dict[str, str] = field(default_factory=dict)  # label -> value printed on the uploaded record


# ------------------------------------------------------------------ value mutations
def _num(s: str) -> float | None:
    m = re.search(r"\d[\d,. ]*", s)
    if not m:
        return None
    t = m.group(0).strip()
    if re.search(r"[.,]\d{2}$", t):
        t = t[:-3]
    try:
        return float(re.sub(r"[^\d]", "", t))
    except ValueError:
        return None


def mutate(f: Fact, doc: Doc, rng: random.Random) -> tuple[str, str]:
    """-> (new printed value, fraud pattern)."""
    loc = R.LOCALES[doc.locale]
    if f.alt:
        return f.alt, "result_flip" if f.kind in ("result", "status") and f.cls == "credential_result" else \
            "credential_inflation"
    if f.kind == "money" and isinstance(f.raw, (int, float)):
        down = rng.random() < DEFLATE.get(doc.family, 0.0)
        x = f.raw * (rng.uniform(0.15, 0.6) if down else rng.choice([1.5, 2, 2.5, 3, 4, 5, 10]) * rng.uniform(0.9, 1.1))
        if x == 0 or f.raw == 0:
            x = R.nice(loc.salary * rng.uniform(1, 8), loc)
        cur = next((c for c in loc.currency if f.value.lstrip("-").startswith(c)), None) or             next((c for c in loc.currency if c.strip() in f.value), loc.currency[0])  # "ZAR 1" is not "R 1"
        return R.fmt_money(R.nice(x, loc), loc, cur), "amount_deflation" if down else \
            PATTERN_BY_FAMILY.get(doc.family, "income_inflation")
    if f.kind == "date" and isinstance(f.raw, date):
        if f.validity:
            d = f.raw + timedelta(days=365 * rng.randint(1, 4) + rng.randint(0, 200))
            pat = "validity_extension"
        else:
            d = f.raw + timedelta(days=rng.choice([-1, 1]) * rng.randint(3, 120))
            pat = "date_shift"
        style = next((s for s in loc.date_styles if R.fmt_date(f.raw, loc, s) == f.value), None)
        return R.fmt_date(d, loc, style), pat
    if f.kind in ("person", "address"):
        for _ in range(20):  # never return the same name
            new = R.person(rng, loc) if f.kind == "person" else \
                f"{rng.randint(2, 480)} {rng.choice(['Palm', 'Cedar', 'Harbour', 'Temple'])} Road, {rng.choice(loc.cities)}"
            if new != f.value:
                break
        return new, "identity_swap"
    if f.kind == "id":
        chars = list(f.value)
        digits = [i for i, c in enumerate(chars) if c.isdigit()]
        for i in rng.sample(digits, k=min(3, len(digits))):
            chars[i] = str((int(chars[i]) + rng.randint(1, 8)) % 10)
        if not digits:  # letters-only code: change one letter
            i = rng.choice([i for i, c in enumerate(chars) if c.isalpha()] or [len(chars) - 1])
            chars[i] = "X" if chars[i] != "X" else "K"
        return "".join(chars), "identity_swap"
    if f.kind == "photo":
        return "[photo of another person]", "identity_swap"
    if f.kind == "signature":
        return "[signature and seal copied from another document]", "authorisation_forgery"
    if f.kind == "grade":
        better = {"B": "A", "B+": "A+", "C": "B+", "A": "A+", "A+": "O", "F": "B", "B-": "A-", "C+": "A", "A-": "A"}
        return better.get(f.value, "A+"), "credential_inflation"
    if f.kind == "number" and isinstance(f.raw, (int, float)):
        if f.cls == "credential_result":
            if f.key == "Accounts in Default":
                return "0", "credential_inflation"
            ceiling = f.scale or 100
            if ceiling < f.raw:  # lower is better (German grades)
                return f"{max(ceiling, f.raw - rng.uniform(0.8, 2)):.1f}", "credential_inflation"
            new = min(ceiling, f.raw + max(ceiling * 0.1, (ceiling - f.raw) * rng.uniform(0.4, 1)))
            if ceiling in (9.0, 10) and "." in f.value and len(f.value.split(".")[1]) == 1:
                new = round(new * 2) / 2
            if "." in f.value:
                return f"{new:.{len(f.value.split('.')[1])}f}", "credential_inflation"
            return (f"{int(round(new)):0{len(f.value)}d}" if f.value.startswith("0") else str(int(round(new)))), \
                "credential_inflation"
        n = f.raw * rng.choice([3, 5, 10, 20])
        return (f"{n:,.0f}" if n >= 100 else f"{n:g}"), PATTERN_BY_FAMILY.get(doc.family, "balance_inflation")
    if f.kind == "text" and f.cls == "credential_result":
        upgrades = {
            "Second Class": "First Class with Distinction", "Junior Developer": "Senior Software Engineer",
            "Accounts Assistant": "Finance Manager", "Team Lead": "Engineering Manager",
            "Lab Technician": "Senior Research Scientist", "Senior Analyst": "Director of Analytics",
            "Product Manager": "Vice President, Product", "Civil Engineer": "Chief Engineer",
            "Registered Nurse": "Nurse Practitioner", "Backend Developer": "Principal Engineer",
            "Branch Manager": "Regional Head", "Bachelor of Science": "Master of Science",
            "Bachelor of Engineering": "Master of Technology", "Bachelor of Commerce": "Master of Business Administration",
            "Master of Science": "Doctor of Philosophy", "Master of Business Administration": "Doctor of Business Administration",
            "Viral fever": "Dengue fever (hospitalisation advised)", "Migraine": "Concussion (no work for 3 weeks)",
            "Acute gastroenteritis": "Typhoid fever (bed rest 3 weeks)", "Lumbar strain": "Slipped disc (surgery advised)",
            "Food service": "Food service and liquor sale", "General trading": "General trading and import/export",
            "IT services": "IT services and payment processing", "Pharmacy retail": "Pharmacy wholesale incl. controlled medicines",
            "Loss of mobile phone": "Theft of mobile phone and laptop", "Lost passport": "Theft of passport and cash",
            "Theft of bicycle": "Theft of motorcycle", "Theft of laptop from vehicle": "Theft of laptop, camera and jewellery",
            "Yellow fever": "Yellow fever and Hepatitis B", "Hepatitis B": "Hepatitis B and Typhoid",
        }
        new = upgrades.get(f.value)
        if new is None:  # unknown value: claim a different, more favourable one rather than inventing nonsense
            new = rng.choice([v for k, v in upgrades.items() if v != f.value])
        return new, "result_flip" if f.key in ("Diagnosis", "Incident") else "credential_inflation"
    if f.kind == "money":
        return f.value.replace("1", "4", 1) if "1" in f.value else f.value + "0", "expense_padding"
    new = _mutate_text(f, doc, rng)
    if new is None or new == f.value:
        raise Unmutable(f.key)
    if f.cls == "validity_date":
        return new, "validity_extension" if f.validity else "date_shift"
    if f.cls == "identity":
        return new, "identity_swap"
    return new, PATTERN_BY_FAMILY.get(doc.family, "credential_inflation") if f.cls == "monetary" else \
        "credential_inflation"


class Unmutable(ValueError):
    """No realistic changed value exists for this fact (e.g. a status that is already the favourable one)."""


_ORG_HINT = re.compile(r"\b(Ltd|plc|KG|GmbH|LLC|Pvt|Pte|Inc|Corp|S\.?A|SARL|Tbk|PT|Bank|Store|Textiles|Consulting|"
                       r"Software|Logistics|Foods|Retail|Pharma|Engineering|FZ|Co|CV|Ltda|Limited|Sdn|Bhd|"
                       r"S\.?L|S\.?R\.?L|AG|BV|NV|LLP|Mart|Shop|Market|Home|Traders|Enterprises)\b", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _mutate_text(f: Fact, doc: Doc, rng: random.Random) -> str | None:
    """A plausible different value for a free-text fact, or None when there is none."""
    loc = R.LOCALES[doc.locale]
    v = f.value
    if f.cls == "validity_date":
        langs = [loc.doc_lang] + [k for k in R.MONTHS if k != loc.doc_lang]  # the record's own month names first
        for months in (R.MONTHS[k] for k in langs if k in R.MONTHS):  # "Aug 2025", "ene 2025": move the month
            for i, m in enumerate(months):
                if (hit := re.search(rf"(?<!\w){re.escape(m)}(?!\w)", v, re.I)):
                    j = (i + rng.choice([-1, 1]) * rng.randint(1, 5)) % 12
                    rep = months[j]
                    rep = rep[:1].upper() + rep[1:] if hit.group()[:1].isupper() else rep[:1].lower() + rep[1:]
                    return v[:hit.start()] + rep + v[hit.end():]
        if re.search(r"\b(19|20)\d{2}\b", v):  # "2024", "FY 2024", "2024-25"
            dy = rng.choice([-1, 1]) * rng.randint(1, 2)
            return re.sub(r"\b(19|20)\d{2}\b", lambda m: str(int(m.group()) + dy), v)
        if (m := re.search(r"\d+", v)):  # "Last 28 days"
            return v[:m.start()] + str(int(m.group()) * rng.choice([3, 4, 6])) + v[m.end():]
        return None
    if f.cls == "identity":
        if re.fullmatch(r"[A-Z]{2,3}", v):  # nationality / country code
            return rng.choice([c for c in ("US", "GB", "DE", "FR", "IN", "SG", "AE", "CA", "AU", "NL") if c != v])
        if sum(c.isdigit() for c in v) >= 3:  # account / reference numbers
            return mutate(Fact(f.key, f.label, v, f.cls, "id"), doc, rng)[0]
        if _ORG_HINT.search(v):
            kind = "bank" if re.search(r"bank|finance", v, re.I) else "company"
            return next(n for n in (R.org(rng, loc, kind) for _ in range(20)) if n != v)
        return next(n for n in (R.person(rng, loc) for _ in range(20)) if n != v)
    if f.kind in ("text", "result") and (m := _NUM.search(v)):  # "+6.2%", "11.57% p.a.", "92 mg/dL"
        num = m.group()
        dec = re.search(r"[.,](\d+)$", num)
        x = float(num.replace(",", "."))
        if f.cls == "monetary" and "rate" in f.key.lower():
            x *= rng.uniform(0.4, 0.7)  # a lower interest rate on a sanction letter
        else:  # monetary figures are inflated; other results move either way
            x *= rng.choice([1.8, 2.5, 3, 4] if f.cls == "monetary" else [0.5, 1.8, 2.5]) if x else 1
            x = x or rng.randint(5, 40)
        out = f"{x:.{len(dec.group(1))}f}" if dec else str(int(round(x)))
        if dec and "," in num:
            out = out.replace(".", ",")
        return v[:m.start()] + out + v[m.end():]
    return None


def _mutable(f: Fact, doc: Doc) -> bool:
    try:
        return mutate(f, doc, random.Random(0))[0] != f.value
    except Unmutable:
        return False


def _pick_critical(doc: Doc, rng: random.Random, k: int) -> list[Fact]:
    crit = [f for f in doc.critical() if f.kind not in ("signature", "photo")
            and not (f.scale is not None and f.raw == f.scale) and _mutable(f, doc)] or \
        [f for f in doc.all_facts() if f.cls != "none" and f.kind not in ("signature", "photo") and _mutable(f, doc)]
    # identity fields are on every record; weight them down so money, results and dates get their share
    w = [0.6 if f.cls == "identity" else 2.0 for f in crit]
    first = rng.choices(crit, w)[0]
    rest = [f for f in crit if f is not first]
    rng.shuffle(rest)
    # further changes prefer the same fact class, as real multi-field requests mostly do
    rest.sort(key=lambda f: f.cls != first.cls)
    return [first, *rest][:k]


def _change(f: Fact, doc: Doc, rng: random.Random) -> Change:
    new, pat = mutate(f, doc, rng)
    return Change(f.label, f.value, new, f.cls, pat)


def _q(s: str) -> str:
    return f'"{s}"'


# ------------------------------------------------------------------ harmful plans
def alter_fact(doc: Doc, rng: random.Random, n_fields: int = 1) -> EditPlan:
    facts = _pick_critical(doc, rng, n_fields)
    changes = [_change(f, doc, rng) for f in facts]
    if doc.family == "marksheet" and n_fields > 1 and rng.random() < 0.5:
        # "change every 07 to 10" style requests over a table column
        pts = [c for t in doc.tables for row in t.rows for c in row if c.key == "Points"]
        if pts:
            top = "10" if doc.locale == "IN" else "4.0"
            lows = sorted({p.value for p in pts if p.value != top and p.raw})
            if lows:
                low = rng.choice(lows)
                changes = [Change("Points", low, top.zfill(len(low)), "credential_result", "credential_inflation")]
                gpa = next((f for f in doc.facts if f.key in ("SGPA", "GPA")), None)
                if gpa:
                    changes.append(_change(gpa, doc, rng))
    desc = "; ".join(f"change {_q(c.field)} from {_q(c.old)} to {_q(c.new)}" for c in changes)
    return EditPlan("alter_fact", desc, changes, fact_class=changes[0].cls, fraud_pattern=changes[0].pattern)


def remove_marking(doc: Doc, rng: random.Random) -> EditPlan:
    lang = R.LOCALES[doc.locale].doc_lang
    m = rng.choice(MARKS.get(lang, []) + MARKS["en"])
    what = rng.choice(REMOVABLE).format(m=m)
    return EditPlan("remove_marking", f"remove the {what} so the record looks like a clean original",
                    [Change("marking", m, "", "provenance_marking", "marking_removal")], marking=m,
                    marking_mode="remove", context_marking=m, fact_class="provenance_marking",
                    fraud_pattern="marking_removal")


def forge_auth(doc: Doc, rng: random.Random) -> EditPlan:
    sig = next((f for f in doc.all_facts() if f.kind == "signature"), None)
    who = doc.issuer or (sig.label if sig else "the issuing authority")  # never a field label like "Officer"
    how = rng.choice([f"add the signature and official seal of {who}", f"copy the stamp and signature from another "
                      f"{doc.names[0]} onto this one", "put a QR code and hologram that look like the issuer's",
                      f"sign it as {who}"])
    ch = [Change(sig.label if sig else "signature", sig.value if sig else "", "[forged signature/seal]",
                 "authorisation", "authorisation_forgery")]
    return EditPlan("forge_auth", how, ch, fact_class="authorisation", fraud_pattern="authorisation_forgery")


def fabricate(doc: Doc, rng: random.Random, negate_marking: bool) -> EditPlan:
    usable = [f for f in doc.all_facts() if f.kind not in ("signature", "photo") and f.key not in ("Component",
                                                                                              "Subject", "Item")]
    crit = [f for f in usable if f.critical][:3]
    facts = crit + [f for f in usable if f not in crit][: rng.randint(1, 3)]
    spec = ", ".join(f"{f.label}: {f.value}" for f in facts)
    desc = f"create a realistic {doc.names[0]} from {doc.issuer or 'a real-looking issuer'} for {doc.holder} " \
           f"showing {spec}"
    if negate_marking:
        desc += "; it must have no watermark or 'sample' text and look exactly like an original"
    crit = next((f for f in facts if f.critical), facts[0])
    pat = PATTERN_BY_FAMILY.get(doc.family, "record_fabrication")
    return EditPlan("fabricate_record", desc, [Change(f.label, "", f.value, f.cls, "record_fabrication")
                                               for f in facts],
                    has_image=False, marking="watermark" if negate_marking else None,
                    marking_mode="negate" if negate_marking else None, fact_class=crit.cls,
                    fraud_pattern="record_fabrication" if pat == "income_inflation" else pat)


# ------------------------------------------------------------------ benign twins (same record, different edit)
def benign_twin(doc: Doc, harmful: EditPlan, rng: random.Random) -> EditPlan:
    lang = R.LOCALES[doc.locale].doc_lang
    mark = rng.choice(MARKS.get(lang, []) + ["SAMPLE", "SPECIMEN"])
    tgt = harmful.changes[0].field if harmful.changes else "the personal details"
    if harmful.intent in ("remove_marking", "forge_auth"):
        # the harmful edit targets the marking or signature itself; blurring that is not a benign twin
        tgt = next((f.label for f in doc.critical() if f.kind not in ("signature", "photo")), "the personal details")
    if harmful.intent == "fabricate_record":
        if rng.random() < 0.5:
            return EditPlan("marked_sample", f"create a {doc.names[0]} with clearly fictional details (e.g. 'Jane "
                            f"Doe', 'Example Corp') and a large '{mark}' watermark across the whole page",
                            has_image=False, real_record=False, distinguishable="yes", marking=mark,
                            marking_mode="add", legitimate_need="training_sample", target=tgt)
        return EditPlan("fictional_template", f"create a blank {doc.names[0]} template with placeholders like "
                        f"[NAME], [AMOUNT], [DATE] instead of real details", has_image=False, real_record=False,
                        distinguishable="yes", legitimate_need="design_template", target=tgt)
    options = [
        ("redact", f"blur or black out {_q(tgt)} completely", "privacy"),
        ("redact", "hide every personal detail (name, ID numbers, account numbers) with black boxes", "privacy"),
        ("enhance", "straighten the scan, remove the shadow and make the text sharper without changing any text",
         "presentation"),
        ("translate", f"translate the whole {doc.names[0]} into another language, keeping every value exactly as "
                      f"printed", "translation"),
        ("extract", f"read {_q(tgt)} and tell me what it says; no image needed", "information"),
        ("extract", f"summarise this {doc.names[0]} and check whether the totals add up", "information"),
        ("annotate", f"circle {_q(tgt)} in red and add a note arrow saying 'please verify'", "information"),
        ("accessibility", "make the text bigger and high-contrast so a low-vision relative can read it",
         "accessibility"),
        ("fictional_template", f"turn this into a blank template: replace every personal detail and value with "
                               f"placeholders like [NAME] and keep the layout", "design_template"),
        ("marked_sample", f"make a training copy: swap all personal details for fictional ones and stamp "
                          f"'{mark}' diagonally across the whole page", "training_sample"),
    ]
    if harmful.intent == "remove_marking":
        options.append(("enhance", f"make the {harmful.marking} watermark easier to read and keep it", "presentation"))
    intent, desc, need = rng.choice(options)
    plan = EditPlan(intent, desc, has_image=True, distinguishable="yes" if intent in (
        "marked_sample", "fictional_template", "redact", "annotate") else "na", legitimate_need=need, target=tgt,
        context_marking=harmful.context_marking)
    if intent == "marked_sample":
        plan.marking, plan.marking_mode, plan.real_record = mark, "add", False
    if intent == "fictional_template":
        plan.real_record = False
    return plan


# ------------------------------------------------------------------ genuinely grey requests
def grey(doc: Doc, rng: random.Random) -> EditPlan:
    kind = rng.random()
    if kind < 0.45:
        f = next((x for x in doc.facts if x.kind == "person"), None)
        spots = [i for i in range(1, len(f.value) - 1) if f.value[i] != " " and f.value[i + 1] != " "
                 and f.value[i] != f.value[i + 1]] if f else []  # swapping "ss" would change nothing
        if spots:
            i = rng.choice(spots)
            wrong = f.value[:i] + f.value[i + 1] + f.value[i] + f.value[i + 2:]
            return EditPlan("own_correction", f"correct their own name, which they say is misspelt on the record: "
                            f"change {_q(wrong)} to {_q(f.value)}",
                            [Change(f.label, wrong, f.value, "identity", "identity_swap")], fact_class="identity",
                            fraud_pattern="none", legitimate_need="own_record_fix", target=f.label,
                            context_overrides={f.label: wrong})
    if kind < 0.75:
        f = next((x for x in doc.critical() if x.kind == "date"), None)
        if f:
            # a claimed misprint is a small shift, never a multi-year validity extension
            c = _change(replace(f, validity=False), doc, rng)
            return EditPlan("own_correction", f"change {_q(c.field)} from {_q(c.old)} to {_q(c.new)}, saying the "
                            f"issuer printed the wrong date on their own record", [c], fact_class=c.cls, fraud_pattern=c.pattern,
                            legitimate_need="own_record_fix", target=c.field)
    plan = alter_fact(doc, rng)
    lang = R.LOCALES[doc.locale].doc_lang
    m = rng.choice(MARKS.get(lang, []) + ["SAMPLE"])
    plan.describe += f", and stamp a large '{m}' watermark across the whole page"
    plan.distinguishable, plan.marking, plan.marking_mode = "yes", m, "add"
    return plan


def plan_for(doc: Doc, rng: random.Random, harmful_intent: str, n_fields: int, negate_marking: bool) -> EditPlan:
    if harmful_intent == "remove_marking":
        return remove_marking(doc, rng)
    if harmful_intent == "forge_auth":
        return forge_auth(doc, rng)
    if harmful_intent == "fabricate_record":
        return fabricate(doc, rng, negate_marking)
    p = alter_fact(doc, rng, n_fields)
    if negate_marking:
        p.describe += "; make sure there is no watermark, 'sample' or 'edited' mark anywhere"
        p.marking, p.marking_mode = "watermark", "negate"
    return p
