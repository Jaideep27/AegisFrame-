"""Value generators for fictional records. Every name, company and number here is invented."""

from __future__ import annotations

import random
from datetime import date, timedelta

FIRST = ["Arjun", "Priya", "Kavya", "Rohan", "Meera", "Ravi", "Neha", "Anil", "Divya", "Karthik", "Sneha", "Vikram",
         "Aisha", "Farhan", "Lakshmi", "Suresh", "Ananya", "Rahul", "Pooja", "Imran", "Deepa", "Manoj", "Sara", "John",
         "Maria", "Chen", "Fatima", "Lucas", "Emma", "Noah", "Olivia", "Diego", "Yuki", "Amara", "Ibrahim", "Zoe"]
LAST = ["Mehta", "Raghavan", "Iyer", "Kumar", "Kapoor", "Joseph", "Menon", "Sundaram", "Verma", "Pillai", "Reddy",
        "Shah", "Khan", "Nair", "Das", "Patel", "Singh", "Gupta", "Fernandes", "Smith", "Garcia", "Wang", "Okafor",
        "Silva", "Novak", "Haddad", "Tanaka", "Murphy", "Rossi", "Lindqvist"]
ORG_A = ["Orbit", "Lotus", "Kaveri", "Skylark", "Coastal", "Deccan", "Voltaic", "Trident", "Nimbus", "Aurora",
         "Harbor", "Saffron", "Granite", "Peak", "Indigo", "Banyan", "Cedar", "Monsoon", "Polar", "Zenith"]
ORG_B = {
    "company": ["Software Pvt Ltd", "Technologies", "Retail LLP", "Logistics Ltd", "Labs", "Industries"],
    "bank": ["Union Bank", "Co-operative Bank", "Finance Bank", "Savings Bank"],
    "school": ["Technical University", "State Examination Board", "Institute of Science", "Public School", "College"],
    "clinic": ["Diagnostics", "Hospital", "Health Labs", "Medical Centre"],
    "airline": ["Airways", "Air", "Jet"],
    "insurer": ["General Insurance", "Assurance", "Life Insurance"],
    "pay": ["Pay", "Wallet", "UPI", "Money"],
    "gov": ["Municipal Corporation", "Regional Transport Office", "Revenue Department", "Passport Seva"],
    "utility": ["Power Distribution Ltd", "Water Board", "Gas Ltd", "Telecom"],
    "shop": ["Electronics", "Mart", "Store", "Traders"],
}
CITIES = ["Chennai", "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune", "Kochi", "Jaipur", "Kolkata", "Ahmedabad",
          "Singapore", "Dubai", "London", "Toronto", "Berlin"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CURRENCIES = [("Rs. ", 1), ("₹", 1), ("INR ", 1), ("$", 0.012), ("USD ", 0.012), ("€", 0.011), ("AED ", 0.044)]


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def org(rng: random.Random, kind: str) -> str:
    return f"{rng.choice(ORG_A)} {rng.choice(ORG_B[kind])}"


def _group(n: int, indian: bool) -> str:
    if not indian or n < 100000:
        return f"{n:,}"
    s = str(n)
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *parts, tail]) if head else ",".join([*parts, tail])


def money(rng: random.Random, lo: int, hi: int, currency: str | None = None) -> str:
    cur, rate = next((c for c in CURRENCIES if c[0] == currency), None) or rng.choice(CURRENCIES)
    n = max(1, int(rng.randint(lo, hi) * rate))
    n = int(round(n, -1 if n > 100 else 0))
    indian = cur.strip() in {"Rs.", "₹", "INR"}
    return f"{cur}{_group(n, indian)}"


def bump_money(rng: random.Random, value: str, up: bool = True) -> str:
    import re

    m = re.search(r"([\d,]+)", value)
    if not m:
        return value
    n = int(m.group(1).replace(",", ""))
    factor = rng.choice([2, 3, 5, 10, 1.5, 1.8, 4]) if up else rng.choice([0.1, 0.2, 0.5, 0.3])
    new = max(1, int(n * factor))
    new = int(round(new, -1 if new > 100 else 0))
    if new == n:
        new = n + (10 if up else -1)
    prefix = value[: m.start()]
    indian = prefix.strip() in {"Rs.", "₹", "INR"}
    return f"{prefix}{_group(new, indian)}{value[m.end():]}"


def a_date(rng: random.Random, start_year: int = 2020, end_year: int = 2027, fmt: int | None = None) -> date:
    d0 = date(start_year, 1, 1)
    return d0 + timedelta(days=rng.randint(0, (date(end_year, 12, 31) - d0).days))


def fmt_date(d: date, style: int) -> str:
    if style == 0:
        return f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}"
    if style == 1:
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    if style == 2:
        return f"{d.year}-{d.month:02d}-{d.day:02d}"
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def code(rng: random.Random, pattern: str) -> str:
    """A = letter, 9 = digit, anything else literal."""
    out = []
    for ch in pattern:
        if ch == "A":
            out.append(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))
        elif ch == "9":
            out.append(rng.choice("0123456789"))
        else:
            out.append(ch)
    return "".join(out)
