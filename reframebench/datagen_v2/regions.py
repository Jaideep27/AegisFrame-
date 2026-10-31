"""Locales: currency and number formats, date formats, fictional names and organisations, label translations.

Every person and organisation is invented by combining common name parts; no real record is used.
Regions are grouped so evaluation can hold out a whole region (leave-one-region-out).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "es": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"],
    "pt": ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"],
    "fr": ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."],
    "de": ["Jan.", "Feb.", "März", "Apr.", "Mai", "Juni", "Juli", "Aug.", "Sept.", "Okt.", "Nov.", "Dez."],
    "id": ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"],
}


@dataclass(frozen=True)
class Locale:
    code: str
    region: str               # region group for leave-one-region-out
    doc_lang: str             # language of field labels on local documents
    currency: tuple[str, ...]  # how amounts are written, e.g. ("₹", "Rs. ", "INR ")
    suffix_currency: bool     # "1.250,00 €" style
    thousands: str            # "," "." " " or "lakh"
    decimals: bool            # amounts shown with cents
    salary: int               # typical monthly gross salary, local currency (sets every money scale)
    date_styles: tuple[str, ...]
    prompt_langs: tuple[tuple[str, float], ...]  # languages users of this locale write requests in
    first: tuple[str, ...]
    last: tuple[str, ...]
    cities: tuple[str, ...]
    company_suffix: tuple[str, ...]
    id_name: str              # what the national ID is called (generic description, not a real form)


L = Locale
LOCALES: dict[str, Locale] = {x.code: x for x in [
    L("US", "north_america", "en", ("$",), False, ",", True, 5200, ("mdy", "mon_d_y"),
      (("en", 0.85), ("es", 0.08), ("spanglish", 0.07)),
      ("James", "Emily", "Michael", "Ashley", "Daniel", "Jessica", "Tyler", "Brianna", "Marcus", "Hannah", "Kevin"),
      ("Johnson", "Miller", "Davis", "Thompson", "Walker", "Brooks", "Carter", "Reyes", "Nguyen", "Coleman"),
      ("Columbus", "Tucson", "Raleigh", "Omaha", "Spokane", "Fresno"), ("Inc.", "LLC", "Corp."), "state ID card"),
    L("UK", "europe", "en", ("£",), False, ",", True, 3100, ("dmy_slash", "d_mon_y"),
      (("en", 1.0),),
      ("Oliver", "Amelia", "Harry", "Isla", "George", "Poppy", "Jack", "Freya", "Callum", "Niamh"),
      ("Hughes", "Wright", "Patel", "Evans", "Clarke", "Turner", "Bennett", "Ahmed", "Fletcher", "Doyle"),
      ("Leeds", "Bristol", "Leicester", "Cardiff", "Norwich", "Derby"), ("Ltd", "plc"), "identity card"),
    L("DE", "europe", "de", ("€",), True, ".", True, 3900, ("dmy_dot",),
      (("de", 0.8), ("en", 0.2)),
      ("Lukas", "Lena", "Jonas", "Mia", "Felix", "Hannah", "Paul", "Lea", "Tim", "Sophie"),
      ("Müller", "Schmidt", "Weber", "Becker", "Hoffmann", "Schäfer", "Koch", "Richter", "Klein", "Wolf"),
      ("Kassel", "Bochum", "Würzburg", "Rostock", "Augsburg", "Mainz"), ("GmbH", "AG", "KG"), "Personalausweis-style ID"),
    L("FR", "europe", "fr", ("€",), True, " ", True, 3000, ("dmy_slash",),
      (("fr", 0.85), ("en", 0.15)),
      ("Lucas", "Chloé", "Hugo", "Manon", "Louis", "Camille", "Nathan", "Inès", "Théo", "Léa"),
      ("Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Lefèvre", "Girard", "Roux", "Fournier", "Mercier"),
      ("Rennes", "Dijon", "Angers", "Nîmes", "Limoges", "Tours"), ("SARL", "SAS", "S.A."), "carte d'identité-style ID"),
    L("ES", "europe", "es", ("€",), True, ".", True, 2300, ("dmy_slash",),
      (("es", 0.85), ("en", 0.15)),
      ("Pablo", "Lucía", "Álvaro", "Carmen", "Javier", "Marta", "Sergio", "Elena", "Diego", "Paula"),
      ("García", "Fernández", "López", "Martínez", "Sánchez", "Romero", "Navarro", "Torres", "Ruiz", "Gil"),
      ("Zaragoza", "Murcia", "Valladolid", "Córdoba", "Alicante", "Vigo"), ("S.L.", "S.A."), "DNI-style ID"),
    L("IN", "south_asia", "en", ("₹", "Rs. ", "INR "), False, "lakh", False, 62000, ("d_mon_y", "dmy_slash", "dmy_dash"),
      (("en", 0.55), ("hinglish", 0.3), ("hi", 0.15)),
      ("Arjun", "Priya", "Rohan", "Kavya", "Vikram", "Sneha", "Karthik", "Ananya", "Imran", "Lakshmi", "Harpreet"),
      ("Mehta", "Iyer", "Reddy", "Sharma", "Nair", "Kulkarni", "Banerjee", "Khan", "Gill", "Pillai", "Joshi"),
      ("Pune", "Indore", "Coimbatore", "Lucknow", "Nagpur", "Vizag"), ("Pvt Ltd", "LLP", "Ltd"), "national ID card"),
    L("ID", "southeast_asia", "id", ("Rp ",), False, ".", False, 9500000, ("d_mon_y", "dmy_slash"),
      (("id", 0.7), ("en", 0.3)),
      ("Budi", "Siti", "Agus", "Dewi", "Rizky", "Putri", "Andi", "Wulan", "Fajar", "Ayu"),
      ("Santoso", "Wijaya", "Saputra", "Hidayat", "Kurniawan", "Lestari", "Pratama", "Nugroho", "Halim", "Siregar"),
      ("Bandung", "Semarang", "Malang", "Makassar", "Medan", "Denpasar"), ("PT", "CV"), "KTP-style ID"),
    L("PH", "southeast_asia", "en", ("₱", "PHP "), False, ",", True, 38000, ("mon_d_y", "mdy"),
      (("en", 0.5), ("taglish", 0.5)),
      ("Juan", "Maria", "Mark", "Angelica", "John Paul", "Kristine", "Carlo", "Patricia", "Paolo", "Jasmine"),
      ("Santos", "Reyes", "Cruz", "Bautista", "Villanueva", "Mendoza", "Aquino", "Ramos", "Dela Cruz", "Garcia"),
      ("Cebu", "Davao", "Iloilo", "Baguio", "Quezon City", "Cagayan de Oro"), ("Inc.", "Corp."), "national ID card"),
    L("SG", "southeast_asia", "en", ("S$", "SGD "), False, ",", True, 6200, ("d_mon_y", "dmy_slash"),
      (("en", 0.9), ("zh", 0.1)),
      ("Wei Ming", "Hui Min", "Jun Jie", "Xin Yi", "Ravi", "Nurul", "Daniel", "Shu Fen", "Hafiz", "Jia Hui"),
      ("Tan", "Lim", "Lee", "Ng", "Ong", "Wong", "Goh", "Chua", "Rahman", "Pillai"),
      ("Singapore",), ("Pte Ltd",), "identity card"),
    L("CN", "east_asia", "en", ("¥", "CNY ", "RMB "), False, ",", True, 11000, ("ymd_dash",),
      (("zh", 0.8), ("en", 0.2)),
      ("Wei", "Fang", "Lei", "Jing", "Hao", "Min", "Yang", "Xiu", "Jun", "Li"),
      ("Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Zhao", "Huang", "Zhou", "Wu"),
      ("Hefei", "Kunming", "Xiamen", "Changsha", "Jinan", "Ningbo"), ("Co., Ltd.",), "resident ID card"),
    L("AE", "middle_east", "en", ("AED ",), False, ",", True, 16000, ("dmy_slash", "d_mon_y"),
      (("en", 0.6), ("ar", 0.4)),
      ("Ahmed", "Fatima", "Omar", "Mariam", "Khalid", "Aisha", "Yousef", "Noura", "Hamdan", "Layla"),
      ("Al Mansoori", "Al Hashimi", "Haddad", "Saleh", "Rahman", "Nasser", "Al Falasi", "Karim", "Aziz", "Said"),
      ("Sharjah", "Al Ain", "Ajman", "Ras Al Khaimah"), ("LLC", "FZE", "FZ-LLC"), "resident ID card"),
    L("SA", "middle_east", "en", ("SAR ", "SR "), False, ",", True, 12500, ("dmy_slash",),
      (("ar", 0.6), ("en", 0.4)),
      ("Abdullah", "Nora", "Faisal", "Reem", "Saud", "Lama", "Turki", "Hessa", "Majed", "Sara"),
      ("Al Qahtani", "Al Harbi", "Al Otaibi", "Al Shehri", "Al Zahrani", "Al Ghamdi", "Al Dosari", "Al Mutairi"),
      ("Dammam", "Abha", "Tabuk", "Buraydah"), ("Co.", "LLC"), "national ID card"),
    L("NG", "africa", "en", ("₦", "NGN "), False, ",", True, 380000, ("dmy_slash", "d_mon_y"),
      (("en", 0.85), ("pidgin", 0.15)),
      ("Chinedu", "Ngozi", "Tunde", "Aisha", "Emeka", "Funmilayo", "Ibrahim", "Chiamaka", "Segun", "Zainab"),
      ("Okafor", "Adeyemi", "Bello", "Eze", "Okonkwo", "Balogun", "Musa", "Nwosu", "Ogunleye", "Abubakar"),
      ("Ibadan", "Enugu", "Kano", "Benin City", "Jos", "Abeokuta"), ("Ltd", "Plc"), "national ID slip"),
    L("KE", "africa", "en", ("KSh ", "KES "), False, ",", True, 65000, ("dmy_slash", "d_mon_y"),
      (("en", 0.75), ("sw", 0.25)),
      ("Brian", "Faith", "Kevin", "Mercy", "Dennis", "Wanjiru", "Collins", "Akinyi", "Samuel", "Njeri"),
      ("Otieno", "Kamau", "Mwangi", "Wafula", "Kiprono", "Achieng", "Njoroge", "Mutua", "Chebet", "Omondi"),
      ("Nakuru", "Eldoret", "Kisumu", "Thika", "Nyeri"), ("Ltd",), "national ID card"),
    L("ZA", "africa", "en", ("R ", "ZAR "), False, " ", True, 27000, ("ymd_slash", "d_mon_y"),
      (("en", 0.9), ("af", 0.1)),
      ("Thabo", "Lerato", "Sipho", "Nomvula", "Pieter", "Anele", "Johan", "Zanele", "Kagiso", "Megan"),
      ("Nkosi", "Dlamini", "van der Merwe", "Botha", "Mokoena", "Naidoo", "Khumalo", "Pillay", "Smit", "Mahlangu"),
      ("Pretoria", "Gqeberha", "Bloemfontein", "Polokwane", "East London"), ("(Pty) Ltd",), "smart ID card"),
    L("BR", "latin_america", "pt", ("R$ ",), False, ".", True, 4600, ("dmy_slash",),
      (("pt", 0.85), ("en", 0.15)),
      ("Lucas", "Ana", "Gabriel", "Juliana", "Mateus", "Larissa", "Rafael", "Beatriz", "Thiago", "Camila"),
      ("Silva", "Souza", "Oliveira", "Pereira", "Costa", "Rodrigues", "Almeida", "Nascimento", "Lima", "Araújo"),
      ("Campinas", "Curitiba", "Recife", "Goiânia", "Belém", "Natal"), ("Ltda.", "S.A."), "RG-style ID"),
    L("MX", "latin_america", "es", ("$", "MXN "), False, ",", True, 18000, ("dmy_slash",),
      (("es", 0.75), ("spanglish", 0.1), ("en", 0.15)),
      ("José", "Guadalupe", "Luis", "Fernanda", "Carlos", "Daniela", "Miguel", "Valeria", "Jorge", "Ximena"),
      ("Hernández", "González", "Ramírez", "Flores", "Morales", "Jiménez", "Vázquez", "Castillo", "Ortiz", "Mendoza"),
      ("Puebla", "Querétaro", "Mérida", "León", "Toluca", "Saltillo"), ("S.A. de C.V.",), "INE-style voter card"),
]}

REGIONS = sorted({x.region for x in LOCALES.values()})

# invented organisation name parts (no real brand)
ORG_PARTS = ("Orbit", "Lumen", "Kestrel", "Halden", "Verano", "Corvid", "Tamsin", "Zephyr", "Aldane", "Novik",
             "Brightwater", "Sable", "Quorra", "Meridia", "Ostra", "Pellucid", "Juniper", "Talvik", "Rivane",
             "Castellan", "Oakhurst", "Sunvale", "Ironleaf", "Bluecrest", "Marlowe", "Vantor", "Elmora", "Torin")
ORG_KIND = {
    "company": ("Software", "Logistics", "Foods", "Textiles", "Consulting", "Engineering", "Retail", "Pharma"),
    "bank": ("Bank", "Savings Bank", "Commercial Bank", "Finance"),
    "school": ("University", "Institute of Technology", "College", "Academy", "Examination Board"),
    "clinic": ("Diagnostics", "Hospital", "Medical Centre", "Health Labs", "Clinic"),
    "insurer": ("Insurance", "Assurance", "General Insurance", "Mutual"),
    "wallet": ("Pay", "Wallet", "Cash", "Money"),
    "gov": ("City Council", "Revenue Office", "Transport Authority", "Civil Registry", "Immigration Service"),
    "utility": ("Power", "Water Utility", "Gas", "Telecom", "Broadband"),
    "shop": ("Electronics", "Mart", "Supermarket", "Pharmacy", "Home Store"),
    "airline": ("Airways", "Air", "Airlines"),
    "broker": ("Securities", "Capital", "Markets", "Exchange"),
    "court": ("District Court", "Magistrate's Court", "Civil Court"),
    "police": ("Police Station", "Police Department"),
    "platform": ("Market", "Bazaar", "Hub", "Studio"),
    "courier": ("Express", "Courier", "Parcel"),
}

LABELS_I18N: dict[str, dict[str, str]] = {
    "Employee Name": {"de": "Name des Mitarbeiters", "fr": "Nom du salarié", "es": "Nombre del empleado",
                      "pt": "Nome do funcionário", "id": "Nama Karyawan"},
    "Employee ID": {"de": "Personalnummer", "fr": "Matricule", "es": "N.º de empleado", "pt": "Matrícula",
                    "id": "NIK Karyawan"},
    "Designation": {"de": "Position", "fr": "Poste", "es": "Puesto", "pt": "Cargo", "id": "Jabatan"},
    "Pay Period": {"de": "Abrechnungszeitraum", "fr": "Période", "es": "Periodo", "pt": "Competência",
                   "id": "Periode"},
    "Gross Pay": {"de": "Bruttolohn", "fr": "Salaire brut", "es": "Total devengado", "pt": "Total de proventos",
                  "id": "Total Pendapatan"},
    "Total Deductions": {"de": "Abzüge gesamt", "fr": "Total retenues", "es": "Total deducciones",
                         "pt": "Total de descontos", "id": "Total Potongan"},
    "Net Pay": {"de": "Nettolohn", "fr": "Net à payer", "es": "Líquido a percibir", "pt": "Líquido a receber",
                "id": "Gaji Bersih"},
    "Account Holder": {"de": "Kontoinhaber", "fr": "Titulaire du compte", "es": "Titular", "pt": "Titular",
                       "id": "Nama Pemilik Rekening"},
    "Account Number": {"de": "IBAN", "fr": "IBAN", "es": "IBAN", "pt": "Conta", "id": "Nomor Rekening"},
    "Opening Balance": {"de": "Anfangssaldo", "fr": "Solde initial", "es": "Saldo inicial", "pt": "Saldo anterior",
                        "id": "Saldo Awal"},
    "Closing Balance": {"de": "Endsaldo", "fr": "Solde final", "es": "Saldo final", "pt": "Saldo final",
                        "id": "Saldo Akhir"},
    "Date": {"de": "Datum", "fr": "Date", "es": "Fecha", "pt": "Data", "id": "Tanggal"},
    "Description": {"de": "Buchungstext", "fr": "Libellé", "es": "Concepto", "pt": "Histórico", "id": "Keterangan"},
    "Amount": {"de": "Betrag", "fr": "Montant", "es": "Importe", "pt": "Valor", "id": "Jumlah"},
    "Balance": {"de": "Saldo", "fr": "Solde", "es": "Saldo", "pt": "Saldo", "id": "Saldo"},
    "Invoice Number": {"de": "Rechnungsnummer", "fr": "N° de facture", "es": "N.º de factura", "pt": "Nº da nota",
                       "id": "No. Faktur"},
    "Subtotal": {"de": "Zwischensumme", "fr": "Sous-total", "es": "Base imponible", "pt": "Subtotal",
                 "id": "Subtotal"},
    "Total": {"de": "Gesamtbetrag", "fr": "Total TTC", "es": "Total", "pt": "Total", "id": "Total"},
    "Student Name": {"de": "Name", "fr": "Nom de l'étudiant", "es": "Nombre del alumno", "pt": "Nome do aluno",
                     "id": "Nama Mahasiswa"},
    "Student ID": {"de": "Matrikelnummer", "fr": "N° étudiant", "es": "N.º de matrícula", "pt": "Matrícula",
                   "id": "NIM"},
    "Patient Name": {"de": "Patient", "fr": "Patient", "es": "Paciente", "pt": "Paciente", "id": "Nama Pasien"},
    "Result": {"de": "Ergebnis", "fr": "Résultat", "es": "Resultado", "pt": "Resultado", "id": "Hasil"},
    "Name": {"de": "Name", "fr": "Nom", "es": "Nombre", "pt": "Nome", "id": "Nama"},
    "Date of Birth": {"de": "Geburtsdatum", "fr": "Date de naissance", "es": "Fecha de nacimiento",
                      "pt": "Data de nascimento", "id": "Tanggal Lahir"},
    "Date of Issue": {"de": "Ausstellungsdatum", "fr": "Date de délivrance", "es": "Fecha de expedición",
                      "pt": "Data de emissão", "id": "Tanggal Terbit"},
    "Date of Expiry": {"de": "Gültig bis", "fr": "Date d'expiration", "es": "Fecha de caducidad",
                       "pt": "Validade", "id": "Berlaku Hingga"},
    "Status": {"de": "Status", "fr": "Statut", "es": "Estado", "pt": "Status", "id": "Status"},
    "Amount Due": {"de": "Fälliger Betrag", "fr": "Montant dû", "es": "Importe a pagar", "pt": "Valor a pagar",
                   "id": "Jumlah Tagihan"},
    "Due Date": {"de": "Fällig am", "fr": "Échéance", "es": "Fecha de vencimiento", "pt": "Vencimento",
                 "id": "Jatuh Tempo"},
    "Address": {"de": "Anschrift", "fr": "Adresse", "es": "Dirección", "pt": "Endereço", "id": "Alamat"},
}


def label(text: str, lang: str) -> str:
    return LABELS_I18N.get(text, {}).get(lang, text)


def pick_lang(rng: random.Random, loc: Locale) -> str:
    langs, weights = zip(*loc.prompt_langs)
    return rng.choices(langs, weights)[0]


def person(rng: random.Random, loc: Locale) -> str:
    return f"{rng.choice(loc.first)} {rng.choice(loc.last)}"


def org(rng: random.Random, loc: Locale, kind: str) -> str:
    name = f"{rng.choice(ORG_PARTS)} {rng.choice(ORG_KIND[kind])}"
    if kind in ("company", "shop", "wallet", "broker", "platform", "courier", "utility", "insurer"):
        name += " " + rng.choice(loc.company_suffix)
    elif kind in ("gov", "court", "police"):
        name = f"{rng.choice(loc.cities)} {rng.choice(ORG_KIND[kind])}"
    return name


def _group(n: int, sep: str) -> str:
    s = str(n)
    if sep == "lakh":
        if n < 100000:
            return f"{n:,}"
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        return ",".join([head, *parts, tail])
    return f"{n:,}".replace(",", sep)


def fmt_money(amount: float, loc: Locale, cur: str | None = None) -> str:
    cur = cur or loc.currency[0]
    if not loc.decimals:
        amount = round(amount)
    whole = int(abs(amount))
    sign = "-" if amount < 0 else ""
    body = _group(whole, loc.thousands)
    if loc.decimals:
        cents = int(round((abs(amount) - whole) * 100)) % 100
        dec = "," if loc.thousands in (".", " ") else "."
        body += f"{dec}{cents:02d}"
    if loc.suffix_currency:
        return f"{sign}{body} {cur.strip()}"
    return f"{sign}{cur}{body}"


def nice(x: float, loc: Locale) -> float:
    """Round an amount the way real records do for its size."""
    if x >= 100000:
        step = 1000 if x < 10_000_000 else 100000
    elif x >= 10000:
        step = 100
    elif x >= 1000:
        step = 10
    else:
        return round(x, 2) if loc.decimals else round(x)
    return round(x / step) * step


def a_date(rng: random.Random, lo: int = 2023, hi: int = 2026) -> date:
    d0 = date(lo, 1, 1)
    return d0 + timedelta(days=rng.randint(0, (date(hi, 12, 31) - d0).days))


def fmt_date(d: date, loc: Locale, style: str | None = None) -> str:
    style = style or loc.date_styles[0]
    mon = MONTHS.get(loc.doc_lang, MONTHS["en"])[d.month - 1]
    return {
        "mdy": f"{d.month:02d}/{d.day:02d}/{d.year}",
        "mon_d_y": f"{mon} {d.day}, {d.year}",
        "dmy_slash": f"{d.day:02d}/{d.month:02d}/{d.year}",
        "dmy_dot": f"{d.day:02d}.{d.month:02d}.{d.year}",
        "dmy_dash": f"{d.day:02d}-{d.month:02d}-{d.year}",
        "d_mon_y": f"{d.day:02d} {mon} {d.year}",
        "ymd_dash": f"{d.year}-{d.month:02d}-{d.day:02d}",
        "ymd_slash": f"{d.year}/{d.month:02d}/{d.day:02d}",
    }[style]


def code(rng: random.Random, pattern: str) -> str:
    """A = letter, 9 = digit, anything else literal."""
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") if c == "A" else rng.choice("0123456789") if c == "9"
                   else c for c in pattern)
