"""Transaction-level classification built on top of keyword models."""

from __future__ import annotations

import bisect
import logging
import re
from collections import defaultdict
from typing import NamedTuple

import pandas as pd

# Support both normal package imports (``from src.statement_classifier ...``)
# and direct module loading from the repository root / ``src`` directory.
if __package__:
    from .predict import KeywordClassifier
    from .prompt_part3 import (
        BANK_LEGAL_NAME_RE,
        REAL_DIGITAL_BRANDS,
        brand_alternation,
        clean_entity,
        detect_explicit_purpose,
        is_company_name,
    )
else:  # pragma: no cover - used by IDEs or scripts loading this file directly
    try:
        from src.predict import KeywordClassifier
        from src.prompt_part3 import (
            BANK_LEGAL_NAME_RE,
            REAL_DIGITAL_BRANDS,
            brand_alternation,
            clean_entity,
            detect_explicit_purpose,
            is_company_name,
        )
    except ModuleNotFoundError:
        from predict import KeywordClassifier
        from prompt_part3 import (
            BANK_LEGAL_NAME_RE,
            REAL_DIGITAL_BRANDS,
            brand_alternation,
            clean_entity,
            detect_explicit_purpose,
            is_company_name,
        )

logger = logging.getLogger(__name__)

PART_COLUMNS = tuple(f"Part_{index}" for index in range(8))

# ---------------------------------------------------------------------------
# Who decides each level (the single place that states the order of decisions)
# ---------------------------------------------------------------------------
# "code"   -> narration rules decide first. A Master Sheet / keyword-model value
#             never overrides them.
# "master" -> the keyword model (Master Sheet) decides first. Code only cleans up
#             a contradiction or fills a blank, except in the code-wins cases
#             named in the second field.
# A level that nothing decides stays blank.
LEVEL_DECISION_ORDER: dict[str, tuple[str, str]] = {
    "Level_0": ("code", "Debit/Credit columns (apply_amount_direction_override)."),
    "Level_1": (
        "code",
        "ATM/cash (apply_atm_withdrawal_overrides), cheque instruments "
        "(apply_cheque_instrument_override) and a narration that starts with the UPI "
        "structure (UPI_START_RE) overwrite the model; other UPI and cheque-return "
        "rules only fill a blank.",
    ),
    "Level_2": ("code", "apply_strict_level2 sets every row; the model value is never used."),
    "Level_3": (
        "master",
        "Code wins for salary, ATM, cheque, cash and PAYTMQR / merchant-only UPI IDs; "
        "other code rules (counterparty type, MERCHANT marker, known hyphen merchants) "
        "are listed in the change log as exceptions.",
    ),
    "Level_4": (
        "code",
        "Platform or gateway first (detect_level_4_platform_or_bank); the bank is used "
        "only when no platform is present. A bank never replaces a platform.",
    ),
    "Level_5": (
        "master",
        "Code fills blanks, clears 'p2p transfer' on merchants, and wins for salary, "
        "ATM, cheque, cash and PAYTMQR.",
    ),
    "Level_6": (
        "master",
        "Code fills blanks and wins for salary, ATM, cheque, cash, PAYTMQR and BBPS "
        "biller names; the institution rule only clears a bank echo. Slash UPI: Master "
        "name for merchant / institution rows, narration payee name for p2p rows.",
    ),
    "Level_7": ("code", "apply_level7_recurrence only."),
}

SALARY_KEYWORDS = re.compile(
    r"\b(?:salary|salaryxce|payroll|payslip|labour|wages|sal)\b",
    flags=re.IGNORECASE,
)
SALARY_ENTITY_STOPWORDS = re.compile(
    r"\b(?:salary|salaryxce|payroll|payslip|labour|wages|sal|credit|credited|via|by|from|for)\b",
    flags=re.IGNORECASE,
)
# Words and month names removed from a salary row's Level 6 (see clean_salary_entity).
SALARY_FILLER_WORDS = frozenset(
    {
        "salary", "salaryxce", "payroll", "payslip", "labour", "wages", "sal",
        "for", "of", "the", "month", "credit", "credited", "via", "by", "from", "to",
    }
)
SALARY_MONTH_WORDS = frozenset(
    {
        "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
        "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
        "oct", "october", "nov", "november", "dec", "december",
    }
)
# "sal ary" / "sala ry" are "salary" broken by statement line wrapping.
SPLIT_SALARY_RE = re.compile(r"\bsal\s+ary\b|\bsala\s+ry\b|\bsalar\s+y\b", flags=re.IGNORECASE)
MIN_SALARY = 10_000
MIN_SALARY_MONTHS = 2
SALARY_DAY_STD_MAX = 7
SLASH_UPI_RE = re.compile(
    # \s* around the bank code: statements wrap long narrations, leaving a stray
    # space (e.g. ".../Flipkart/ YESB/paytm-5650/...").
    r"\bUPI\b(?:/\d+)?/(?:DR|CR)(?:/\d+)?/([^/\d\-][^/]{1,19})/\s*([A-Z]{2,6})\s*/(\S+)",
    flags=re.IGNORECASE,
)
# Payee names that carry no identity. A slash-format UPI row whose payee is one
# of these gives no evidence of a person, so it is not asserted to be P2P.
UPI_PLACEHOLDER_PAYEES = frozenset({"unknown", "na", "nil", "null", "none"})
# UPI IDs issued only to merchants: PhonePe business QR codes (Q + digits @ybl,
# @ibl, @axl), BharatPe merchant IDs and Paytm QR IDs (paytmqr...). Matched on
# the narration with spaces removed, because statements wrap long IDs
# ("BHARATPE0991555 5394@YESBANKLTD").
MERCHANT_VPA_RE = re.compile(
    r"(?:^|[-/])(?:q\d{6,}@(?:ybl|ibl|axl)|bharatpe[a-z0-9.]*@|paytmqr[a-z0-9]*@)",
    flags=re.IGNORECASE,
)
# "paytm-<digits>@paytm" is treated as a merchant only together with a
# merchant order id ("...-OID202404010903130"); on its own it may be personal.
PAYTM_NUMERIC_VPA_RE = re.compile(r"(?:^|[-/])paytm-?\d+@paytm", flags=re.IGNORECASE)
MERCHANT_ORDER_ID_RE = re.compile(r"\bOID\d{6,}", flags=re.IGNORECASE)


# Google Pay for Business: the "GPAY-<id>" payee prefix and the business-only handles are
# evidence of a merchant. Handles are read from UPI_APP_HANDLES' business list below.
GOOGLE_PAY_BUSINESS_HANDLES = ("okbizaxis", "okpayaxis")
GOOGLE_PAY_BUSINESS_RE = re.compile(
    r"(?:^|[-/])gpay-?\d*[a-z0-9.]*@|@(?:" + "|".join(GOOGLE_PAY_BUSINESS_HANDLES) + r")(?![a-z0-9])",
    flags=re.IGNORECASE,
)
# A wallet top-up ("ADD MONEY TO WALLET") loads the payer's own wallet: a merchant payment.
WALLET_TOPUP_RE = re.compile(
    r"\badd\s*money\s*to\s*wallet\b|\bwallet\s*(?:top\s*up|topup|load|recharge)\b",
    flags=re.IGNORECASE,
)
# Level 5 given to a wallet top-up. The Master Sheet's allowed Level 5 list has no wallet-load
# value, so it stays blank; set it here if the list gains one.
WALLET_TOPUP_LEVEL5 = ""


def has_merchant_vpa(particulars: str) -> bool:
    """Return True when the narration's UPI ID is a merchant-only ID."""

    compact = re.sub(r"\s+", "", str(particulars or ""))
    if MERCHANT_VPA_RE.search(compact) or GOOGLE_PAY_BUSINESS_RE.search(compact):
        return True
    return bool(PAYTM_NUMERIC_VPA_RE.search(compact) and MERCHANT_ORDER_ID_RE.search(compact))
MERCHANT_MARKER_RE = re.compile(
    r"(?:^|[/|])\s*MERCHANT\s*(?:[/|]|$)",
    flags=re.IGNORECASE,
)
# Whole words only, so "cashfree" or "cashback" never match. WDL / WITHDRAWAL /
# EAW / NFS are cash only when the narration is not a charge or fee.
ATM_WITHDRAWAL_RE = re.compile(
    r"\bATM[\s\-/|]*WDL\b|\bATM\s*[/|]\s*DR\s*(?:[/|]|$)|\bATW(?![A-Za-z])|\bNWD(?![A-Za-z])"
    r"|\b(?:WDL|WITHDRAWAL|EAW|NFS)(?![A-Za-z])(?!.*\b(?:CHG|CHGS|CHARGES?|FEES?)\b)",
    flags=re.IGNORECASE,
)
SYSTEM_INITIATED_RE = re.compile(
    r"\b(?:"
    r"E?NACH|ACH|ECS|"
    r"AUTO[\s-]?DEBIT|AUTODEBIT|"
    r"AUTO[\s-]?PAY|AUTOPAY|"
    r"STANDING\s+INSTRUCTION|"
    r"RECURRING\s+PAYMENT|"
    r"UPI\s+MANDATE|"
    r"SI[\s-]?DEBIT|"
    r"MANDATE|"
    r"EMI(?:\s+DEBIT|\s+PAYMENT)?|"
    r"LOAN\s+INSTAL(?:L)?MENT"
    r")\b",
    flags=re.IGNORECASE,
)

EXPLICIT_UPI_RE = re.compile(r"\bUPI\b", flags=re.IGNORECASE)
MANUAL_TRANSFER_RE = re.compile(
    r"\b(?:NEFT|IMPS|RTGS)\b",
    flags=re.IGNORECASE,
)


def _marker_text(particulars: str, part_texts: list[str] | None = None) -> str:
    """Return the narration plus its split parts, joined for marker detection."""

    pieces = [str(particulars or "")]
    if part_texts:
        pieces.append(" | ".join(part.strip() for part in part_texts if part and part.strip()))
    return " | ".join(piece for piece in pieces if piece)


GATEWAY_VPA_FRAGMENTS = frozenset(
    {
        "gpay",
        "google",
        "paytm",
        "phonepe",
        "razorpay",
        "amazon",
        "flipkart",
        "zomato",
        "swiggy",
        "blinkit",
        "zepto",
        "bharatpe",
        "mobikwik",
        "cred",
        "irctc",
        "airtel",
        "jio",
        "netflix",
        "spotify",
        "payu",
        "cashfree",
        "pharmeasy",
    }
)
UPI_P2_MERCHANT_MARKERS = frozenset({"P2M", "P2V"})
UPI_P2_P2P_MARKERS = frozenset({"P2A", "P2P"})
KNOWN_UPI_MERCHANTS = {
    "kettoorg": "ketto",
    "milaap": "milaap",
    "lalpathlabs": "lal path labs",
    "artofliving": "art of living",
    "freshtohome": "fresh to home",
    "urbancompany": "urban company",
    "simpl": "simpl",
    "indigo": "indigo",
    "dlfavenuesaket": "dlf avenue saket",
    "impactguru": "impact guru",
    "lop eragreenpark".replace(" ", ""): "l opera green park",
    "cdhpromenade": "cdh promenade",
    "petmojomangopawtec": "petmojo",
    "delfiniumfertility": "delfinium fertility",
    "ferticityivfclinic": "ferticity ivf clinic",
}
PAYMENT_INTERMEDIARY_FRAGMENTS = frozenset(
    {"gpay", "google", "paytm", "one97", "phonepe", "razorpay", "bharatpe", "mobikwik", "cred", "payu", "cashfree"}
)
KNOWN_MERCHANT_FRAGMENTS = GATEWAY_VPA_FRAGMENTS - PAYMENT_INTERMEDIARY_FRAGMENTS
MERCHANT_DEFAULT_PURPOSES = {
    "amazon": "retail",
    "flipkart": "retail",
    "zomato": "dining",
    "swiggy": "dining",
    "blinkit": "grocery",
    "zepto": "grocery",
    "bigbasket": "grocery",
    "grofers": "grocery",
    "jiomart": "grocery",
    "instamart": "grocery",
    "irctc": "transport",
    "uber": "transport",
    "ola": "transport",
    "rapido": "transport",
    "olacabs": "transport",
    "pharmeasy": "pharmacy",
    "netmeds": "pharmacy",
    "medplus": "pharmacy",
    "apollo": "healthcare",
    "1mg": "pharmacy",
    "netflix": "subscription",
    "spotify": "subscription",
    "hotstar": "subscription",
    "primevideo": "subscription",
    "youtube": "subscription",
    "bookmyshow": "entertainment",
    "myntra": "retail",
    "meesho": "retail",
    "ajio": "retail",
    "nykaa": "retail",
    "makemytrip": "travel",
    "goibibo": "travel",
    "yatra": "travel",
    "cleartrip": "travel",
    "ixigo": "travel",
    "indigo": "travel",
    "spicejet": "travel",
    "airindia": "travel",
    "oyo": "travel",
    "airbnb": "travel",
    "groww": "financial services",
    "zerodha": "financial services",
    "upstox": "financial services",
    "kuvera": "financial services",
    "smallcase": "financial services",
    "paytmmoney": "financial services",
    "mirae": "financial services",
}
HYPHEN_UPI_RE = re.compile(r"^\s*UPI\s*-", flags=re.IGNORECASE)
# The narration itself starts with the UPI structure ("UPI/..." or "UPI-..."),
# optionally after the bank's "TO TRANSFER-" / "BY TRANSFER-" prefix. A "UPI" word
# later in the text does not count.
UPI_START_RE = re.compile(r"^\s*(?:(?:TO|BY)\s+TRANSFER\s*-\s*)?UPI\s*[/-]", flags=re.IGNORECASE)
BANK_CODE_TO_NAME = {
    "sbin": "state bank of india",
    "hdfc": "hdfc bank",
    "icici": "icici bank",
    "utib": "axis bank",
    "yesb": "yes bank",
    "cnrb": "canara bank",
    "punb": "punjab national bank",
    "idfc": "idfc first bank",
    "kotak": "kotak mahindra bank",
    "bob": "bank of baroda",
    "boi": "bank of india",
    "union": "union bank of india",
    "paytm": "paytm",
}
BANK_NAMES = frozenset(BANK_CODE_TO_NAME.values())
IFSC_PREFIX_TO_BANK = {**BANK_CODE_TO_NAME, "icic": "icici bank"}
IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

PLATFORM_ALIASES = {
    "google pay": ("google pay", "googlepay", "gpay", "tez", "okbizaxis", "okpayaxis"),
    "phonepe": ("phonepe", "phone pe"),
    "jupiter": ("jupiteraxis",),
    "paytm": ("paytm", "paytm upi", "paytm payments bank", "one97", "one97 communications"),
    "amazon pay": ("amazon pay", "amazonpay"),
    "bhim": ("bhim", "bharat interface for money"),
    "bharatpe": ("bharatpe", "bharat pe"),
    "mobikwik": ("mobikwik", "mobi kwik"),
    "cred": ("cred pay", "credpay", "cred"),
    "whatsapp pay": ("whatsapp pay", "whatsapp payments"),
    "jiofinance": ("jiofinance", "jio finance"),
    # "rz p" / "r zp" / "p ayu" / "pa yu" are gateway codes split by statement line wrapping.
    "razorpay": ("razorpay", "razor pay", "rzp", "rz p", "r zp"),
    "getepay": ("getepay", "get epay"),
    "cashfree": ("cashfree", "cash free"),
    "payu": ("payu", "pay u", "p ayu", "pa yu"),
    "billdesk": ("billdesk", "bill desk"),
    "juspay": ("juspay", "jus pay"),
    "pine labs": ("pine labs", "pinelabs"),
    "payzapp": ("payzapp", "pay zapp"),
    "freecharge": ("freecharge", "free charge"),
    "mobikwik zip": ("mobikwik zip",),
    "mswipe": ("mswipe", "m swipe"),
}

EMBEDDED_PLATFORM_ALIASES = {
    "razorpay": "razorpay",
    "getepay": "getepay",
    "cashfree": "cashfree",
    "billdesk": "billdesk",
    "phonepe": "phonepe",
    "bharatpe": "bharatpe",
    "mobikwik": "mobikwik",
    "juspay": "juspay",
    "freecharge": "freecharge",
    "payzapp": "payzapp",
    "mswipe": "mswipe",
    "googlepay": "google pay",
    "amazonpay": "amazon pay",
    "pinelabs": "pine labs",
    "paytm": "paytm",
}

UPI_APP_HANDLES = {
    "okicici": "google pay",
    "okhdfcbank": "google pay",
    "okhdfc": "google pay",
    "oksbi": "google pay",
    "okaxis": "google pay",
    "ybl": "phonepe",
    "ibl": "phonepe",
    "axl": "phonepe",
    # Paytm handles name the platform, not the bank behind them.
    "ptyes": "paytm",
    "ptys": "paytm",
    "ptybl": "paytm",
    "ptsbi": "paytm",
    "ptaxis": "paytm",
    "pty": "paytm",
    "paytm": "paytm",
    "slc": "slice",
    "okbizaxis": "google pay",
    "okpayaxis": "google pay",
    "jupiteraxis": "jupiter",
    # Google Pay handles that statements cut mid-word ("@OKH", "@OKHDF", "@OKI", "@OKS").
    "okh": "google pay",
    "okhdf": "google pay",
    "oki": "google pay",
    "oks": "google pay",
}
# Every handle is matched with spaces allowed between its letters ("@JUPITERA XIS",
# "@OKBIZAX IS", "@OKHDF CBANK"), i.e. as if the spaces were removed.
UPI_SPACE_TOLERANT_HANDLES = frozenset(UPI_APP_HANDLES)
# Platforms known from a UPI handle (slice has no narration alias of its own).
UPI_HANDLE_PLATFORMS = frozenset(UPI_APP_HANDLES.values())
UPI_APP_HANDLE_RE = re.compile(
    r"@\s*("
    + "|".join(
        r"\s*".join(re.escape(char) for char in handle) if handle in UPI_SPACE_TOLERANT_HANDLES else re.escape(handle)
        for handle in sorted(UPI_APP_HANDLES, key=len, reverse=True)
    )
    + r")(?![a-z0-9])"
)

BANK_ALIASES = {
    "state bank of india": ("state bank of india", "sbi", "sbin"),
    "hdfc bank": ("hdfc bank", "hdfc", "hdf"),
    "icici bank": ("icici bank", "icici", "icic", "ici"),
    "axis bank": ("axis bank", "axis", "utib", "axi"),
    "kotak mahindra bank": ("kotak mahindra bank", "kotak bank", "kotak", "kkbk"),
    "punjab national bank": ("punjab national bank", "pnb", "punb"),
    "bank of baroda": ("bank of baroda", "bob", "barb"),
    "bank of india": ("bank of india", "boi"),
    # "cab" is the cheque-clearing code ("CLG/.../CAB/<date>"). A taxi narration
    # ("MERU CAB") also carries a longer bank code, which wins on alias length.
    "canara bank": ("canara bank", "canara", "cnrb", "cab"),
    "union bank of india": ("union bank of india", "union bank", "ubin"),
    "indian bank": ("indian bank", "idib"),
    "indian overseas bank": ("indian overseas bank", "iob", "ioba"),
    "central bank of india": ("central bank of india", "cbi", "cbin"),
    "uco bank": ("uco bank", "uco"),
    "bank of maharashtra": ("bank of maharashtra", "mahabank", "mahb"),
    "punjab and sind bank": ("punjab and sind bank", "psb", "psib"),
    "yes bank": ("yes bank", "yesb"),
    "idfc first bank": ("idfc first bank", "idfc bank", "idfc", "idfb", "idf"),
    "indusind bank": ("indusind bank", "indusind", "inbk", "indb"),
    "federal bank": ("federal bank", "federal", "fdr", "fdrl"),
    "bandhan bank": ("bandhan bank", "bandhan", "bdbl"),
    "rbl bank": ("rbl bank", "rbl", "ratn"),
    "south indian bank": ("south indian bank", "sib", "sibl"),
    "karnataka bank": ("karnataka bank", "ktkb", "karb"),
    "city union bank": ("city union bank", "ciub", "kcu"),
    "karur vysya bank": ("karur vysya bank", "kvb", "kvbl"),
    "dcb bank": ("dcb bank", "dcb"),
    "tamilnad mercantile bank": ("tamilnad mercantile bank", "tmb", "tmbl"),
    "au small finance bank": ("au small finance bank", "au bank", "aubl"),
    "idbi bank": ("idbi bank", "idbi", "ibkl", "idb"),
    "cosmos co-operative bank": ("cosmos co-operative bank", "cosmos bank", "cosb"),
    "equitas small finance bank": ("equitas small finance bank", "equitas", "esfb"),
    "ujjivan small finance bank": ("ujjivan small finance bank", "ujjivan", "usfb"),
    "jana small finance bank": ("jana small finance bank", "jana", "jsfb"),
    "paytm payments bank": ("paytm payments bank", "paytm bank"),
    "india post payments bank": ("india post payments bank", "ippb"),
    "fino payments bank": ("fino payments bank", "fino"),
    "airtel payments bank": ("airtel payments bank", "airtel bank", "airp"),
    "dbs bank": ("dbs bank", "dbsbank", "dbs", "dbss"),
    "maharashtra gramin bank": ("maharashtra gramin bank", "mahg"),
    "esaf small finance bank": ("esaf small finance bank", "esaf", "esmf", "esb"),
    "unity small finance bank": ("unity small finance bank",),
}


def _contains_alias(text: str, alias: str) -> bool:
    """Match an alias as a phrase/token, not as part of a longer word."""
    escaped = re.escape(alias.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text))


def match_bank_name(segment: str) -> str:
    """Return the full bank name a narration segment spells out, or "".

    Two rules, both against the full bank names (keys of ``BANK_ALIASES``):

    - Fused: the letters of the segment (spaces and punctuation dropped, a
      trailing "ltd" / "limited" removed) equal a full bank name written
      without spaces, and there are 8 letters or more
      ("STATEBANKOFINDIA", "AirtelPaymentsBank", "KARURVYSYABANKLTD").
    - Truncated: a segment of at least 2 words and 6 letters that is a prefix
      of exactly one full bank name ("BANK OF B", "STATE BAN", "PUNJAB NA").
      A prefix shared by several banks ("BANK OF") gives nothing.

    A segment containing a digit is a reference number, never a bank name.
    """

    text = str(segment or "").strip().lower()
    if not text or re.search(r"\d", text):
        return ""
    words = re.findall(r"[a-z]+", text)
    if not words:
        return ""

    letters = "".join(words)
    for suffix in ("limited", "ltd"):
        if letters.endswith(suffix) and len(letters) > len(suffix):
            letters = letters[: -len(suffix)]
            break
    if len(letters) >= 8:
        for name in BANK_ALIASES:
            if letters == re.sub(r"[^a-z]", "", name):
                return name

    spaced = " ".join(words)
    if len(words) >= 2 and len("".join(words)) >= 6:
        hits = [name for name in BANK_ALIASES if " ".join(re.findall(r"[a-z]+", name)).startswith(spaced)]
        if len(hits) == 1:
            return hits[0]
    return ""


def detect_level_4_platform_or_bank(text: str) -> str:
    """Return platform first; if no platform is found, return the bank."""
    normalized = str(text or "").lower()
    if not normalized.strip():
        return ""

    platform_matches = []
    for canonical, aliases in PLATFORM_ALIASES.items():
        for alias in aliases:
            if _contains_alias(normalized, alias):
                platform_matches.append((len(alias), canonical))
    if platform_matches:
        platform_matches.sort(reverse=True)
        return platform_matches[0][1]

    for token in re.findall(r"[a-z0-9]+", normalized):
        for alias, canonical in EMBEDDED_PLATFORM_ALIASES.items():
            if alias in token:
                platform_matches.append((len(alias), canonical))
    if platform_matches:
        platform_matches.sort(reverse=True)
        return platform_matches[0][1]

    handle_match = UPI_APP_HANDLE_RE.search(normalized)
    if handle_match:
        return UPI_APP_HANDLES[re.sub(r"\s+", "", handle_match.group(1))]

    bank_matches = []
    for canonical, aliases in BANK_ALIASES.items():
        for alias in aliases:
            if _contains_alias(normalized, alias):
                bank_matches.append((len(alias), canonical))
    if bank_matches:
        bank_matches.sort(reverse=True)
        return bank_matches[0][1]

    # Last resort, after no platform and no bank alias matched: a bank name that
    # is fused ("AirtelPaymentsBank") or cut short ("BANK OF B") in one segment.
    for segment in re.split(r"[/|\-:]", str(text or "")):
        bank = match_bank_name(segment)
        if bank:
            return bank

    return ""


def _is_platform_label(label: str) -> bool:
    """Return True for a Level 4 label that names a platform or gateway, not a bank.

    Unlike ``is_payment_intermediary_label`` this counts every ``PLATFORM_ALIASES``
    name (paytm, billdesk, getepay ...), because for Level 4 the platform always
    wins over the bank.
    """

    value = _normalize_compare(label)
    return bool(value) and (
        value in PLATFORM_ALIASES
        or value in UPI_HANDLE_PLATFORMS
        or is_payment_intermediary_label(value)
    )


NARRATION_LEVEL5_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:credit\s*card|cc\s*bill|ccbp|ccpmt|cc\s*payment|card\s*payment"
            r"|hdfc\s*cc|icici\s*cc|axis\s*cc|sbi\s*cc|kotak\s*cc)\b",
            re.IGNORECASE,
        ),
        "credit card",
    ),
    (
        re.compile(
            r"\b(?:emi|loan\s*repay|loan\s*emi|home\s*loan|personal\s*loan"
            r"|car\s*loan|auto\s*loan|education\s*loan|loan\s*instal+ment"
            r"|equated\s*monthly|loan\s*recovery|ecs\s*emi)\b",
            re.IGNORECASE,
        ),
        "loan repayment",
    ),
    (
        re.compile(
            r"\b(?:insurance|insur|policy\s*premium|premium\s*payment"
            r"|lic\s*premium|\blic\b|acko|ergo|bupa|niva\s*bupa|star\s*health"
            r"|icici\s*lombard|hdfc\s*ergo|bajaj\s*allianz|new\s*india\s*assurance)\b",
            re.IGNORECASE,
        ),
        "insurance",
    ),
    (
        re.compile(
            r"\b(?:salary|payroll|wages|\bsal\b|payslip)\b",
            re.IGNORECASE,
        ),
        "salary",
    ),
    (
        re.compile(
            r"\b(?:income\s*tax|advance\s*tax|tds\s*payment|gst\s*payment"
            r"|itdtax|stax|tax\s*payment)\b",
            re.IGNORECASE,
        ),
        "tax",
    ),
    (
        re.compile(
            r"\b(?:topup|top\s*up|prepaid\s*recharge|mobile\s*recharge"
            r"|sim\s*recharge)\b"
            r"|\brecharge\b(?!.*\b(?:dth|dish\s*tv|tata\s*sky|d2h|videocon\s*d2h)\b)",
            re.IGNORECASE,
        ),
        "mobile recharge",
    ),
    (
        re.compile(
            r"\b(?:electricity|electric\s*bill|water\s*bill|gas\s*bill|lpg"
            r"|broadband|internet\s*bill|wifi\s*bill|dth\s*(?:bill|recharge)|utility"
            r"|piped\s*gas|png\s*bill|mseb|bescom|bses|torrent\s*power"
            r"|adani\s*gas|mahanagar\s*gas|indraprastha\s*gas)\b",
            re.IGNORECASE,
        ),
        "utilities",
    ),
    (
        re.compile(
            r"\b(?:pharmacy|pharma|medicos?|medicals?|medi\s*store|chemist|drugstore"
            r"|pharmeasy|medplus|netmeds|1mg|apollopharmacy|lenskar"
            r"|ayurveda\s*kendra|ayurved|homeopath|ayurvedic)\b",
            re.IGNORECASE,
        ),
        "pharmacy",
    ),
    (
        re.compile(
            r"\b(?:hospital|clinic|medicare|healthcare|diagnostic"
            r"|pathlab|path\s*lab|ivf|fertility|nursing\s*home|medical\s*centre"
            r"|medical\s*center|dental|dentist|optician|optical|eye\s*care"
            r"|ags\s*medicare|ms\s*son\s*x\s*medical)\b",
            re.IGNORECASE,
        ),
        "healthcare",
    ),
    (
        re.compile(
            r"\b(?:grocery|groceries|blinkit|zepto|bigbasket|grofers"
            r"|jiomart|instamart|swiggy\s*instamart|supermarket|super\s*mark(?:et)?|marts?|kirana|fresh\s*mart|sabji"
            r"|vegetable|fruit\s*shop|daily\s*needs|ever\s*green\s*fresh"
            r"|milk\s*booth|dairy|sorabh\s*dairy)\b",
            re.IGNORECASE,
        ),
        "grocery",
    ),
    (
        re.compile(
            r"\b(?:restaurant|zomato|swiggy|cafe|dhaba|food\s*court|foods?|kachori|coffee|brew|mithai"
            r"|pizza|burger|kfc|mcdonalds|dominos?|biryani|diner"
            r"|bakery|confectionery|sweet\s*shop|juice\s*bar|tea\s*stall"
            r"|mess|canteen|tiffin|bar\s*and\s*grill"
            r"|jubilant\s*food|saravana\s*bhava|dyu\s*art|frozen\s*bottle"
            r"|atithi\s*dhaba|peshawar\s*restaurant|burger\s*king)\b"
            r"|\bhotel\b(?!\s*(?:booking|stay|accommodation|reservation|check))",
            re.IGNORECASE,
        ),
        "dining",
    ),
    (
        re.compile(
            r"\b(?:petrol|diesel|fuels?|filling|pumps?|hpcl|iocl|bpcl|hp\s*petrol"
            r"|cng\s*station|gas\s*station|shell|petrol\s*pump"
            r"|fuel\s*station|bharat\s*petroleum|indian\s*oil)\b"
            # fused compound name: "NELLOREPOLICEFUELST" (fuel + station suffix)
            r"|[a-z]fuel(?:st|stn|stat|station|pump)",
            re.IGNORECASE,
        ),
        "fuel",
    ),
    (
        re.compile(
            r"\b(?:makemytrip|mmt|goibibo|yatra|cleartrip|ixigo"
            r"|spicejet|air\s*india|vistara|akasa|go\s*first|air\s*asia"
            r"|hotel\s*booking|hotel\s*stay|oyo\s*room|airbnb"
            r"|flight\s*ticket|airline\s*ticket|holiday\s*package|tour\s*package)\b",
            re.IGNORECASE,
        ),
        "travel",
    ),
    (
        re.compile(
            r"\b(?:uber|olacabs|ola\s*cabs?|rapido|irctc|indian\s*rail"
            r"|metro\s*card|metro\s*rail|city\s*bus|pmpml|msrtc|dtc\s*bus"
            r"|cab\s*service|taxi|airport\s*transfer|wti\s*cabs?)\b",
            re.IGNORECASE,
        ),
        "transport",
    ),
    (
        re.compile(
            r"\b(?:school\s*fee|college\s*fee|tuition|tuition\s*fee"
            r"|university\s*fee|course\s*fee|coaching|byju|unacademy|vedantu"
            r"|education\s*fee|exam\s*fee|admission\s*fee)\b",
            re.IGNORECASE,
        ),
        "education",
    ),
    (
        re.compile(
            r"\b(?:groww|zerodha|upstox|kuvera|smallcase|slice|paytm\s*money"
            r"|mutual\s*fund|sip\s*payment|sip\s*debit|nps\s*contribution"
            r"|demat|ppf\s*deposit|fd\s*deposit|rd\s*deposit|investment)\b",
            re.IGNORECASE,
        ),
        "financial services",
    ),
    (
        re.compile(
            r"\b(?:bookmyshow|pvr|inox|cinepolis|paytm\s*insider|district)\b",
            re.IGNORECASE,
        ),
        "entertainment",
    ),
    (
        re.compile(
            r"\b(?:netflix|spotify|hotstar|primevideo|youtube\s*premium"
            r"|jiocinema|zee5|sonyliv|voot|mxplayer"
            r"|subscription|annual\s*plan)\b",
            re.IGNORECASE,
        ),
        "subscription",
    ),
    (
        re.compile(
            r"\b(?:amazon|flipkart|myntra|meesho|ajio|nykaa|snapdeal"
            r"|reliance\s*digital|croma|vijay\s*sales|decathlon"
            r"|shoppers\s*stop|lifestyle|westside|max\s*fashion"
            r"|big\s*bazar|dmart|ansari\s*clothes)\b",
            re.IGNORECASE,
        ),
        "retail",
    ),
    (
        # Real digital brands only; a payment platform never sets Level 5.
        re.compile(r"(?<![a-z])(?:" + brand_alternation(REAL_DIGITAL_BRANDS) + r")(?![a-z])", re.IGNORECASE),
        "digital services",
    ),
]


# IDFC FIRST with a LOAN ACC / LOAN A/C reference is a loan repayment, in either order.
# Name evidence strong enough to make a row a merchant even when the model said p2p:
# a fuel-station compound ("NELLOREPOLICEFUELST"), "fuel station", "petrol pump".
P2P_OVERRIDING_EVIDENCE: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"[a-z]fuel(?:st|stn|stat|station|pump)|(?<![a-z])fuel\s*(?:station|pump)(?![a-z])"
            r"|(?<![a-z])petrol\s*pump(?![a-z])",
            flags=re.IGNORECASE,
        ),
        "fuel",
    ),
)
JIO_EXACT_PARTS = frozenset({"jio", "jiop"})
IDFC_LENDER_NAME = "idfc first bank"
IDFC_LOAN_ACC_RE = re.compile(
    r"(?=.*\bidfc\b)(?=.*\bloan\s*a(?:/?c|cc(?:ount)?)\b)", flags=re.IGNORECASE
)
# Jio as a merchant is a mobile recharge (like Airtel). Jio Payments Bank, Jio Finance,
# JioMart, JioCinema, JioSaavn and Jio Fiber are other things, so they do not count.
JIO_MERCHANT_RE = re.compile(
    r"(?<![a-z])jio(?![a-z])(?!\s*(?:payments?\s*bank|finance|financial|mart|cinema|saavn|fib(?:er|re)))",
    flags=re.IGNORECASE,
)


REAL_DIGITAL_BRAND_RE = re.compile(
    r"(?<![a-z])(?:" + brand_alternation(REAL_DIGITAL_BRANDS) + r")(?![a-z])", flags=re.IGNORECASE
)


# When several purpose patterns match one text, the more specific purpose wins over the
# generic one ("SUPRA FUEL MART" is fuel, not grocery). Data, not logic: extend per purpose.
PURPOSE_SPECIFICITY: dict[str, frozenset[str]] = {
    "fuel": frozenset({"grocery", "retail", "services"}),
    "pharmacy": frozenset({"retail", "services"}),
    "healthcare": frozenset({"services"}),
}
# Purposes that must never be inferred from a merchant name alone.
NAME_EVIDENCE_EXCLUDED_PURPOSES = frozenset({"salary", "credit card", "digital services"})


def _platform_name_regex() -> re.Pattern:
    """Every platform / gateway alias from PLATFORM_ALIASES as one token-bounded regex, so
    the platform list lives in one place."""

    aliases = {a for names in PLATFORM_ALIASES.values() for a in names} | set(PLATFORM_ALIASES)
    body = "|".join(
        r"\s*".join(re.escape(ch) for ch in alias.replace(" ", ""))
        for alias in sorted(aliases, key=len, reverse=True)
    )
    return re.compile(r"(?<![a-z])(?:" + body + r")(?![a-z])", re.IGNORECASE)


PLATFORM_NAME_RE = _platform_name_regex()


def _matched_purposes(text: str, skip: frozenset[str] = frozenset()) -> list[str]:
    """Every purpose in NARRATION_LEVEL5_PATTERNS whose pattern matches the text, in order."""

    found: list[str] = []
    for pattern, purpose in NARRATION_LEVEL5_PATTERNS:
        if purpose not in skip and purpose not in found and pattern.search(text):
            found.append(purpose)
    return found


def _most_specific_purpose(purposes: list[str]) -> str:
    for purpose in purposes:
        if any(purpose in PURPOSE_SPECIFICITY.get(other, ()) for other in purposes):
            continue
        return purpose
    return purposes[0] if purposes else ""


def apply_platform_purpose_guard(
    particulars: str, result: dict[str, str], part_texts: list[str] | None = None
) -> dict[str, str]:
    """A payment platform (Paytm, BharatPe, Amazon Pay, Google Pay, PhonePe...) never sets
    Level 5. `digital services` stays only for a real digital brand in the narration or
    the Level 6 name; otherwise it is cleared so evidence-only logic can fill it."""

    if result.get("Level_5") != "digital services":
        return result
    evidence = f"{_marker_text(particulars, part_texts)} {result.get('Level_6', '')}"
    if not REAL_DIGITAL_BRAND_RE.search(evidence):
        result["Level_5"] = ""
    return result


def apply_narration_purpose_override(
    particulars: str,
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
    part_texts: list[str] | None = None,
) -> dict[str, str]:
    """Fill a blank Level 5 purpose from the full narration when keyword model left it empty."""

    text = _marker_text(particulars, part_texts)
    direction = derive_direction_from_amounts(debit_value, credit_value)

    # IDFC FIRST loan account: the narration states the purpose, so it replaces a
    # model purpose too. A credit (disbursal, refund) is not a repayment.
    if (
        result.get("Level_1") != "cash"
        and direction != "credit"
        and IDFC_LOAN_ACC_RE.search(text)
    ):
        result["Level_5"] = "loan repayment"
        if result.get("Level_3") in {"", "p2p"}:
            result["Level_3"] = "institution"
        # Level 6 is the full lender name, never a cut piece of it ("idfc firs").
        current = re.sub(r"[^a-z]", "", result.get("Level_6", "").lower())
        lender = re.sub(r"[^a-z]", "", IDFC_LENDER_NAME)
        if not current or (len(current) >= 4 and lender.startswith(current)):
            result["Level_6"] = IDFC_LENDER_NAME
        return result

    # A narration part that is exactly "jio" is the Jio merchant row of the Master Sheet;
    # a p2p label on it is a mislabel (mobile recharge, like Airtel).
    if (
        result.get("Level_3") == "p2p"
        and result.get("Level_5") in {"", "p2p transfer"}
        and any(_normalize_compare(part) in JIO_EXACT_PARTS for part in (part_texts or []))
    ):
        result["Level_3"] = "merchant"
        result["Level_5"] = "mobile recharge"
        return result

    # Unambiguous business-name evidence beats the default "p2p transfer" label.
    if (
        result.get("Level_1") != "cash"
        and direction != "credit"
        and result.get("Level_3") in {"", "p2p"}
        and result.get("Level_5") in {"", "p2p transfer"}
    ):
        for pattern, purpose in P2P_OVERRIDING_EVIDENCE:
            if pattern.search(text):
                result["Level_3"] = "merchant"
                result["Level_5"] = purpose
                return result

    if result.get("Level_5"):
        return result

    if result.get("Level_1") == "cash":
        return result

    if result.get("Level_3") == "merchant" and JIO_MERCHANT_RE.search(text):
        result["Level_5"] = "mobile recharge"
        return result

    skip = set()
    if direction == "debit":
        skip.add("salary")
    if direction == "credit":
        skip.add("credit card")
    # Platform names are rails, not evidence of what was bought: read the rest of the text.
    purpose = _most_specific_purpose(
        _matched_purposes(PLATFORM_NAME_RE.sub(" ", text), frozenset(skip))
    )
    if purpose:
        result["Level_5"] = purpose

    # A merchant's own name is evidence too (KACHORI, COFFEE, SUPER MARK, FILLING...),
    # read with the same whole-token patterns, so the result does not depend on the app.
    if not result.get("Level_5") and result.get("Level_3") == "merchant" and result.get("Level_6"):
        purpose = _most_specific_purpose(
            _matched_purposes(
                PLATFORM_NAME_RE.sub(" ", str(result["Level_6"])), NAME_EVIDENCE_EXCLUDED_PURPOSES
            )
        )
        if purpose:
            result["Level_5"] = purpose

    return result


def parse_amount(value: object) -> float:
    """Convert a debit or credit cell into a positive float."""

    if pd.isna(value):
        return 0.0

    cleaned = re.sub(r"[^0-9.\-]", "", str(value).replace(",", ""))
    try:
        parsed = float(cleaned)
    except ValueError:
        return 0.0
    return parsed if parsed > 0 else 0.0


def _empty_levels() -> dict[str, str]:
    """Return a blank row-level prediction dictionary."""

    return {f"Level_{index}": "" for index in range(8)}


def _normalize_compare(value: str) -> str:
    """Normalize a label for duplicate checks."""

    return "" if value is None else str(value).strip().lower()


# Channel words that may precede the rail ("INET-IMPS-CR-<ref>-<NAME>"): not an entity.
HYPHEN_RAIL_CHANNEL_PREFIXES = ("INET",)
# Rail prefixes that banks glue onto the rail name: word -> (Level 1, layout of the rest).
# "slash" = PREFIX/<ref>/<name or IFSC>/<account>; "hyphen" = PREFIX-<NAME> <ref> UTR ...
# Sure ones only; add a row here to teach a new bank layout (it also feeds the Level 6 stop list).
LEVEL1_PREFIX_WORDS: dict[str, tuple[str, str]] = {
    "impsar": ("imps", "slash"),
    "impsab": ("imps", "slash"),
    "upiab": ("upi", "slash"),
    "rtgso": ("rtgs", "hyphen"),
    "nefto": ("neft", "hyphen"),
}
_LEAD = r"^(?:\d{6,}\s*[-/ ]\s*)?"
LEVEL1_STRUCTURAL_RULES: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), value)
    for pattern, value in (
        (_LEAD + r"(?:POS|ME\s+DC\s+SI)\b", "card"),
        (_LEAD + r"ACH\s*[DC]\s*-", "nach"),
        (_LEAD + r"IB\s+BILLPAY\b", "bbps"),
        (_LEAD + r"IB\b.*\bFUNDS\s+TRANSFER\b", "internal transfer"),
        (_LEAD + r"TPT(?:\s|-|$)", "internal transfer"),
    )
) + tuple(
    # Rail prefixes from LEVEL1_PREFIX_WORDS (IMPSAR, UPIAB, RTGSO, NEFTO ...).
    (re.compile(_LEAD + re.escape(word) + r"(?=[/\-\s:])", re.IGNORECASE), level_1)
    for word, (level_1, _) in LEVEL1_PREFIX_WORDS.items()
)
# Channel / product prefixes that never name a payee, whether or not Level 1 is decided for them.
CHANNEL_PREFIX_WORDS = frozenset(
    {"mobft", "mobcc", "etxn", "imps", "neft", "rtgs", "upi", "nach", "atm", "trf"}
)
# Labels a bank prints instead of a payee: phrase -> (Level 3, Level 5, direction) it implies.
# An empty Level 3 / Level 5 means the phrase is only a label (never a payee). Leading phrases
# are stripped from a name ("trf to X" -> "x"); a value made only of them is not a payee.
# direction "debit" = the implied labels apply to debits only (a credit is not a repayment).
# Another bank's loan or charge phrase only needs a row here.
LABEL_PHRASES: dict[str, tuple[str, str, str]] = {
    "loan recovery for": ("institution", "loan repayment", "debit"),
    "loan account payments for": ("institution", "loan repayment", "debit"),
    "charges for": ("institution", "charges", "any"),
    "general charges recovery": ("institution", "charges", "any"),
    "general charges": ("institution", "charges", "any"),
    "locker rent": ("institution", "charges", "any"),
    "locker": ("institution", "charges", "any"),
    "vetting fees": ("institution", "charges", "any"),
    "credit card payment": ("", "", ""),
    "self": ("", "", ""),
    "trf to": ("", "", ""),
    "trf frm": ("", "", ""),
    "trf from": ("", "", ""),
    "mobft from": ("", "", ""),
    "mobft to": ("", "", ""),
    "auto debit for": ("", "", ""),
}


def _rule_prefix_words(pattern: re.Pattern) -> set[str]:
    """Prefix words a Level 1 structural rule is keyed on ("ACH\\s*[DC]\\s*-" gives ach)."""

    source = pattern.pattern
    if source.startswith(_LEAD):
        source = source[len(_LEAD):]
    group = re.match(r"\(\?:([^()]*)\)", source)
    head = group.group(1) if group else re.split(r"\\b|\(\?|\\s\*\[|\[|-|\.|\$", source, maxsplit=1)[0]
    return {w.lower() for w in re.findall(r"[A-Za-z]{2,}", head.replace("\\s", " "))}


# Every prefix of every Level 1 structural rule is a stop word: no manual list to keep in step.
LEVEL6_STOP_WORDS = frozenset(
    {w for pattern, _ in LEVEL1_STRUCTURAL_RULES for w in _rule_prefix_words(pattern)}
    | CHANNEL_PREFIX_WORDS
    | {w for ph in LABEL_PHRASES for w in ph.split()}
)
# Prefix words stripped from the front of a name (statement-specific ones only; the standard
# channel words stay in the stop list but are never cut out of a longer value).
STRIP_PREFIX_WORDS = frozenset(LEVEL1_PREFIX_WORDS) | {"mobft", "mobcc", "etxn"}
_LABEL_PHRASES_LONGEST_FIRST = tuple(
    ph.split() for ph in sorted(LABEL_PHRASES, key=lambda x: -len(x.split())) if len(ph.split()) > 1
)


def strip_leading_labels(value: str) -> str:
    """Drop leading label phrases and prefix words from a name: "trf to pma" -> "pma"."""

    tokens = re.findall(r"[a-z0-9&.\-]+", str(value or "").lower())
    changed = True
    while changed and tokens:
        changed = False
        for phrase in _LABEL_PHRASES_LONGEST_FIRST:
            if tokens[: len(phrase)] == phrase:
                tokens = tokens[len(phrase):]
                changed = True
                break
        else:
            if len(tokens) > 1 and tokens[0] in STRIP_PREFIX_WORDS:
                tokens = tokens[1:]
                changed = True
    return " ".join(tokens)


def is_stop_value(value: str) -> bool:
    """True when a Level 6 candidate is empty, or only rail words, labels and reference numbers."""

    tokens = [t for t in re.findall(r"[a-z0-9]+", str(value or "").lower()) if not t.isdigit()]
    return not tokens or all(t in LEVEL6_STOP_WORDS for t in tokens)


def guard_level6(value: str, particulars: str = "") -> str:
    """Return a clean payee, or "" when the candidate is a rail word, label or reference."""

    candidate = strip_leading_labels(value)
    if is_stop_value(candidate):
        return ""
    first = re.match(r"\s*([a-z0-9]+)", str(particulars or "").lower())
    if first and candidate == first.group(1):
        return ""
    return candidate


GENERIC_LEVEL6_WORDS = frozenset(
    {
        "to", "by", "from", "for", "via", "bill", "bills", "payment", "payments",
        "pay", "payme", "transfer", "transfers", "trf", "neft", "imps", "rtgs",
        "upi", "atm", "wdl", "withdrawal", "cash", "merchant", "dr", "cr", "bbps",
        "ach", "nach", "ecs", "clearing", "deposit", "chq", "cheque", "issued",
        "recvd", "clrd", "bouncd", "pdc", "intl", "ref", "credit", "debit", "card",
        "recharge", "mobile", "premium", "insurance", "electricity", "broadband",
        "dth", "online", "txn", "transaction",
        # Bank channel / product codes that prefix narrations (not entities).
        "inf", "inft", "bil", "ift", "inb", "mmt", "clg", "trfr", "cms", "gib",
        "onl", "impstxn", "emi", "loan", "return", "returned", "rtn", "chg",
        "charges", "gst",
        # Card purchase prefix "ECOM PUR/<merchant>/...".
        "ecom", "pur",
        # Internet-banking channel prefix ("INET-IMPS-CR-...").
        *(word.lower() for word in HYPHEN_RAIL_CHANNEL_PREFIXES),
    }
)
# Counterparty position in common bank narration layouts (used for Level 6).
#   NEFT-<UTR>-<NAME>-...          RTGS RETURN-<UTR>-<NAME>
#   CLG|TRF/<DRAWER>/<chq no>/...  cheque instruments
#   INF|BIL/INFT/<ref>/.../<NAME>  in-bank net-banking transfers (name is last)
#   MMT/IMPS/<rrn>/<remark>/<NAME>/<bank>
#   TRFR TO|FROM: <NAME>           in-bank transfers
# "INET-IMPS-CR/<NAME>/<bank>/<ref>...": the payee is the first slash segment after the rail.
INET_SLASH_ENTITY_RE = re.compile(
    r"^\s*(?:" + "|".join(HYPHEN_RAIL_CHANNEL_PREFIXES) + r")\s*-\s*(?:NEFT|RTGS|IMPS)\s*-\s*(?:CR|DR)\s*/([^/]+)/",
    flags=re.IGNORECASE,
)
HYPHEN_RAIL_ENTITY_RE = re.compile(
    r"^\s*(?:(?:" + "|".join(HYPHEN_RAIL_CHANNEL_PREFIXES) + r")-)?(?:NEFT|RTGS|IMPS)(?:\s+RETURN)?(?:-(?:CR|DR))?"
    r"-[A-Z0-9]*\d{5,}[A-Z0-9]*-([^-]+)",
    flags=re.IGNORECASE,
)
CHEQUE_ENTITY_RE = re.compile(r"^\s*(?:CLG|TRF)/([^/]+)/", flags=re.IGNORECASE)
INFT_ENTITY_RE = re.compile(r"^\s*(?:INF|BIL)/INFT/[^/]*\d[^/]*/(.*)$", flags=re.IGNORECASE)
MMT_IMPS_ENTITY_RE = re.compile(r"^\s*MMT/IMPS/\d{6,}/[^/]*/([^/]+)", flags=re.IGNORECASE)
INTERNAL_TRANSFER_ENTITY_RE = re.compile(r"^\s*TRFR\s+(?:TO|FROM)\s*:\s*(.+)$", flags=re.IGNORECASE)
#   ECOM PUR/<MERCHANT>/<pincode>/...   card purchases (merchant name is first)
ECOM_PUR_ENTITY_RE = re.compile(r"^\s*ECOM\s+PUR/([^/]+)/", flags=re.IGNORECASE)
SLASH_ENTITY_RAIL_RE = re.compile(r"\b(UPI|NEFT|IMPS|RTGS|ATM)/(.*)", flags=re.IGNORECASE)
# The biller code is the name only: digits inside it are kept ("D2HME2025031234"
# gives "D2HME"), but the date/reference digits that follow it are not
# ("LIC2025032389" gives "LIC", "MAHANAGAR GAS2025031201" gives "MAHANAGAR GAS").
BBPS_BILLER_RE = re.compile(r"\bBBPS/(?:DR|CR)/([A-Z](?:[A-Z0-9 ]*[A-Z])?)\s*\d+", flags=re.IGNORECASE)
BBPS_BILLER_NAMES = {"jiofiber": "jio fiber", "tatasky": "tata sky"}
# Aliases for well-known BBPS biller codes that appear verbatim in narrations.
BBPS_BILLER_ALIASES: dict[str, str] = {
    "d2hme": "d2h",
    "maxlifei": "max life insurance",
    "mcgmwate": "mcgm water",
    "sbilife": "sbi life",
    "lic": "lic of india",
}


def is_generic_level6(value: str) -> bool:
    """Return True when a Level 6 value names no entity (e.g. "to bill payment")."""

    text = _normalize_compare(value)
    if len(re.sub(r"[^a-z]", "", text)) < 3:
        return True
    if is_stop_value(text):
        return True
    return all(token in GENERIC_LEVEL6_WORDS for token in re.findall(r"[a-z0-9]+", text))


def extract_narration_entity(particulars: str) -> str:
    """Return the payee/payer/biller explicitly named by a structured narration."""

    text = str(particulars or "")

    biller = BBPS_BILLER_RE.search(text)
    if biller:
        code = re.sub(r"\s+", " ", biller.group(1)).strip().lower()
        compact_code = re.sub(r"\s+", "", code)
        if code.endswith("cc") and any(code[:-2] in aliases for aliases in BANK_ALIASES.values()):
            return f"{code[:-2]} credit card"
        return BBPS_BILLER_ALIASES.get(compact_code, BBPS_BILLER_NAMES.get(code, code))

    rail = SLASH_ENTITY_RAIL_RE.search(text)
    if not rail:
        return ""
    segments = [segment.strip() for segment in rail.group(2).split("/")]
    direction_index = next(
        (index for index, segment in enumerate(segments) if segment.upper() in {"DR", "CR"}),
        None,
    )
    if direction_index is None:
        return ""
    for segment in segments[direction_index + 1 :]:
        if re.search(r"\d{5,}", segment):
            continue
        name = re.sub(r"\s+", " ", segment).strip().lower()
        if name and not is_generic_level6(name):
            return name
        return ""
    return ""


def _clean_layout_name(value: str) -> str:
    """Normalise a name taken from a narration layout, or "" if it is not a name.

    Reference/account numbers are dropped and leading/trailing generic words
    are trimmed. A segment that still contains generic words in the middle
    (e.g. "yasodhaa from va") is free-text remark, not a reliable name.
    """

    text = re.sub(r"[^a-z0-9&.\- ]+", " ", str(value or "").lower())
    tokens = [token for token in text.split() if not token.isdigit()]
    while tokens and re.sub(r"[^a-z0-9]", "", tokens[0]) in GENERIC_LEVEL6_WORDS:
        tokens.pop(0)
    while tokens and re.sub(r"[^a-z0-9]", "", tokens[-1]) in GENERIC_LEVEL6_WORDS:
        tokens.pop()
    name = " ".join(tokens).strip(" .-&")
    if not name or is_generic_level6(name):
        return ""
    if any(word in GENERIC_LEVEL6_WORDS for word in re.findall(r"[a-z0-9]+", name)):
        return ""
    return name


# Layouts where the payee sits between a prefix and a reference: group 1 is the name.
MOBFT_NAME_RE = re.compile(r"^\s*MOBFT\s+(?:from|to)\s*:\s*(.+?)\s*/\s*\d{6,}", flags=re.IGNORECASE)
_HYPHEN_PREFIXES = [w for w, (_, layout) in LEVEL1_PREFIX_WORDS.items() if layout == "hyphen"]
_SLASH_PREFIXES = [w for w, (_, layout) in LEVEL1_PREFIX_WORDS.items() if layout == "slash"]
HYPHEN_PREFIX_NAME_RE = re.compile(
    r"^\s*(?:" + "|".join(_HYPHEN_PREFIXES) + r")\s*-\s*(.+?)(?=\s+\S*\d{6,}|\s+UTR\b|$)", flags=re.IGNORECASE
)
SLASH_PREFIX_RE = re.compile(r"^\s*(?:" + "|".join(_SLASH_PREFIXES) + r")/(.*)$", flags=re.IGNORECASE)
IFSC_LIKE_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$|^[A-Z]{4}\d{3,}[A-Z0-9]*$", flags=re.IGNORECASE)


def _slash_prefix_name(rest: str) -> str:
    """First segment of PREFIX/<ref>/<name or IFSC>/<account> that is a name, else ""."""

    for segment in rest.split("/"):
        seg = segment.strip()
        if (
            not seg
            or re.fullmatch(r"[\d\s]+", seg)
            or "@" in seg
            or seg.upper() in {"CR", "DR"}
            or IFSC_LIKE_RE.match(seg.replace(" ", ""))
            or is_bank_only_part(seg)
        ):
            continue
        return _clean_name_text(seg)
    return ""


# Words after which a name continues with a relative's name ("GAURAV S/O SH DINESH"): cut there.
RELATIONSHIP_WORDS = frozenset({"so", "wo", "do", "wife", "son"})
RELATIONSHIP_INITIALS = frozenset({"s", "w", "d"})


def _trim_person_name(tokens: list[str]) -> list[str]:
    """Cut a name at a relationship word (S/O, W/O, D/O, WIFE, SON) and drop single-letter tails."""

    kept: list[str] = []
    for index, token in enumerate(tokens):
        if index >= 1 and (
            token in RELATIONSHIP_WORDS
            or (token in RELATIONSHIP_INITIALS and index + 1 < len(tokens) and tokens[index + 1] == "o")
        ):
            break
        kept.append(token)
    while len(kept) > 1 and len(kept[-1]) == 1:
        kept.pop()
    return kept


def _clean_name_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9&.\- ]+", " ", str(value or "").lower())
    tokens = [t for t in text.split() if not t.isdigit()]
    return " ".join(_trim_person_name(tokens)).strip(" .-&")


# ---------------------------------------------------------------------------
# Level 6 (payee or brand name): layout-aware extraction and normalisation.
#
# Each rail writes the counterparty in one fixed field. LAYOUT_RULES names that
# field per template, so the name is taken only from there. References, VPAs,
# bank handles, IFSC codes, account numbers, cities, prefix words and bank names
# are never candidates. A new bank layout needs one row in this table.
# ---------------------------------------------------------------------------
class LayoutRule(NamedTuple):
    label: str
    pattern: re.Pattern
    has_name: bool = True  # False: the template carries no payee, Level 6 is blank
    strip_words: tuple[str, ...] = ()  # leading words that belong to the template, not the name
    city_group: str = ""  # group holding the city (never a candidate), for checks
    bank_group: str = ""  # group holding the bank field (evidence, never a candidate)


def _layout(
    label: str, pattern: str, has_name: bool = True, strip: tuple[str, ...] = (), city: str = "", bank: str = ""
) -> LayoutRule:
    return LayoutRule(label, re.compile(pattern, flags=re.IGNORECASE), has_name, strip, city, bank)


_HYPHEN_PREFIX_ALT = "|".join(_HYPHEN_PREFIXES) or r"(?!)"
LAYOUT_RULES: tuple[LayoutRule, ...] = (
    # --- templates that carry no payee -------------------------------------------------
    _layout("atm_card", r"^\s*(?:NWD|ATW|ATM)\s*[-/ ]", has_name=False),
    _layout("reversal_upi_ref", r"^\s*REV-UPI-\d", has_name=False),
    _layout("billpay", r"^\s*IB\s+BILLPAY", has_name=False),
    _layout("tax_line", r"^\s*\d{6}[A-Z]\d+\s+DPO\d+\s+[CSI]GST", has_name=False),
    _layout("ach_return_charge", r"^\s*\.?ACH\s+DEBIT\s+RETURN", has_name=False),
    _layout("upi_vpa_first", r"^\s*UPI\s*-\s*[^-]*@", has_name=False),
    # --- name field named by the template ---------------------------------------------
    # UPI-<NAME>-<VPA>@<handle>-<IFSC>-<ref>-<remark>
    _layout("upi_hyphen", r"^\s*UPI\s*-\s*(?P<name>[^-@]+?)\s*-(?=.*@)"),
    # IMPS-<ref>-<NAME>-<bank>-<account>-<remark>
    _layout("imps_hyphen", r"^\s*IMPS\s*-\s*\d{6,}\s*-\s*(?P<name>[^-]+)"),
    # NEFT CR-<IFSC>-<NAME>-<remitter>-<ref>
    _layout("neft_cr", r"^\s*(?:NEFT|RTGS)\s+(?:CR|DR)\s*-\s*[A-Z]{4}0[A-Z0-9]{6}\s*-\s*(?P<name>[^-]+)"),
    # RTGSO-<NAME> <ref> UTR ...   NEFTO-<NAME> <ref> ...
    _layout("rail_out_hyphen", r"^\s*(?:" + _HYPHEN_PREFIX_ALT + r")\s*-\s*(?P<name>.+?)(?=\s+\S*\d{6,}|\s+UTR\b|$)"),
    # ACH D- <NAME>-<mandate or reference>
    _layout("ach_mandate", r"^\s*\.?ACH\s+[DC]\s*-\s*(?P<name>[^-]+)", strip=("tp ach", "tp")),
    # <account>-TPT-<remark>-<NAME>
    _layout("tpt", r"^\s*\d{6,}\s*-\s*TPT\s*-\s*[^-]*-\s*(?P<name>[^-]+)\s*$"),
    # A2AINT01-<NAME>-<remark>-<period>
    _layout("a2a", r"^\s*A2AINT\w*\s*-\s*(?P<name>[^-]+)"),
    # POS/<MERCHANT>/<CITY>/<date>/<time>
    _layout("pos_slash", r"^\s*POS/(?P<name>[^/]+)/(?P<city>[^/]*)/", city="city"),
    # POS <masked card> <GATEWAY*>MERCHANT
    _layout("pos_card", r"^\s*POS\s+\d{4,}X+\d{4}\s+(?:[A-Z]{2,5}\*)?(?P<name>.+?)\s*$"),
    # MOBFT from or to: <NAME>/<ref>
    _layout("mobft", r"^\s*MOBFT\s+(?:from|to)\s*:\s*(?P<name>.+?)\s*/\s*\d{6,}"),
    # UPI|IMPS/P2A|P2M|P2P/<ref>/<NAME>/<bank>/...
    _layout("slash_p2x", r"^\s*(?:UPI|IMPS)/(?:P2A|P2M|P2P)/\d{6,}/(?P<name>[^/]*)/(?P<bank>[^/]*)", bank="bank"),
    # NEFT|RTGS/<ref>/<NAME>/<bank>/...
    # (the IFSC field some banks put before the name is skipped)
    _layout(
        "slash_neft",
        r"^\s*(?:NEFT|RTGS)/[A-Z0-9]*\d{5,}[A-Z0-9]*/(?:[A-Z]{4}0[A-Z0-9]{6}/)?(?P<name>[^/]*)(?:/(?P<bank>[^/]*))?",
        bank="bank",
    ),
    # UPI/DR|CR/<ref>/<NAME>/<bank code>/<masked VPA>/...
    _layout("upi_dr_slash", r"^\s*UPI/(?:DR|CR)/\d{6,}/(?P<name>[^/]*)/(?P<bank>[^/]*)", bank="bank"),
)

# Words a template adds in front of, or around, a name.
TITLE_WORDS = frozenset({"mr", "mrs", "ms", "miss", "mister", "shri", "smt", "sh", "dr", "late"})
# A generic label sitting in the name field ("Bank Acco", "Payment"): no payee.
PLACEHOLDER_NAME_PHRASES = ("bank account", "payment", "upi payment", "upi", "transfer", "fund transfer", "paid via upi")
# Words that complete a name split by line wrapping ("LIMIT ED", "PRI VATE", "SA LARY").
WRAP_VOCABULARY = frozenset(
    {
        "limited", "private", "services", "service", "finance", "financial", "finlease", "fincap",
        "enterprises", "enterprise", "industries", "solutions", "technologies", "trading", "traders",
        "corporation", "company", "leasing", "capital", "commercial", "india", "salary", "payment",
        "payments", "consultancy", "systems", "software", "infotech", "holdings", "securities",
        "insurance", "credit", "loans", "stores", "agency", "associates", "foundation",
    }
)
# Words that make a name a company, not a person (only a weak hint for Level 3).
COMPANY_SUFFIX_WORDS = frozenset(
    {"ltd", "limited", "pvt", "private", "llp", "inc", "finlease", "fincap", "finance", "financial",
     "finserv", "leasing", "capital", "services", "enterprises", "industries", "solutions", "technologies"}
)
LEVEL6_TAIL_STOP = frozenset({"ltd", "limited", "pvt", "private"})


def _letters(value: str) -> str:
    return re.sub(r"[^a-z]", "", str(value or "").lower())


def rejoin_wrapped_words(tokens: list[str], vocabulary: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Join two neighbouring fragments when together they spell a known word.

    "limit ed" -> "limited", "pri vate" -> "private". The join needs at least one
    fragment that is not a word by itself, so "ram kumar" is never fused.
    """

    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index + 1 < len(tokens):
            nxt = tokens[index + 1]
            joined = token + nxt
            # Company words always rejoin. A word seen intact elsewhere in the statement
            # rejoins when one fragment is not itself a word of the statement ("m ohd",
            # "mar keting"); two intact words ("ram", "kumar") are never fused.
            static_join = joined in WRAP_VOCABULARY and (token not in WRAP_VOCABULARY or nxt not in WRAP_VOCABULARY)
            vocab_join = joined in vocabulary and (token not in vocabulary or nxt not in vocabulary)
            if static_join or vocab_join:
                out.append(joined)
                index += 2
                continue
        out.append(token)
        index += 1
    return out


def normalize_level6(value: str, strip_words: tuple[str, ...] = (), vocabulary: frozenset[str] | set[str] = frozenset()) -> str:
    """Return a clean Level 6 name: lowercase, no punctuation, titles, relatives or reference tails.

    "MR. RAJESH KUMAR S/O RAM" -> "rajesh kumar"; "M/S ABC TRADERS" -> "abc traders";
    a token with a digit (a reference) ends the name; a single-letter tail is cut;
    when nothing valid is left the result is "".
    """

    text = re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower().replace("'", ""))
    tokens = text.split()
    for phrase in strip_words:
        words = phrase.split()
        if len(tokens) > len(words) and tokens[: len(words)] == words:
            tokens = tokens[len(words):]
    # leading titles ("mr", "m s")
    while tokens and (tokens[0] in TITLE_WORDS or (tokens[0] == "m" and len(tokens) > 1 and tokens[1] == "s")):
        tokens = tokens[2:] if tokens[0] == "m" else tokens[1:]
    # a reference (token with a digit) ends the name
    kept: list[str] = []
    for token in tokens:
        if re.search(r"\d", token):
            break
        kept.append(token)
    kept = _trim_person_name(rejoin_wrapped_words(kept, vocabulary))
    # drop a trailing placeholder tail
    while len(kept) > 1 and kept[-1] in {"o", "a"} and len(kept[-1]) == 1:
        kept.pop()
    name = " ".join(kept).strip()
    if len(_letters(name)) < 3 or re.fullmatch(r"x+", _letters(name)):
        return ""
    return name


def _is_placeholder_name(name: str) -> bool:
    letters = _letters(name)
    if len(letters) < 4:
        return False
    return any(_letters(phrase).startswith(letters) for phrase in PLACEHOLDER_NAME_PHRASES)


def _is_bank_with_branch_tail(name: str) -> bool:
    """A bank name followed only by "limited" or one short branch / centre code ("yes bank limited rac")."""

    tokens = name.split()
    for size in range(len(tokens) - 1, 1, -1):
        head = " ".join(tokens[:size])
        if match_bank_name(head) or head in BANK_ALIASES:
            rest = [t for t in tokens[size:] if t not in COMPANY_SUFFIX_WORDS]
            return len(rest) <= 1 and all(len(t) <= 4 for t in rest)
    return False


def _name_is_valid_payee(name: str) -> bool:
    """False for stop values, labels, placeholders, bank names and charge words."""

    if not name or is_stop_value(name) or _is_placeholder_name(name):
        return False
    if is_bank_only_part(name) or _is_bank_with_branch_tail(name):
        return False
    tokens = name.split()
    return not all(t in GENERIC_LEVEL6_WORDS or t in LEVEL6_TAIL_STOP for t in tokens)


def layout_level6_name(particulars: str) -> tuple[bool, str, str]:
    """Return (matched, name, city) for the first LAYOUT_RULES template the narration fits.

    ``matched`` is True when a template applies. ``name`` is "" when the template
    has no payee or its name field holds only a label, bank name or reference.
    """

    text = str(particulars or "")
    for rule in LAYOUT_RULES:
        match = rule.pattern.match(text)
        if not match:
            continue
        city = match.group(rule.city_group).strip() if rule.city_group else ""
        if not rule.has_name:
            return True, "", city
        name = normalize_level6(match.group("name"), rule.strip_words)
        return True, (name if _name_is_valid_payee(name) else ""), city
    return False, "", ""


VPA_LOCAL_RE = re.compile(r"([A-Za-z0-9._\-]+)\s*@")
REFERENCE_NAME_RE = re.compile(r"^(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{6,}$|^\d{5,}$")


def sanitize_level6(value: str, particulars: str = "") -> str:
    """Blank a Level 6 that is a VPA, reference id, rail prefix, label, city or bank name."""

    name = str(value or "").strip().lower()
    if not name:
        return ""
    compact = re.sub(r"[^a-z0-9]", "", name)
    if any(compact == re.sub(r"[^a-z0-9]", "", local.lower()) for local in VPA_LOCAL_RE.findall(str(particulars or ""))):
        return ""
    if any(REFERENCE_NAME_RE.match(token) for token in name.split()):
        return ""
    if not _name_is_valid_payee(name):
        return ""
    _, _, city = layout_level6_name(particulars)
    if city and compact == re.sub(r"[^a-z0-9]", "", city.lower()):
        return ""
    return name


def apply_layout_level6_override(
    particulars: str,
    result: dict[str, str],
    master_locked: bool = False,
) -> tuple[dict[str, str], bool]:
    """Set Level 6 from the field the narration template names.

    Returns the result and whether the layout name now sits in Level 6 (those
    rows may be merged to one canonical name later). An exact Master match
    keeps its Level 6; a fuller existing name that contains the layout name is
    kept as well; otherwise the template's field replaces what the model took
    from other text, and a template without a valid payee leaves it blank.
    """

    current = str(result.get("Level_6", "") or "")
    if master_locked:
        return result, False
    matched, name, _ = layout_level6_name(particulars)
    if not matched:
        if current and result.get("Level_5") == "loan repayment" and is_bank_only_part(current):
            return result, False  # a loan paid to a bank: the bank is the lender
        cleaned = sanitize_level6(re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", current.lower())).strip(), particulars) if current else ""
        if cleaned != current:
            result["Level_6"] = cleaned
        return result, False
    if name and current and not is_generic_level6(current) and sanitize_level6(current, particulars):
        # An existing name that is the same entity as the template's field (the same
        # words ignoring "limited" / "pvt"), or that contains it ("swiggy genie" for
        # "swiggy"), is a Master or brand spelling: it stays.
        same_entity = (
            entity_merge_score(current, name) >= 1.0
            or _letters(name) in _letters(current)
            or current in KNOWN_MERCHANT_FRAGMENTS  # a known brand inside a fuller template name
        )
        if same_entity and _letters(current) != _letters(name):
            return result, False
    if not name and current and result.get("Level_5") == "loan repayment" and is_bank_only_part(current):
        return result, False  # a loan paid to a bank: the bank is the lender (IDFC FIRST BANK)
    if (
        not name
        and current
        and sanitize_level6(current, particulars)
        and (
            current in KNOWN_MERCHANT_FRAGMENTS
            or (result.get("Level_3") == "merchant" and result.get("Level_5") not in {"", "p2p transfer"})
        )
    ):
        return result, False  # the field holds only a bank, but the row has a known brand ("airtel", recharge)
    result["Level_6"] = name
    return result, bool(name)


def extract_level6_entity(particulars: str) -> str:
    """Return the counterparty named by the narration, for Level 6 (never a rail word or label)."""

    return guard_level6(_extract_level6_entity_raw(particulars), particulars)


def _extract_level6_entity_raw(particulars: str) -> str:
    """Return the counterparty named by the narration, for Level 6.

    Uses ``extract_narration_entity`` (slash rails, BBPS billers) first, then
    the bank layouts above. Kept separate from ``extract_narration_entity`` so
    the Level 7 counterparty key, which calls that function, is unchanged.
    """

    matched, layout_name, _ = layout_level6_name(particulars)
    if matched:
        return layout_name

    entity = extract_narration_entity(particulars)
    if entity:
        return entity

    text = str(particulars or "")
    for pattern in (MOBFT_NAME_RE, HYPHEN_PREFIX_NAME_RE):
        named = pattern.match(text)
        if named:
            return _clean_name_text(named.group(1))
    slash = SLASH_PREFIX_RE.match(text)
    if slash:
        return _slash_prefix_name(slash.group(1))
    for pattern in (
        INET_SLASH_ENTITY_RE,
        HYPHEN_RAIL_ENTITY_RE,
        CHEQUE_ENTITY_RE,
        MMT_IMPS_ENTITY_RE,
        INTERNAL_TRANSFER_ENTITY_RE,
        ECOM_PUR_ENTITY_RE,
    ):
        match = pattern.match(text)
        if match:
            return _clean_layout_name(match.group(1))

    inft = INFT_ENTITY_RE.match(text)
    if inft:
        segments = [segment for segment in inft.group(1).split("/") if segment.strip()]
        return _clean_layout_name(segments[-1]) if segments else ""
    return ""


def _echoes_narration_part(value: str, particulars: str, part_texts: list[str] | None) -> bool:
    """Return True when a Level 6 value is just a copy of one narration fragment."""

    def norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))

    target = norm(value)
    if not target:
        return False
    pieces = list(part_texts or []) + re.split(r"[/|:]|-(?=\S)", str(particulars or ""))
    return any(norm(piece) == target for piece in pieces)


def fill_specific_level6(
    particulars: str,
    result: dict[str, str],
    part_texts: list[str] | None = None,
) -> dict[str, str]:
    """Replace a blank, generic or category Level 6 with the narration's entity.

    A specific Level 6 is kept, unless it is merely a copy of one narration
    fragment (e.g. "inft", "trfr to adimoolam n", an MMT remark) and the
    narration layout names the counterparty explicitly.
    """

    current = result.get("Level_6", "")
    stripped = strip_leading_labels(current) if current else current
    had_label = bool(current) and stripped != _normalize_compare(current)
    if stripped != current:
        current = result["Level_6"] = stripped
    is_category = bool(current) and _normalize_compare(current) == _normalize_compare(result.get("Level_5", ""))
    biller = BBPS_BILLER_RE.search(str(particulars or ""))
    is_raw_biller_code = bool(biller) and re.sub(r"[^a-z0-9]", "", _normalize_compare(current)) == re.sub(
        r"[^a-z0-9]", "", biller.group(1).lower()
    )
    entity = extract_level6_entity(particulars)
    is_fragment_copy = (
        bool(entity)
        and _normalize_compare(current) != entity
        and not (current and entity_merge_score(current, entity) >= 1.0)  # same entity, Master spelling stays
        and (had_label or _echoes_narration_part(current, particulars, part_texts))
    )
    if (
        current
        and not is_category
        and not is_raw_biller_code
        and not is_fragment_copy
        and not is_generic_level6(current)
    ):
        return result
    if entity:
        result["Level_6"] = entity
    elif is_category or is_stop_value(current):
        result["Level_6"] = ""
    return result


def derive_direction_from_amounts(debit_value: object, credit_value: object) -> str:
    """Return debit or credit from the numeric transaction columns."""

    debit_num = parse_amount(debit_value)
    credit_num = parse_amount(credit_value)
    if debit_num > 0 and credit_num <= 0:
        return "debit"
    if credit_num > 0 and debit_num <= 0:
        return "credit"
    return ""


# Platforms that are themselves the counterparty: when the payee field names one, Level 6
# keeps it even though Level 4 is the same platform.
PAYEE_PLATFORM_LEVEL6 = frozenset({"slice"})


def _apply_transaction_rules(
    result: dict[str, str],
    default_p2p_purpose: bool = True,
    confirmed_level6: str = "",
) -> dict[str, str]:
    """Apply row-level consistency rules after aggregation.

    ``confirmed_level6`` is a counterparty the narration layout names
    explicitly; it survives the institution rule below.
    """

    if default_p2p_purpose and result["Level_3"] == "p2p" and not result["Level_5"]:
        result["Level_5"] = "p2p transfer"

    # A merchant is never a person-to-person transfer, whichever narration
    # format (slash or hyphen UPI) or Master Sheet row put the label there.
    if result["Level_3"] == "merchant" and result["Level_5"] == "p2p transfer":
        result["Level_5"] = ""

    if result["Level_1"] == "cash":
        result["Level_3"] = ""
        if is_payment_intermediary_label(result["Level_4"]):
            result["Level_4"] = ""
        result["Level_5"] = ""
        result["Level_6"] = ""

    # For an institution row Level 6 is cleared only when it just repeats the
    # bank (or is a generic / category word). A named counterparty (a person, a
    # company, a known merchant such as "BLUE APPARELS") is kept.
    if (
        result["Level_3"] == "institution"
        and result["Level_4"]
        and not (confirmed_level6 and _normalize_compare(result["Level_6"]) == confirmed_level6)
        and (
            _normalize_compare(result["Level_6"]) == _normalize_compare(result["Level_4"])
            or is_bank_only_part(result["Level_6"])
            or is_generic_level6(result["Level_6"])
            or _normalize_compare(result["Level_6"]) == _normalize_compare(result["Level_5"])
        )
    ):
        result["Level_6"] = ""

    if result["Level_6"] and is_stop_value(result["Level_6"]):
        result["Level_6"] = ""

    if (
        result["Level_4"]
        and result["Level_6"]
        and _normalize_compare(result["Level_4"]) == _normalize_compare(result["Level_6"])
        and not (
            confirmed_level6
            and _normalize_compare(result["Level_6"]) == confirmed_level6
            and confirmed_level6 in PAYEE_PLATFORM_LEVEL6
        )
    ):
        result["Level_6"] = ""

    return result


# Four-letter IFSC bank codes. A narration part made of one of these (e.g. the
# UTR prefix "HDFCN" = HDFC + NEFT, "KVBLR" = Karur Vysya + RTGS, or a full
# IFSC "HDFC0001304") names the bank the money moved through, not the
# counterparty.
BANK_IFSC_CODES = frozenset(
    {alias for aliases in BANK_ALIASES.values() for alias in aliases if re.fullmatch(r"[a-z]{4}", alias)}
    | {code for code in IFSC_PREFIX_TO_BANK if len(code) == 4}
    | {"fdrl", "idfb", "kkbk", "barb", "ibkl", "ucba", "indb", "ratn", "cosb", "jaka", "csbk", "dlxb"}
)
BANK_NAME_WORDS = frozenset(
    {word for aliases in BANK_ALIASES.values() for alias in aliases for word in alias.split()}
    | {"bank", "ltd", "limited", "the", "co", "op", "coop", "cooperative", "of", "and"}
)


def is_bank_only_part(part_text: str) -> bool:
    """Return True when a narration part names only a bank (code, IFSC, UTR prefix or name)."""

    compact = re.sub(r"[^a-z0-9]", "", str(part_text or "").lower())
    if not compact:
        return False
    if match_bank_name(part_text):
        return True
    if compact[:4] in BANK_IFSC_CODES and re.fullmatch(r"[a-z]{4}(?:[a-z]?\d{6,}|[a-z]?|0[a-z0-9]{6})", compact):
        return True
    detected = detect_level_4_platform_or_bank(part_text)
    if not detected or is_payment_intermediary_label(detected):
        return False
    words = re.findall(r"[a-z0-9]+", str(part_text).lower())
    return all(word in BANK_NAME_WORDS for word in words)


# Marker the row classifier puts on a part prediction that came from an exact Master lookup.
EXACT_MASTER_KEY = "_exact_master"


def aggregate_part_predictions(
    part_predictions: list[tuple[str, dict[str, str]]],
) -> dict[str, str]:
    """Combine part-level predictions into one row-level result."""

    result = _empty_levels()
    # A part that only names a bank (the UTR prefix "HDFCN", a full IFSC,
    # "AirtelPaymentsBank", "BANK OF B") says which bank moved the money, not who
    # the counterparty is or what the money was for. It may contribute Level 4
    # only; its Level 3, 5 and 6 are ignored.
    bank_only_flags = [is_bank_only_part(part_text) for part_text, _ in part_predictions]
    # A part that is an exact Master Sheet match and names a counterparty type (Level 3)
    # is read first, so no earlier part can outvote the Master row.
    priority = sorted(
        range(len(part_predictions)),
        key=lambda i: not (part_predictions[i][1].get(EXACT_MASTER_KEY) and part_predictions[i][1].get("Level_3")),
    )
    part_predictions = [part_predictions[i] for i in priority]
    bank_only_flags = [bank_only_flags[i] for i in priority]
    for (_, prediction), bank_only in zip(part_predictions, bank_only_flags):
        for level_index in range(8):
            level_name = f"Level_{level_index}"
            if bank_only and level_name != "Level_4":
                continue
            if not result[level_name] and prediction.get(level_name):
                result[level_name] = prediction[level_name]

    specific_level6 = next(
        (
            prediction["Level_6"]
            for (_, prediction), bank_only in zip(part_predictions, bank_only_flags)
            if not bank_only
            and prediction.get("Level_6")
            and not is_generic_level6(prediction["Level_6"])
        ),
        "",
    )
    if specific_level6:
        result["Level_6"] = specific_level6

    intermediary = next(
        (
            prediction["Level_4"]
            for _, prediction in part_predictions
            if prediction.get("Level_4") and is_payment_intermediary_label(prediction["Level_4"])
        ),
        "",
    )
    if intermediary:
        result["Level_4"] = intermediary

    return _apply_transaction_rules(result)


def apply_structured_upi_overrides(
    part_predictions: list[tuple[str, dict[str, str]]],
    keyword_classifier: KeywordClassifier,
    result: dict[str, str],
) -> dict[str, str]:
    """Use stable UPI/P2X structure when it is present in split parts."""

    if len(part_predictions) < 3:
        return result

    ordered_parts = [part_text.strip() for part_text, _ in part_predictions if part_text.strip()]
    ordered_predictions = [prediction for part_text, prediction in part_predictions if part_text.strip()]
    if len(ordered_parts) < 3 or ordered_parts[0].upper() != "UPI":
        return result

    marker = ordered_parts[1].upper().replace(" ", "")
    if marker not in UPI_P2_MERCHANT_MARKERS | UPI_P2_P2P_MARKERS:
        return result

    entity_text = ordered_parts[2]
    entity_prediction = keyword_classifier.classify(entity_text)
    bank_name = ""
    inherited_purpose = ""

    for prediction in ordered_predictions[3:]:
        if not bank_name and prediction.get("Level_4"):
            bank_name = prediction["Level_4"]
        if not inherited_purpose and prediction.get("Level_5") not in {"", "p2p transfer"}:
            inherited_purpose = prediction["Level_5"]

    result["Level_1"] = "upi"
    # Platform first: a bank from a later part never replaces a platform.
    if bank_name and not (_is_platform_label(result["Level_4"]) and not _is_platform_label(bank_name)):
        result["Level_4"] = bank_name

    if marker in UPI_P2_MERCHANT_MARKERS:
        result["Level_3"] = "merchant"
        result["Level_6"] = clean_entity(entity_text, result["Level_4"])
        purpose = entity_prediction.get("Level_5", "") or inherited_purpose or detect_explicit_purpose(entity_text)
        if purpose and purpose != "p2p transfer":
            result["Level_5"] = purpose
    else:
        result["Level_3"] = "p2p"
        result["Level_6"] = clean_entity(entity_text)
        result["Level_5"] = "p2p transfer"

    return _apply_transaction_rules(result)


# A payment gateway named as a separate word inside a payee / VPA segment.
GATEWAY_WORD_RE = re.compile(
    r"(?<![a-z])(?:rzp|razorpay|payu|cashfree|billdesk|getepay|bharatpe|phonepe|paytm|mobikwik)(?![a-z])"
)


# A gateway name fused to the front of a word ("razorpayzomato").
GATEWAY_PREFIX_RE = re.compile(r"^(?:razorpay|payu|cashfree|billdesk|getepay|bharatpe|phonepe|paytm|mobikwik|rzp)")


def extract_hyphen_upi_merchant(particulars: str) -> str:
    """Return the known merchant named in a hyphen-delimited UPI narration."""

    text = str(particulars or "")
    if not HYPHEN_UPI_RE.match(text):
        return ""

    segments = [segment.strip() for segment in text.split("-")]
    vpa_segment = next((segment for segment in segments[1:] if "@" in segment), "")
    if not vpa_segment:
        return ""

    payee_segments = segments[1 : segments.index(vpa_segment)]
    if payee_segments and (
        payee_segments[0].upper().replace(" ", "") in UPI_P2_MERCHANT_MARKERS | UPI_P2_P2P_MARKERS
    ):
        return ""

    for candidate in [vpa_segment.split("@", 1)[0], *payee_segments]:
        candidate = candidate.lower()
        # "<merchant> <gateway> <bank handle>" (e.g. "cleartrip rzp rxairtel"): the
        # text after a gateway word is the gateway's bank handle, not the merchant,
        # so a brand found only there (airtel) is not the merchant. When the
        # gateway comes first ("razorpay zomato") the whole text is searched.
        gateway = GATEWAY_WORD_RE.search(candidate)
        if gateway and re.search(r"[a-z]", candidate[: gateway.start()]):
            candidate = candidate[: gateway.start()]
        # A brand must start a word, or follow a gateway name fused to it
        # ("razorpayzomato"). A brand buried inside a bank handle or in letters
        # joined across words ("rxairtel", "saji okicici" -> "jio") is not the merchant.
        tokens = re.findall(r"[a-z]+", candidate)
        tokens += [GATEWAY_PREFIX_RE.sub("", token, count=1) for token in tokens]
        for fragment in sorted(KNOWN_MERCHANT_FRAGMENTS, key=lambda item: (-len(item), item)):
            if any(token.startswith(fragment) for token in tokens):
                return fragment
    return ""


def is_payment_intermediary_label(label: str) -> bool:
    """Return True for a label that names a payment platform/intermediary.

    Bank names win over platform aliases: "paytm" appears in both tables and is
    treated as a bank here.
    """

    value = _normalize_compare(label)
    if not value or value in BANK_NAMES:
        return False
    if value in PLATFORM_ALIASES:
        return True
    letters = re.sub(r"[^a-z0-9]", "", value)  # digits stay: "one97" is a brand token
    return any(fragment in letters for fragment in PAYMENT_INTERMEDIARY_FRAGMENTS)


def derive_upi_bank(particulars: str) -> str:
    """Return the bank named by a UPI narration's IFSC prefix or VPA handle."""

    segments = [segment.strip() for segment in str(particulars or "").split("-")]

    for segment in segments:
        compact = re.sub(r"\s+", "", segment).upper()
        if IFSC_RE.match(compact):
            bank = IFSC_PREFIX_TO_BANK.get(compact[:4].lower())
            if bank:
                return bank

    vpa_segment = next((segment for segment in segments if "@" in segment), "")
    handle = re.sub(r"[^a-z]", "", vpa_segment.split("@", 1)[1].lower()) if vpa_segment else ""
    if handle:
        if handle in BANK_CODE_TO_NAME:
            return BANK_CODE_TO_NAME[handle]
        for code, name in BANK_CODE_TO_NAME.items():
            if len(code) >= 4 and handle.startswith(code):
                return name
    return ""


def apply_hyphen_upi_merchant_override(
    particulars: str,
    part_predictions: list[tuple[str, dict[str, str]]],
    keyword_classifier: KeywordClassifier,
    result: dict[str, str],
) -> dict[str, str]:
    """Classify hyphen-format UPI payments to a known merchant as merchant rows."""

    merchant = extract_hyphen_upi_merchant(particulars)
    if not merchant:
        return result

    purpose = keyword_classifier.classify(merchant).get("Level_5", "")
    if not purpose or purpose in {"p2p transfer", "salary"}:
        purpose = MERCHANT_DEFAULT_PURPOSES.get(merchant, "")

    level4_candidates = [result["Level_4"]] + [
        prediction.get("Level_4", "") for _, prediction in part_predictions
    ]
    # Level 4 platform first: the platform or gateway named by the narration or
    # predicted by the model (razorpay, payu, cred, getepay ...) wins; the bank
    # (model, then narration IFSC/handle) is used only when no platform exists.
    detected = detect_level_4_platform_or_bank(particulars)
    platform = detected if _is_platform_label(detected) else ""
    platform = platform or next((label for label in level4_candidates if _is_platform_label(label)), "")
    bank = next((label for label in level4_candidates if label and not _is_platform_label(label)), "")
    result["Level_4"] = platform or bank or derive_upi_bank(particulars)

    result["Level_1"] = "upi" if UPI_START_RE.match(str(particulars or "")) else (result["Level_1"] or "upi")
    result["Level_3"] = "merchant"
    # Master first: a fuller Master Sheet name ("swiggy dine in") is kept over the
    # shorter brand ("swiggy") found in the narration.
    current_level6 = result.get("Level_6", "")
    brand_letters = re.sub(r"[^a-z0-9]", "", merchant.lower())
    current_letters = re.sub(r"[^a-z0-9]", "", current_level6.lower())
    if not (
        current_level6
        and not is_generic_level6(current_level6)
        and brand_letters
        and brand_letters in current_letters
    ):
        result["Level_6"] = merchant
    # Master first: a purpose from the Master Sheet is kept. Code fills a blank,
    # and replaces "p2p transfer" / "salary", which a merchant cannot have.
    if result["Level_5"] in {"", "p2p transfer", "salary"}:
        result["Level_5"] = purpose

    return _apply_transaction_rules(result)


def apply_debit_salary_guard(
    debit_value: object,
    credit_value: object,
    result: dict[str, str],
) -> dict[str, str]:
    """Drop a salary purpose from rows whose amounts say they are debits."""

    if derive_direction_from_amounts(debit_value, credit_value) == "debit":
        if _normalize_compare(result.get("Level_5", "")) == "salary":
            result["Level_5"] = ""
    return result


def apply_raw_transaction_overrides(
    particulars: str,
    result: dict[str, str],
    part_texts: list[str] | None = None,
    debit_value: object = None,
    credit_value: object = None,
) -> dict[str, str]:
    """Apply row-level overrides only when narration provides useful evidence."""

    text = str(particulars or "")
    direction = derive_direction_from_amounts(debit_value, credit_value)
    is_debit = direction == "debit"
    marker_text = _marker_text(text, part_texts)

    if re.search(r"\bACH\b", marker_text, flags=re.IGNORECASE) and direction == "credit":
        if re.search(r"\bDIVIDEND(?:S)?\b", marker_text, flags=re.IGNORECASE):
            result["Level_5"] = "dividend"
        elif result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    automated_debit = bool(SYSTEM_INITIATED_RE.search(marker_text)) and is_debit
    if automated_debit:
        if result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    match = SLASH_UPI_RE.search(text.upper())
    if match:
        name = match.group(1).strip().lower()
        bank_code = match.group(2).strip().lower()
        vpa = match.group(3).strip().lower()
        bank_name = BANK_CODE_TO_NAME.get(bank_code, "")
        # A bare merchant-QR id (PhonePe business "q<digits>", "paytmqr...", BharatPe)
        # is a merchant even when the statement cut off the "@handle".
        is_gateway = any(fragment in vpa for fragment in GATEWAY_VPA_FRAGMENTS) or bool(
            re.match(r"(?:q\d{6,}|paytmqr|bharatpe)", vpa)
        )

        result["Level_1"] = "upi" if UPI_START_RE.match(text) else (result["Level_1"] or "upi")
        if not _is_platform_label(result["Level_4"]):
            result["Level_4"] = bank_name or result["Level_4"]

        if is_gateway:
            result["Level_3"] = "merchant"
            if result.get("Level_5") == "p2p transfer":
                result["Level_5"] = ""
        else:
            purpose = result.get("Level_5", "")
            if (
                purpose
                and purpose not in {"p2p transfer", "salary"}
                and result.get("Level_3") != "institution"
            ):
                result["Level_3"] = "merchant"
            elif (
                result.get("Level_3") not in {"institution", "merchant"}
                and purpose != "salary"
                and not (SALARY_KEYWORDS.search(text) and not is_debit)
                and not automated_debit
            ):
                if name in UPI_PLACEHOLDER_PAYEES:
                    # No identifiable payee: nothing supports a person-to-person
                    # claim, so withdraw an inferred p2p label instead of
                    # asserting one.
                    if result.get("Level_3") == "p2p":
                        result["Level_3"] = ""
                    if purpose == "p2p transfer":
                        result["Level_5"] = ""
                else:
                    result["Level_3"] = "p2p"
                    result["Level_5"] = "p2p transfer"

    if MERCHANT_MARKER_RE.search(marker_text):
        result["Level_3"] = "merchant"
        if result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    # Slash UPI Level 6: the Master Sheet holds the clean brand name, so for a
    # merchant or institution row it is kept. A p2p row (or a blank / generic
    # Level 6) takes the payee name from the narration.
    if match and name != "merchant":
        keep_master_name = (
            result.get("Level_3") in {"merchant", "institution"}
            and bool(result.get("Level_6"))
            and not is_generic_level6(result["Level_6"])
        )
        if not keep_master_name:
            result["Level_6"] = name

    if SALARY_KEYWORDS.search(text) and not is_debit:
        entity = extract_level6_entity(text)
        if not entity:
            entity = SALARY_KEYWORDS.sub("", text)
            entity = re.sub(r"[\-/|]+", " ", entity)
            entity = re.sub(r"\s+", " ", entity).strip().lower()

        result["Level_5"] = "salary"
        result["Level_6"] = clean_salary_entity(entity)

    # Level 4 platform first: the narration's platform or gateway (RAZORPAYZOMATO,
    # "rzp icici") is kept as Level 4, and a bank found in the narration never
    # replaces a platform already in Level 4.
    detected_level4 = detect_level_4_platform_or_bank(marker_text)
    if detected_level4 and not (
        _is_platform_label(result["Level_4"]) and not _is_platform_label(detected_level4)
    ):
        result["Level_4"] = detected_level4

    upper_text = text.upper()
    parts = upper_text.split("-", 2)
    first_part = parts[1].strip() if len(parts) > 1 else ""
    merchant_key = re.sub(r"[^a-z0-9]", "", first_part.lower())

    merchant_name = KNOWN_UPI_MERCHANTS.get(merchant_key)
    is_paytm_qr = "PAYTMQR" in upper_text

    if upper_text.startswith("UPI-") and (merchant_name or is_paytm_qr):
        result["Level_3"] = "merchant"

        if merchant_name:
            result["Level_6"] = merchant_name
        elif not result.get("Level_6"):
            result["Level_6"] = first_part.lower()

        if result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    # A merchant-only UPI ID (PhonePe business QR, BharatPe, Paytm merchant) is
    # direct evidence of a merchant payment, whatever the payee name looks like.
    if has_merchant_vpa(text) and result.get("Level_3") != "institution":
        result["Level_3"] = "merchant"
        if result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    if WALLET_TOPUP_RE.search(marker_text) and result.get("Level_3") != "institution":
        result["Level_3"] = "merchant"
        result["Level_5"] = WALLET_TOPUP_LEVEL5

    # A platform whose app IS the counterparty (slice) keeps its name as Level 6 when the
    # payee field gives it; Level 6 is cleared only for a name read from a bank code or handle.
    payee_field = _normalize_compare(match.group(1) if match else first_part)
    payee_platform = payee_field if payee_field in PAYEE_PLATFORM_LEVEL6 else ""
    if payee_platform and result.get("Level_3") == "merchant" and not result.get("Level_6"):
        result["Level_6"] = payee_platform

    fill_specific_level6(text, result, part_texts)

    return _apply_transaction_rules(
        result,
        default_p2p_purpose=not automated_debit,
        confirmed_level6=extract_level6_entity(text) or payee_platform,
    )


def apply_atm_withdrawal_overrides(
    particulars: str,
    part_predictions: list[tuple[str, dict[str, str]]],
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
) -> dict[str, str]:
    """Mark ATM cash withdrawals as cash and drop contradictory P2P labels."""

    text = _marker_text(particulars, [part_text for part_text, _ in part_predictions])
    if not ATM_WITHDRAWAL_RE.search(text):
        return result
    if derive_direction_from_amounts(debit_value, credit_value) == "credit":
        return result

    bank_name = result["Level_4"]
    if not bank_name:
        bank_name = detect_level_4_platform_or_bank(text)
    for _, prediction in part_predictions:
        if not bank_name and prediction.get("Level_4"):
            bank_name = prediction["Level_4"]

    result["Level_1"] = "cash"
    result["Level_3"] = ""
    result["Level_5"] = ""
    result["Level_4"] = bank_name
    # Level 6 is a brand or person. A cash withdrawal has neither: the ATM
    # ("ICICI BANK ATM") is the bank, already in Level 4, and terminal or
    # location codes ("S1AWMN16-KULLU") are not entities.
    result["Level_6"] = ""

    return result


# Level 3 evidence that the counterparty cannot be a person.
#   Bank charges: "RTN CHG-...", "IMPS Chg May-25+GST", "Chq rtn Chg ..."
BANK_CHARGE_RE = re.compile(r"^\s*(?:[a-z]+\s+){0,3}(?:chg|chgs|charges?)\b", flags=re.IGNORECASE)
#   Bank fee lines BANK_CHARGE_RE misses: "ATM / IMPS TRANSACTION CHARGES", "SMS CHARGES",
#   "IMPS SC CHQ" and the reversal "REV PART ATM / IMPS TRANSACTION CHARGE".
BANK_FEE_NARRATION_RE = re.compile(
    r"^\s*(?:REV(?:ERSAL)?\s+(?:PART\s+)?)?"
    r"(?:ATM\s*/\s*IMPS\s+TRANSACTION\s+CHARGES?|SMS\s+CHARGES?|IMPS\s+SC\s+CHQ)\b",
    flags=re.IGNORECASE,
)
#   Government payee: ".../CHALLAN 280 TAX/GOVT/", ".../PF CONTRIBUTION/GOVT/"
GOVERNMENT_PAYEE_RE = re.compile(r"(?:^|/)\s*GOVT\s*(?:/|$)|\bCHALLAN\b", flags=re.IGNORECASE)
#   Mandate debits (NACH / ECS / auto-debit / standing instruction). Mandates are
#   registered by institutions and companies, never by an individual.
MANDATE_DEBIT_RE = re.compile(
    r"\b(?:E?NACH|ECS|ACH|ACHDR|AUTO[\s-]?DEBIT|AUTODEBIT|AUTO[\s-]?PAY|AUTOPAY|"
    r"STANDING\s+INSTRUCTION|SI[\s-]?(?:DEBIT|MAD)|UPI\s+MANDATE|MANDATE)\b",
    flags=re.IGNORECASE,
)
#   Bulk corporate credits ("ACH C- TCSFINDIV...", "ACH C- IRFC LTD-..."):
#   dividends, interest and refunds paid by companies, never by a person.
BULK_CORPORATE_CREDIT_RE = re.compile(r"\b(?:ACH|NACH)\s*CR?\b", flags=re.IGNORECASE)
INSTITUTIONAL_PURPOSE_RE = re.compile(
    r"\b(?:loan|emi|sip|mutual\s*fund|mf|insurance|premium|policy|credit\s*card|cc|tax|challan|gst)\b",
    flags=re.IGNORECASE,
)
INSTITUTIONAL_LEVEL5 = frozenset(
    {"loan repayment", "credit card", "insurance", "financial services", "investment", "tax", "dividend"}
)
# Card payments ("ECOM PUR/...", "POS 4160...", "ME DC SI ...") are always paid
# to a merchant, never to a person.
CARD_PURCHASE_RE = re.compile(r"^\s*(?:ECOM\s+PUR|POS|ME\s+DC\s+SI)\b", flags=re.IGNORECASE)


def apply_counterparty_type_override(
    particulars: str,
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
    part_texts: list[str] | None = None,
) -> dict[str, str]:
    """Correct a P2P (or blank) Level 3 when the narration shows a non-person counterparty.

    - bank charges -> institution (the bank)
    - government payee (GOVT / CHALLAN) -> institution
    - mandate debit (NACH/ECS/auto-debit/SI) labelled p2p -> institution when the
      purpose is institutional (loan, EMI, SIP, insurance, card, tax), otherwise
      blank: a mandate is never person-to-person, but the type is not stated.
    - ACH/NACH credit -> institution (a company's dividend/interest payout).
    - card purchase (ECOM PUR / POS / ME DC SI) -> merchant.
    A merchant or institution label already on the row is never changed. When
    Level 3 stops being p2p, a "p2p transfer" purpose is removed.
    """

    level_3 = result.get("Level_3", "")
    if level_3 not in {"p2p", ""}:
        return result

    text = str(particulars or "")
    marker_text = _marker_text(text, part_texts)
    direction = derive_direction_from_amounts(debit_value, credit_value)
    is_debit = direction == "debit"

    new_level_3 = None
    if CARD_PURCHASE_RE.match(text):
        new_level_3 = "merchant"
    elif BANK_CHARGE_RE.match(text) or GOVERNMENT_PAYEE_RE.search(text):
        new_level_3 = "institution"
    elif direction == "credit" and BULK_CORPORATE_CREDIT_RE.search(text):
        new_level_3 = "institution"
    elif is_debit and MANDATE_DEBIT_RE.search(marker_text):
        institutional = bool(INSTITUTIONAL_PURPOSE_RE.search(marker_text)) or (
            _normalize_compare(result.get("Level_5", "")) in INSTITUTIONAL_LEVEL5
        )
        if institutional:
            new_level_3 = "institution"
        elif level_3 == "p2p":
            new_level_3 = ""

    if new_level_3 is None or new_level_3 == level_3:
        return result
    result["Level_3"] = new_level_3
    if result.get("Level_5") == "p2p transfer":
        result["Level_5"] = ""
    return result


# Narration-start patterns for the LABEL_PHRASES that imply a Level 3 / Level 5, longest first.
LABEL_IMPLIED_RULES: tuple[tuple[re.Pattern, tuple[str, str, str]], ...] = tuple(
    (re.compile(r"^\s*" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"(?![a-z])", re.IGNORECASE), implied)
    for phrase, implied in sorted(LABEL_PHRASES.items(), key=lambda item: -len(item[0]))
    if implied[0] or implied[1]
)
# A business name hint: legal words, proprietor tags and common business stems.
BUSINESS_NAME_STEM_RE = re.compile(
    r"\b(?:prop|proprietor|\w*infra\w*|\w*beverage\w*|\w*industr\w*|\w*enterpri\w*|\w*traders?"
    r"|\w*stores?|\w*collect\w*)\b",
    flags=re.IGNORECASE,
)


def _name_has_business_evidence(name: str) -> bool:
    """A business word, a business stem or a purpose pattern on the name itself."""

    return bool(
        BUSINESS_WORD_RE.search(name)
        or BUSINESS_NAME_STEM_RE.search(name)
        or _has_long_business_word(name)
        or _matched_purposes(PLATFORM_NAME_RE.sub(" ", name), NAME_EVIDENCE_EXCLUDED_PURPOSES)
    )


def apply_label_layout_override(
    particulars: str,
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
    master_locked: bool = False,
) -> dict[str, str]:
    """Settle rows whose layout alone decides the counterparty type.

    - LABEL_PHRASES that imply labels (Loan Recovery For, Charges for, Locker, Vetting
      Fees ...): the Level 3 / Level 5 in the table, Level 6 blank.
    - MOBFT from|to: <PERSON NAME>/<ref>: p2p / p2p transfer. A business-looking
      name, or one the row already settled as merchant or institution, is left alone.
    ``master_locked`` is True when an exact Master Sheet match set Level 3: nothing here
    changes it.
    """

    if master_locked:
        return result
    text = str(particulars or "")
    direction = derive_direction_from_amounts(debit_value, credit_value)
    for pattern, (level_3, level_5, directions) in LABEL_IMPLIED_RULES:
        if pattern.match(text):
            if directions == "debit" and direction == "credit":
                return result
            result.update({"Level_3": level_3, "Level_5": level_5, "Level_6": ""})
            return result
    named = MOBFT_NAME_RE.match(text)
    if named and result.get("Level_3") in {"", "p2p"} and not _name_has_business_evidence(named.group(1)):
        result["Level_3"] = "p2p"
        result["Level_5"] = "p2p transfer"
    return result


def _compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _drop_tail_exact_matches(
    particulars: str, row_parts: list[tuple[str, dict[str, str]]]
) -> list[tuple[str, dict[str, str]]]:
    """In a "<RAILPREFIX>-<NAME> <ref> UTR ... Beneficiary Bank/Branch ..." row the payee is the
    text right after the rail word. An exact Master match on a part that does not contain the
    payee (a branch or city fragment such as "NEW DELHI") describes the tail, not the
    counterparty, so it loses its exact-match priority."""

    named = HYPHEN_PREFIX_NAME_RE.match(str(particulars or ""))
    if not named:
        return row_parts
    payee = _compact_key(_clean_name_text(named.group(1)))
    if not payee:
        return row_parts
    return [
        (text, {k: v for k, v in prediction.items() if k != EXACT_MASTER_KEY})
        if prediction.get(EXACT_MASTER_KEY) and payee not in _compact_key(text)
        else (text, prediction)
        for text, prediction in row_parts
    ]


def apply_rail_payee_override(
    particulars: str,
    result: dict[str, str],
    master_locked: bool = False,
) -> dict[str, str]:
    """Counterparty type for "<RAILPREFIX>-<NAME> <ref> UTR ..." rows (RTGSO-, NEFTO- ...).

    A merchant needs evidence in the name (business word / stem, or a purpose pattern).
    Without it the payee is a person: Level 3 p2p, Level 5 p2p transfer, never retail.
    With it, Level 3 is merchant and Level 5 is kept only when a purpose pattern on the
    name supports it, otherwise blank. Institutions and exact Master matches are left alone.
    """

    if master_locked:
        return result
    named = HYPHEN_PREFIX_NAME_RE.match(str(particulars or ""))
    if not named or result.get("Level_3") == "institution" or result.get("Level_5") == "salary":
        return result
    name = named.group(1)
    # The payee is the text right after the rail word; bank, branch, IFSC and account
    # text further on never names the counterparty.
    payee = guard_level6(_clean_name_text(name), particulars)
    if payee:
        result["Level_6"] = payee
    if _name_has_business_evidence(name):
        supported = _matched_purposes(PLATFORM_NAME_RE.sub(" ", name), NAME_EVIDENCE_EXCLUDED_PURPOSES)
        result["Level_3"] = "merchant"
        if result.get("Level_5") not in supported:
            result["Level_5"] = _most_specific_purpose(supported)
    else:
        result["Level_3"] = "p2p"
        result["Level_5"] = "p2p transfer"
    return result


def apply_initiation_mode_override(
    particulars: str,
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
    part_texts: list[str] | None = None,
) -> dict[str, str]:
    """Classify Level 2 when transaction narration provides clear evidence."""

    text = _marker_text(particulars, part_texts)
    direction = derive_direction_from_amounts(debit_value, credit_value)

    if SYSTEM_INITIATED_RE.search(text):
        result["Level_2"] = "System generated"
        return result

    if SALARY_KEYWORDS.search(text) and direction == "credit":
        result["Level_2"] = "System generated"
        return result

    if direction == "debit":
        if (
            ATM_WITHDRAWAL_RE.search(text)
            or EXPLICIT_UPI_RE.search(text)
            or MANUAL_TRANSFER_RE.search(text)
        ):
            result["Level_2"] = "User-generated"

    return result


# A cheque instrument narration: <CLG|TRF>/<drawer>/<6-digit cheque no>/<drawee
# bank code>/<dd.mm.yyyy>, e.g. "TRF/SUNRAY ENGINEERING/000757/ICI/10.06.2025".
# "TRF/<name>/ICI" without a cheque number and date is a plain transfer and
# does not match.
CHEQUE_INSTRUMENT_RE = re.compile(
    r"^\s*(?:CLG|TRF)/[^/]+/(\d{6})/[A-Z]{2,4}/\d{2}\.\d{2}\.\d{4}\s*$",
    flags=re.IGNORECASE,
)
# A returned instrument: "REJECT:<cheque no>:<return reason>".
CHEQUE_RETURN_RE = re.compile(r"^\s*REJECT:(\d{1,6}):", flags=re.IGNORECASE)


# (pattern, Level 1) tried in order against the narration, only when Level 1 is blank.
# An optional leading account / reference number ("50100047610192-TPT-...") is skipped.
# Adding a bank's prefix is one row here; no code change.
# A payment-gateway narration (PAYU...) gets the rail it spells out, only if exactly one.
GATEWAY_NARRATION_RE = re.compile(r"(?<![a-z])payu", re.IGNORECASE)
GATEWAY_RAIL_WORDS = ("upi", "imps", "neft", "rtgs")


def apply_level1_structural_fill(particulars: str, result: dict[str, str]) -> dict[str, str]:
    """Fill a blank Level 1 from structural prefixes. Never changes an existing value."""

    if result.get("Level_1"):
        return result
    text = str(particulars or "").strip()
    for pattern, value in LEVEL1_STRUCTURAL_RULES:
        if pattern.search(text):
            result["Level_1"] = value
            return result
    if GATEWAY_NARRATION_RE.search(text):
        rails = [w for w in GATEWAY_RAIL_WORDS if re.search(rf"(?<![a-z]){w}(?![a-z])", text, re.IGNORECASE)]
        if len(rails) == 1:
            result["Level_1"] = rails[0]
    return result


def apply_cheque_instrument_override(particulars: str, result: dict[str, str]) -> dict[str, str]:
    """Set Level_1 to "cheque" for a full cheque-instrument narration.

    The full instrument layout (CLG|TRF/<drawer>/<cheque no>/<bank>/<date>) is
    unambiguous, so it overrides a Level_1 from the Master Sheet.
    """

    if CHEQUE_INSTRUMENT_RE.match(str(particulars or "")):
        result["Level_1"] = "cheque"
    return result


def apply_cheque_return_level1(df: pd.DataFrame) -> pd.DataFrame:
    """Mark "REJECT:<no>:..." rows as cheque when <no> is a cheque in the statement.

    A REJECT line alone could be a cheque or a mandate return, so it is labelled
    only when its number matches the cheque number of a CLG/TRF cheque
    instrument in the same statement (per ``Source_File`` when present). Only
    blank Level_1 values are filled.
    """

    if "Particulars" not in df.columns or "Level_1" not in df.columns:
        return df

    output_df = df.copy()
    text = output_df["Particulars"].fillna("").astype(str)
    groups = output_df["Source_File"].fillna("") if "Source_File" in output_df.columns else pd.Series("", index=output_df.index)
    level_1 = output_df["Level_1"].fillna("").astype(str).str.strip()

    cheque_numbers: dict[str, set[str]] = defaultdict(set)
    for index, value in text.items():
        match = CHEQUE_INSTRUMENT_RE.match(value)
        if match:
            cheque_numbers[groups[index]].add(match.group(1).lstrip("0"))

    for index, value in text.items():
        match = CHEQUE_RETURN_RE.match(value)
        if match and not level_1[index] and match.group(1).lstrip("0") in cheque_numbers[groups[index]]:
            output_df.at[index, "Level_1"] = "cheque"
    return output_df


def apply_amount_direction_override(
    debit_value: object,
    credit_value: object,
    result: dict[str, str],
) -> dict[str, str]:
    """Set direction from debit and credit columns when it is explicit."""

    direction = derive_direction_from_amounts(debit_value, credit_value)
    if direction:
        result["Level_0"] = direction
    return result


# ---------------------------------------------------------------------------
# One entity, one name: statement-level canonical Level 6 and Level 3 per entity.
# ---------------------------------------------------------------------------
ENTITY_MERGE_THRESHOLD = 0.8
ENTITY_PREFIX_MIN_LETTERS = 6
# Legal-form words and titles: ignored when two names are compared, never part of the identity.
LEGAL_FORM_WORDS = frozenset({"ltd", "limited", "pvt", "private", "llp", "inc"})
_NAME_NOISE_WORDS = LEGAL_FORM_WORDS | TITLE_WORDS
IFSC_EVIDENCE_RE = re.compile(r"\b([A-Z]{4})\s?(0[A-Z0-9]{6})\b")
VPA_HANDLE_RE = re.compile(r"@\s*([A-Za-z][A-Za-z ]{1,24})")
BUSINESS_HANDLE_RE = re.compile(r"@\s*ok\s*biz", flags=re.IGNORECASE)
LOAN_PHRASE_RE = re.compile(r"\bloan\s+(?:disburse\w*|repay\w*|closure|emi|interest)|\bemi\b", flags=re.IGNORECASE)
P2M_TAG_RE = re.compile(r"/P2M/", flags=re.IGNORECASE)
P2A_TAG_RE = re.compile(r"/P2A/", flags=re.IGNORECASE)
ACH_RAIL_RE = re.compile(r"^\s*\.?ACH\s+[DC]\s*-", flags=re.IGNORECASE)
# Structural evidence for Level 3, strongest first (Master matches rank above all of these).
STRUCTURAL_L3_ORDER = ("p2m_tag", "p2a_tag", "ach_rail", "business_handle", "loan_phrase")
STRUCTURAL_L3_VALUE = {
    "p2m_tag": "merchant",
    "p2a_tag": "p2p",
    "ach_rail": "institution",
    "business_handle": "merchant",
    "loan_phrase": "institution",
}


def _name_tokens(name: str) -> list[str]:
    tokens = [t for t in str(name or "").split() if t not in _NAME_NOISE_WORDS]
    return tokens or str(name or "").split()


def entity_merge_score(first: str, second: str) -> float:
    """How surely two Level 6 names are one entity: 1.0 equal token sets; prefix scores by length ratio."""

    a, b = _name_tokens(first), _name_tokens(second)
    if not a or not b:
        return 0.0
    if frozenset(a) == frozenset(b):
        return 1.0
    la, lb = "".join(a), "".join(b)
    short, long_ = (la, lb) if len(la) <= len(lb) else (lb, la)
    if len(short) >= ENTITY_PREFIX_MIN_LETTERS and long_.startswith(short):
        return 0.6 + 0.4 * len(short) / len(long_)
    return 0.0


def _row_evidence(row: pd.Series) -> dict[str, str]:
    """Bank, IFSC and VPA handle of a row: facts that tell two same-looking names apart."""

    text = str(row.get("Particulars", "") or "")
    evidence: dict[str, str] = {}
    bank = ""
    for rule in LAYOUT_RULES:
        found = rule.pattern.match(text) if rule.bank_group else None
        if found:
            bank = _letters(found.group(rule.bank_group))  # the template's own bank field
            break
    if len(bank) < 4:
        level4 = str(row.get("Level_4", "") or "").strip().lower()
        bank = _letters(level4) if level4 and not _is_platform_label(level4) else ""
    if bank:
        evidence["bank"] = bank
    ifsc = IFSC_EVIDENCE_RE.search(text.upper())
    if ifsc:
        evidence["ifsc"] = ifsc.group(1) + ifsc.group(2)
    handle = VPA_HANDLE_RE.search(text)
    if handle:
        evidence["vpa"] = re.sub(r"[^a-z]", "", handle.group(1).lower())
    return evidence


def _bank_values_agree(left: set[str], right: set[str]) -> bool:
    """Bank fields are cut at different widths ("bankofi", "bankofindia"): a shared prefix is agreement."""

    return any(a.startswith(b) or b.startswith(a) for a in left for b in right)


def _evidence_conflict(left: dict[str, set[str]], right: dict[str, set[str]]) -> bool:
    for kind in ("bank", "ifsc", "vpa"):
        if not (left.get(kind) and right.get(kind)):
            continue
        agree = _bank_values_agree(left[kind], right[kind]) if kind == "bank" else bool(left[kind] & right[kind])
        if not agree:
            return True
    return False


def _statement_vocabulary(names: list[str]) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    for name in names:
        for token in set(name.split()):
            counts[token] += 1
    return {token for token, count in counts.items() if count >= 2 and len(token) >= 4}


def _entity_clusters(df: pd.DataFrame, rows: list) -> list[list]:
    """Group rows into entities: merge compatible names that no evidence contradicts."""

    nodes: dict[tuple[str, str], dict] = {}
    for idx in rows:
        row = df.loc[idx]
        name = str(row.get("Level_6", "") or "")
        if not name:
            continue
        evidence = _row_evidence(row)
        node = nodes.setdefault((name, evidence.get("bank", "")), {"name": name, "rows": [], "ev": defaultdict(set)})
        node["rows"].append(idx)
        for kind, value in evidence.items():
            node["ev"][kind].add(value)
    items = list(nodes.values())
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cluster_ev = [{k: set(v) for k, v in item["ev"].items()} for item in items]
    # Candidate pairs without an all-pairs scan: equal token sets share a key; a prefix
    # pair sits next to each other in the sorted letter strings.
    candidates: set[tuple[int, int]] = set()
    by_tokens: dict[frozenset, list[int]] = defaultdict(list)
    spelled: list[tuple[str, int]] = []
    for i, item in enumerate(items):
        tokens = _name_tokens(item["name"])
        by_tokens[frozenset(tokens)].append(i)
        spelled.append(("".join(tokens), i))
    for members in by_tokens.values():
        candidates.update((a, b) for pos, a in enumerate(members) for b in members[pos + 1 :])
    spelled.sort()
    keys = [letters for letters, _ in spelled]
    for pos, (letters, i) in enumerate(spelled):
        if len(letters) < ENTITY_PREFIX_MIN_LETTERS:
            continue
        end = bisect.bisect_left(keys, letters + "\uffff", pos + 1)
        for _, j in spelled[pos + 1 : end]:
            candidates.add((min(i, j), max(i, j)))
    pairs = []
    for i, j in candidates:
        score = entity_merge_score(items[i]["name"], items[j]["name"])
        if score >= ENTITY_MERGE_THRESHOLD:
            pairs.append((score, i, j))
    for _, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        ri, rj = find(i), find(j)
        if ri == rj or _evidence_conflict(cluster_ev[ri], cluster_ev[rj]):
            continue
        parent[rj] = ri
        for kind, values in cluster_ev[rj].items():
            cluster_ev[ri].setdefault(kind, set()).update(values)
    groups: dict[int, list] = defaultdict(list)
    for i, item in enumerate(items):
        groups[find(i)].extend(item["rows"])
    return list(groups.values())


def apply_canonical_level6(df: pd.DataFrame, named_rows: set | None = None) -> pd.DataFrame:
    """Give every row of one entity the same Level 6: the longest full name seen.

    Only rows whose Level 6 came from a layout name field (``named_rows``) take
    part; Master names are never touched. Two names merge when their token sets
    are equal, or one is a prefix of the other (6 letters or more), the merge
    score reaches ``ENTITY_MERGE_THRESHOLD`` and no bank, IFSC or VPA evidence
    differs. Split words are rejoined from the statement's own vocabulary
    ("m ohd" -> "mohd" when "mohd" appears intact elsewhere).
    """

    output_df = df.copy()
    rows = [idx for idx in output_df.index if (named_rows is None or idx in named_rows) and str(output_df.at[idx, "Level_6"] or "")]
    if not rows:
        return output_df
    vocabulary = _statement_vocabulary([str(output_df.at[idx, "Level_6"]) for idx in rows])
    for idx in rows:
        name = str(output_df.at[idx, "Level_6"])
        rejoined = " ".join(rejoin_wrapped_words(name.split(), vocabulary))
        if rejoined != name:
            output_df.at[idx, "Level_6"] = rejoined
    for cluster in _entity_clusters(output_df, rows):
        names = {str(output_df.at[idx, "Level_6"]) for idx in cluster}
        if len(names) < 2:
            continue
        counts = {n: sum(str(output_df.at[idx, "Level_6"]) == n for idx in cluster) for n in names}
        display = max(names, key=lambda n: (len(_letters(n)), counts[n], len(n.split()), n))
        for idx in cluster:
            output_df.at[idx, "Level_6"] = display
    return output_df


def _structural_l3_tag(text: str, level1: str, name: str) -> str:
    if P2M_TAG_RE.search(text):
        return "p2m_tag"
    if P2A_TAG_RE.search(text):
        # An account-to-account transfer says nothing about a business-named account.
        company = bool(set(name.split()) & COMPANY_SUFFIX_WORDS) or _name_has_business_evidence(name)
        return "" if company else "p2a_tag"
    if ACH_RAIL_RE.match(text) or level1 == "nach":
        return "ach_rail"
    if BUSINESS_HANDLE_RE.search(text):
        return "business_handle"
    if LOAN_PHRASE_RE.search(text) and (set(name.split()) & COMPANY_SUFFIX_WORDS or _name_has_business_evidence(name)):
        return "loan_phrase"
    return ""


def apply_entity_level3(
    df: pd.DataFrame,
    named_rows: set | None = None,
    locked_rows: set | None = None,
) -> pd.DataFrame:
    """Make Level 3 agree across the rows of one entity, evidence first.

    Order: a Master match, then structural evidence (P2M / P2A tag, ACH rail,
    business VPA handle, loan phrase), then a company suffix, which is only a
    weak hint: it may choose between values the entity already has but never
    replaces a value every row agrees on. When nothing decides, rows keep their
    value. Master-matched rows are never changed.
    """

    output_df = df.copy()
    locked = locked_rows or set()
    rows = [idx for idx in output_df.index if (named_rows is None or idx in named_rows) and str(output_df.at[idx, "Level_6"] or "")]
    for cluster in _entity_clusters(output_df, rows):
        name = str(output_df.at[cluster[0], "Level_6"])
        master = [str(output_df.at[i, "Level_3"] or "") for i in cluster if i in locked and output_df.at[i, "Level_3"]]
        tags = {}
        for i in cluster:
            if i in locked:
                continue
            tag = _structural_l3_tag(
                str(output_df.at[i, "Particulars"] or ""), str(output_df.at[i, "Level_1"] or "").lower(), str(output_df.at[i, "Level_6"])
            )
            if tag:
                tags[tag] = tags.get(tag, 0) + 1
        existing = [str(output_df.at[i, "Level_3"] or "") for i in cluster if i not in locked and output_df.at[i, "Level_3"]]
        value = ""
        if master:
            value = max(set(master), key=master.count)
        elif tags:
            value = STRUCTURAL_L3_VALUE[next(tag for tag in STRUCTURAL_L3_ORDER if tag in tags)]
        elif existing and set(existing) != {"p2p"} and "p2p" in existing:
            company = any(set(str(output_df.at[i, "Level_6"]).split()) & COMPANY_SUFFIX_WORDS for i in cluster)
            if company:
                others = [v for v in existing if v != "p2p"]
                value = max(set(others), key=others.count)
        elif existing and len(set(existing)) == 1 and len(existing) < len(cluster):
            value = existing[0]  # fill the blanks of a cluster whose rows all agree
        if not value:
            continue
        for i in cluster:
            if i in locked or str(output_df.at[i, "Level_5"] or "") == "salary" or str(output_df.at[i, "Level_1"] or "") == "cash":
                continue
            if str(output_df.at[i, "Level_3"] or "") == value:
                continue
            output_df.at[i, "Level_3"] = value
            if value in {"merchant", "institution"} and str(output_df.at[i, "Level_5"] or "") == "p2p transfer":
                output_df.at[i, "Level_5"] = ""
    return output_df


def extract_salary_entity(text: str) -> str:
    """Extract a likely salary source string from particulars."""

    value = str(text or "")
    value = SALARY_ENTITY_STOPWORDS.sub(" ", value)
    value = re.sub(r"[\d/|_\-*]+", " ", value)
    value = re.sub(r"\b(?:neft|imps|upi|rtgs|ecs|nach|bank|payment|transfer)\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def clean_salary_entity(value: str) -> str:
    """Return the salary payer name only, for Level 6 of a salary row.

    Removes the salary words, "for / of / the / month", a month name that follows
    them ("salary for dec"), rail words, reference numbers and x-masks, and a
    stray one- or two-letter tail left after a salary word ("labour an").
    "ctqgg a salary for dec" gives "ctqgg a".
    """

    text = SPLIT_SALARY_RE.sub("salary", str(value or "").lower())
    kept: list[str] = []
    prev_filler = False
    prev_salary_word = False
    for token in (item for item in re.split(r"[^a-z0-9]+", text) if item):
        if token in SALARY_FILLER_WORDS:
            prev_filler = True
            prev_salary_word = token not in {"for", "of", "the", "month", "credit", "credited", "via", "by", "from", "to"}
            continue
        if token in SALARY_MONTH_WORDS and prev_filler:
            continue
        if (
            re.search(r"\d", token)
            or re.fullmatch(r"x{2,}", token)
            or token in {"neft", "imps", "upi", "rtgs", "ecs", "nach", "bank", "payment", "transfer"}
            or (prev_salary_word and len(token) <= 2)
        ):
            prev_filler = prev_salary_word = False
            continue
        kept.append(token)
        prev_filler = prev_salary_word = False
    name = " ".join(kept)
    return "" if is_stop_value(name) else name


class FirstLevelRule(NamedTuple):
    """One First Level rule: a narration pattern and the label for each direction ("" = no label)."""

    name: str
    pattern: re.Pattern
    debit: str
    credit: str


def _wrap_safe(*words: str) -> str:
    """Alternation of words that tolerates a line-wrap space inside a word ("ASSURA NCE")."""

    return "|".join(r"\s?".join(re.escape(ch) for ch in word) for word in words)


def _fl(name: str, pattern: str, debit: str, credit: str = "") -> FirstLevelRule:
    return FirstLevelRule(name, re.compile(pattern, flags=re.IGNORECASE), debit, credit)


# First Level Classification is compared with the reference statement analysis, so the label set
# follows it: "<rail> Payment" / "<rail> Receipt" plus purpose labels. The first rule whose pattern
# matches and that has a label for the row's direction wins. Order is the priority: specific
# purposes first, then the payment rail. A new bank prefix or keyword needs one row here.
FIRST_LEVEL_RULES: tuple[FirstLevelRule, ...] = (
    # --- returns, reversals, charges, interest ---------------------------------------
    _fl("cheque_return", r"^\s*REJECT\b", "Outward Cheque Return"),
    _fl("investment", r"ZERODHA|DHANI\s+STOCKS|MARGADARSI|\bCHITS?\b|^\s*IB\s+FD\b|FD\s+PREMAT|SUPERMON|MUTUAL\s+FUND|\bSIP\b|BROKING|\bDEMAT\b|GROWW|UPSTOX", "Investment", "Investment"),
    _fl("bank_charges", r"^\s*(?:RTN|RETURN)\s+CHG|\bECS\b.*\bSC\b|^(?!\s*(?:(?:TO|BY)\s+[^-]*-\s*)?UPI).*(?:\bCHR?GS?\b|\bCHARGES?\b|\bSMS\s+ALERT|\bPENAL|\bFEES?\b)", "Bank Charges"),
    _fl("interest", r"\bINT\.?\s*(?:PD|COLL)\b|:INT\.|\bINTE?RE?ST\b|\bINTREST\b|\bSBINT\b|\bINT\s+ON\b|\bINT:", "Interest Paid", "Interest Received"),
    _fl("online_return_in", r"^\s*UPIRET|^\s*REV-UPI|RTGS\s+RETURN|\bRETURN\b.*\b(?:RTGS|NEFT|UPI)\b", "", "Inward Online Return"),
    _fl("refund", r"REFUND|REVERSING|REVERSAL|^\s*REV\b|^\s*CRV\b|SUBSIDY|^\s*APBS\b", "", "Refund/Reversal"),
    _fl("online_return_out", r"^(?!\s*(?:IMPS|NEFT|RTGS))(?!.*LOAN).*(?:\bRETURN(?:ED)?\b|\bREFUND)", "Outward Online Return"),
    _fl("salary", r"\bSA\s?LARY\b", "Salary", "Salary"),
    _fl("bill_statement", r"^\s*TO\s+BILL\s+PAYMENT", "Bill Payment"),
    _fl("credit_card", r"CRED\s?CLUB|CRED\.CLUB|CREDIT[\s_]?CARD|MOBCC|^\s*CC\s+\d|BILLDK\w*CARDS|\bAUTOPAY\b.*\bSI\b", "Credit Card Payment"),
    _fl("insurance", _wrap_safe("INSURANCE", "ASSURANCE", "PREMIUM") + r"|\bLIC\b|PMSBY|PMJJBY|HDFCERGO|GODIGIT|SUNLIFE", "Insurance", "Insurance"),
    _fl("rent", r"\bRENT\b|RENTL\b", "Rent Payment"),
    _fl("tax", r"^\s*GIB/.*(?:GST|TAX)|\bITDTAX|\bTAX\b", "Tax", "Tax"),
    _fl("forex", r"^\s*INW\b|\bUSD\d|\bFOREX\b", "Forex/Remittances", "Forex/Remittances"),
    # --- mandates and loans -----------------------------------------------------------
    _fl("ach", r"^\s*\.?ACH\s*[DC]\b|\bNACH\b|\bECS\b|^\s*CMS/|\bEMIS?\b", "ECS/NACH Payment", "ECS/NACH Receipt"),
    # A loan word on a UPI line is a remark to a person ("UPI Payment"); on other rails it is a loan.
    _fl("loan_on_bank_rail", r"^(?!\s*(?:(?:TO|BY)\s+[^-]*-\s*)?UPI)(?:.*(?:LOAN|" + _wrap_safe("REPAYMENT") + r"))", "Loan", "Loan"),
    _fl("loan_account", r"\bLOAN\s+ACC", "Loan", "Loan"),
    _fl("rtgs_in_transfer", r"INFT/.*\bRTGS\b", "RTGS Payment", "RTGS Receipt"),
    _fl("neft_in_transfer", r"INFT/.*\bNEFT\b", "NEFT Payment", "NEFT Receipt"),
    # --- utilities (telecom, fuel, bill payments) --------------------------------------
    _fl("utility", r"^(?!\s*(?:IMPS|NEFT|RTGS))(?:.*)(?:AIRTEL(?!\s*PAYMENTS)|PETROL|FILLING\s?ST|FUEL(?!\s*ST)|FREECHARGE|" + _wrap_safe("POSTPAID", "ELECTRICITY", "RECHARGE", "BROADBAND") + r"|TATA\s?(?:PLAY|SKY)|\bBSNL\b|VODAFONE|\bIOCL\b|\bBPCL\b|\bHPCL\b)", "Utility Expenses", "Utility Expenses"),
    # --- cash, card, wallet ------------------------------------------------------------
    _fl("cash_return", r"^\s*CASH-REVERSAL", "", "Cash Return"),
    _fl("atm_statement", r"^\s*TO\s+ATM\b", "ATM Withdrawal"),
    _fl("cash_out", r"^\s*(?:ATM|ATW|NWD|EAW|CWDR|SELF)\b|\bCASH\s+WITHDRAWAL", "Cash Withdrawal"),
    _fl("cash_in", r"^\s*(?:CASH|CAM)\b|\bCASH\s+DEP|^\s*BY\s+CASH", "", "Cash Deposit"),
    _fl("pos", r"^\s*(?:POS|ECOM\s+PUR)\b", "POS Expense"),
    _fl("wallet", r"^\s*DRP/|^\s*MCRM\d", "Other Payment", "Other Receipt"),
    _fl("me_debit", r"^\s*ME\s+DC\b", "Other Payment"),
    # --- payment rail ------------------------------------------------------------------
    _fl("upi", r"^\s*(?:(?:TO|BY)\s+[^-]*-\s*)?UPI(?:AB)?\b", "UPI Payment", "UPI Receipt"),
    _fl("imps", r"^\s*(?:(?:TO|BY)\s+[^-]*-\s*)?(?:IMPS|IMPSAR|IMPSAB|MMT/IMPS|(?:INET|MOB)-\s*IMPS)\b", "IMPS Payment", "IMPS Receipt"),
    _fl("neft", r"^\s*(?:(?:TO|BY)\s+[^-]*-\s*)?(?:NEFT|NEFTO|(?:INET|MOB)-\s*NEFT)\b", "NEFT Payment", "NEFT Receipt"),
    _fl("rtgs", r"^\s*(?:(?:TO|BY)\s+[^-]*-\s*)?(?:RTGS|RTGSO)\b", "RTGS Payment", "RTGS Receipt"),
    _fl("cheque_clearing", r"^\s*CLG\b|CLEARING\s*/\s*CHEQUE|^\s*(?:TRFR?|ME)\b", "Cheque Payment", ""),
    _fl("cheque_credit", r"^\s*CLG\b|CLEARING\s*/\s*CHEQUE", "", "Cheque Receipt"),
    _fl("mobile_banking", r"^\s*(?:MB|MOB)\b", "Mobile Payment", "Mobile Receipt"),
    _fl("mobft", r"^\s*MOBFT\b", "Online Payment", "Mobile Receipt"),
    _fl("online_transfer", r"^\s*(?:INF|IFT|INB|BIL|IB|TRFR?|NET|FT)\b|-TPT-|/PAYU\w*$|^\s*REF-|BILLDK|/RAZP[A-Z]|\bINW\b", "Online Payment", "Online Receipt"),
    # A narration that is only a name (no rail, no digits): a cheque or transfer entry.
    _fl("name_only", r"^[A-Za-z][A-Za-z.&,\s]*[A-Za-z]$", "Cheque Payment", "Online Receipt"),
)

# Rail label from Level 1 when no narration rule applies.
FIRST_LEVEL_BY_LEVEL1: dict[str, tuple[str, str]] = {
    "upi": ("UPI Payment", "UPI Receipt"),
    "imps": ("IMPS Payment", "IMPS Receipt"),
    "neft": ("NEFT Payment", "NEFT Receipt"),
    "rtgs": ("RTGS Payment", "RTGS Receipt"),
    "cheque": ("Cheque Payment", "Cheque Receipt"),
    "nach": ("ECS/NACH Payment", "ECS/NACH Receipt"),
    "ach": ("ECS/NACH Payment", "ECS/NACH Receipt"),
    "auto-debit": ("ECS/NACH Payment", "ECS/NACH Receipt"),
    "cash": ("Cash Withdrawal", "Cash Deposit"),
    "card": ("POS Expense", ""),
    "wallet": ("Other Payment", "Other Receipt"),
    "internal transfer": ("Online Payment", "Online Receipt"),
    "bbps": ("Bill Payment", ""),
}


def derive_first_level_classification(row: pd.Series) -> str:
    """Return the First Level label: purpose keywords, then the rail, with Payment / Receipt by direction."""

    debit_num = parse_amount(row.get("Debit"))
    credit_num = parse_amount(row.get("Credit"))
    level_1 = str(row.get("Level_1", "") or "").strip().lower()
    level_5 = str(row.get("Level_5", "") or "").strip().lower()
    particulars = str(row.get("Particulars", "") or "")
    is_debit = debit_num > 0 and credit_num <= 0
    is_credit = credit_num > 0 and not is_debit

    for rule in FIRST_LEVEL_RULES:
        label = rule.debit if is_debit else rule.credit if is_credit else ""
        if label and rule.pattern.search(particulars):
            return label
    label = FIRST_LEVEL_BY_LEVEL1.get(level_1, ("", ""))[0 if is_debit else 1] if (is_debit or is_credit) else ""
    if label:
        return label
    if level_5 == "loan repayment":
        return "Loan"
    return derive_first_level_fallback(particulars, level_1, level_5)


# Fallback First Level labels, used only when none of the rules above applies.
# Each needs explicit narration evidence; anything else stays blank.
FIRST_LEVEL_FALLBACKS: list[tuple[re.Pattern[str], str]] = [
    # Reversals of UPI / card transactions: "UPIRET-...", "CRV POS ..."
    (re.compile(r"^\s*(?:UPI\s*RET|CRV\s+POS)\b", re.IGNORECASE), "Refund/Reversal"),
    # Bank charges: "RTN CHG-...", "IMPS Chg May-25+GST", "Chq rtn Chg ..."
    (BANK_CHARGE_RE, "Bank Charges"),
    (BANK_FEE_NARRATION_RE, "Bank Charges"),
    # Interest: "...:Int.Coll:...", "...:Int.Pd:...", "INTEREST PAID TILL ..."
    (re.compile(r"\bInt\.\s*(?:Coll|Pd)\b|\bINTEREST\s+(?:PAID|CREDITED|COLLECTED)\b", re.IGNORECASE), "Interest"),
    # Dividends: "DR REDDYS DIV 23 ...", "ACH C- HCL 1ST INTDIV24 ...", and codes
    # run together with the company name ("TCSFINDIV...", "TCS3INTSPLDIV...").
    (re.compile(r"\bDIV(?:IDEND)?S?\d*\b|(?:INT|FIN|SPL)DIV", re.IGNORECASE), "Other Receipt"),
    # Card payments: "POS 416021XXXXXX3640 ...", "ME DC SI ...", "ECOM PUR/..."
    (re.compile(r"^\s*(?:POS|ME\s+DC\s+SI|ECOM\s+PUR)\b", re.IGNORECASE), "Card Payment"),
    # International remittances: ".../INTL/DR/.../INTL TRANSFER-"
    (re.compile(r"\bINTL\s*/|\bINTL\s+TRANSFER\b", re.IGNORECASE), "International Transfer"),
    # In-bank / net-banking / mobile-banking fund transfers.
    (
        re.compile(
            r"^\s*(?:(?:INF|BIL)/INFT/|TRFR\s+(?:TO|FROM)\b|TRF/|IFT/|INB/IFT/|MOB/TPFT/|IB\s+FUNDS\s+TRANSFER\b)",
            re.IGNORECASE,
        ),
        "Fund Transfer",
    ),
]


def derive_first_level_fallback(particulars: str, level_1: str = "", level_5: str = "") -> str:
    """Return a First Level label for rows the main mapping leaves blank."""

    if level_1 in {"nach", "auto-debit"}:
        return "ECS/NACH Payment"
    for pattern, label in FIRST_LEVEL_FALLBACKS:
        if pattern.search(str(particulars or "")):
            return label
    return ""


def derive_second_level_classification(row: pd.Series) -> str:
    """Create a readable secondary label from the fine-grained prediction."""

    level_5 = str(row.get("Level_5", "") or "").strip()
    level_3 = str(row.get("Level_3", "") or "").strip()

    if level_5:
        if level_5 == "p2p transfer":
            return "P2P Transfer"
        return level_5.title()
    if level_3 == "p2p":
        return "P2P"
    if level_3:
        return level_3.title()
    # Nothing known about purpose or counterparty: describe the transaction type instead.
    return SECOND_LEVEL_BY_FIRST.get(str(row.get("First Level Classification", "") or "").strip(), "")


# Second Level when Level 5 and Level 3 are blank: the kind of transaction the First Level names.
SECOND_LEVEL_BY_FIRST: dict[str, str] = {
    "Cash Withdrawal": "Cash", "ATM Withdrawal": "Cash", "Cash Deposit": "Cash", "Cash Return": "Cash",
    "ECS/NACH Payment": "Mandate", "ECS/NACH Receipt": "Mandate", "Contra": "Mandate Reversal",
    "Bank Charges": "Charges", "Interest Paid": "Interest", "Interest Received": "Interest",
    "Salary": "Salary", "Investment": "Investment", "Tax": "Tax", "Insurance": "Insurance",
    "Loan": "Loan Repayment", "Credit Card Payment": "Credit Card", "Utility Expenses": "Utilities",
    "Bill Payment": "Utilities", "POS Expense": "Card Purchase", "Refund/Reversal": "Refund",
    "Inward Online Return": "Refund", "Outward Online Return": "Refund", "Outward Cheque Return": "Cheque Return",
    "Rent Payment": "Rent", "Other Receipt": "Wallet", "Other Payment": "Wallet",
    "RTGS Payment": "Bank Transfer", "RTGS Receipt": "Bank Transfer", "NEFT Payment": "Bank Transfer",
    "NEFT Receipt": "Bank Transfer", "IMPS Payment": "Bank Transfer", "IMPS Receipt": "Bank Transfer",
    "Online Payment": "Bank Transfer", "Online Receipt": "Bank Transfer", "Mobile Payment": "Bank Transfer",
    "Mobile Receipt": "Bank Transfer", "Cheque Payment": "Cheque", "Cheque Receipt": "Cheque",
    "UPI Payment": "UPI", "UPI Receipt": "UPI", "Forex/Remittances": "Remittance",
}


CONTRA_WINDOW_DAYS = 7


def apply_first_level_contra(df: pd.DataFrame) -> pd.DataFrame:
    """Label a mandate debit and its same-amount reversal credit "Contra".

    A debit and a credit with the same narration and amount, within
    ``CONTRA_WINDOW_DAYS`` of each other, cancel out: both rows are Contra.
    Only mandate (ECS/NACH) rows are paired.
    """

    output_df = df.copy()
    labels = output_df["First Level Classification"].astype(str)
    candidates = output_df.index[labels.isin({"ECS/NACH Payment", "ECS/NACH Receipt"})]
    if len(candidates) < 2:
        return output_df
    debit = output_df.loc[candidates, "Debit"].map(parse_amount)
    credit = output_df.loc[candidates, "Credit"].map(parse_amount)
    dates = None
    if "Date" in output_df.columns:
        raw_dates = output_df.loc[candidates, "Date"]
        try:
            iso_dates = pd.to_datetime(raw_dates, errors="coerce", format="ISO8601")
        except (ValueError, TypeError):
            iso_dates = pd.Series(pd.NaT, index=raw_dates.index)
        dates = iso_dates
        if iso_dates.isna().any():
            dates = iso_dates.fillna(pd.to_datetime(raw_dates, errors="coerce", dayfirst=True))
    groups: dict[tuple[str, float], dict[str, list]] = defaultdict(lambda: {"d": [], "c": []})
    for idx in candidates:
        amount = debit[idx] if debit[idx] > 0 and credit[idx] <= 0 else credit[idx]
        if amount <= 0:
            continue
        side = "d" if debit[idx] > 0 and credit[idx] <= 0 else "c"
        key = (" ".join(str(output_df.at[idx, "Particulars"]).split()).lower(), round(float(amount), 2))
        groups[key][side].append(idx)
    for sides in groups.values():
        pool = list(sides["c"])
        for d_idx in sides["d"]:
            for c_idx in pool:
                near = True
                if dates is not None and pd.notna(dates[d_idx]) and pd.notna(dates[c_idx]):
                    near = abs((dates[d_idx] - dates[c_idx]).days) <= CONTRA_WINDOW_DAYS
                if near:
                    output_df.at[d_idx, "First Level Classification"] = "Contra"
                    output_df.at[c_idx, "First Level Classification"] = "Contra"
                    pool.remove(c_idx)
                    break
    return output_df


# ---------------------------------------------------------------------------
# Pre-detect-salary overrides (rules 1-6 from the 2024-10-24 spec)
# ---------------------------------------------------------------------------
# Rule 1: "RTN CHG" rows (bank charge return).
RTN_CHG_RE = re.compile(r"^\s*(?:RTN\s+CHG|RETURN\s+CHARGE)\b", flags=re.IGNORECASE)
# Rule 2: ACH credit with a dividend keyword.
ACH_CR_DIV_RE = re.compile(r"\b(?:ACH|NACH)\s*C(?:R|REDIT)?\b", flags=re.IGNORECASE)
DIV_KEYWORD_RE = re.compile(r"\bDIV(?:IDEND)?S?\b|(?:INT|FIN|SPL)DIV", flags=re.IGNORECASE)
# Rule 3: Interest credit rows.
INT_COLL_RE = re.compile(
    r"\bInt\.\s*(?:Coll|Pd)\b|\bINTEREST\s+(?:PAID|CREDITED|COLLECTED)\b",
    flags=re.IGNORECASE,
)
# Rule 4: DRP wallet load/unload.
DRP_WALLET_RE = re.compile(r"^\s*DRP/", flags=re.IGNORECASE)
DRP_LOAD_RE = re.compile(r"\bLoad\b", flags=re.IGNORECASE)
DRP_UNLOAD_RE = re.compile(r"\bUnLoad\b", flags=re.IGNORECASE)
# Rule 5: Business-name words that disqualify a CLG/NEFT/RTGS/IMPS/TRF payee from salary.
BUSINESS_WORD_RE = re.compile(
    r"\b(?:LTD|LIMITED|PVT|PRIVATE|TRADING|ENTERPRISES?|CONSTRUCTION|CORPO(?:RATE)?|"
    r"TEXTILES?|FINANCE|FINANCIALS?|SOCIETY|STORES?|INDUSTRIES|INFRA(?:STRUCTURE)?|"
    r"EXPORTS?|IMPORTS?|BUILDERS?|DEVELOPERS?|REALTORS?|ASSOCIATES?)\b",
    flags=re.IGNORECASE,
)
BUSINESS_RAIL_RE = re.compile(r"^\s*(?:CLG|NEFT|RTGS|IMPS|TRF)/", flags=re.IGNORECASE)
# Rule 6: Generic Level-6 values that should be blanked.
GENERIC_LEVEL6_EXACT: frozenset[str] = frozenset({
    "unload", "load", "funds insufficient", "exceeds arrangement",
    "inft", "upiret", "ib funds transfer dr", "ib billpay dr",
})

# Narration prefixes that should never be promoted to salary by detect_salary.
_SALARY_EXCLUSION_RE = re.compile(
    r"^\s*(?:CLG|TRF|DRP|NEFT|RTGS)/"
    r"|\bUPI\b.*\bP2A\b"
    r"|\bATW(?![A-Za-z])\b",
    flags=re.IGNORECASE,
)


def _company_name_in_narration(text: str) -> bool:
    """Return True when the narration contains a business-word on a transfer rail."""
    return bool(BUSINESS_RAIL_RE.search(text) and BUSINESS_WORD_RE.search(text))


# Level 5 purposes that only a business can sell. A person-to-person row that
# carries one of these is a contradiction (see apply_counterparty_purpose_compat).
MERCHANT_LEVEL5_PURPOSES = frozenset(
    {
        "retail", "services", "transport", "dining", "grocery", "pharmacy", "healthcare",
        "entertainment", "subscription", "digital services", "fuel", "travel", "mobile recharge",
    }
)
# Business words of 7+ letters. They count when a narration token starts or ends
# with one, because statements fuse names ("SAGARENTERPRISE", "ABCTRADERS").
LONG_BUSINESS_WORDS = frozenset(
    {
        "enterprise", "enterprises", "traders", "trading", "industries", "agencies",
        "lubricants", "associates", "engineering", "electricals", "electronics",
        "hardware", "textiles", "garments", "distributors", "suppliers", "services",
        "solutions", "logistics", "constructions", "construction", "pharmacy",
        "furniture", "jewellers", "jewellery", "stationery", "provisions",
    }
)
# A real Airtel narration (recharge, bill, merchant) is an Airtel merchant. The
# bank "Airtel Payments Bank" (also fused as "AirtelPaymentsBank") is not.
AIRTEL_MERCHANT_RE = re.compile(
    r"(?<![a-z])airtel(?!\s*(?:payments?\s*)?bank)", flags=re.IGNORECASE
)


def _has_long_business_word(text: str) -> bool:
    """Return True when a token starts or ends with a long business word."""

    for token in re.findall(r"[a-z]+", str(text or "").lower()):
        for word in LONG_BUSINESS_WORDS:
            if len(word) >= 7 and len(token) >= len(word) and (token.startswith(word) or token.endswith(word)):
                return True
    return False


def apply_counterparty_purpose_compat(
    particulars: str,
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
    part_texts: list[str] | None = None,
) -> dict[str, str]:
    """Settle a p2p row that carries a merchant-only Level 5 purpose.

    A person is not a retailer, a restaurant or a fuel station. For a Level 3
    ``p2p`` row whose Level 5 is a merchant purpose:

    - if the narration supports that purpose (a NARRATION_LEVEL5_PATTERNS match
      for the same purpose, a real Airtel narration for mobile recharge) or
      names a business (BUSINESS_WORD_RE, or a long business word at the start
      or end of a token), the counterparty is a merchant: Level 3 -> merchant;
    - otherwise the purpose is unsupported: Level 5 is cleared, Level 3 stays
      p2p, and the default "p2p transfer" purpose is restored.

    Runs as the last step of the per-row chain.
    """

    if result.get("Level_3") != "p2p":
        return result
    purpose = _normalize_compare(result.get("Level_5", ""))
    # An institutional purpose (financial services, loan, insurance...) is never person-to-person.
    if purpose in INSTITUTIONAL_LEVEL5:
        result["Level_3"] = "institution"
        return result
    if purpose not in MERCHANT_LEVEL5_PURPOSES:
        return result

    text = _marker_text(particulars, part_texts)
    supported = any(
        pattern_purpose == purpose and pattern.search(text)
        for pattern, pattern_purpose in NARRATION_LEVEL5_PATTERNS
    )
    if purpose == "mobile recharge" and (AIRTEL_MERCHANT_RE.search(text) or JIO_MERCHANT_RE.search(text)):
        supported = True

    if supported or BUSINESS_WORD_RE.search(text) or _has_long_business_word(text):
        result["Level_3"] = "merchant"
        return result

    result["Level_5"] = ""
    direction = derive_direction_from_amounts(debit_value, credit_value)
    automated_debit = bool(SYSTEM_INITIATED_RE.search(text)) and direction == "debit"
    if not automated_debit:
        result["Level_5"] = "p2p transfer"
    return result


SALARY_CHEQUE_RAIL_RE = re.compile(r"\b(?:CLG|CLEARING|CHEQUE|CHQ|CHEQ)\b", flags=re.IGNORECASE)


def _is_non_salary_counterparty(row: pd.Series) -> bool:
    """Cheque credits and business counterparties are never inferred salary.

    Only used for rows without a salary word: such rows may not become
    salary on recurrence and date consistency alone.
    """

    text = str(row.get("Particulars", "") or "")
    # A cash deposit names no payer, so recurrence alone cannot make it salary.
    if str(row.get("Level_1", "") or "").strip().lower() in {"cheque", "cash"} or SALARY_CHEQUE_RAIL_RE.search(text):
        return True
    if str(row.get("Level_3", "") or "").strip().lower() in {"merchant", "institution"}:
        return True
    return bool(BUSINESS_WORD_RE.search(text))


def apply_pre_detect_salary_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply structural overrides to output_df before detect_salary runs.

    Rules applied in order (a later rule can overwrite an earlier one only
    when a row matches multiple patterns, which is rare):

    1. RTN CHG -> Level_1=cheque, Level_3=institution, Level_5=charges,
       Level_6="bank charges".
    2. ACH C + DIV -> Level_1=ach, Level_3=institution, Level_5=dividend,
       Level_6=company extracted from narration (or blank).
    3. Int.Coll / INTEREST PAID -> Level_3=institution, Level_5=interest,
       Level_6="".
    4. DRP/Load -> Level_1=wallet, Level_3=institution,
       Level_5=wallet load/unload, Level_6="".
    5. CLG/NEFT/RTGS/IMPS/TRF with business-word payee -> Level_3=merchant,
       remove p2p transfer.
    6. Blank Level_6 when it is a known generic placeholder or contains "xxxx".
    """

    output_df = df.copy()
    particulars = output_df["Particulars"].fillna("").astype(str)
    level_6 = output_df["Level_6"].fillna("").astype(str)
    credit_num = output_df["Credit"].apply(parse_amount)

    for idx in output_df.index:
        text = particulars[idx]
        text_lo = text.lower()

        # Rule 1: RTN CHG bank charge
        if RTN_CHG_RE.match(text):
            output_df.at[idx, "Level_1"] = level1_from_rail_word(text)
            output_df.at[idx, "Level_3"] = "institution"
            output_df.at[idx, "Level_5"] = "charges"
            output_df.at[idx, "Level_6"] = "bank charges"
            continue

        # Rule 1b: bank fee narrations -> institution / charges, no counterparty name
        if BANK_FEE_NARRATION_RE.match(text):
            output_df.at[idx, "Level_3"] = "institution"
            output_df.at[idx, "Level_5"] = "charges"
            output_df.at[idx, "Level_6"] = ""
            row = output_df.loc[idx]
            output_df.at[idx, "First Level Classification"] = derive_first_level_classification(row)
            output_df.at[idx, "Second Level Classification"] = derive_second_level_classification(row)
            continue

        # Rule 2: ACH credit + dividend
        if credit_num[idx] > 0 and ACH_CR_DIV_RE.search(text) and DIV_KEYWORD_RE.search(text):
            company = extract_level6_entity(text)
            output_df.at[idx, "Level_1"] = "nach"
            output_df.at[idx, "Level_3"] = "institution"
            output_df.at[idx, "Level_5"] = "dividend"
            output_df.at[idx, "Level_6"] = company
            continue

        # Rule 3: Interest credit/collection
        if INT_COLL_RE.search(text):
            output_df.at[idx, "Level_3"] = "institution"
            output_df.at[idx, "Level_5"] = "interest"
            output_df.at[idx, "Level_6"] = ""
            continue

        # Rule 4: DRP wallet load / unload
        if DRP_WALLET_RE.match(text):
            output_df.at[idx, "Level_1"] = "wallet"
            output_df.at[idx, "Level_3"] = "institution"
            if DRP_UNLOAD_RE.search(text):
                output_df.at[idx, "Level_5"] = "wallet unload"
            elif DRP_LOAD_RE.search(text):
                output_df.at[idx, "Level_5"] = "wallet load"
            else:
                output_df.at[idx, "Level_5"] = "wallet load"
            output_df.at[idx, "Level_6"] = ""
            # DRP rows must never be flagged as salary
            output_df.at[idx, "IsSalary"] = 0
            output_df.at[idx, "SalaryConfidence"] = 0.0
            continue

        # Rule 5: Business-name payee on a transfer rail -> merchant
        if _company_name_in_narration(text):
            current_l3 = str(output_df.at[idx, "Level_3"] or "")
            if current_l3 not in {"institution"}:
                output_df.at[idx, "Level_3"] = "merchant"
                if str(output_df.at[idx, "Level_5"] or "") == "p2p transfer":
                    output_df.at[idx, "Level_5"] = ""

    # Rule 6: Blank generic / placeholder Level_6 values.
    for idx in output_df.index:
        raw_l6 = str(output_df.at[idx, "Level_6"] or "").strip().lower()
        if raw_l6 in GENERIC_LEVEL6_EXACT or "xxxx" in raw_l6:
            output_df.at[idx, "Level_6"] = ""

    return output_df


def detect_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Flag salary-like credits using explicit text plus recurring credits.

    - A salary word in the narration on a credit is salary (confidence 1.0).
    - Otherwise credits >= MIN_SALARY are clustered by rounded amount. A
      cluster recurs when it spans MIN_SALARY_MONTHS distinct calendar
      months (+0.4); day-of-month std <= SALARY_DAY_STD_MAX adds +0.3 and a
      salary word +0.3; accepted at score >= 0.6.
    - CLG, DRP, TRF, UPI-P2A, cheque, business-word and merchant/institution
      credits are excluded from inference.
    - IsSalary, Level_5 = salary and First Level = Salary always agree.
    """

    output_df = df.copy()
    output_df["IsSalary"] = 0
    output_df["SalaryConfidence"] = 0.0

    credit_num = output_df["Credit"].apply(parse_amount)
    debit_num = (
        output_df["Debit"].apply(parse_amount)
        if "Debit" in output_df.columns
        else pd.Series(0.0, index=output_df.index)
    )
    debit_mask = (debit_num > 0) & (credit_num <= 0)

    # Salary text is read from the narration only. First/Second Level are
    # derived from the model's Level_5, so searching them would let a
    # predicted "salary" label count as its own evidence.
    salary_text = output_df["Particulars"].astype(str).str.contains(SALARY_KEYWORDS, na=False)
    credit_only = (credit_num > 0) & ~debit_mask

    # Explicit evidence on a credit: a salary word in the narration, or a
    # Level_5 = salary predicted by the keyword model (codebook employers).
    predicted_salary = (
        output_df["Level_5"].astype(str).str.strip().str.lower().eq("salary")
        if "Level_5" in output_df.columns
        else pd.Series(False, index=output_df.index)
    )
    explicit_mask = (salary_text | predicted_salary) & credit_only
    output_df.loc[explicit_mask, "IsSalary"] = 1
    output_df.loc[explicit_mask, "SalaryConfidence"] = 1.0

    # Without a Date column no row has a usable date, so recurring-credit
    # inference is skipped; explicit salary evidence above still applies.
    # Statement dates are day-first text ("2/3/25" is 2 March) mixed with
    # real dates, so they are parsed the same way as Level 7 does.
    date_column = (
        output_df["Date"]
        if "Date" in output_df.columns
        else pd.Series(pd.NaT, index=output_df.index)
    )
    dates = pd.to_datetime(date_column.map(_level7_parse_date), errors="coerce")
    candidates = output_df.index[
        (~explicit_mask) & credit_only & (credit_num >= MIN_SALARY) & dates.notna()
    ]

    if len(candidates) >= 2:
        # Exclude narration patterns that should never be salary, plus cheque
        # credits and business counterparties: without a salary word they
        # must not pass on recurrence and date consistency alone.
        exclusion_mask = output_df.loc[candidates].apply(
            lambda row: bool(_SALARY_EXCLUSION_RE.search(str(row["Particulars"])))
            or _company_name_in_narration(str(row["Particulars"]))
            or _is_non_salary_counterparty(row),
            axis=1,
        )
        candidates = candidates[~exclusion_mask.values]

    if len(candidates) >= 2:
        amount_keys = ((credit_num.loc[candidates] / 1000).round() * 1000).astype(int)
        clusters: dict[int, list[int]] = defaultdict(list)
        for index, amount_key in amount_keys.items():
            clusters[int(amount_key)].append(index)

        for indices in clusters.values():
            # "Recurs" means the same rounded amount in at least
            # MIN_SALARY_MONTHS distinct calendar months (year + month);
            # two credits in one month count as one month.
            distinct_months = dates.loc[indices].dt.to_period("M").nunique()
            if distinct_months < MIN_SALARY_MONTHS:
                continue

            day_std = dates.loc[indices].dt.day.std(ddof=0)
            text_hit = bool(salary_text.loc[indices].any())

            score = 0.40
            score += 0.30 if (pd.notna(day_std) and day_std <= SALARY_DAY_STD_MAX) else 0.0
            score += 0.30 if text_hit else 0.0

            if score >= 0.60:
                output_df.loc[indices, "IsSalary"] = 1
                output_df.loc[indices, "SalaryConfidence"] = round(score, 2)

    output_df.loc[debit_mask, "IsSalary"] = 0
    output_df.loc[debit_mask, "SalaryConfidence"] = 0.0

    # Rows that are not salary must not keep a salary label from the model or
    # from an earlier step: clear Level_5 and re-derive First/Second Level.
    not_salary = output_df["IsSalary"].ne(1)
    stale_level_5 = (
        output_df["Level_5"].astype(str).str.strip().str.lower().eq("salary")
        if "Level_5" in output_df.columns
        else pd.Series(False, index=output_df.index)
    )
    stale_first_level = (
        output_df["First Level Classification"].astype(str).str.strip().str.lower().eq("salary")
    )
    stale_mask = not_salary & (stale_level_5 | stale_first_level)
    if stale_mask.any():
        if "Level_5" in output_df.columns:
            output_df.loc[stale_mask & stale_level_5, "Level_5"] = ""
        stale_rows = output_df.loc[stale_mask]
        output_df.loc[stale_mask, "First Level Classification"] = stale_rows.apply(
            derive_first_level_classification, axis=1
        )
        output_df.loc[stale_mask, "Second Level Classification"] = stale_rows.apply(
            derive_second_level_classification, axis=1
        )

    salary_mask = output_df["IsSalary"].eq(1)
    salary_entity = output_df["Particulars"].astype(str).map(
        lambda text: clean_salary_entity(extract_level6_entity(text) or text)
    )
    output_df.loc[salary_mask, "Level_0"] = "credit"
    # Level_1 is the payment rail (rtgs/neft/imps/cheque...); salary is a
    # purpose, so an identified rail is kept rather than blanked.
    output_df.loc[salary_mask, "Level_2"] = "System generated"
    output_df.loc[salary_mask, "Level_5"] = "salary"
    output_df.loc[salary_mask, "Level_7"] = ""
    output_df.loc[salary_mask, "Level_6"] = salary_entity.loc[salary_mask]
    # First Level says "Salary" only when the narration itself says so; a recurring credit that
    # is salary by Level 5 keeps its rail label ("RTGS Receipt").
    if salary_mask.any():
        output_df.loc[salary_mask, "First Level Classification"] = output_df.loc[salary_mask].apply(
            derive_first_level_classification, axis=1
        )
    output_df.loc[salary_mask, "Second Level Classification"] = ""

    return output_df


def build_salary_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a salary-only summary sheet."""

    salary_df = df[df.get("IsSalary", 0).astype(int).eq(1)].copy()
    if salary_df.empty:
        return salary_df

    salary_df["Month"] = (
        pd.to_datetime(salary_df["Date"].map(_level7_parse_date), errors="coerce")
        .dt.to_period("M")
        .astype(str)
    )
    summary_columns = [
        "Date",
        "Month",
        "Particulars",
        "Credit",
        "SalaryConfidence",
        "Level_5",
        "Level_6",
        "First Level Classification",
        "Source_File",
    ]
    available_columns = [column for column in summary_columns if column in salary_df.columns]
    return salary_df[available_columns].reset_index(drop=True)


USER_INITIATED_RE = re.compile(
    r"\b(?:ATM[\s/-]*WDL|ATM\s+CASH\s+WITHDRAWAL|"
    r"POS\s+PURCHASE|CARD\s+PURCHASE)\b",
    flags=re.IGNORECASE,
)


# Events the bank itself originates, whatever the customer did: charges and fees
# (BANK_CHARGE_RE: "Dr Card Charges GST", "ATM Decline Chgs Incl GST", "RTN CHG"),
# interest (INT_COLL_RE: "Int.Pd", "Int.Coll"; "Int on delayed ..."), and
# reversals of UPI / card transactions ("UPIRET-...", "CRV POS ...").
BANK_ORIGINATED_RE = re.compile(
    "|".join(
        f"(?:{pattern})"
        for pattern in (
            BANK_CHARGE_RE.pattern,
            BANK_FEE_NARRATION_RE.pattern,
            INT_COLL_RE.pattern,
            r"\bInt\s+on\s+delayed\b",
            r"^\s*(?:UPI\s*RET|CRV\s+POS)\b",
        )
    ),
    flags=re.IGNORECASE,
)
BANK_ORIGINATED_LEVEL5 = frozenset({"charges", "interest", "dividend"})
# Rows that keep the customer-initiated default even if a bank-originated word
# appears: wallet load / unload and UPI P2A (person-to-account) transfers.
LEVEL2_USER_RE = re.compile(r"\bUPI\b.*\bP2A\b|\bP2A\b", flags=re.IGNORECASE)


def _is_bank_originated(row: pd.Series, particulars: str) -> bool:
    """Return True for a bank-originated event (see BANK_ORIGINATED_RE)."""

    level_1 = str(row.get("Level_1", "") or "").strip().lower()
    level_5 = str(row.get("Level_5", "") or "").strip().lower()
    if level_1 == "wallet" or level_5.startswith("wallet") or LEVEL2_USER_RE.search(particulars):
        return False
    return bool(BANK_ORIGINATED_RE.search(particulars)) or level_5 in BANK_ORIGINATED_LEVEL5


def apply_strict_level2(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a complete, rule-based initiation mode to every transaction."""
    output_df = df.copy()
    output_df["Level_2"] = "User-generated"

    for idx, row in output_df.iterrows():
        particulars = str(row.get("Particulars", "") or "").strip()

        if SYSTEM_INITIATED_RE.search(particulars):
            output_df.at[idx, "Level_2"] = "System generated"

        elif row.get("IsSalary", 0) == 1:
            output_df.at[idx, "Level_2"] = "System generated"

        elif _is_bank_originated(row, particulars):
            output_df.at[idx, "Level_2"] = "System generated"

    return output_df


# ---------------------------------------------------------------------------
# Level 7: Recurrence Classification (recurring / one-time)
# ---------------------------------------------------------------------------
LEVEL7_RECURRING = "recurring"
LEVEL7_ONE_TIME = "one-time"

# (name, min gap days, max gap days, min distinct dates in the chain)
LEVEL7_CADENCES = (
    ("monthly", 25, 35, 2),
    ("bimonthly", 50, 68, 2),
    ("quarterly", 75, 100, 2),
    ("fortnightly", 12, 16, 3),
    ("weekly", 5, 9, 3),
)
LEVEL7_AMOUNT_SPREAD = 0.10  # fixed-amount tolerance for non-bill chains
LEVEL7_P2P_AMOUNT_SPREAD = 0.05  # tighter for person-to-person transfers
LEVEL7_EXACT_SPREAD = 0.01  # "same amount" for a two-date pair
LEVEL7_PAIR_DAY_TOLERANCE = 3  # day-of-month drift allowed for a two-date pair
LEVEL7_REVERSAL_WINDOW_DAYS = 10

# Refunds, reversals and failed or returned items are never a recurrence.
LEVEL7_REVERSAL_RE = re.compile(
    r"\b(?:revers(?:al|ed)?|rev|refund(?:ed)?|return(?:ed)?|rtn|chargeback|"
    r"failed|reject(?:ed)?|bounce[d]?|unpaid|insufficient|declin\w*|cancel\w*|"
    r"unsuccess\w*|dishonor\w*|disput\w*)\b",
    flags=re.IGNORECASE,
)
# Fees and one-off loan events that mention EMI but are not an instalment.
LEVEL7_NOT_INSTRUMENT_RE = re.compile(
    r"\b(?:fee|fees|charge|charges|penalt(?:y|ies)|processing|conversion|"
    r"foreclos\w*|pre[\s-]?payment|prepay\w*|part[\s-]?payment|waiver|"
    r"regist\w*|cancel\w*)\b",
    flags=re.IGNORECASE,
)
# Standing-instruction style evidence that is reliable even on manual transfer rails.
LEVEL7_MANDATE_RE = re.compile(
    r"\b(?:E?NACH|ACH|ECS|AUTO[\s-]?DEBIT|AUTODEBIT|AUTO[\s-]?PAY|AUTOPAY|"
    r"STANDING\s+INSTRUCTION|RECURRING\s+PAYMENT|UPI\s+MANDATE|SI[\s-]?DEBIT|MANDATE)\b",
    flags=re.IGNORECASE,
)
LEVEL7_MANUAL_RAIL_RE = re.compile(r"\b(?:UPI|IMPS|NEFT|RTGS)\b", flags=re.IGNORECASE)

# Specific bill/loan/utility keywords (excluding broad terms like bare 'bill' or 'payment').
LEVEL7_BILL_RE = re.compile(
    r"\b(?:emi|loan|instal(?:l)?ments?|premium|insurance|policy|sip|mutual\s+fund|"
    r"subscription|broadband|electricity|water|gas|dth|piped\s+gas|lpg|"
    r"credit\s+card|bbps)\b",
    flags=re.IGNORECASE,
)
LEVEL7_BILL_LEVEL5 = frozenset(
    {"loan repayment", "utilities", "mobile recharge", "insurance", "credit card", "subscription"}
)

LEVEL7_KEY_STOPWORDS = frozenset(
    {
        "inf", "inft", "bil", "paid", "clg", "mmt", "trf", "trfr", "rtn", "to", "by", "from",
        "via", "for", "ref", "txn", "dr", "cr", "debit", "credit", "upi", "neft", "imps",
        "rtgs", "payment", "transfer", "pay", "bank", "ltd", "pvt", "private", "limited",
        "reversal", "reversed", "rev", "refund", "return", "returned", "failed",
        "reject", "rejected", "bounced", "bounce", "unpaid", "insufficient", "funds",
        "gpay", "googlepay", "phonepe", "paytm", "razorpay", "billdesk", "cashfree",
        "payu", "bhim", "cred", "mobikwik", "amazonpay", "payzapp", "freecharge",
        "pinelabs", "juspay", "bharatpe",
    }
)
LEVEL7_INFT_RE = re.compile(r"^\s*INF/INFT/[^/]*/([^/]*)/(.+)$", flags=re.IGNORECASE)
LEVEL7_WIRE_RE = re.compile(r"^\s*(?:NEFT|RTGS)[-:][A-Za-z0-9]{8,}[-/](.+)$", flags=re.IGNORECASE)
LEVEL7_CLEARING_RE = re.compile(r"^\s*CLG/([^/]+)/", flags=re.IGNORECASE)


def _level7_parse_date(value: object) -> pd.Timestamp | None:
    """Return a normalised date, or None. Text dates are read day-first."""

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if not isinstance(value, str):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.normalize()
    text = value.strip()
    if not text:
        return None
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(pd.to_datetime(text, format=fmt)).normalize()
        except (ValueError, TypeError):
            continue
    stamp = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return None if pd.isna(stamp) else pd.Timestamp(stamp).normalize()


def _level7_clean_key(text: str) -> str:
    """Lower-case identity with reference numbers, rails and filler removed."""

    tokens = []
    for raw in re.split(r"[^A-Za-z0-9]+", str(text or "").lower()):
        if not raw:
            continue
        if re.search(r"\d", raw) and len(raw) >= 6:
            continue
        word = re.sub(r"\d+", "", raw)
        if not word or word in LEVEL7_KEY_STOPWORDS:
            continue
        tokens.append(word)
    return " ".join(tokens)


def _normalize_level7_key(key: str) -> str:
    """Normalise a counterparty key for Level 7 grouping.

    - Lowercase
    - Remove trailing single-letter suffixes (e.g. "acme ltd n" -> "acme ltd")
    - Prefix deduplication: if two keys share a prefix of 5+ characters,
      the shorter one maps to the longer (done at the call-site via a
      shared reference dict; this helper only strips trailing junk).
    """

    k = key.strip().lower()
    # Drop a trailing single letter separated by a space (bank-statement
    # artefact like "ramesh kumar n" or "sharma k").
    k = re.sub(r"\s+[a-z]$", "", k).strip()
    return k


def level7_counterparty(particulars: object) -> str:
    """Return a normalised payee/payer identity that ignores changing references."""

    text = str(particulars or "").strip()
    if not text:
        return ""
    entity = extract_narration_entity(text)
    if entity:
        return _normalize_level7_key(_level7_clean_key(entity))
    inft = LEVEL7_INFT_RE.match(text)
    if inft:
        return _normalize_level7_key(_level7_clean_key(inft.group(2)))
    wire = LEVEL7_WIRE_RE.match(text)
    if wire:
        for segment in re.split(r"[-/]", wire.group(1)):
            key = _normalize_level7_key(_level7_clean_key(segment))
            if key:
                return key
        return ""
    clearing = LEVEL7_CLEARING_RE.match(text)
    if clearing:
        return _normalize_level7_key(_level7_clean_key(clearing.group(1)))
    return _normalize_level7_key(_level7_clean_key(text))


def _level7_explicit_evidence(text: str, is_debit: bool) -> bool:
    """Return True for reliable EMI / standing-instruction evidence on a debit."""

    if not is_debit or not text:
        return False
    if LEVEL7_REVERSAL_RE.search(text) or LEVEL7_NOT_INSTRUMENT_RE.search(text):
        return False
    if LEVEL7_MANUAL_RAIL_RE.search(text):
        return bool(LEVEL7_MANDATE_RE.search(text))
    return bool(SYSTEM_INITIATED_RE.search(text))


def _is_month_end(ts: pd.Timestamp) -> bool:
    """Return True if timestamp is within 3 days of the end of its month."""
    return (ts + pd.Timedelta(days=4)).month != ts.month


def _level7_chain_qualifies(
    cadence: str,
    count: int,
    amounts: list[float],
    dates: list[pd.Timestamp],
    bill_like: bool,
    is_p2p: bool,
) -> bool:
    """Apply the corroboration rules to one cadence chain."""

    if bill_like and not is_p2p:
        return True

    ordered = sorted(amounts)
    median = ordered[len(ordered) // 2]
    spread = (ordered[-1] - ordered[0]) / median if median > 0 else float("inf")

    if is_p2p:
        return count >= 3 and spread <= LEVEL7_P2P_AMOUNT_SPREAD

    if count >= 3:
        return spread <= LEVEL7_AMOUNT_SPREAD

    if cadence in ("monthly", "bimonthly") and count == 2:
        d1, d2 = dates[0], dates[1]
        days_of_month = [d1.day, d2.day]
        drift = max(days_of_month) - min(days_of_month)
        drift = min(drift, 31 - drift)
        if _is_month_end(d1) and _is_month_end(d2):
            drift = 0
        return spread <= LEVEL7_EXACT_SPREAD and drift <= LEVEL7_PAIR_DAY_TOLERANCE

    return False


def _level7_chains(dates: list[pd.Timestamp], low: int, high: int) -> list[list[int]]:
    """Split sorted unique dates into runs whose consecutive gaps are in [low, high]."""

    chains: list[list[int]] = []
    run = [0]
    for position in range(1, len(dates)):
        gap = (dates[position] - dates[position - 1]).days
        if low <= gap <= high:
            run.append(position)
        else:
            if len(run) > 1:
                chains.append(run)
            run = [position]
    if len(run) > 1:
        chains.append(run)
    return chains


def decide_level7(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return a (label, reason) pair per row. Reason is "" for non-transactions."""

    size = len(df)
    columns = df.columns
    debit_num = df["Debit"].map(parse_amount) if "Debit" in columns else pd.Series(0.0, index=df.index)
    credit_num = df["Credit"].map(parse_amount) if "Credit" in columns else pd.Series(0.0, index=df.index)
    text_col = df["Particulars"] if "Particulars" in columns else pd.Series("", index=df.index)
    date_col = df["Date"] if "Date" in columns else pd.Series(None, index=df.index)
    level1 = df["Level_1"] if "Level_1" in columns else pd.Series("", index=df.index)
    level3 = df["Level_3"] if "Level_3" in columns else pd.Series("", index=df.index)
    level5 = df["Level_5"] if "Level_5" in columns else pd.Series("", index=df.index)

    rows = []
    for position in range(size):
        debit, credit = float(debit_num.iloc[position]), float(credit_num.iloc[position])
        text = "" if pd.isna(text_col.iloc[position]) else str(text_col.iloc[position])
        if debit <= 0 and credit <= 0:
            rows.append(None)
            continue
        direction = "debit" if debit > 0 and credit <= 0 else "credit" if credit > 0 and debit <= 0 else "mixed"
        l1 = "" if pd.isna(level1.iloc[position]) else str(level1.iloc[position]).strip().lower()
        l3 = "" if pd.isna(level3.iloc[position]) else str(level3.iloc[position]).strip().lower()
        l5 = "" if pd.isna(level5.iloc[position]) else str(level5.iloc[position]).strip().lower()
        rows.append(
            {
                "text": text,
                "date": _level7_parse_date(date_col.iloc[position]),
                "amount": max(debit, credit),
                "direction": direction,
                "key": level7_counterparty(text),
                "reversal": bool(LEVEL7_REVERSAL_RE.search(text)),
                "cash": l1 == "cash" or bool(ATM_WITHDRAWAL_RE.search(text)),
                "p2p": l3 == "p2p" or l5 == "p2p transfer",
                "bill": bool(LEVEL7_BILL_RE.search(text)) or l5 in LEVEL7_BILL_LEVEL5,
                "reversed": False,
            }
        )

    by_amount: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        if row and not row["reversal"] and row["date"] is not None:
            by_amount[round(row["amount"] * 100)].append(position)
    for position, row in enumerate(rows):
        if not row or not row["reversal"] or row["date"] is None:
            continue
        candidates = [
            other
            for other in by_amount.get(round(row["amount"] * 100), [])
            if not rows[other]["reversed"]
            and rows[other]["direction"] != row["direction"]
            and abs((row["date"] - rows[other]["date"]).days) <= LEVEL7_REVERSAL_WINDOW_DAYS
            and (not row["key"] or rows[other]["key"] == row["key"])
        ]
        if candidates:
            best_candidate = min(candidates, key=lambda other: abs((row["date"] - rows[other]["date"]).days))
            rows[best_candidate]["reversed"] = True

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        if (
            row
            and row["date"] is not None
            and row["key"]
            and not row["reversal"]
            and not row["reversed"]
            and not row["cash"]
            and row["direction"] != "mixed"
        ):
            groups[(row["direction"], row["key"])].append(position)

    history: dict[int, str] = {}
    for members in groups.values():
        by_date: dict[pd.Timestamp, list[int]] = defaultdict(list)
        for position in members:
            by_date[rows[position]["date"]].append(position)
        dates = sorted(by_date)
        if len(dates) < 2:
            continue
        for cadence, low, high, minimum in LEVEL7_CADENCES:
            for chain in _level7_chains(dates, low, high):
                if len(chain) < minimum:
                    continue
                chain_rows = [position for index in chain for position in by_date[dates[index]]]
                chain_dates = [dates[index] for index in chain]
                amounts = []
                for index in chain:
                    distinct = {round(rows[p]["amount"], 2) for p in by_date[dates[index]]}
                    amounts.append(sum(distinct))
                if _level7_chain_qualifies(
                    cadence,
                    len(chain),
                    amounts,
                    chain_dates,
                    any(rows[p]["bill"] for p in chain_rows),
                    any(rows[p]["p2p"] for p in chain_rows),
                ):
                    for position in chain_rows:
                        history.setdefault(position, f"{cadence}_history")

    decisions: list[tuple[str, str]] = []
    for position, row in enumerate(rows):
        if row is None:
            decisions.append(("", "not_a_transaction"))
        elif row["reversal"]:
            decisions.append((LEVEL7_ONE_TIME, "reversal_or_failed"))
        elif row["reversed"]:
            decisions.append((LEVEL7_ONE_TIME, "reversed_original"))
        elif _level7_explicit_evidence(row["text"], row["direction"] == "debit"):
            decisions.append((LEVEL7_RECURRING, "explicit_instruction"))
        elif position in history:
            decisions.append((LEVEL7_RECURRING, history[position]))
        elif row["date"] is None:
            decisions.append((LEVEL7_ONE_TIME, "missing_date"))
        elif not row["key"]:
            decisions.append((LEVEL7_ONE_TIME, "no_counterparty"))
        else:
            decisions.append((LEVEL7_ONE_TIME, "no_reliable_pattern"))
    return decisions


def apply_level7_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Set Level_7 to "recurring" or "one-time" for every valid transaction.

    Only the Level_7 column is written. Rows without a positive Debit or Credit
    are not transactions and get a blank Level_7.
    """

    output_df = df.copy()
    output_df["Level_7"] = [label for label, _ in decide_level7(output_df)]
    return output_df


# ---------------------------------------------------------------------------
# Final Level 1 to Level 5 rules. Every rule is a row in a table below; adding a
# bank's prefix or a new purpose cue is one row, not a code change.
# ---------------------------------------------------------------------------
_LEAD_REF = r"^\s*(?:\d{6,}\s*[-:/ ]\s*)?"
# "Chq: 0" / "Chq: 518512326492" at the end of a narration is the statement's own cheque-number
# column, not evidence of a cheque.
LEVEL1_TRAILING_CHQ_RE = re.compile(r"(?:\s*Chq\s*:?\s*[A-Za-z0-9]*)+\s*$", flags=re.IGNORECASE)
# A returned (bounced) cheque: "REJECT:<no>:<reason>". The reason text is never a payee.
REJECT_RE = re.compile(r"^\s*REJECT\s*:", flags=re.IGNORECASE)
BOUNCED_CHEQUE_LEVELS = {
    "Level_1": "cheque",
    "Level_2": "System generated",
    "Level_3": "",
    "Level_4": "",
    "Level_5": "",
    "Level_6": "",
    "Level_7": "one-time",
}
# The rail a charge, interest or fee line names in its own text (first listed wins).
LEVEL1_RAIL_WORDS: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, flags=re.IGNORECASE), value)
    for pattern, value in (
        (r"(?<![A-Za-z])imps(?![A-Za-z])", "imps"),
        (r"(?<![A-Za-z])neft(?![A-Za-z])", "neft"),
        (r"(?<![A-Za-z])rtgs(?![A-Za-z])", "rtgs"),
        (r"(?<![A-Za-z])upi(?![A-Za-z])", "upi"),
        (r"(?<![A-Za-z])(?:ach|ecs|nach|enach|mandate)(?![A-Za-z])", "nach"),
    )
)


def level1_from_rail_word(text: str) -> str:
    """Return the rail a narration spells out (imps, nach ...), or "" when it names none."""

    core = LEVEL1_TRAILING_CHQ_RE.sub("", str(text or ""))
    for pattern, value in LEVEL1_RAIL_WORDS:
        if pattern.search(core):
            return value
    return ""


# (name, pattern, Level 1) tried in order against the narration without its "Chq:" tail.
# "@rail" = the rail word in the text, blank when there is none. "" = leave blank on purpose:
# these prefixes are not understood well enough to guess.
LEVEL1_FINAL_RULES: tuple[tuple[str, re.Pattern, str], ...] = tuple(
    (name, re.compile(pattern, flags=re.IGNORECASE), value)
    for name, pattern, value in (
        (
            "charge_or_interest",
            _LEAD_REF + r"(?:SBINT(?![A-Za-z])|INT\.?\s*COLL(?![A-Za-z])|SMS\s+CHARGES?(?![A-Za-z])"
            r"|IMPS\s+SC(?![A-Za-z])|ECS\s+MANDATE\s+SC(?![A-Za-z])|ECS\s+DR\s+RTN\s+SC(?![A-Za-z])"
            r"|RTN\s+CHG(?![A-Za-z])|RETURN\s+CHARGE(?![A-Za-z]))",
            "@rail",
        ),
        ("not_guessed", r"^\s*(?:MB/|CMS(?![A-Za-z])|GIB(?![A-Za-z])|TAB-|BIL/)", ""),
        ("cheque", r"^\s*(?:(?:TO|BY)\s+)?(?:CHQ|CLG|CHEQUE|CLEARING)(?![A-Za-z])", "cheque"),
        (
            "internal_transfer",
            _LEAD_REF + r"(?:IFT|INF/INFT|INFT|TRFR|TRF|TPT|IB\s+FUNDS\s+TRANSFER)(?![A-Za-z])",
            "internal transfer",
        ),
        ("cash", r"(?<![A-Za-z])(?:NWD|ATW|ATM[\s\-/|]*WDL|CASH\s+DEP(?:OSIT)?)(?![A-Za-z])", "cash"),
        ("nach", r"(?<![A-Za-z])(?:ACH|ECS|NACH|ENACH|MANDATE)(?![A-Za-z])", "nach"),
    )
)


def decide_level1(text: str, current: str = "") -> str:
    """Return the final Level 1 for a narration; ``current`` is kept when no rule speaks."""

    core = LEVEL1_TRAILING_CHQ_RE.sub("", str(text or "")).strip()
    if CHEQUE_INSTRUMENT_RE.match(core):
        return "cheque"
    for name, pattern, value in LEVEL1_FINAL_RULES:
        if not pattern.search(core):
            continue
        if name == "nach" and UPI_START_RE.match(core):
            continue  # "UPI/.../MANDATE" is a UPI mandate; the UPI structure decides
        return level1_from_rail_word(core) if value == "@rail" else value
    held = str(current or "").strip().lower()
    if held == "auto-debit":
        return "nach"
    if held == "cheque":
        # "cheque" needs a cheque word at the start; a stray flag is dropped, then the
        # structural prefixes get another chance.
        refilled = apply_level1_structural_fill(core, {"Level_1": ""})
        return refilled["Level_1"]
    return str(current or "")


# Level 3: a company-form word in the payee is a business on every rail.
P2M_PAYEE_RE = re.compile(r"/P2M/[^/]*/([^/]+)/", flags=re.IGNORECASE)
BANK_PAYEE_RE = re.compile(r"^[A-Za-z&. ]*\bBANK(?:\s+(?:LTD|LIMITED))?\.?$", flags=re.IGNORECASE)
REFUND_RE = re.compile(r"(?<![A-Za-z])(?:REFUNDS?|RFND|REVERSAL)(?![A-Za-z])", flags=re.IGNORECASE)
BANK_WORD_RE = re.compile(r"\bBANK\b", flags=re.IGNORECASE)
# Level 5: purposes only one direction can have.
DEBIT_ONLY_LEVEL5 = frozenset({"credit card", "loan repayment"})
CREDIT_NARRATION_PURPOSE_OK_RE = re.compile(r"\brepay(?:ment|ments|ed)?\b", flags=re.IGNORECASE)
# (pattern, direction, Level 5, Level 3 used when Level 3 is blank or p2p, extra condition)
LEVEL5_FINAL_RULES: tuple[tuple[re.Pattern, str, str, str, re.Pattern | None], ...] = (
    (
        re.compile(r"(?<![A-Za-z])CC(?![A-Za-z]).{0,40}?AUTO[\s-]?PAY", flags=re.IGNORECASE),
        "debit",
        "credit card",
        "institution",
        None,
    ),
    (DIV_KEYWORD_RE, "credit", "dividend", "institution", ACH_CR_DIV_RE),
    (re.compile(r"^\s*DRP/.*\bUN\s*LOAD\b", flags=re.IGNORECASE), "any", "wallet unload", "", None),
    (re.compile(r"^\s*DRP/.*\bLOAD\b", flags=re.IGNORECASE), "any", "wallet load", "", None),
)
# Level 4: a label whose alias collides with an ordinary word ("cred" inside "credit", "credi",
# "credited") is kept only when that alias stands alone as a token. Add a label here to protect it.
LEVEL4_COLLISION_LABELS = frozenset({"cred"})
# A platform named in the text wins over a bank or a blank: label -> its text aliases.
LEVEL4_TEXT_PLATFORM_ALIASES: dict[str, tuple[str, ...]] = {"paytm": ("one97", "one97 communications")}
# A digit-bearing word standing alone (ONE97), never part of a UPI id ("VIVEN82@OKHDFCBANK",
# "ZOMATO4.PAYU@ICICI") or a longer reference.
BRAND_DIGIT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9@.])([A-Za-z]{2,})(\d{1,4})(?=\s+[A-Za-z]|\s*[/|]|$)")


def restore_digit_brand_tokens(level6: str, particulars: str) -> str:
    """Put the digits back in a brand token the name lost them in (one -> one97)."""

    name = str(level6 or "")
    if not name:
        return name
    for match in BRAND_DIGIT_TOKEN_RE.finditer(str(particulars or "")):
        base, full = match.group(1).lower(), match.group(0).lower()
        if re.search(rf"(?<![a-z0-9]){base}(?![a-z0-9])", name):
            name = re.sub(rf"(?<![a-z0-9]){base}(?![a-z0-9])", full, name, count=1)
    return name


def apply_final_level_rules(
    particulars: str,
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
    master_locked: bool = False,
) -> dict[str, str]:
    """Settle Level 1 to Level 5 with the tables above. Master matches keep their names."""

    text = str(particulars or "")
    core = LEVEL1_TRAILING_CHQ_RE.sub("", text).strip()
    direction = derive_direction_from_amounts(debit_value, credit_value)
    is_credit, is_debit = direction == "credit", direction == "debit"

    # 1. Bounced cheque: nothing but the instrument is known.
    if REJECT_RE.match(core):
        result.update(BOUNCED_CHEQUE_LEVELS)
        return result

    # 2. Level 1.
    result["Level_1"] = decide_level1(text, result.get("Level_1", ""))

    # 3. ATM cash rows: no counterparty, purpose or payee; Level 4 keeps the bank when named.
    if is_debit and result["Level_1"] == "cash" and ATM_WITHDRAWAL_RE.search(core):
        result["Level_3"] = result["Level_5"] = result["Level_6"] = ""
        if not result.get("Level_4"):
            result["Level_4"] = detect_level_4_platform_or_bank(core)

    # 4. Level 3.
    if not master_locked:
        payee_match = P2M_PAYEE_RE.search(core)
        payee_text = BANK_LEGAL_NAME_RE.sub(" ", core)
        if payee_match and BANK_PAYEE_RE.match(payee_match.group(1).strip()):
            result["Level_3"], result["Level_5"] = "institution", ""
        elif result.get("Level_3") == "p2p" and is_company_name(payee_text):
            is_body = bool(BANK_WORD_RE.search(payee_text) or GOVERNMENT_PAYEE_RE.search(core))
            result["Level_3"] = "institution" if is_body else "merchant"
            if result.get("Level_5") == "p2p transfer":
                result["Level_5"] = ""
        if is_credit and REFUND_RE.search(core):
            result["Level_5"] = ""

    # 5. Level 5.
    if not master_locked:
        for pattern, rule_direction, purpose, level_3, condition in LEVEL5_FINAL_RULES:
            if rule_direction == "debit" and not is_debit:
                continue
            if rule_direction == "credit" and not is_credit:
                continue
            if pattern.search(core) and (condition is None or condition.search(core)):
                result["Level_5"] = purpose
                if level_3 and result.get("Level_3", "") in {"", "p2p"}:
                    result["Level_3"] = level_3
                break
    if is_credit:
        purpose = _normalize_compare(result.get("Level_5", ""))
        if purpose == "credit card":
            result["Level_5"] = "p2p transfer" if result.get("Level_3") == "p2p" else ""
        elif purpose in DEBIT_ONLY_LEVEL5 and not CREDIT_NARRATION_PURPOSE_OK_RE.search(core):
            result["Level_5"] = ""

    # 6. Level 4: whole-token platforms; a platform in the text beats a bank.
    level_4 = _normalize_compare(result.get("Level_4", ""))
    if level_4 in LEVEL4_COLLISION_LABELS and not master_locked:
        lowered = core.lower()
        named = any(_contains_alias(lowered, alias) for alias in PLATFORM_ALIASES[level_4])
        if not named and detect_level_4_platform_or_bank(core) != level_4:
            result["Level_4"] = detect_level_4_platform_or_bank(core)
    for label, aliases in LEVEL4_TEXT_PLATFORM_ALIASES.items():
        if any(_contains_alias(core.lower(), alias) for alias in aliases):
            result["Level_4"] = label
            break

    # 7. Digits inside a brand token stay.
    if result.get("Level_6"):
        result["Level_6"] = restore_digit_brand_tokens(result["Level_6"], core)
    return result


def apply_final_level_rules_df(df: pd.DataFrame, locked_rows: set | None = None) -> pd.DataFrame:
    """Re-run the final rules on a classified frame (entity merging can undo a row's labels)."""

    needed = {"Particulars", "Level_1", "Level_3", "Level_4", "Level_5", "Level_6"}
    if not needed.issubset(df.columns):
        return df
    output_df = df.copy()
    locked = locked_rows or set()
    debit = output_df["Debit"] if "Debit" in output_df.columns else pd.Series(None, index=output_df.index)
    credit = output_df["Credit"] if "Credit" in output_df.columns else pd.Series(None, index=output_df.index)
    columns = [c for c in ("Level_1", "Level_2", "Level_3", "Level_4", "Level_5", "Level_6", "Level_7") if c in output_df.columns]
    for index in output_df.index:
        row = {c: ("" if pd.isna(output_df.at[index, c]) else str(output_df.at[index, c])) for c in columns}
        before = dict(row)
        after = apply_final_level_rules(output_df.at[index, "Particulars"], row, debit[index], credit[index], index in locked)
        for column in columns:
            if after.get(column, before[column]) != before[column]:
                output_df.at[index, column] = after[column]
    return output_df



class BankStatementClassifier:
    """Classify full bank statement rows by aggregating split keyword predictions."""

    def __init__(self, models_dir: str = "models/") -> None:
        """Load the underlying keyword classifier."""

        self.keyword_classifier = KeywordClassifier(models_dir=models_dir)

    def _classify_parts(self, flat_parts: list[str]) -> list[dict[str, str]]:
        """Classify the parts, marking the ones that are exact Master Sheet matches."""

        if not flat_parts:
            return []
        with_metadata = getattr(self.keyword_classifier, "classify_batch_with_metadata", None)
        if with_metadata is None:
            return self.keyword_classifier.classify_batch(flat_parts)
        predictions = []
        for row in with_metadata(flat_parts):
            prediction = {name: row.get(name, "") for name in _empty_levels()}
            if row.get("decision_source") == "exact_lookup":
                prediction[EXACT_MASTER_KEY] = True
            predictions.append(prediction)
        return predictions

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a row-level classified dataframe."""

        output_df = df.copy()
        missing_parts = [column for column in PART_COLUMNS if column not in output_df.columns]
        if missing_parts:
            raise KeyError(
                "Missing split columns. Run the statement splitter before classification."
            )

        flat_parts: list[str] = []
        part_positions: list[tuple[int, str]] = []

        for row_index, row in output_df.iterrows():
            for part_column in PART_COLUMNS:
                part_value = str(row.get(part_column, "") or "").strip()
                if part_value:
                    flat_parts.append(part_value)
                    part_positions.append((row_index, part_column))

        predictions = self._classify_parts(flat_parts)

        grouped_predictions: dict[int, list[tuple[str, dict[str, str]]]] = defaultdict(list)
        for (row_index, _), part_text, prediction in zip(part_positions, flat_parts, predictions):
            grouped_predictions[row_index].append((part_text, prediction))

        row_level_predictions = []
        named_rows: set = set()
        locked_rows: set = set()
        for row_index in output_df.index:
            row_parts = _drop_tail_exact_matches(
                str(output_df.loc[row_index, "Particulars"]), grouped_predictions.get(row_index, [])
            )
            row_prediction = aggregate_part_predictions(row_parts)
            row_prediction = apply_structured_upi_overrides(
                row_parts,
                self.keyword_classifier,
                row_prediction,
            )
            row_prediction = apply_hyphen_upi_merchant_override(
                str(output_df.loc[row_index, "Particulars"]),
                row_parts,
                self.keyword_classifier,
                row_prediction,
            )
            row_prediction = apply_raw_transaction_overrides(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                [part_text for part_text, _ in row_parts],
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
            )
            row_prediction = apply_atm_withdrawal_overrides(
                str(output_df.loc[row_index, "Particulars"]),
                row_parts,
                row_prediction,
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
            )
            row_prediction = apply_counterparty_type_override(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                [part_text for part_text, _ in row_parts],
            )
            master_locked = any(
                prediction.get(EXACT_MASTER_KEY) and prediction.get("Level_3")
                for _, prediction in row_parts
            )
            row_prediction = apply_label_layout_override(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                master_locked,
            )
            row_prediction = apply_rail_payee_override(
                str(output_df.loc[row_index, "Particulars"]), row_prediction, master_locked
            )
            row_prediction = apply_initiation_mode_override(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                [part_text for part_text, _ in row_parts],
            )
            row_prediction = apply_debit_salary_guard(
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                row_prediction,
            )
            row_prediction = apply_platform_purpose_guard(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                [part_text for part_text, _ in row_parts],
            )
            row_prediction = apply_narration_purpose_override(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                [part_text for part_text, _ in row_parts],
            )
            row_prediction = apply_cheque_instrument_override(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
            )
            row_prediction = apply_level1_structural_fill(
                str(output_df.loc[row_index, "Particulars"]), row_prediction
            )
            row_prediction = apply_amount_direction_override(
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                row_prediction,
            )
            row_prediction = apply_counterparty_purpose_compat(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                [part_text for part_text, _ in row_parts],
            )
            row_prediction = apply_final_level_rules(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                master_locked,
            )
            row_prediction, layout_named = apply_layout_level6_override(
                str(output_df.loc[row_index, "Particulars"]), row_prediction, master_locked
            )
            if layout_named:
                named_rows.add(row_index)
            if master_locked:
                locked_rows.add(row_index)
            row_level_predictions.append(row_prediction)

        merchant_rows = sum(
            prediction["Level_3"] == "merchant"
            and bool(MERCHANT_MARKER_RE.search(_marker_text(
                str(output_df.loc[row_index, "Particulars"]),
                [part_text for part_text, _ in grouped_predictions.get(row_index, [])],
            )))
            for row_index, prediction in zip(output_df.index, row_level_predictions)
        )
        atm_rows = sum(
            prediction["Level_1"] == "cash"
            and bool(ATM_WITHDRAWAL_RE.search(_marker_text(
                str(output_df.loc[row_index, "Particulars"]),
                [part_text for part_text, _ in grouped_predictions.get(row_index, [])],
            )))
            for row_index, prediction in zip(output_df.index, row_level_predictions)
        )
        logger.info(
            "Explicit-marker overrides applied: %s MERCHANT rows, %s ATM withdrawal rows",
            merchant_rows,
            atm_rows,
        )

        levels_df = pd.DataFrame(row_level_predictions, index=output_df.index)
        output_df = pd.concat([output_df, levels_df], axis=1)
        output_df = apply_cheque_return_level1(output_df)
        output_df = apply_canonical_level6(output_df, named_rows)
        output_df = apply_entity_level3(output_df, named_rows, locked_rows)
        output_df = apply_final_level_rules_df(output_df, locked_rows)
        output_df["First Level Classification"] = output_df.apply(
            derive_first_level_classification,
            axis=1,
        )
        output_df = apply_first_level_contra(output_df)
        output_df["Second Level Classification"] = output_df.apply(
            derive_second_level_classification,
            axis=1,
        )

        output_df = apply_pre_detect_salary_overrides(output_df)
        output_df = detect_salary(output_df)
        output_df = apply_strict_level2(output_df)
        output_df = apply_level7_recurrence(output_df)
        output_df = apply_final_level_rules_df(output_df, locked_rows)
        logger.info("Classified %s statement rows", len(output_df))
        return output_df