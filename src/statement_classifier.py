"""Transaction-level classification built on top of keyword models."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

# Support both normal package imports (``from src.statement_classifier ...``)
# and direct module loading from the repository root / ``src`` directory.
if __package__:
    from .predict import KeywordClassifier
    from .prompt_part3 import clean_entity, detect_explicit_purpose
else:  # pragma: no cover - used by IDEs or scripts loading this file directly
    try:
        from src.predict import KeywordClassifier
        from src.prompt_part3 import clean_entity, detect_explicit_purpose
    except ModuleNotFoundError:
        from predict import KeywordClassifier
        from prompt_part3 import clean_entity, detect_explicit_purpose

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


def has_merchant_vpa(particulars: str) -> bool:
    """Return True when the narration's UPI ID is a merchant-only ID."""

    compact = re.sub(r"\s+", "", str(particulars or ""))
    if MERCHANT_VPA_RE.search(compact):
        return True
    return bool(PAYTM_NUMERIC_VPA_RE.search(compact) and MERCHANT_ORDER_ID_RE.search(compact))
MERCHANT_MARKER_RE = re.compile(
    r"(?:^|[/|])\s*MERCHANT\s*(?:[/|]|$)",
    flags=re.IGNORECASE,
)
# Whole words only, so "cashfree" or "cashback" never match. WDL / WITHDRAWAL /
# EAW / NFS are cash only when the narration is not a charge or fee.
ATM_WITHDRAWAL_RE = re.compile(
    r"\bATM[\s\-/|]*WDL\b|\bATM\s*[/|]\s*DR\s*(?:[/|]|$)|\bATW(?![A-Za-z])"
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
    {"gpay", "google", "paytm", "phonepe", "razorpay", "bharatpe", "mobikwik", "cred", "payu", "cashfree"}
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
    "google pay": ("google pay", "googlepay", "gpay", "tez"),
    "phonepe": ("phonepe", "phone pe"),
    "paytm": ("paytm", "paytm upi", "paytm payments bank"),
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
}
UPI_APP_HANDLE_RE = re.compile(
    r"@\s*(okhdfcbank|okhdfc|okicici|oksbi|okaxis|ybl|ibl|axl)(?![a-z0-9])"
)

BANK_ALIASES = {
    "state bank of india": ("state bank of india", "sbi", "sbin"),
    "hdfc bank": ("hdfc bank", "hdfc"),
    "icici bank": ("icici bank", "icici"),
    "axis bank": ("axis bank", "axis", "utib"),
    "kotak mahindra bank": ("kotak mahindra bank", "kotak bank", "kotak", "kkbk"),
    "punjab national bank": ("punjab national bank", "pnb", "punb"),
    "bank of baroda": ("bank of baroda", "bob", "barb"),
    "bank of india": ("bank of india", "boi"),
    "canara bank": ("canara bank", "canara", "cnrb"),
    "union bank of india": ("union bank of india", "union bank", "ubin"),
    "indian bank": ("indian bank", "idib"),
    "indian overseas bank": ("indian overseas bank", "iob", "ioba"),
    "central bank of india": ("central bank of india", "cbi", "cbin"),
    "uco bank": ("uco bank", "uco"),
    "bank of maharashtra": ("bank of maharashtra", "mahabank", "mahb"),
    "punjab and sind bank": ("punjab and sind bank", "psb", "psib"),
    "yes bank": ("yes bank", "yesb"),
    "idfc first bank": ("idfc first bank", "idfc bank", "idfc", "idfb"),
    "indusind bank": ("indusind bank", "indusind", "inbk"),
    "federal bank": ("federal bank", "federal", "fdr", "fdrl"),
    "bandhan bank": ("bandhan bank", "bandhan", "bdbl"),
    "rbl bank": ("rbl bank", "rbl", "ratn"),
    "south indian bank": ("south indian bank", "sib", "sibl"),
    "karnataka bank": ("karnataka bank", "ktkb"),
    "city union bank": ("city union bank", "ciub"),
    "karur vysya bank": ("karur vysya bank", "kvb", "kvbl"),
    "dcb bank": ("dcb bank", "dcb"),
    "tamilnad mercantile bank": ("tamilnad mercantile bank", "tmb", "tmbl"),
    "au small finance bank": ("au small finance bank", "au bank", "aubl"),
    "idbi bank": ("idbi bank", "idbi", "ibkl"),
    "cosmos co-operative bank": ("cosmos co-operative bank", "cosmos bank", "cosb"),
    "equitas small finance bank": ("equitas small finance bank", "equitas", "esfb"),
    "ujjivan small finance bank": ("ujjivan small finance bank", "ujjivan", "usfb"),
    "jana small finance bank": ("jana small finance bank", "jana", "jsfb"),
    "paytm payments bank": ("paytm payments bank", "paytm bank"),
    "india post payments bank": ("india post payments bank", "ippb"),
    "fino payments bank": ("fino payments bank", "fino"),
    "airtel payments bank": ("airtel payments bank", "airtel bank"),
    "dbs bank": ("dbs bank", "dbsbank", "dbs", "dbss"),
}


def _contains_alias(text: str, alias: str) -> bool:
    """Match an alias as a phrase/token, not as part of a longer word."""
    escaped = re.escape(alias.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text))


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
        return UPI_APP_HANDLES[handle_match.group(1)]

    bank_matches = []
    for canonical, aliases in BANK_ALIASES.items():
        for alias in aliases:
            if _contains_alias(normalized, alias):
                bank_matches.append((len(alias), canonical))
    if bank_matches:
        bank_matches.sort(reverse=True)
        return bank_matches[0][1]

    return ""


def _is_platform_label(label: str) -> bool:
    """Return True for a Level 4 label that names a platform or gateway, not a bank.

    Unlike ``is_payment_intermediary_label`` this counts every ``PLATFORM_ALIASES``
    name (paytm, billdesk, getepay ...), because for Level 4 the platform always
    wins over the bank.
    """

    value = _normalize_compare(label)
    return bool(value) and (value in PLATFORM_ALIASES or is_payment_intermediary_label(value))


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
            r"|jiomart|instamart|swiggy\s*instamart|supermarket|kirana|fresh\s*mart|sabji"
            r"|vegetable|fruit\s*shop|daily\s*needs|ever\s*green\s*fresh"
            r"|milk\s*booth|dairy|sorabh\s*dairy)\b",
            re.IGNORECASE,
        ),
        "grocery",
    ),
    (
        re.compile(
            r"\b(?:restaurant|zomato|swiggy|cafe|dhaba|food\s*court"
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
            r"\b(?:petrol|diesel|fuel|hpcl|iocl|bpcl|hp\s*petrol"
            r"|cng\s*station|gas\s*station|shell|petrol\s*pump"
            r"|fuel\s*station|bharat\s*petroleum|indian\s*oil)\b",
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
            r"\b(?:groww|zerodha|upstox|kuvera|smallcase|paytm\s*money"
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
        re.compile(
            r"\b(?:paytm|gpay|google\s*pay|phonepe|razorpay|cashfree"
            r"|freecharge|payu|bharatpe|billdesk|juspay|mobikwik)\b",
            re.IGNORECASE,
        ),
        "digital services",
    ),
]


def apply_narration_purpose_override(
    particulars: str,
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
    part_texts: list[str] | None = None,
) -> dict[str, str]:
    """Fill a blank Level 5 purpose from the full narration when keyword model left it empty."""

    if result.get("Level_5"):
        return result

    if result.get("Level_1") == "cash":
        return result

    text = _marker_text(particulars, part_texts)
    direction = derive_direction_from_amounts(debit_value, credit_value)

    for pattern, purpose in NARRATION_LEVEL5_PATTERNS:
        if not pattern.search(text):
            continue

        if purpose == "salary" and direction == "debit":
            continue
        if purpose == "credit card" and direction == "credit":
            continue

        result["Level_5"] = purpose
        break

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
    }
)
# Counterparty position in common bank narration layouts (used for Level 6).
#   NEFT-<UTR>-<NAME>-...          RTGS RETURN-<UTR>-<NAME>
#   CLG|TRF/<DRAWER>/<chq no>/...  cheque instruments
#   INF|BIL/INFT/<ref>/.../<NAME>  in-bank net-banking transfers (name is last)
#   MMT/IMPS/<rrn>/<remark>/<NAME>/<bank>
#   TRFR TO|FROM: <NAME>           in-bank transfers
HYPHEN_RAIL_ENTITY_RE = re.compile(
    r"^\s*(?:NEFT|RTGS|IMPS)(?:\s+RETURN)?-[A-Z0-9]*\d{5,}[A-Z0-9]*-([^-]+)",
    flags=re.IGNORECASE,
)
CHEQUE_ENTITY_RE = re.compile(r"^\s*(?:CLG|TRF)/([^/]+)/", flags=re.IGNORECASE)
INFT_ENTITY_RE = re.compile(r"^\s*(?:INF|BIL)/INFT/[^/]*\d[^/]*/(.*)$", flags=re.IGNORECASE)
MMT_IMPS_ENTITY_RE = re.compile(r"^\s*MMT/IMPS/\d{6,}/[^/]*/([^/]+)", flags=re.IGNORECASE)
INTERNAL_TRANSFER_ENTITY_RE = re.compile(r"^\s*TRFR\s+(?:TO|FROM)\s*:\s*(.+)$", flags=re.IGNORECASE)
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


def extract_level6_entity(particulars: str) -> str:
    """Return the counterparty named by the narration, for Level 6.

    Uses ``extract_narration_entity`` (slash rails, BBPS billers) first, then
    the bank layouts above. Kept separate from ``extract_narration_entity`` so
    the Level 7 counterparty key, which calls that function, is unchanged.
    """

    entity = extract_narration_entity(particulars)
    if entity:
        return entity

    text = str(particulars or "")
    for pattern in (HYPHEN_RAIL_ENTITY_RE, CHEQUE_ENTITY_RE, MMT_IMPS_ENTITY_RE, INTERNAL_TRANSFER_ENTITY_RE):
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
    is_category = bool(current) and _normalize_compare(current) == _normalize_compare(result.get("Level_5", ""))
    biller = BBPS_BILLER_RE.search(str(particulars or ""))
    is_raw_biller_code = bool(biller) and re.sub(r"[^a-z0-9]", "", _normalize_compare(current)) == re.sub(
        r"[^a-z0-9]", "", biller.group(1).lower()
    )
    entity = extract_level6_entity(particulars)
    is_fragment_copy = (
        bool(entity)
        and _normalize_compare(current) != entity
        and _echoes_narration_part(current, particulars, part_texts)
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
    elif is_category:
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

    if (
        result["Level_4"]
        and result["Level_6"]
        and _normalize_compare(result["Level_4"]) == _normalize_compare(result["Level_6"])
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
    if compact[:4] in BANK_IFSC_CODES and re.fullmatch(r"[a-z]{4}(?:[a-z]?\d{6,}|[a-z]?|0[a-z0-9]{6})", compact):
        return True
    detected = detect_level_4_platform_or_bank(part_text)
    if not detected or is_payment_intermediary_label(detected):
        return False
    words = re.findall(r"[a-z0-9]+", str(part_text).lower())
    return all(word in BANK_NAME_WORDS for word in words)


def aggregate_part_predictions(
    part_predictions: list[tuple[str, dict[str, str]]],
) -> dict[str, str]:
    """Combine part-level predictions into one row-level result."""

    result = _empty_levels()
    for _, prediction in part_predictions:
        for level_index in range(8):
            level_name = f"Level_{level_index}"
            if not result[level_name] and prediction.get(level_name):
                result[level_name] = prediction[level_name]

    # Level 3: a part that only names a bank (e.g. the UTR prefix "HDFCN") says
    # which bank moved the money, not who the counterparty is. If it supplied
    # "institution", use the counterparty type from a later part that names an
    # actual payer/payee (e.g. "BLUE APPARELS" -> merchant), when one exists.
    first_level3 = next(
        ((part_text, prediction) for part_text, prediction in part_predictions if prediction.get("Level_3")),
        None,
    )
    if (
        first_level3
        and first_level3[1]["Level_3"] == "institution"
        and is_bank_only_part(first_level3[0])
    ):
        counterparty_level3 = next(
            (
                prediction["Level_3"]
                for part_text, prediction in part_predictions
                if prediction.get("Level_3") and not is_bank_only_part(part_text)
            ),
            "",
        )
        if counterparty_level3:
            result["Level_3"] = counterparty_level3

    specific_level6 = next(
        (
            prediction["Level_6"]
            for _, prediction in part_predictions
            if prediction.get("Level_6") and not is_generic_level6(prediction["Level_6"])
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
    letters = re.sub(r"[^a-z]", "", value)
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

    fill_specific_level6(text, result, part_texts)

    return _apply_transaction_rules(
        result,
        default_p2p_purpose=not automated_debit,
        confirmed_level6=extract_level6_entity(text),
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
    if BANK_CHARGE_RE.match(text) or GOVERNMENT_PAYEE_RE.search(text):
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
    return " ".join(kept)


def derive_first_level_classification(row: pd.Series) -> str:
    """Map row-level predictions into a coarse business transaction label."""

    debit_num = parse_amount(row.get("Debit"))
    credit_num = parse_amount(row.get("Credit"))
    level_1 = str(row.get("Level_1", "") or "").strip().lower()
    level_2 = str(row.get("Level_2", "") or "").strip().lower()
    level_5 = str(row.get("Level_5", "") or "").strip().lower()
    particulars = str(row.get("Particulars", "") or "")
    is_debit = debit_num > 0 and credit_num <= 0

    if level_5 == "salary" and not is_debit:
        return "Salary"
    if SALARY_KEYWORDS.search(particulars) and credit_num > 0:
        return "Salary"
    if level_1 == "upi" and credit_num > 0:
        return "UPI Receipt"
    if level_1 == "upi":
        return "UPI Payment"
    if level_1 == "neft":
        return "NEFT Payment"
    if level_1 == "rtgs":
        return "RTGS Payment"
    if level_1 == "imps":
        return "IMPS Payment"
    if level_1 == "cheque":
        return "Cheque Payment"
    if level_1 == "cash" and debit_num > 0:
        return "Cash Withdrawal"
    if level_1 == "cash" and credit_num > 0:
        return "Cash Deposit"
    # Level_2 holds "System generated" / "User-generated", never "auto-debit": a
    # system-generated row whose narration is a mandate debit is the ECS/NACH row.
    if level_2 in {"auto-debit", "system generated"} and is_debit and MANDATE_DEBIT_RE.search(particulars):
        return "ECS/NACH Payment"
    if level_5 == "loan repayment":
        return "Loan"
    if level_5 == "credit card":
        return "Credit Card Payment"
    if level_5 in {"utilities", "mobile recharge", "bill payment"}:
        return "Utility Expenses"
    if level_5 == "insurance":
        return "Insurance"
    if level_5 == "tax":
        return "Tax"
    if level_5 in {"investment", "financial services"}:
        return "Investment"
    if "refund" in particulars.lower() or level_5 == "refund":
        return "Refund/Reversal"
    return derive_first_level_fallback(particulars, level_1, level_5)


# Fallback First Level labels, used only when none of the rules above applies.
# Each needs explicit narration evidence; anything else stays blank.
FIRST_LEVEL_FALLBACKS: list[tuple[re.Pattern[str], str]] = [
    # Reversals of UPI / card transactions: "UPIRET-...", "CRV POS ..."
    (re.compile(r"^\s*(?:UPI\s*RET|CRV\s+POS)\b", re.IGNORECASE), "Refund/Reversal"),
    # Bank charges: "RTN CHG-...", "IMPS Chg May-25+GST", "Chq rtn Chg ..."
    (BANK_CHARGE_RE, "Bank Charges"),
    # Interest: "...:Int.Coll:...", "...:Int.Pd:...", "INTEREST PAID TILL ..."
    (re.compile(r"\bInt\.\s*(?:Coll|Pd)\b|\bINTEREST\s+(?:PAID|CREDITED|COLLECTED)\b", re.IGNORECASE), "Interest"),
    # Dividends: "DR REDDYS DIV 23 ...", "ACH C- HCL 1ST INTDIV24 ...", and codes
    # run together with the company name ("TCSFINDIV...", "TCS3INTSPLDIV...").
    (re.compile(r"\bDIV(?:IDEND)?S?\d*\b|(?:INT|FIN|SPL)DIV", re.IGNORECASE), "Dividend"),
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
    return level_3.title() if level_3 else ""


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


SALARY_CHEQUE_RAIL_RE = re.compile(r"\b(?:CLG|CLEARING|CHEQUE|CHQ|CHEQ)\b", flags=re.IGNORECASE)


def _is_non_salary_counterparty(row: pd.Series) -> bool:
    """Cheque credits and business counterparties are never inferred salary.

    Only used for rows without a salary word: such rows may not become
    salary on recurrence and date consistency alone.
    """

    text = str(row.get("Particulars", "") or "")
    if str(row.get("Level_1", "") or "").strip().lower() == "cheque" or SALARY_CHEQUE_RAIL_RE.search(text):
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
            output_df.at[idx, "Level_1"] = "cheque"
            output_df.at[idx, "Level_3"] = "institution"
            output_df.at[idx, "Level_5"] = "charges"
            output_df.at[idx, "Level_6"] = "bank charges"
            continue

        # Rule 2: ACH credit + dividend
        if credit_num[idx] > 0 and ACH_CR_DIV_RE.search(text) and DIV_KEYWORD_RE.search(text):
            company = extract_level6_entity(text)
            output_df.at[idx, "Level_1"] = "ach"
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
    output_df.loc[salary_mask, "First Level Classification"] = "Salary"
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


class BankStatementClassifier:
    """Classify full bank statement rows by aggregating split keyword predictions."""

    def __init__(self, models_dir: str = "models/") -> None:
        """Load the underlying keyword classifier."""

        self.keyword_classifier = KeywordClassifier(models_dir=models_dir)

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

        predictions = self.keyword_classifier.classify_batch(flat_parts) if flat_parts else []

        grouped_predictions: dict[int, list[tuple[str, dict[str, str]]]] = defaultdict(list)
        for (row_index, _), part_text, prediction in zip(part_positions, flat_parts, predictions):
            grouped_predictions[row_index].append((part_text, prediction))

        row_level_predictions = []
        for row_index in output_df.index:
            row_parts = grouped_predictions.get(row_index, [])
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
            row_prediction = apply_amount_direction_override(
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
                row_prediction,
            )
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
        output_df["First Level Classification"] = output_df.apply(
            derive_first_level_classification,
            axis=1,
        )
        output_df["Second Level Classification"] = output_df.apply(
            derive_second_level_classification,
            axis=1,
        )

        output_df = apply_pre_detect_salary_overrides(output_df)
        output_df = detect_salary(output_df)
        output_df = apply_strict_level2(output_df)
        output_df = apply_level7_recurrence(output_df)
        logger.info("Classified %s statement rows", len(output_df))
        return output_df