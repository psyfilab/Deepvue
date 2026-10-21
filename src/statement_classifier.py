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
SALARY_KEYWORDS = re.compile(
    r"\b(?:salary|salaryxce|payroll|payslip|labour|wages|sal)\b",
    flags=re.IGNORECASE,
)
SALARY_ENTITY_STOPWORDS = re.compile(
    r"\b(?:salary|salaryxce|payroll|payslip|labour|wages|sal|credit|credited|via|by|from|for)\b",
    flags=re.IGNORECASE,
)
MIN_SALARY = 10_000
MIN_SALARY_MONTHS = 2
SALARY_DAY_STD_MAX = 7
SLASH_UPI_RE = re.compile(
    r"\bUPI\b(?:/\d+)?/(?:DR|CR)(?:/\d+)?/([^/\d\-][^/]{1,19})/([A-Z]{2,6})/(\S+)",
    flags=re.IGNORECASE,
)
# Explicit merchant marker as its own narration segment, e.g. ".../MERCHANT/...".
# Segments may be separated by "/" (raw slash format) or "|" (pipe format /
# joined split parts), so both are accepted.
MERCHANT_MARKER_RE = re.compile(
    r"(?:^|[/|])\s*MERCHANT\s*(?:[/|]|$)",
    flags=re.IGNORECASE,
)
# ATM cash withdrawal narrations, e.g. "ATM WDL ...", "ATM-WDL", "ATM/DR/...",
# the same segments in pipe format ("ATM | DR | ..."), or the "ATW" form used
# by several banks and the NFS network ("ATW-512345XXXXXX6789-...", "NFS/ATW/...").
# ATW must stand alone as a token so words such as "ATWOOD" do not match.
ATM_WITHDRAWAL_RE = re.compile(
    r"\bATM[\s\-/|]*WDL\b|\bATM\s*[/|]\s*DR\s*(?:[/|]|$)|\bATW(?![A-Za-z])",
    flags=re.IGNORECASE,
)
SYSTEM_INITIATED_RE = re.compile(
    r"\b(?:"
    r"NACH|ACH|ECS|"
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
    """Return the narration plus its split parts, joined for marker detection.

    The split parts keep each narration segment intact even when the
    Particulars text has been normalised, so explicit markers such as
    MERCHANT or ATM/DR are detected in either form.
    """

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
# Payment intermediaries sit between the payer and the shop. They show up in
# narrations (e.g. "RAZORPAYZOMATO", "PAYVIARAZORPAY") but are never the
# merchant the person actually paid, so they are not used as the Level_6 name.
PAYMENT_INTERMEDIARY_FRAGMENTS = frozenset(
    {"gpay", "google", "paytm", "phonepe", "razorpay", "bharatpe", "mobikwik", "cred", "payu", "cashfree"}
)
KNOWN_MERCHANT_FRAGMENTS = GATEWAY_VPA_FRAGMENTS - PAYMENT_INTERMEDIARY_FRAGMENTS
# Fallback Level_5 purpose for a recognised merchant when the keyword model has
# no purpose for the merchant name itself.
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
# Hyphen-delimited UPI narrations, e.g. "UPI-<payee>-<vpa>@<psp>-<ifsc>-<ref>-<note>".
HYPHEN_UPI_RE = re.compile(r"^\s*UPI\s*-", flags=re.IGNORECASE)
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
# IFSC prefixes are not always the same as the short bank codes above
# (ICICI's IFSC starts with ICIC, while its VPA handle is "icici").
IFSC_PREFIX_TO_BANK = {**BANK_CODE_TO_NAME, "icic": "icici bank"}
IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

# Level 4 precedence:
# 1. Explicit payment platform / payment intermediary
# 2. Explicit bank name or IFSC bank code
# 3. Blank if neither is identifiable

PLATFORM_ALIASES = {
    "google pay": (
        "google pay",
        "googlepay",
        "gpay",
        "tez",
    ),
    "phonepe": (
        "phonepe",
        "phone pe",
    ),
    "paytm": (
        "paytm",
        "paytm upi",
        "paytm payments bank",
    ),
    "amazon pay": (
        "amazon pay",
        "amazonpay",
    ),
    "bhim": (
        "bhim",
        "bharat interface for money",
    ),
    "bharatpe": (
        "bharatpe",
        "bharat pe",
    ),
    "mobikwik": (
        "mobikwik",
        "mobi kwik",
    ),
    "cred": (
        "cred pay",
        "cred",
    ),
    "whatsapp pay": (
        "whatsapp pay",
        "whatsapp payments",
    ),
    "jiofinance": (
        "jiofinance",
        "jio finance",
    ),
    "razorpay": (
        "razorpay",
        "razor pay",
    ),
    "cashfree": (
        "cashfree",
        "cash free",
    ),
    "payu": (
        "payu",
        "pay u",
    ),
    "billdesk": (
        "billdesk",
        "bill desk",
    ),
    "juspay": (
        "juspay",
        "jus pay",
    ),
    "pine labs": (
        "pine labs",
        "pinelabs",
    ),
    "payzapp": (
        "payzapp",
        "pay zapp",
    ),
    "freecharge": (
        "freecharge",
        "free charge",
    ),
    "mobikwik zip": (
        "mobikwik zip",
    ),
}

# Distinctive platform names that are also recognised when glued to other
# letters inside one narration token, e.g. "RAZORPAYZOMATO" or "PAYVIARAZORPAY".
# Short or ambiguous aliases (cred, payu, bhim, tez, gpay) are deliberately
# excluded because they occur inside ordinary words such as "CREDIT".
EMBEDDED_PLATFORM_ALIASES = {
    "razorpay": "razorpay",
    "cashfree": "cashfree",
    "billdesk": "billdesk",
    "phonepe": "phonepe",
    "bharatpe": "bharatpe",
    "mobikwik": "mobikwik",
    "juspay": "juspay",
    "freecharge": "freecharge",
    "payzapp": "payzapp",
    "googlepay": "google pay",
    "amazonpay": "amazon pay",
    "pinelabs": "pine labs",
    "paytm": "paytm",
}

# UPI handles (the part after "@" in a VPA) issued only by one app. Matched
# only directly after "@", so words such as "OKAY" or "SYBIL" never count.
# "okhdfc" covers narrations that truncate "@okhdfcbank".
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

# Alias -> normalized bank name.
# Keep IFSC prefixes and bank-name aliases together so both can be detected.
BANK_ALIASES = {
    "state bank of india": (
        "state bank of india",
        "sbi",
        "sbin",
    ),
    "hdfc bank": (
        "hdfc bank",
        "hdfc",
    ),
    "icici bank": (
        "icici bank",
        "icici",
    ),
    "axis bank": (
        "axis bank",
        "axis",
        "utib",
    ),
    "kotak mahindra bank": (
        "kotak mahindra bank",
        "kotak bank",
        "kotak",
    ),
    "punjab national bank": (
        "punjab national bank",
        "pnb",
        "punb",
    ),
    "bank of baroda": (
        "bank of baroda",
        "bob",
    ),
    "bank of india": (
        "bank of india",
        "boi",
    ),
    "canara bank": (
        "canara bank",
        "canara",
        "cnrb",
    ),
    "union bank of india": (
        "union bank of india",
        "union bank",
        "ubin",
    ),
    "indian bank": (
        "indian bank",
        "idib",
    ),
    "indian overseas bank": (
        "indian overseas bank",
        "iob",
        "ioba",
    ),
    "central bank of india": (
        "central bank of india",
        "cbi",
        "cbin",
    ),
    "uco bank": (
        "uco bank",
        "uco",
    ),
    "bank of maharashtra": (
        "bank of maharashtra",
        "mahabank",
        "mahb",
    ),
    "punjab and sind bank": (
        "punjab and sind bank",
        "psb",
        "psib",
    ),
    "yes bank": (
        "yes bank",
        "yesb",
    ),
    "idfc first bank": (
        "idfc first bank",
        "idfc bank",
        "idfc",
    ),
    "indusind bank": (
        "indusind bank",
        "indusind",
        "inbk",
    ),
    "federal bank": (
        "federal bank",
        "federal",
        "fdr",
    ),
    "bandhan bank": (
        "bandhan bank",
        "bandhan",
        "bdbl",
    ),
    "rbl bank": (
        "rbl bank",
        "rbl",
    ),
    "south indian bank": (
        "south indian bank",
        "sib",
        "sibl",
    ),
    "karnataka bank": (
        "karnataka bank",
        "ktkb",
    ),
    "city union bank": (
        "city union bank",
        "ciub",
    ),
    "karur vysya bank": (
        "karur vysya bank",
        "kvb",
        "kvbl",
    ),
    "dcb bank": (
        "dcb bank",
        "dcb",
    ),
    "tamilnad mercantile bank": (
        "tamilnad mercantile bank",
        "tmb",
        "tmbl",
    ),
    "au small finance bank": (
        "au small finance bank",
        "au bank",
        "aubl",
    ),
    "equitas small finance bank": (
        "equitas small finance bank",
        "equitas",
        "esfb",
    ),
    "ujjivan small finance bank": (
        "ujjivan small finance bank",
        "ujjivan",
        "usfb",
    ),
    "jana small finance bank": (
        "jana small finance bank",
        "jana",
        "jsfb",
    ),
    "paytm payments bank": (
        "paytm payments bank",
        "paytm bank",
    ),
    "india post payments bank": (
        "india post payments bank",
        "ippb",
    ),
    "fino payments bank": (
        "fino payments bank",
        "fino",
    ),
    "airtel payments bank": (
        "airtel payments bank",
        "airtel bank",
    ),
}


def _contains_alias(text: str, alias: str) -> bool:
    """Match an alias as a phrase/token, not as part of a longer word."""
    escaped = re.escape(alias.strip())
    # Permit spaces in multi-word names to match repeated whitespace.
    escaped = escaped.replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text))


def detect_level_4_platform_or_bank(text: str) -> str:
    """
    Return platform first; if no platform is found, return the bank.
    Returns an empty string when there is insufficient evidence.
    """
    normalized = str(text or "").lower()
    if not normalized.strip():
        return ""

    # Longest alias first to avoid shorter aliases winning prematurely.
    platform_matches = []
    for canonical, aliases in PLATFORM_ALIASES.items():
        for alias in aliases:
            if _contains_alias(normalized, alias):
                platform_matches.append((len(alias), canonical))
    if platform_matches:
        platform_matches.sort(reverse=True)
        return platform_matches[0][1]

    # A distinctive platform name glued inside a longer token still counts as
    # platform evidence and must win over any bank in the same narration.
    for token in re.findall(r"[a-z0-9]+", normalized):
        for alias, canonical in EMBEDDED_PLATFORM_ALIASES.items():
            if alias in token:
                platform_matches.append((len(alias), canonical))
    if platform_matches:
        platform_matches.sort(reverse=True)
        return platform_matches[0][1]

    # A UPI handle that only one app issues is reliable platform evidence.
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


# ---------------------------------------------------------------------------
# Narration-level Level 5 purpose inference
# ---------------------------------------------------------------------------
# These patterns run on the *full narration* (Particulars), which typically
# contains more evidence than any single keyword part.  They are applied only
# when Level 5 is blank after all keyword-model and structured-UPI overrides.
# Order matters: more specific patterns come before broader ones that share
# keywords.  Labels reuse the ALLOWED_PURPOSES vocabulary from prompt_part3.py.
NARRATION_LEVEL5_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Credit-card bill payment (must beat generic "loan repayment" rule)
    (
        re.compile(
            r"\b(?:credit\s*card|cc\s*bill|ccbp|ccpmt|cc\s*payment|card\s*payment"
            r"|hdfc\s*cc|icici\s*cc|axis\s*cc|sbi\s*cc|kotak\s*cc)\b",
            re.IGNORECASE,
        ),
        "credit card",
    ),
    # Loan / EMI repayment
    (
        re.compile(
            r"\b(?:emi|loan\s*repay|loan\s*emi|home\s*loan|personal\s*loan"
            r"|car\s*loan|auto\s*loan|education\s*loan|loan\s*instal+ment"
            r"|equated\s*monthly|loan\s*recovery|ecs\s*emi)\b",
            re.IGNORECASE,
        ),
        "loan repayment",
    ),
    # Insurance premium
    (
        re.compile(
            r"\b(?:insurance|insur|policy\s*premium|premium\s*payment"
            r"|lic\s*premium|\blic\b|acko|ergo|bupa|niva\s*bupa|star\s*health"
            r"|icici\s*lombard|hdfc\s*ergo|bajaj\s*allianz|new\s*india\s*assurance)\b",
            re.IGNORECASE,
        ),
        "insurance",
    ),
    # Salary / payroll (applied only to credits — guarded at call site)
    (
        re.compile(
            r"\b(?:salary|payroll|wages|\bsal\b|payslip)\b",
            re.IGNORECASE,
        ),
        "salary",
    ),
    # Tax payments
    (
        re.compile(
            r"\b(?:income\s*tax|advance\s*tax|tds\s*payment|gst\s*payment"
            r"|itdtax|stax|tax\s*payment)\b",
            re.IGNORECASE,
        ),
        "tax",
    ),
    # Mobile / DTH recharge (recharge keyword only; dth recharge belongs to utilities)
    (
        re.compile(
            r"\b(?:topup|top\s*up|prepaid\s*recharge|mobile\s*recharge"
            r"|sim\s*recharge)\b"
            r"|\brecharge\b(?!.*\b(?:dth|dish\s*tv|tata\s*sky|d2h|videocon\s*d2h)\b)",
            re.IGNORECASE,
        ),
        "mobile recharge",
    ),
    # Utilities & bills (electricity, gas, broadband, DTH including DTH recharge)
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
    # Pharmacy / medicine — before dining/healthcare to prevent cafe/health-bar overlap
    (
        re.compile(
            r"\b(?:pharmacy|pharma|medicos?|medicals?|medi\s*store|chemist|drugstore"
            r"|pharmeasy|medplus|netmeds|1mg|apollopharmacy|lenskar"
            r"|ayurveda\s*kendra|ayurved|homeopath|ayurvedic)\b",
            re.IGNORECASE,
        ),
        "pharmacy",
    ),
    # Healthcare (hospitals, clinics — distinct from pharmacy)
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
    # Grocery — placed BEFORE dining so "Swiggy Instamart" hits grocery not dining
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
    # Dining / food — placed after grocery
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
    # Fuel / petrol
    (
        re.compile(
            r"\b(?:petrol|diesel|fuel|hpcl|iocl|bpcl|hp\s*petrol"
            r"|cng\s*station|gas\s*station|shell|petrol\s*pump"
            r"|fuel\s*station|bharat\s*petroleum|indian\s*oil)\b",
            re.IGNORECASE,
        ),
        "fuel",
    ),
    # Travel / flight / accommodation bookings
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
    # Transport / ride-hailing / metro / railways
    (
        re.compile(
            r"\b(?:uber|olacabs|ola\s*cabs?|rapido|irctc|indian\s*rail"
            r"|metro\s*card|metro\s*rail|city\s*bus|pmpml|msrtc|dtc\s*bus"
            r"|cab\s*service|taxi|airport\s*transfer|wti\s*cabs?)\b",
            re.IGNORECASE,
        ),
        "transport",
    ),
    # Education / school / college fees
    (
        re.compile(
            r"\b(?:school\s*fee|college\s*fee|tuition|tuition\s*fee"
            r"|university\s*fee|course\s*fee|coaching|byju|unacademy|vedantu"
            r"|education\s*fee|exam\s*fee|admission\s*fee)\b",
            re.IGNORECASE,
        ),
        "education",
    ),
    # Investments / SIP / mutual funds
    (
        re.compile(
            r"\b(?:groww|zerodha|upstox|kuvera|smallcase|paytm\s*money"
            r"|mutual\s*fund|sip\s*payment|sip\s*debit|nps\s*contribution"
            r"|demat|ppf\s*deposit|fd\s*deposit|rd\s*deposit|investment)\b",
            re.IGNORECASE,
        ),
        "financial services",
    ),
    # Subscriptions / streaming / OTT
    (
        re.compile(
            r"\b(?:netflix|spotify|hotstar|primevideo|youtube\s*premium"
            r"|jiocinema|zee5|sonyliv|voot|mxplayer|bookmyshow"
            r"|subscription|annual\s*plan)\b",
            re.IGNORECASE,
        ),
        "subscription",
    ),
    # Retail / shopping (broad — after more specific retail sub-categories above)
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
    # Digital services / payment platforms as last resort
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
    """Fill a blank Level 5 purpose from the full narration when the keyword
    model left it empty.

    This is the last-resort fill pass: it only ever writes into a *blank*
    Level 5 and never overwrites a value set by the keyword model or any
    earlier override.  It applies ordered regex patterns to the full narration
    (which contains more context than any individual keyword part).

    Direction awareness:
    - ``salary`` is only applied to credit rows.
    - ``credit card`` (bill payment) is only applied to debit rows.
    - All other categories apply regardless of direction.
    """

    # Nothing to do if Level 5 is already set.
    if result.get("Level_5"):
        return result

    # ATM withdrawals are already handled by apply_atm_withdrawal_overrides.
    if result.get("Level_1") == "cash":
        return result

    text = _marker_text(particulars, part_texts)
    direction = derive_direction_from_amounts(debit_value, credit_value)

    for pattern, purpose in NARRATION_LEVEL5_PATTERNS:
        if not pattern.search(text):
            continue

        # salary is income — only on credits
        if purpose == "salary" and direction == "debit":
            continue
        # credit card *bill payment* — only on debits
        if purpose == "credit card" and direction == "credit":
            continue

        result["Level_5"] = purpose
        break  # First match wins; patterns ordered by specificity.

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


# Level 6 holds the specific payee/payer/brand. These words describe the
# transaction (narration prefixes, rails, bill types), not an entity.
GENERIC_LEVEL6_WORDS = frozenset(
    {
        "to", "by", "from", "for", "via", "bill", "bills", "payment", "payments",
        "pay", "payme", "transfer", "transfers", "trf", "neft", "imps", "rtgs",
        "upi", "atm", "wdl", "withdrawal", "cash", "merchant", "dr", "cr", "bbps",
        "ach", "nach", "ecs", "clearing", "deposit", "chq", "cheque", "issued",
        "recvd", "clrd", "bouncd", "pdc", "intl", "ref", "credit", "debit", "card",
        "recharge", "mobile", "premium", "insurance", "electricity", "broadband",
        "dth", "online", "txn", "transaction",
    }
)
# Structured slash narrations: <RAIL>/<DR|CR>/<ref>/<NAME>/<BANK>/..., where the
# reference may sit before or after DR/CR (e.g. "UPI/449682688795/DR/ARUN/SBIN").
SLASH_ENTITY_RAIL_RE = re.compile(r"\b(UPI|NEFT|IMPS|RTGS|ATM)/(.*)", flags=re.IGNORECASE)
# BBPS biller id, e.g. "BBPS/DR/HDFCCC2025030457/CREDIT CARD BILL/".
BBPS_BILLER_RE = re.compile(r"\bBBPS/(?:DR|CR)/([A-Z][A-Z ]*[A-Z])\d", flags=re.IGNORECASE)
BBPS_BILLER_NAMES = {"jiofiber": "jio fiber", "tatasky": "tata sky"}


def is_generic_level6(value: str) -> bool:
    """Return True when a Level 6 value names no entity (e.g. "to bill payment")."""

    text = _normalize_compare(value)
    if len(re.sub(r"[^a-z]", "", text)) < 3:
        return True
    return all(token in GENERIC_LEVEL6_WORDS for token in re.findall(r"[a-z0-9]+", text))


def extract_narration_entity(particulars: str) -> str:
    """Return the payee/payer/biller explicitly named by a structured narration.

    Handles slash narrations (UPI/NEFT/IMPS/RTGS/ATM: the first segment after
    DR/CR that is not a reference number) and BBPS biller ids ("LIC2025...",
    "HDFCCC2025..." -> "hdfc credit card"). Returns "" when the narration does
    not identify an entity, so callers keep their existing fallback.
    """

    text = str(particulars or "")

    biller = BBPS_BILLER_RE.search(text)
    if biller:
        code = re.sub(r"\s+", " ", biller.group(1)).strip().lower()
        if code.endswith("cc") and any(code[:-2] in aliases for aliases in BANK_ALIASES.values()):
            return f"{code[:-2]} credit card"
        return BBPS_BILLER_NAMES.get(code, code)

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
        if re.search(r"\d{5,}", segment):  # transaction / NEFT reference
            continue
        name = re.sub(r"\s+", " ", segment).strip().lower()
        if name and not is_generic_level6(name):
            return name
        return ""
    return ""


def fill_specific_level6(particulars: str, result: dict[str, str]) -> dict[str, str]:
    """Replace a blank, generic or category Level 6 with the narration's entity.

    A specific Level 6 is never changed. When the narration names no entity,
    the existing value is kept, except that a copy of the Level 5 category is
    removed because a category is not an entity.
    """

    current = result.get("Level_6", "")
    is_category = bool(current) and _normalize_compare(current) == _normalize_compare(result.get("Level_5", ""))
    biller = BBPS_BILLER_RE.search(str(particulars or ""))
    # A raw copy of the BBPS biller code ("hdfccc") is replaced by its readable form.
    is_raw_biller_code = bool(biller) and re.sub(r"[^a-z0-9]", "", _normalize_compare(current)) == re.sub(
        r"[^a-z0-9]", "", biller.group(1).lower()
    )
    if current and not is_category and not is_raw_biller_code and not is_generic_level6(current):
        return result
    entity = extract_narration_entity(particulars)
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
) -> dict[str, str]:
    """Apply a few row-level consistency rules after aggregation.

    ``default_p2p_purpose`` lets a caller that has just cleared an unsupported
    "p2p transfer" purpose stop the default below from putting it back.
    """

    if default_p2p_purpose and result["Level_3"] == "p2p" and not result["Level_5"]:
        result["Level_5"] = "p2p transfer"

    if result["Level_1"] == "cash":
        result["Level_3"] = ""
        # Keep a bank (e.g. the bank of an ATM or cash deposit machine); a
        # payment platform cannot apply to a cash transaction.
        if is_payment_intermediary_label(result["Level_4"]):
            result["Level_4"] = ""
        result["Level_5"] = ""
        result["Level_6"] = ""

    if result["Level_3"] == "institution" and result["Level_4"]:
        result["Level_6"] = ""

    if (
        result["Level_4"]
        and result["Level_6"]
        and _normalize_compare(result["Level_4"]) == _normalize_compare(result["Level_6"])
    ):
        result["Level_6"] = ""

    return result


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

    # Level 6: the first part is usually the narration prefix ("TO TRANSFER",
    # "TO BILL PAYMENT"), so prefer the first specific entity from any part and
    # keep the first generic value only when no part names an entity.
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

    # Level 4 precedence: a payment intermediary predicted for any part wins
    # over a bank predicted for an earlier part.
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
    if bank_name:
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


def extract_hyphen_upi_merchant(particulars: str) -> str:
    """Return the known merchant named in a hyphen-delimited UPI narration.

    The narration is ``UPI-<payee>-<vpa>@<psp>-<ifsc>-<ref>-<note>``. Only the
    VPA handle and the payee segment are searched, VPA first, because the
    trailing note often names the payment gateway ("PAYVIARAZORPAY") or is
    truncated ("REQUEST FROM AMAZO"). Payment intermediaries are not in the
    merchant set, so a gateway can never be returned as the merchant. Returns
    an empty string when no known merchant is found, or when an explicit
    P2M/P2P marker means the structured UPI override already owns the row.
    """

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
        letters = re.sub(r"[^a-z]", "", candidate.lower())
        # Longest fragment first so the result does not depend on set ordering.
        for fragment in sorted(KNOWN_MERCHANT_FRAGMENTS, key=lambda item: (-len(item), item)):
            if fragment in letters:
                return fragment
    return ""


def is_payment_intermediary_label(label: str) -> bool:
    """Return True for a label that names a payment platform/intermediary.

    Level_4 holds the intermediary when one is identified, else the bank. A
    canonical platform name from ``PLATFORM_ALIASES`` (including "paytm") is an
    intermediary; any other known bank name is not.
    """

    value = _normalize_compare(label)
    if value in PLATFORM_ALIASES:
        return True
    if not value or value in BANK_NAMES:
        return False
    letters = re.sub(r"[^a-z]", "", value)
    return any(fragment in letters for fragment in PAYMENT_INTERMEDIARY_FRAGMENTS)


def derive_upi_bank(particulars: str) -> str:
    """Return the bank named by a UPI narration's IFSC prefix or VPA handle.

    The IFSC (e.g. ``ICIC0DC0099``, even when split as ``ICI C0DC0099``) is
    tried first, then the handle after ``@`` (e.g. ``@ICICI``). Returns an
    empty string when neither identifies a bank in ``BANK_CODE_TO_NAME``.
    """

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
    """Classify hyphen-format UPI payments to a known merchant as merchant rows.

    Without this, only ``UPI/DR/...`` (slash) and ``UPI|P2M|...`` (split-marker)
    narrations reach a merchant rule. A hyphen narration such as
    ``UPI-YESBANK-AMAZON@YAPL-...`` keeps the bank-style labels of its parts
    (``institution`` plus a bank in Level_4), and the institution rule in
    ``_apply_transaction_rules`` then blanks Level_6.

    Level_6 is the merchant, never the payment intermediary. Level_4 is a
    predicted payment intermediary (e.g. "razorpay") from the row or any part,
    unless the narration's IFSC or VPA handle names a bank; otherwise the first
    predicted bank, else the bank derived from the narration, else blank.
    A platform named in the narration itself is applied afterwards by
    ``apply_raw_transaction_overrides`` and takes precedence over all of these.
    """

    merchant = extract_hyphen_upi_merchant(particulars)
    if not merchant:
        return result

    purpose = keyword_classifier.classify(merchant).get("Level_5", "")
    if not purpose or purpose in {"p2p transfer", "salary"}:
        purpose = MERCHANT_DEFAULT_PURPOSES.get(merchant, "")

    level4_candidates = [result["Level_4"]] + [
        prediction.get("Level_4", "") for _, prediction in part_predictions
    ]
    intermediary = next(
        (label for label in level4_candidates if label and is_payment_intermediary_label(label)),
        "",
    )
    bank = next(
        (label for label in level4_candidates if label and not is_payment_intermediary_label(label)),
        "",
    )
    narration_bank = derive_upi_bank(particulars)
    # A predicted intermediary is a model guess; a bank named in the narration
    # is evidence, so the predicted intermediary is used only without one.
    result["Level_4"] = (intermediary if not narration_bank else "") or bank or narration_bank

    result["Level_1"] = result["Level_1"] or "upi"
    result["Level_3"] = "merchant"
    result["Level_6"] = merchant
    if purpose:
        result["Level_5"] = purpose
    elif result["Level_5"] in {"p2p transfer", "salary"}:
        result["Level_5"] = ""

    return _apply_transaction_rules(result)


def apply_debit_salary_guard(
    debit_value: object,
    credit_value: object,
    result: dict[str, str],
) -> dict[str, str]:
    """Drop a salary purpose from rows whose amounts say they are debits.

    Salary is income, so it can only apply to credit rows. Rows with no usable
    amounts (direction unknown) are left unchanged.
    """

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

    # ACH credits are institutional receipts, not person-to-person transfers.
    # Only assign dividend purpose when the narration explicitly supports it.
    if re.search(r"\bACH\b", marker_text, flags=re.IGNORECASE) and direction == "credit":
        if re.search(r"\bDIVIDEND(?:S)?\b", marker_text, flags=re.IGNORECASE):
            result["Level_5"] = "dividend"
        elif result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    # System-initiated debits, including card autopay, are not P2P transfers.
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
        is_gateway = any(fragment in vpa for fragment in GATEWAY_VPA_FRAGMENTS)

        result["Level_1"] = result["Level_1"] or "upi"
        result["Level_4"] = bank_name or result["Level_4"]

        if name != "merchant":
            result["Level_6"] = name

        if is_gateway:
            result["Level_3"] = "merchant"
            if result.get("Level_5") == "p2p transfer":
                result["Level_5"] = ""
        else:
            # Preserve a useful merchant purpose already found by the classifier.
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
                # Plain UPI: a named payee at a bank, with no gateway handle,
                # merchant purpose or merchant/institution label, is a P2P
                # transfer. Explicit MERCHANT markers below still win.
                result["Level_3"] = "p2p"
                result["Level_5"] = "p2p transfer"

    # An explicit MERCHANT marker takes precedence over a generic P2P label.
    if MERCHANT_MARKER_RE.search(marker_text):
        result["Level_3"] = "merchant"
        if result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    # Salary is income and must not be inferred from a debit.
    if SALARY_KEYWORDS.search(text) and not is_debit:
        # Prefer the payer named by a structured narration ("NISHA MA");
        # otherwise keep the existing cleaned-narration fallback.
        entity = extract_narration_entity(text)
        if not entity:
            entity = SALARY_KEYWORDS.sub("", text)
            entity = re.sub(r"[\-/|]+", " ", entity)
            entity = re.sub(r"\s+", " ", entity).strip().lower()

        # Salary purpose does not establish counterparty type.
        result["Level_5"] = "salary"
        result["Level_6"] = entity

    # Level 4: payment platform / bank detection (salary credits included).
    detected_level4 = detect_level_4_platform_or_bank(marker_text)
    if detected_level4:
        result["Level_4"] = detected_level4

    # Correct known merchant UPI transactions without changing
    # their already-detected purpose.
    upper_text = text.upper()
    parts = upper_text.split("-", 2)
    first_part = parts[1].strip() if len(parts) > 1 else ""
    merchant_key = re.sub(r"[^a-z0-9]", "", first_part.lower())

    merchant_name = KNOWN_UPI_MERCHANTS.get(merchant_key)

    # PAYTMQR is evidence of a QR merchant payment.
    is_paytm_qr = "PAYTMQR" in upper_text

    if upper_text.startswith("UPI-") and (merchant_name or is_paytm_qr):
        result["Level_3"] = "merchant"

        if merchant_name:
            result["Level_6"] = merchant_name
        elif not result.get("Level_6"):
            result["Level_6"] = first_part.lower()

        # Remove an unsupported P2P purpose but preserve useful
        # purposes such as grocery, retail, or digital services.
        if result.get("Level_5") == "p2p transfer":
            result["Level_5"] = ""

    # Level 6: replace a generic label ("to transfer", "to bill payment") with
    # the payee/biller the narration names, if it names one.
    fill_specific_level6(text, result)

    # Do not let the p2p default re-add a purpose the automated-debit rule cleared.
    return _apply_transaction_rules(result, default_p2p_purpose=not automated_debit)


def apply_atm_withdrawal_overrides(
    particulars: str,
    part_predictions: list[tuple[str, dict[str, str]]],
    result: dict[str, str],
    debit_value: object = None,
    credit_value: object = None,
) -> dict[str, str]:
    """Mark ATM cash withdrawals as cash and drop contradictory P2P labels.

    Runs after the other row-level overrides and does not re-apply the
    generic cash rule, so bank and ATM details (Level_4 / Level_6) are kept.
    Credit rows (e.g. failed ATM reversals) are left unchanged.
    """

    text = _marker_text(particulars, [part_text for part_text, _ in part_predictions])
    if not ATM_WITHDRAWAL_RE.search(text):
        return result
    if derive_direction_from_amounts(debit_value, credit_value) == "credit":
        return result

    bank_name = result["Level_4"]
    location = result["Level_6"]
    if not bank_name:
        # The generic cash rule may have blanked a bank detected from the
        # narration (e.g. "HDFC BANK ATM"); re-detect it before falling back to
        # the part predictions.
        bank_name = detect_level_4_platform_or_bank(text)
    for _, prediction in part_predictions:
        if not bank_name and prediction.get("Level_4"):
            bank_name = prediction["Level_4"]
        if (
            (not location or is_generic_level6(location))
            and prediction.get("Level_6")
            and not is_generic_level6(prediction["Level_6"])
            and prediction.get("Level_3") != "p2p"
        ):
            location = prediction["Level_6"]
    if not location or is_generic_level6(location):
        # e.g. "ATM/DR/729425/ICICI BANK ATM/..." names the ATM itself.
        location = extract_narration_entity(particulars) or location

    result["Level_1"] = "cash"
    # An ATM withdrawal is a cash transaction, not a P2P/merchant payment or
    # a salary transaction. Clear all purpose/entity-type labels that can
    # contradict this (including "retail", which appeared in File 4).
    result["Level_3"] = ""
    result["Level_5"] = ""
    result["Level_4"] = bank_name
    result["Level_6"] = location
    if bank_name and location and _normalize_compare(bank_name) == _normalize_compare(location):
        result["Level_6"] = ""

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

    # Explicit automatic-payment mechanisms take priority.
    if SYSTEM_INITIATED_RE.search(text):
        result["Level_2"] = "System generated"
        return result

    # Agreed project rule: explicitly identified salary credits are system-generated.
    if SALARY_KEYWORDS.search(text) and direction == "credit":
        result["Level_2"] = "System generated"
        return result

    # Outgoing payments and ATM withdrawals are treated as user-initiated.
    # Do not infer initiation mode for incoming transfers from the rail alone.
    if direction == "debit":
        if (
            ATM_WITHDRAWAL_RE.search(text)
            or EXPLICIT_UPI_RE.search(text)
            or MANUAL_TRANSFER_RE.search(text)
        ):
            result["Level_2"] = "User-generated"

    # Ambiguous transactions remain blank.
    return result


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
    if level_2 == "auto-debit":
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


def detect_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Flag salary-like credits using explicit text plus recurring credits."""

    output_df = df.copy()
    output_df["IsSalary"] = 0
    output_df["SalaryConfidence"] = 0.0

    credit_num = output_df["Credit"].apply(parse_amount)
    debit_num = (
        output_df["Debit"].apply(parse_amount)
        if "Debit" in output_df.columns
        else pd.Series(0.0, index=output_df.index)
    )
    # Salary is income: a row that is a debit can never be salary.
    debit_mask = (debit_num > 0) & (credit_num <= 0)

    # Remove stale salary labels from debit rows and re-derive their labels.
    stale_level_5 = (
        output_df["Level_5"].astype(str).str.strip().str.lower().eq("salary")
        if "Level_5" in output_df.columns
        else pd.Series(False, index=output_df.index)
    )
    stale_first_level = (
        output_df["First Level Classification"].astype(str).str.strip().str.lower().eq("salary")
    )
    stale_mask = debit_mask & (stale_level_5 | stale_first_level)
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

    output_df["NormText"] = (
        output_df["Particulars"].astype(str)
        + " "
        + output_df["First Level Classification"].astype(str)
        + " "
        + output_df["Second Level Classification"].astype(str)
    ).str.lower()

    explicit_mask = (
        output_df.get("Level_5", pd.Series("", index=output_df.index))
        .astype(str)
        .str.contains("salary", case=False, na=False)
        | output_df.get("First Level Classification", pd.Series("", index=output_df.index))
        .astype(str)
        .str.contains("salary", case=False, na=False)
        | (output_df["NormText"].str.contains(SALARY_KEYWORDS, na=False) & (credit_num >= MIN_SALARY))
    ) & ~debit_mask

    output_df.loc[explicit_mask, "IsSalary"] = 1
    output_df.loc[explicit_mask, "SalaryConfidence"] = 1.0

    dates = pd.to_datetime(output_df.get("Date"), errors="coerce")
    candidates = output_df.index[
        (~explicit_mask) & (~debit_mask) & (credit_num >= MIN_SALARY) & dates.notna()
    ]

    if len(candidates) >= 2:
        amount_keys = ((credit_num.loc[candidates] / 1000).round() * 1000).astype(int)
        clusters: dict[int, list[int]] = defaultdict(list)
        for index, amount_key in amount_keys.items():
            clusters[int(amount_key)].append(index)

        for indices in clusters.values():
            if len(indices) < MIN_SALARY_MONTHS:
                continue

            day_std = dates.loc[indices].dt.day.std(ddof=0)
            text_hit = output_df.loc[indices, "NormText"].str.contains(SALARY_KEYWORDS, na=False).any()

            score = 0.40
            score += 0.30 if (pd.notna(day_std) and day_std <= SALARY_DAY_STD_MAX) else 0.0
            score += 0.30 if text_hit else 0.0

            if score >= 0.60:
                output_df.loc[indices, "IsSalary"] = 1
                output_df.loc[indices, "SalaryConfidence"] = round(score, 2)

    output_df.loc[debit_mask, "IsSalary"] = 0
    output_df.loc[debit_mask, "SalaryConfidence"] = 0.0

    salary_mask = output_df["IsSalary"].eq(1)
    # Level 6: the payer named by a structured narration, else the existing
    # cleaned-narration fallback.
    salary_entity = output_df["Particulars"].astype(str).map(
        lambda text: extract_narration_entity(text) or extract_salary_entity(text)
    )
    output_df.loc[salary_mask, "Level_0"] = "credit"
    output_df.loc[salary_mask, "Level_1"] = ""
    output_df.loc[salary_mask, "Level_2"] = "System generated"
    output_df.loc[salary_mask, "Level_5"] = "salary"
    output_df.loc[salary_mask, "Level_7"] = ""
    output_df.loc[salary_mask, "Level_6"] = salary_entity.loc[salary_mask]
    output_df.loc[salary_mask, "First Level Classification"] = "Salary"
    output_df.loc[salary_mask, "Second Level Classification"] = ""

    return output_df.drop(columns=["NormText"])


def build_salary_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a salary-only summary sheet."""

    salary_df = df[df.get("IsSalary", 0).astype(int).eq(1)].copy()
    if salary_df.empty:
        return salary_df

    salary_df["Month"] = pd.to_datetime(salary_df["Date"], errors="coerce").dt.to_period("M").astype(str)
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


# Level 2: Initiation Mode (SYSTEM_INITIATED_RE is defined near the top of the file)
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

        # Explicit automated-payment evidence takes priority.
        if SYSTEM_INITIATED_RE.search(particulars):
            output_df.at[idx, "Level_2"] = "System generated"

        # Salary credits identified by the existing salary-detection pipeline.
        elif row.get("IsSalary", 0) == 1:
            output_df.at[idx, "Level_2"] = "System generated"

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
        output_df["First Level Classification"] = output_df.apply(
            derive_first_level_classification,
            axis=1,
        )
        output_df["Second Level Classification"] = output_df.apply(
            derive_second_level_classification,
            axis=1,
        )

        output_df = detect_salary(output_df)
        output_df = apply_strict_level2(output_df)
        logger.info("Classified %s statement rows", len(output_df))
        return output_df