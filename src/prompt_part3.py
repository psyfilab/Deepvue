"""Prompt-aligned PART_3 keyword dataset and classifier helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .preprocess import KeywordPreprocessor
from .rule_engine import RuleEngine

logger = logging.getLogger(__name__)

PROMPT_PRIMARY_PURPOSES = (
    "tax",
    "salary",
    "loan repayment",
    "credit card",
    "mobile recharge",
    "utilities",
    "pharmacy",
    "dining",
    "grocery",
    "subscription",
    "p2p transfer",
    "transport",
    "digital services",
    "retail",
    "services",
)

PROMPT_SECONDARY_PURPOSES = (
    "financial services",
    "insurance",
    "healthcare",
    "education",
    "fuel",
    "refund",
)

ALLOWED_PURPOSES = frozenset(PROMPT_PRIMARY_PURPOSES + PROMPT_SECONDARY_PURPOSES)
ALLOWED_LEVEL3 = frozenset({"p2p", "merchant", "institution"})

GENERIC_ENTITY_VALUES = frozenset(
    {
        "",
        "nan",
        "unknown",
        "merchant",
        "institution",
        "p2p",
        "retail",
        "services",
        "digital services",
        "transaction meta",
        "upi meta",
        "paymen",
        "payment",
        "bank",
    }
)

P2P_HANDLE_TOKENS = (
    "okicici",
    "okhdfcbank",
    "okhdfc",
    "oksbi",
    "okaxis",
    "okaxisbank",
    "okbizaxis",
    "okyes",
    "okybl",
    "okbizax",
    "okhdfcban",
    "okhdfcba",
    "okicic",
    "ybl",
    "ibl",
    "axl",
    "apl",
    "rapl",
    "ptybl",
    "yesbyblupi",
    "yesbptmupi",
    "yesbyesupi",
    "yesbaplupi",
    "@okicici",
    "@okhdfcbank",
    "@oksbi",
    "@okaxis",
)

RAIL_TOKENS = frozenset({"upi", "imps", "neft", "rtgs", "nach", "cash", "cheque", "chq"})
GENERIC_PLATFORM_TOKENS = frozenset(
    {
        "gpay",
        "googlepay",
        "google",
        "phonepe",
        "paytm",
        "amazonpay",
        "amazon",
        "razorpay",
        "payu",
        "cashfree",
        "freecharge",
        "bharatpe",
        "cred",
    }
)

BUSINESS_TOKENS = frozenset(
    {
        "agency",
        "agencies",
        "airtel",
        "amazon",
        "apparel",
        "apparels",
        "auto",
        "bakery",
        "bank",
        "bazaar",
        "bill",
        "biz",
        "cafe",
        "capital",
        "center",
        "centre",
        "communications",
        "computers",
        "consulting",
        "corporation",
        "delivery",
        "digital",
        "electric",
        "electronics",
        "enterprise",
        "enterprises",
        "fashion",
        "finance",
        "foods",
        "fuel",
        "fuels",
        "global",
        "grocery",
        "health",
        "home",
        "hospital",
        "hotel",
        "house",
        "impex",
        "india",
        "industries",
        "insurance",
        "limited",
        "ltd",
        "mall",
        "market",
        "mart",
        "medical",
        "medicals",
        "mobile",
        "motors",
        "nagar",
        "online",
        "pay",
        "payment",
        "payments",
        "petrol",
        "pharma",
        "power",
        "private",
        "pump",
        "pvt",
        "rail",
        "razorpay",
        "recharge",
        "restaurant",
        "retail",
        "ride",
        "service",
        "services",
        "shop",
        "shops",
        "corner",
        "stall",
        "station",
        "store",
        "stores",
        "supermarket",
        "tech",
        "technologies",
        "technology",
        "telecom",
        "trader",
        "traders",
        "trade",
        "travels",
        "utility",
        "restaurant",
        "restaurants",
        "tea",
        "chai",
        "bhandar",
        "manpower",
        "marketplace",
        "engineering",
        "systems",
    }
)

BUSINESS_SUBSTRINGS = (
    "bank",
    "bill",
    "cafe",
    "capital",
    "corporat",
    "deliver",
    "digit",
    "electr",
    "enterpris",
    "fashion",
    "financ",
    "fuel",
    "groc",
    "health",
    "hotel",
    "house",
    "impex",
    "indust",
    "insur",
    "market",
    "medic",
    "medicals",
    "mobil",
    "manpower",
    "marketplace",
    "payment",
    "petrol",
    "pharm",
    "power",
    "recharg",
    "retail",
    "service",
    "shop",
    "store",
    "systems",
    "tech",
    "telecom",
    "trade",
    "travel",
    "utility",
)

NON_PERSON_SUBSTRINGS = (
    "airtel",
    "amazon",
    "bharatpe",
    "bill",
    "blink",
    "cashfree",
    "cred",
    "ekart",
    "flipk",
    "freecharge",
    "google",
    "groww",
    "instamart",
    "irctc",
    "jio",
    "netbank",
    "paymen",
    "paytm",
    "payu",
    "phonep",
    "razorpay",
    "swigg",
    "utility",
    "zepto",
    "zomat",
)

MANUAL_PLATFORM_ALIASES: dict[str, tuple[str, ...]] = {
    "hdfc bank": ("hdfcbank", "okhdfcbank", "okhdfc", "hdfcbank", "hdfc"),
    "icici bank": ("okicici", "icicibank", "icici", "icic", "icicib"),
    "state bank of india": ("oksbi", "sbin", "sbi", "statebankofindia"),
    "axis bank": ("okaxis", "axisbank", "utib", "axis"),
    "yes bank": ("yesbank", "yesb"),
    "bank of baroda": ("bankofbaroda", "barb", "bob"),
    "bank of india": ("bankofindia", "bkid", "boi"),
    "canara bank": ("canarabank", "cnrb"),
    "punjab national bank": ("punjabnationalbank", "pnb", "punb", "punjabnational"),
    "union bank of india": ("unionbankofindia", "ubin", "ubi"),
    "federal bank": ("federalbank", "fdrl", "fed"),
    "idfc first bank": ("idfcfirstbank", "idfc", "idfb"),
    "idbi bank": ("idbibank", "idbi"),
    "indian bank": ("indianbank",),
    "indian overseas bank": ("indianoverseasbank", "iob"),
    "kotak mahindra bank": ("kotakmahindrabank", "kotak", "kmb"),
    "airtel payments bank": ("airtelpaymentsbank",),
    "india post payments bank": ("indiapostpaymentsbank", "ippb"),
    "paytm": ("paytm",),
    "google pay": ("googlepay", "gpay"),
    "phonepe": ("phonepe",),
    "amazon pay": ("amazonpay",),
    "razorpay": ("razorpay", "rzp"),
    "payu": ("payu",),
    "cashfree": ("cashfree",),
    "bharatpe": ("bharatpe",),
    "freecharge": ("freecharge",),
    "mobikwik": ("mobikwik", "ikwik"),
    "cred": ("cred",),
    "billdesk": ("billdesk",),
    "au small finance bank": ("aubl",),
    "karur vysya bank": ("kvbl",),
    "central bank of india": ("cbin",),
    "uco bank": ("ucba", "ucobank", "uco"),
    "india post payments bank": ("ipos", "ippb"),
    "fino payments bank": ("fino",),
    "esaf small finance bank": ("esfb",),
    "indian bank": ("idib",),
    "idbi bank": ("ibkl",),
    "hsbc": ("hsbc",),
    "city union bank": ("ciub", "cityunionbank", "cityunionban"),
    "csb bank": ("csb", "csbk"),
    "bandhan bank": ("bdbl",),
    "dbs bank india": ("dbs", "dbssin", "dbsbank"),
    "tamilnad mercantile bank": ("tmbl", "tmb", "tamilnadmerca"),
    "south indian bank": ("sib",),
    "rbl bank": ("rblbank", "rbl"),
}

MANUAL_MERCHANT_HINTS: dict[str, dict[str, str]] = {
    "zomato": {"Level_3": "merchant", "Level_5": "dining", "Level_6": "zomato"},
    "swiggyinstamart": {
        "Level_3": "merchant",
        "Level_5": "grocery",
        "Level_6": "swiggy instamart",
    },
    "swiggystores": {"Level_3": "merchant", "Level_5": "grocery", "Level_6": "swiggy stores"},
    "swiggygenie": {"Level_3": "merchant", "Level_5": "services", "Level_6": "swiggy genie"},
    "swiggy": {"Level_3": "merchant", "Level_5": "dining", "Level_6": "swiggy"},
    "blinkit": {"Level_3": "merchant", "Level_5": "grocery", "Level_6": "blinkit"},
    "zepto": {"Level_3": "merchant", "Level_5": "grocery", "Level_6": "zepto"},
    "pharmeasy": {"Level_3": "merchant", "Level_5": "pharmacy", "Level_6": "pharmeasy"},
    "medplus": {"Level_3": "merchant", "Level_5": "pharmacy", "Level_6": "medplus"},
    "irctc": {"Level_3": "merchant", "Level_5": "transport", "Level_6": "irctc"},
    "netflix": {"Level_3": "merchant", "Level_5": "subscription", "Level_6": "netflix"},
    "spotify": {"Level_3": "merchant", "Level_5": "subscription", "Level_6": "spotify"},
    "amazonpayrecharge": {
        "Level_3": "merchant",
        "Level_5": "mobile recharge",
        "Level_6": "amazon pay",
    },
    "amazonpaygrocery": {
        "Level_3": "merchant",
        "Level_5": "grocery",
        "Level_6": "amazon pay",
    },
    "amazonpay": {
        "Level_3": "merchant",
        "Level_5": "digital services",
        "Level_6": "amazon pay",
    },
    "amazon": {"Level_3": "merchant", "Level_5": "retail", "Level_6": "amazon"},
    "flipkart": {"Level_3": "merchant", "Level_5": "retail", "Level_6": "flipkart"},
    "paytmrecharge": {
        "Level_3": "institution",
        "Level_4": "paytm",
        "Level_5": "mobile recharge",
    },
    "paytmutilitybill": {
        "Level_3": "institution",
        "Level_4": "paytm",
        "Level_5": "utilities",
    },
    "paytmqr": {
        "Level_3": "institution",
        "Level_4": "paytm",
        "Level_5": "digital services",
    },
    "googlepay": {
        "Level_3": "institution",
        "Level_4": "google pay",
        "Level_5": "digital services",
    },
    "phonepe": {
        "Level_3": "institution",
        "Level_4": "phonepe",
        "Level_5": "digital services",
    },
    "bharatpe": {
        "Level_3": "institution",
        "Level_4": "bharatpe",
        "Level_5": "digital services",
    },
    "freecharge": {
        "Level_3": "institution",
        "Level_4": "freecharge",
        "Level_5": "digital services",
    },
    "cashfree": {
        "Level_3": "institution",
        "Level_4": "cashfree",
        "Level_5": "digital services",
    },
    "razorpay": {
        "Level_3": "institution",
        "Level_4": "razorpay",
        "Level_5": "digital services",
    },
    "groww": {"Level_3": "merchant", "Level_5": "financial services", "Level_6": "groww"},
    "msrtc": {"Level_3": "merchant", "Level_5": "transport", "Level_6": "msrtc"},
    "pmpml": {"Level_3": "merchant", "Level_5": "transport", "Level_6": "pmpml"},
    "ekart": {"Level_3": "merchant", "Level_5": "retail", "Level_6": "ekart"},
    "wellness": {"Level_3": "merchant", "Level_5": "pharmacy", "Level_6": "wellness"},
    "wellnessforever": {"Level_3": "merchant", "Level_5": "pharmacy", "Level_6": "wellness forever"},
    "cafecreme": {"Level_3": "merchant", "Level_5": "dining", "Level_6": "cafe creme"},
}

PURPOSE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btax\b|gst|itdtax|stax", re.IGNORECASE), "tax"),
    (re.compile(r"\bsalary|payroll|wages|labour|sal\b", re.IGNORECASE), "salary"),
    (re.compile(r"credit card|ccbp|cc payment", re.IGNORECASE), "credit card"),
    (re.compile(r"\bemi\b|loan repay|loan emi|intrest emi|interest emi", re.IGNORECASE), "loan repayment"),
    (re.compile(r"recharge|topup", re.IGNORECASE), "mobile recharge"),
    (re.compile(r"electricity|water|gas|lpg|billdesk|billpay|utility|broadband|dth", re.IGNORECASE), "utilities"),
    (re.compile(r"pharm|medplus|1mg|pharmeasy", re.IGNORECASE), "pharmacy"),
    (re.compile(r"zomato|swiggy|food|restaurant|cafe|pizza|burger", re.IGNORECASE), "dining"),
    (re.compile(r"blinkit|zepto|grofers|grocery|instamart|mart", re.IGNORECASE), "grocery"),
    (re.compile(r"netflix|spotify|prime|monthly|subscription", re.IGNORECASE), "subscription"),
    (re.compile(r"irctc|rail|metro|uber|ola|rapido|bus|ticket|travel", re.IGNORECASE), "transport"),
    (re.compile(r"paytm|gpay|googlepay|phonepe|razorpay|cashfree|freecharge|payu|bharatpe", re.IGNORECASE), "digital services"),
    (re.compile(r"amazon|flipkart|myntra|meesho|retail", re.IGNORECASE), "retail"),
    (re.compile(r"service|vyapar|professional|consult", re.IGNORECASE), "services"),
    (re.compile(r"insurance|policy|premium|lic|acko|ergo|bupa", re.IGNORECASE), "insurance"),
    (re.compile(r"groww|zerodha|upstox|mutual fund|sip|nps|investment", re.IGNORECASE), "financial services"),
]

PURPOSE_NORMALIZATION: dict[str, str] = {
    "dining/bar": "dining",
    "dining/kirana": "grocery",
    "food delivery": "dining",
    "ecommerce": "retail",
    "ecommerce/blinkit": "grocery",
    "retail/fashion": "retail",
    "retail/grocery": "grocery",
    "transport/vehicle": "transport",
    "transport/ride": "transport",
    "transport/railways": "transport",
    "transport/flight": "transport",
    "bill payment": "utilities",
    "fuel/petrol": "fuel",
    "digital wallet": "digital services",
    "digital payment": "digital services",
    "government payments": "services",
    "financial services": "financial services",
    "investment": "financial services",
    "health": "healthcare",
    "educational": "education",
}

NAME_LIKE_TOKEN_RE = re.compile(r"^[a-z][a-z\s]{1,40}$")
NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")
COMPACT_RE = re.compile(r"[^a-z0-9]")
WHITESPACE_RE = re.compile(r"\s+")
VOWEL_RE = re.compile(r"[aeiou]")


@dataclass
class PromptArtifacts:
    """Serialized prompt classifier components."""

    preprocessor: KeywordPreprocessor
    exact_lookup: dict[str, dict[str, str]]
    level3_model: Any | None
    level5_model: Any | None
    merchant_aliases: dict[str, dict[str, str]]
    platform_aliases: dict[str, str]
    metadata: dict[str, Any]


def empty_result() -> dict[str, str]:
    """Return a blank prompt output row."""

    return {
        "Level_0": "",
        "Level_1": "",
        "Level_2": "",
        "Level_3": "",
        "Level_4": "",
        "Level_5": "",
        "Level_6": "",
        "Level_7": "",
    }


def normalize_spaces(text: str) -> str:
    """Return a lowercase, space-normalized text string."""

    lowered = text.strip().lower()
    lowered = NON_ALPHA_RE.sub(" ", lowered)
    return WHITESPACE_RE.sub(" ", lowered).strip()


def compact_text(text: str) -> str:
    """Return a compact alphanumeric text string."""

    return COMPACT_RE.sub("", text.lower())


def normalize_purpose(value: str) -> str:
    """Map noisy purpose values into the prompt purpose space."""

    purpose = str(value or "").strip().lower()
    if not purpose:
        return ""
    purpose = PURPOSE_NORMALIZATION.get(purpose, purpose)
    return purpose if purpose in ALLOWED_PURPOSES else ""


def clean_entity(value: str, level_4_value: str = "") -> str:
    """Clean entity text and drop generic or duplicated values."""

    entity = normalize_spaces(str(value or ""))
    if entity in GENERIC_ENTITY_VALUES:
        return ""
    if level_4_value and entity == normalize_spaces(level_4_value):
        return ""
    return entity


def keyword_supports_bank(keyword_compact: str, bank_name: str, alias: str) -> bool:
    """Return whether an alias is explicit enough to support a bank/platform label."""

    if not alias:
        return False
    if len(alias) <= 3:
        return (
            keyword_compact == alias
            or keyword_compact.startswith(alias)
            or keyword_compact.endswith(alias)
        )
    if alias in {"axis", "amazon", "kotak"} and alias == keyword_compact:
        return True
    if alias in {"axis", "amazon", "kotak"}:
        return alias in keyword_compact and len(keyword_compact) <= len(alias) + 6
    return alias in keyword_compact or compact_text(bank_name) in keyword_compact


def detect_platform_or_bank(keyword: str, merchant_aliases: dict[str, dict[str, str]] | None = None) -> str:
    """Detect explicit banks or payment platforms from the keyword."""

    keyword_compact = compact_text(keyword)
    if merchant_aliases:
        for alias in sorted(merchant_aliases, key=len, reverse=True):
            if alias in keyword_compact:
                return ""

    for canonical_name, aliases in MANUAL_PLATFORM_ALIASES.items():
        for alias in aliases:
            if keyword_supports_bank(keyword_compact, canonical_name, compact_text(alias)):
                return canonical_name
    return ""


def detect_manual_merchant(keyword: str) -> dict[str, str]:
    """Detect strong merchant aliases from the keyword."""

    keyword_compact = compact_text(keyword)
    for alias, labels in sorted(MANUAL_MERCHANT_HINTS.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in keyword_compact:
            return labels.copy()
    return {}


def detect_explicit_purpose(keyword: str) -> str:
    """Detect explicit purpose cues directly from the keyword."""

    for pattern, purpose in PURPOSE_PATTERNS:
        if pattern.search(keyword):
            return purpose
    return ""


def strip_handle_tokens(keyword: str) -> str:
    """Remove handle-like tokens from a keyword."""

    text = normalize_spaces(keyword)
    for handle in P2P_HANDLE_TOKENS:
        handle_pattern = re.escape(normalize_spaces(handle))
        text = re.sub(rf"\b{handle_pattern}\b", " ", text)
    text = re.sub(r"\b(?:ok\s*sbi|ok\s*icici|ok\s*hdfc\s*bank|ok\s*axis)\b", " ", text)
    text = re.sub(r"\b(?:sbin|hdfc|icici|axisbank|yesbank|yesb|utib|bkid|punb|cnrb|fdrl|ubin)\b$", " ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def detect_p2p_entity(keyword: str, merchant_aliases: dict[str, dict[str, str]] | None = None) -> str:
    """Return a likely p2p entity name when the keyword looks person-like."""

    stripped = strip_handle_tokens(keyword)
    keyword_compact = compact_text(keyword)
    has_handle_token = any(
        token in keyword_compact for token in map(compact_text, P2P_HANDLE_TOKENS)
    )

    if has_handle_token and stripped and looks_like_person_name(stripped):
        return stripped

    if merchant_aliases:
        for alias in merchant_aliases:
            if alias in keyword_compact:
                return ""

    if detect_manual_merchant(keyword):
        return ""

    normalized = normalize_spaces(keyword)
    if detect_platform_or_bank(keyword, merchant_aliases):
        return ""
    if looks_like_person_name(normalized):
        return normalized
    return ""


def is_rail_platform_only_keyword(keyword: str) -> bool:
    """Return whether a keyword is only a rail plus a generic platform token."""

    tokens = set(normalize_spaces(keyword).split())
    if not tokens:
        return False
    return tokens.issubset(RAIL_TOKENS | GENERIC_PLATFORM_TOKENS)


def looks_like_person_name(text: str) -> bool:
    """Return whether a cleaned string looks like a person name."""

    normalized = normalize_spaces(text)
    tokens = normalized.split()
    if not NAME_LIKE_TOKEN_RE.match(normalized):
        return False
    if not 1 <= len(tokens) <= 4:
        return False
    if len(normalized.replace(" ", "")) < 4:
        return False
    if not VOWEL_RE.search(normalized):
        return False
    token_set = set(tokens)
    if token_set.issubset(RAIL_TOKENS | GENERIC_PLATFORM_TOKENS):
        return False
    if token_set & BUSINESS_TOKENS:
        return False
    if any(substring in normalized.replace(" ", "") for substring in BUSINESS_SUBSTRINGS):
        return False
    if any(substring in normalized.replace(" ", "") for substring in NON_PERSON_SUBSTRINGS):
        return False
    return True


def looks_like_business_name(text: str) -> bool:
    """Return whether a keyword looks more like a business than a person."""

    normalized = normalize_spaces(text)
    if not normalized:
        return False
    if looks_like_person_name(normalized):
        return False
    token_set = set(normalized.split())
    if token_set & BUSINESS_TOKENS:
        return True
    compact = normalized.replace(" ", "")
    return any(substring in compact for substring in BUSINESS_SUBSTRINGS)


def derive_prompt_labels(
    keyword: str,
    weak_level_3: str = "",
    weak_level_4: str = "",
    weak_level_5: str = "",
    weak_level_6: str = "",
    merchant_aliases: dict[str, dict[str, str]] | None = None,
    rule_engine: RuleEngine | None = None,
) -> dict[str, str]:
    """Build prompt-aligned labels from a raw keyword and weak labels."""

    rules = rule_engine or RuleEngine()
    result = empty_result()
    result.update(rules.classify(keyword))

    weak_level_3 = normalize_spaces(weak_level_3)
    weak_level_4 = normalize_spaces(weak_level_4)
    weak_level_5 = normalize_purpose(weak_level_5)
    weak_level_6 = clean_entity(weak_level_6, weak_level_4)

    if not str(keyword or "").strip():
        result["Level_0"] = "unknown"
        return result

    if result["Level_1"] and is_rail_platform_only_keyword(keyword):
        return result

    manual_merchant = detect_manual_merchant(keyword)
    if manual_merchant:
        result.update({key: value for key, value in manual_merchant.items() if value})

    if not result["Level_3"]:
        p2p_entity = detect_p2p_entity(keyword, merchant_aliases=merchant_aliases)
        if p2p_entity:
            result["Level_3"] = "p2p"
            result["Level_6"] = p2p_entity

    if not result["Level_3"]:
        platform_or_bank = detect_platform_or_bank(keyword, merchant_aliases=merchant_aliases)
        if platform_or_bank:
            result["Level_3"] = "institution"
            result["Level_4"] = platform_or_bank

    if not result["Level_3"] and looks_like_business_name(keyword):
        result["Level_3"] = "merchant"
        result["Level_6"] = result["Level_6"] or clean_entity(keyword)

    if not result["Level_3"] and weak_level_3 in ALLOWED_LEVEL3:
        if weak_level_3 == "institution" and detect_p2p_entity(keyword, merchant_aliases=merchant_aliases):
            result["Level_3"] = "p2p"
        else:
            result["Level_3"] = weak_level_3

    if weak_level_4 and not result["Level_4"]:
        detected = detect_platform_or_bank(keyword, merchant_aliases=merchant_aliases)
        if detected and detected == weak_level_4:
            result["Level_4"] = detected
            result["Level_3"] = result["Level_3"] or "institution"

    explicit_purpose = detect_explicit_purpose(keyword)
    if explicit_purpose and not result["Level_5"]:
        result["Level_5"] = explicit_purpose

    if weak_level_5 and not result["Level_5"]:
        result["Level_5"] = weak_level_5

    if weak_level_6 and not result["Level_6"]:
        if result["Level_3"] in {"p2p", "merchant"}:
            result["Level_6"] = weak_level_6
        elif result["Level_3"] == "institution" and not result["Level_4"]:
            result["Level_6"] = weak_level_6

    if result["Level_3"] == "institution" and result["Level_4"]:
        result["Level_6"] = ""

    if result["Level_3"] == "p2p" and result["Level_5"] == "p2p transfer":
        if "transfer" not in normalize_spaces(keyword):
            result["Level_5"] = ""

    if result["Level_1"] and result["Level_3"] == "institution" and result["Level_4"] in {"paytm", "google pay", "phonepe"}:
        result["Level_3"] = ""
        result["Level_4"] = ""
        if result["Level_5"] == "digital services":
            result["Level_5"] = ""

    if not any(result.values()):
        result["Level_0"] = "unknown"

    return result


def build_initial_prompt_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a prompt-aligned PART_3 dataset from the master sheet."""

    part3_df = df[df["Source"].astype(str).str.upper() == "PART_3"].copy()
    rule_engine = RuleEngine()

    records = []
    for _, row in part3_df.iterrows():
        labels = derive_prompt_labels(
            keyword=str(row["Key Word"]),
            weak_level_3=str(row["Level 3"]),
            weak_level_4=str(row["Level 4"]),
            weak_level_5=str(row["Level 5"]),
            weak_level_6=str(row["Level 6"]),
            merchant_aliases=None,
            rule_engine=rule_engine,
        )
        records.append(
            {
                "Source": row["Source"],
                "Count": int(pd.to_numeric(row["Count"], errors="coerce") or 1),
                "Key Word": str(row["Key Word"]),
                **labels,
            }
        )

    return pd.DataFrame(records)


def build_merchant_aliases(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Build a conservative merchant alias dictionary from the prompt dataset."""

    alias_votes: dict[str, dict[str, int]] = {}
    alias_payloads: dict[str, dict[str, str]] = {}
    platform_aliases = {
        compact_text(alias)
        for aliases in MANUAL_PLATFORM_ALIASES.values()
        for alias in aliases
    } | {compact_text(name) for name in MANUAL_PLATFORM_ALIASES}

    merchant_rows = df[
        (df["Level_3"] == "merchant")
        & (df["Level_6"].astype(str).str.strip() != "")
    ].copy()

    for _, row in merchant_rows.iterrows():
        alias = compact_text(str(row["Level_6"]))
        if len(alias) < 4:
            continue
        if alias in platform_aliases:
            continue
        payload = {
            "Level_3": "merchant",
            "Level_5": str(row["Level_5"]),
            "Level_6": str(row["Level_6"]),
        }
        alias_votes.setdefault(alias, {})
        key = json_key(payload)
        alias_votes[alias][key] = alias_votes[alias].get(key, 0) + int(row["Count"])
        alias_payloads[key] = payload

    merchant_aliases = {alias: payload.copy() for alias, payload in MANUAL_MERCHANT_HINTS.items()}

    for alias, vote_map in alias_votes.items():
        best_key, best_count = max(vote_map.items(), key=lambda item: item[1])
        total = sum(vote_map.values())
        if best_count < 2 and alias not in merchant_aliases:
            continue
        if best_count / total < 0.75:
            continue
        merchant_aliases[alias] = alias_payloads[best_key]

    logger.info("Built %s merchant aliases", len(merchant_aliases))
    return merchant_aliases


def json_key(payload: dict[str, str]) -> str:
    """Return a stable key for a label payload."""

    return "|".join(f"{key}={payload.get(key, '')}" for key in ("Level_3", "Level_4", "Level_5", "Level_6"))


def rebuild_prompt_dataset(df: pd.DataFrame, merchant_aliases: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Re-derive prompt labels with merchant aliases available."""

    rule_engine = RuleEngine()
    records = []
    for _, row in df.iterrows():
        labels = derive_prompt_labels(
            keyword=str(row["Key Word"]),
            weak_level_3=str(row["Level_3"]),
            weak_level_4=str(row["Level_4"]),
            weak_level_5=str(row["Level_5"]),
            weak_level_6=str(row["Level_6"]),
            merchant_aliases=merchant_aliases,
            rule_engine=rule_engine,
        )
        records.append(
            {
                "Source": row["Source"],
                "Count": int(row["Count"]),
                "Key Word": str(row["Key Word"]),
                **labels,
            }
        )
    return pd.DataFrame(records)


def build_exact_lookup(df: pd.DataFrame, preprocessor: KeywordPreprocessor) -> dict[str, dict[str, str]]:
    """Build an exact keyword lookup from cleaned prompt labels."""

    lookup: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = preprocessor.transform(str(row["Key Word"]))
        lookup[key] = {
            "Level_0": str(row["Level_0"]),
            "Level_1": str(row["Level_1"]),
            "Level_2": str(row["Level_2"]),
            "Level_3": str(row["Level_3"]),
            "Level_4": str(row["Level_4"]),
            "Level_5": str(row["Level_5"]),
            "Level_6": str(row["Level_6"]),
            "Level_7": str(row["Level_7"]),
        }
    return lookup


def format_prompt_tsv(keywords: list[str], predictions: list[dict[str, str]]) -> str:
    """Render predictions in the exact TSV format used by the prompt."""

    header = [
        "Key Word",
        "Level 0",
        "Level 1",
        "Level 2",
        "Level 3",
        "Level 4",
        "Level 5",
        "Level 6",
        "Level 7",
    ]
    rows = ["\t".join(header)]
    for keyword, prediction in zip(keywords, predictions):
        rows.append(
            "\t".join(
                [
                    keyword,
                    prediction["Level_0"],
                    prediction["Level_1"],
                    prediction["Level_2"],
                    prediction["Level_3"],
                    prediction["Level_4"],
                    prediction["Level_5"],
                    prediction["Level_6"],
                    prediction["Level_7"],
                ]
            )
        )
    return "\n".join(rows)
