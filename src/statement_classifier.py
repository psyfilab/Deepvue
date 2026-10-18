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


def derive_direction_from_amounts(debit_value: object, credit_value: object) -> str:
    """Return debit or credit from the numeric transaction columns."""

    debit_num = parse_amount(debit_value)
    credit_num = parse_amount(credit_value)
    if debit_num > 0 and credit_num <= 0:
        return "debit"
    if credit_num > 0 and debit_num <= 0:
        return "credit"
    return ""


def _apply_transaction_rules(result: dict[str, str]) -> dict[str, str]:
    """Apply a few row-level consistency rules after aggregation."""

    if result["Level_3"] == "p2p" and not result["Level_5"]:
        result["Level_5"] = "p2p transfer"

    if not result["Level_3"] and result["Level_5"] == "salary":
        result["Level_3"] = "institution"

    if result["Level_1"] == "cash":
        result["Level_3"] = ""
        result["Level_4"] = ""
        result["Level_5"] = ""
        result["Level_6"] = ""

    if result["Level_5"] == "salary":
        result["Level_3"] = "institution"
        result["Level_4"] = ""

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
    """Return True for a label that names a payment gateway rather than a bank.

    Level_4 holds a bank. A label that is a known bank name (including
    "paytm", which is a bank value in ``BANK_CODE_TO_NAME``) is never treated
    as an intermediary.
    """

    value = _normalize_compare(label)
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

    Level_6 is the merchant, never the payment intermediary. Level_4 is a bank:
    a valid bank already on the row is kept; an intermediary label such as
    "razorpay" (or a blank) is replaced by the first valid bank any part
    predicted, else the bank derived from the narration's IFSC or VPA handle,
    else left blank.
    """

    merchant = extract_hyphen_upi_merchant(particulars)
    if not merchant:
        return result

    purpose = keyword_classifier.classify(merchant).get("Level_5", "")
    if not purpose or purpose in {"p2p transfer", "salary"}:
        purpose = MERCHANT_DEFAULT_PURPOSES.get(merchant, "")

    bank = "" if is_payment_intermediary_label(result["Level_4"]) else result["Level_4"]
    if not bank:
        bank = next(
            (
                prediction["Level_4"]
                for _, prediction in part_predictions
                if prediction.get("Level_4") and not is_payment_intermediary_label(prediction["Level_4"])
            ),
            "",
        )
    result["Level_4"] = bank or derive_upi_bank(particulars)

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
    """Override part aggregation when the raw narration structure is explicit.

    ``part_texts`` (the row's split parts) is used only to detect an explicit
    MERCHANT marker when it is not visible in the Particulars text.
    ``debit_value`` / ``credit_value`` are optional; when they show a debit, the
    salary keyword override is skipped because salary is income.
    """

    text = str(particulars or "")
    is_debit = derive_direction_from_amounts(debit_value, credit_value) == "debit"
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
        else:
            result["Level_3"] = "p2p"
            result["Level_5"] = "p2p transfer"

    if MERCHANT_MARKER_RE.search(_marker_text(text, part_texts)):
        result["Level_3"] = "merchant"
        if result["Level_5"] == "p2p transfer":
            result["Level_5"] = ""

    if SALARY_KEYWORDS.search(text) and not is_debit:
        entity = SALARY_KEYWORDS.sub("", text)
        entity = re.sub(r"[\-/|]+", " ", entity)
        entity = re.sub(r"\s+", " ", entity).strip().lower()
        result["Level_3"] = "institution"
        result["Level_4"] = ""
        result["Level_5"] = "salary"
        result["Level_6"] = entity

    return _apply_transaction_rules(result)


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
    for _, prediction in part_predictions:
        if not bank_name and prediction.get("Level_4"):
            bank_name = prediction["Level_4"]
        if not location and prediction.get("Level_6") and prediction.get("Level_3") != "p2p":
            location = prediction["Level_6"]

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
    salary_entity = output_df["Particulars"].astype(str).map(extract_salary_entity)
    output_df.loc[salary_mask, "Level_0"] = "credit"
    output_df.loc[salary_mask, "Level_1"] = ""
    output_df.loc[salary_mask, "Level_2"] = ""
    output_df.loc[salary_mask, "Level_3"] = "institution"
    output_df.loc[salary_mask, "Level_4"] = ""
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
            row_prediction = apply_debit_salary_guard(
                output_df.loc[row_index, "Debit"],
                output_df.loc[row_index, "Credit"],
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
        output_df["First Level Classification"] = output_df.apply(
            derive_first_level_classification,
            axis=1,
        )
        output_df["Second Level Classification"] = output_df.apply(
            derive_second_level_classification,
            axis=1,
        )

        output_df = detect_salary(output_df)
        logger.info("Classified %s statement rows", len(output_df))
        return output_df