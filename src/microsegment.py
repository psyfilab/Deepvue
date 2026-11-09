"""Micro-segmentation enrichment for classified bank statements.

Rule order for every segment flag:
  1. Level 5 wins. When the classifier set a Level 5, the segment comes from the Level 5 table below.
  2. Only when Level 5 is blank: the purpose tail of a UPI narration ("Groc-", "Trvl-"...), then keywords.
  3. Keywords read the payee name part only, as whole words. The UPI handle, bank name, reference and
     purpose tail are never read (a bank name never triggers a flag).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

from .salary_timeseries import (
    payee_info,
    purpose_tail,
    purpose_text,
    refine_salary,
    strip_noise,
    word_variants,
)
from .statement_classifier import parse_amount

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------ Level 5 -> segment
# One table for debit and credit; SEGMENT_DIRECTION says which side a segment may fire on.
L5_SEGMENT_MAP = {
    "food": {"dining", "dining/bar", "dining/kirana", "food delivery"},
    "grocery": {"grocery", "groceries", "ecommerce/blinkit"},
    "ecomm": {"retail", "ecommerce"},
    "entertainment": {"entertainment", "entertainment/retail"},
    "sub": {"subscription"},
    "transport": {"transport", "transport/vehicle"},
    "travel": {"travel"},
    "utility": {"utilities", "mobile recharge", "bill payment", "telecom"},
    "fuel": {"fuel"},
    "pharmacy": {"pharmacy", "healthcare"},
    "insurance": {"insurance"},
    "investment": {"investment", "financial services", "dividend"},
    "loan": {"loan repayment"},
    "creditcard": {"credit card"},
    "tax": {"tax"},
    "education": {"education", "education fees", "educational"},
    "refund": {"refund", "cashback"},
    "penalty": {"interest", "charges"},
    "own_account": {"wallet load", "wallet unload"},
}
L5_TO_SEGMENT = {value: segment for segment, values in L5_SEGMENT_MAP.items() for value in values}
assert len(L5_TO_SEGMENT) == sum(len(v) for v in L5_SEGMENT_MAP.values()), "a Level 5 value is in two segments"

# Which side of the statement each segment may fire on: debit, credit or any.
SEGMENT_DIRECTION = {
    "food": "debit", "grocery": "debit", "quick_comm": "debit", "ecomm": "debit", "entertainment": "debit",
    "sub": "debit", "saving": "debit", "loan": "debit", "emi": "debit", "creditcard": "debit",
    "utility": "debit", "fuel": "debit", "transport": "debit", "travel": "debit", "pharmacy": "debit",
    "tax": "debit", "education": "debit", "refund": "credit",
    "penalty": "any", "insurance": "any", "investment": "any", "own_account": "any",
}
SEGMENT_FLAG = {segment: f"exp_{segment}" for segment in (
    "food", "grocery", "quick_comm", "ecomm", "entertainment", "sub", "saving", "investment", "loan",
    "creditcard", "emi", "utility", "fuel", "transport", "travel", "pharmacy", "insurance", "tax",
    "education", "penalty", "refund")}
SEGMENT_FLAG["sub"] = "exp_sub"

# Level 5 values too generic to decide a segment: keywords may read the payee, as if Level 5 were blank.
GENERIC_LEVEL_5 = frozenset({"services", "other", "misc", "miscellaneous"})
# When no segment came from Level 5 or keywords, the First Level Classification decides (credit card
# payments, insurance, tax and refunds the classifier already identified).
FIRST_LEVEL_SEGMENT = {
    "credit card payment": "creditcard", "insurance": "insurance", "tax": "tax",
    "refund/reversal": "refund", "inward online return": "refund", "refund": "refund",
}

# Level 5 blank rows: the purpose tag a UPI narration ends with decides the segment (None = no segment).
PURPOSE_TAIL_SEGMENT = {
    "groc": "grocery", "trvl": "travel", "wall": "own_account", "sub": "sub", "cab": "transport",
    "inv": "investment", "cryp": "investment", "misc": None,
    "hlth": "pharmacy", "phar": "pharmacy", "pharm": "pharmacy", "edu": "education", "shop": "ecomm",
    "food": "food", "tran": "transport", "trav": "travel", "ente": "entertainment", "fuel": "fuel",
    "rfnd": "refund",
}

# Extra keywords, merged with the Master sheet vocabulary (whole words, payee name only).
BUILTIN_KEYWORDS = {
    "grocery": {"bigbasket"},
    "travel": {"makemytrip", "cleartrip", "yatra", "goibibo", "hotel", "hotels"},
    "transport": {"uber", "ola", "rapido", "metro", "fastag"},
}
QUICK_COMMERCE_WORDS = ("blinkit", "grofers", "zepto", "instamart", "swiggy instamart", "bbnow", "dunzo",
                        "flipkart minutes", "fk minutes", "amazon fresh")
# A "financial services" Level 5 is a loan when the payee says so.
LOAN_FINANCE_WORDS = ("emi", "loan", "fin", "finance", "finserv", "fintech", "kissht", "slice")
EMI_WORDS = ("emi", "emis", "loan", "loans", "loan emi", "hand loan", "loan repay", "loan payment")
PENALTY_WORDS = ("nach rtn", "bounce", "penalty", "late fee", "return chg", "sms chrg", "sms chrgs",
                 "alert charge", "overdue", "chrg", "chrgs", "chg", "chgs", "charges", "surcharge",
                 "markup", "imps sc", "transaction charges")
SAVING_WORDS = ("fd", "fixed deposit", "rd", "recurring deposit", "ppf", "nps", "mutual fund", "sip")
INSURANCE_WORDS = ("lic", "insurance", "policy", "premium", "acko", "policybazaar", "star health",
                   "niva bupa", "hdfc ergo")
REFUND_WORDS = ("refund", "reversal", "cashback", "reversed")
ATM_WORDS = ("atm", "atm wdl", "cash wd", "cash withdrawal", "cash withd", "nwd", "atw")
MEDICAL_WORDS = ("lab", "labs", "laboratory", "diagnostic", "diagnostics", "hospital", "hospitals", "clinic",
                 "clinics", "fertility", "ivf", "pharmacy")

# --------------------------------------------------------------------------- labels and priorities
SEGMENT_PRIORITY = [
    ("exp_penalty", "Penalty / Charges"),
    ("exp_quick_comm", "Quick Commerce"),
    ("exp_food", "Food"),
    ("exp_grocery", "Grocery"),
    ("exp_ecomm", "Ecommerce"),
    ("exp_entertainment", "Entertainment"),
    ("exp_sub", "Subscription"),
    ("exp_investment", "Investment"),
    ("exp_saving", "Saving / Insurance"),
    ("exp_insurance", "Saving / Insurance"),
    ("exp_loan", "Loan"),
    ("exp_creditcard", "Credit Card Payment"),
    ("exp_emi", "EMI"),
    ("exp_utility", "Utility"),
    ("exp_fuel", "Fuel"),
    ("exp_transport", "Transport"),
    ("exp_travel", "Travel"),
    ("exp_pharmacy", "Pharmacy"),
    ("exp_tax", "Tax"),
    ("exp_education", "Education"),
    ("exp_atm", "ATM Withdrawal"),
    ("exp_refund", "Refund"),
    ("exp_luxury", "Luxury"),
    ("exp_micro", "Micro-spend"),
    ("exp_upi_velocity", "UPI Velocity"),
]

BIAS_WEIGHTS = {
    "exp_food": 2.0,
    "exp_quick_comm": 1.5,
    "exp_micro": 2.0,
    "exp_luxury": 1.5,
    "exp_upi_velocity": 1.5,
    "exp_ecomm": 1.0,
    "exp_entertainment": 1.0,
    "exp_penalty": 1.0,
    "exp_grocery": 0.5,
    "exp_sub": 0.5,
    "exp_emi": 0.5,
    "exp_atm": 0.5,
    "exp_creditcard": 0.5,
    "exp_pharmacy": 0.0,
    "exp_transport": 0.0,
    "exp_travel": 0.0,
    "exp_refund": 0.0,
    "exp_tax": -0.5,
    "exp_utility": -0.5,
    "exp_fuel": -0.5,
    "exp_education": -1.0,
    "exp_insurance": -1.5,
    "exp_saving": -2.0,
    "exp_investment": -2.0,
}

# First Level values are lower-case; they come from the First Level classifier.
LUXURY_EXCLUDED_FIRST_LEVEL = frozenset({
    "rtgs payment", "cheque payment", "neft payment", "ecs/nach payment", "insurance", "tax", "loan",
    "salary", "rent payment", "utility expenses", "contra", "credit card payment", "investment",
})
LUXURY_LEVEL_5 = frozenset({"", "retail", "services"})
LUXURY_BLOCKING_SEGMENTS = frozenset({
    "penalty", "loan", "emi", "investment", "saving", "tax", "insurance", "pharmacy", "own_account", "creditcard",
})
LUXURY_MIN, LUXURY_MAX = 5000, 500000
CASH_WITHDRAWAL_FIRST_LEVEL = frozenset({"cash withdrawal", "atm withdrawal"})
INWARD_FIRST_LEVEL = frozenset({"loan", "imps receipt", "neft receipt", "rtgs receipt", "cheque receipt", "online receipt"})
NEUTRAL_BIAS_LABELS = frozenset({"P2P Transfer", "Own Account Transfer"})  # no present-bias signal

MEANINGFUL_HASH_RE = re.compile(r"^[a-z0-9]{12,}$")
# A "/Inv-" style purpose tag is handled by PURPOSE_TAIL_SEGMENT; these two read the payee text.
MF_HOUSE_RE = re.compile(r"(?<= )(?:[a-z]{2,12}mf|mf sip)(?= )")


def is_meaningful_keyword(keyword: str) -> bool:
    """Filter out very noisy tokens when building regex vocabularies."""

    value = keyword.strip().lower()
    if len(value) < 3 or len(value) > 45:
        return False
    if MEANINGFUL_HASH_RE.match(value):
        return False
    if not re.search(r"[a-z]", value):
        return False
    digit_ratio = sum(char.isdigit() for char in value) / max(len(value), 1)
    return digit_ratio <= 0.35


def build_vocab(master_path: str) -> dict[str, frozenset[str]]:
    """Build segment-specific keyword vocabularies from the master sheet plus BUILTIN_KEYWORDS."""

    master_df = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    master_df.columns = [column.strip() for column in master_df.columns]
    vocab: dict[str, set[str]] = defaultdict(set)

    for _, row in master_df.iterrows():
        level_5 = str(row.get("Level 5", "")).strip().lower()
        segment = L5_TO_SEGMENT.get(level_5)
        keyword = str(row.get("Key Word", "")).strip().lower()
        if segment and segment != "own_account" and is_meaningful_keyword(keyword):
            vocab[segment].add(keyword)
    for segment, words in BUILTIN_KEYWORDS.items():
        vocab[segment] |= set(words)

    return {segment: frozenset(values) for segment, values in vocab.items()}


def phrase_regex(words) -> re.Pattern[str]:
    """Whole-word regex over word_variants() text (every word has a space on both sides)."""

    parts = []
    for word in sorted({" ".join(re.sub(r"[^a-z0-9]+", " ", w.lower()).split()) for w in words}, key=len, reverse=True):
        if word:
            parts.append(re.escape(word))
    return re.compile(r"(?<= )(?:" + "|".join(parts) + r")(?= )")


def build_regexes(vocab: dict[str, frozenset[str]]) -> dict[str, re.Pattern[str]]:
    """Compile one whole-word regex per segment."""

    return {segment: phrase_regex(keywords) for segment, keywords in vocab.items() if keywords}


QUICK_COMM_RE = phrase_regex(QUICK_COMMERCE_WORDS)
LOAN_FINANCE_RE = phrase_regex(LOAN_FINANCE_WORDS)
EMI_RE = phrase_regex(EMI_WORDS)
PENALTY_RE = phrase_regex(PENALTY_WORDS)
SAVING_RE = phrase_regex(SAVING_WORDS)
INSURANCE_RE = phrase_regex(INSURANCE_WORDS)
REFUND_RE = phrase_regex(REFUND_WORDS)
ATM_RE = phrase_regex(ATM_WORDS)
MEDICAL_RE = phrase_regex(MEDICAL_WORDS)


def _label(row: pd.Series, column_name: str) -> str:
    """Read a lowercase label from a dataframe row."""

    return str(row.get(column_name, "") or "").strip().lower()


def _particulars(row: pd.Series) -> str:
    """Read the raw particulars string from a row."""

    return str(row.get("Particulars", "") or "").strip()


def _is_debit(row: pd.Series) -> bool:
    """True for a money-out row."""

    return float(row.get("debit_num", 0) or 0) > 0


def _is_credit(row: pd.Series) -> bool:
    return float(row.get("credit_num", 0) or 0) > 0


def _direction_ok(segment: str, debit: bool, credit: bool) -> bool:
    side = SEGMENT_DIRECTION.get(segment, "any")
    return side == "any" or (side == "debit" and debit) or (side == "credit" and credit)


def _assign_microsegment(row: pd.Series) -> str:
    """Assign one final micro-segment label."""

    level_5, level_3 = _label(row, "Level_5"), _label(row, "Level_3")
    first_level = _label(row, "First Level Classification")
    # Structural facts first: Level 5 salary or wallet moves, and cash withdrawals by First Level.
    if level_5 == "salary" and _is_credit(row):
        return "Salary"
    if level_5 in L5_SEGMENT_MAP["own_account"] or row.get("_own_account") == 1:
        return "Own Account Transfer"
    if _is_debit(row) and first_level in CASH_WITHDRAWAL_FIRST_LEVEL:
        return "ATM Withdrawal"
    l5_has_segment = level_5 in L5_TO_SEGMENT
    if level_5 == "digital services" and level_3 != "p2p":
        return "Digital Payment"
    if level_3 == "p2p" and not l5_has_segment:
        return "P2P Transfer"  # Level 5 wins over p2p only when Level 5 maps to a segment
    for column_name, label in SEGMENT_PRIORITY:
        if int(row.get(column_name, 0) or 0) == 1:
            return label
    if _is_credit(row) and first_level in INWARD_FIRST_LEVEL:
        return "Inward Transfer"
    return "Other"


def _compute_bias(row: pd.Series) -> float:
    """Present-bias score: weighted spend flags. Credits, P2P and own-account moves are neutral."""

    if row.get("MicroSegment") in NEUTRAL_BIAS_LABELS or not _is_debit(row):
        return 0.0
    return round(
        sum(weight for column_name, weight in BIAS_WEIGHTS.items() if int(row.get(column_name, 0) or 0) == 1),
        2,
    )


class Microsegmenter:
    """Enrich classified transactions with consumer micro-segmentation flags."""

    def __init__(self, master_sheet_path: str) -> None:
        """Build whole-word vocabularies from the local master sheet."""

        self.vocab = build_vocab(master_sheet_path)
        self.regexes = build_regexes(self.vocab)

    # ------------------------------------------------------------------ one row
    def _segments(self, row: pd.Series) -> tuple[set[str], str, bool, str]:
        """Segments for one row, plus the payee text, whether the payee is a bank, and the purpose text."""

        debit, credit = _is_debit(row), _is_credit(row)
        narration = _particulars(row)
        info = payee_info(narration, row.get("Level_6", ""))
        payee = word_variants(info.text)
        purpose = word_variants(purpose_text(narration))
        words = word_variants(strip_noise(narration))
        level_5 = _label(row, "Level_5")
        segments: set[str] = set()

        if level_5 and level_5 not in GENERIC_LEVEL_5:  # Level 5 always wins
            segment = L5_TO_SEGMENT.get(level_5)
            if segment == "grocery" and QUICK_COMM_RE.search(payee + purpose):
                segment = "quick_comm"
            if level_5 == "financial services" and LOAN_FINANCE_RE.search(payee + purpose):
                segment = "loan"
            if segment:
                segments.add(segment)
            if level_5 in {"digital services"} and _label(row, "Level_7") == "recurring":
                segments.add("sub")
        else:
            tail = purpose_tail(narration)
            if tail in PURPOSE_TAIL_SEGMENT:
                if PURPOSE_TAIL_SEGMENT[tail]:
                    segments.add(PURPOSE_TAIL_SEGMENT[tail])
            else:
                if not info.is_bank:
                    for segment, pattern in self.regexes.items():
                        if segment != "own_account" and pattern.search(payee):
                            segments.add(segment)
                    if QUICK_COMM_RE.search(payee):
                        segments.add("quick_comm")
                    if MF_HOUSE_RE.search(payee):
                        segments.add("investment")
                    if PENALTY_RE.search(payee):
                        segments.add("penalty")
                    if SAVING_RE.search(payee):
                        segments.add("saving")
                    if INSURANCE_RE.search(payee):
                        segments.add("insurance")
                if PENALTY_RE.search(words) and not info.name:  # bank-written lines carry no payee
                    segments.add("penalty")
                if EMI_RE.search(payee + purpose) or (not info.name and EMI_RE.search(words)):
                    segments.add("emi")
                if REFUND_RE.search(words):
                    segments.add("refund")
        if "quick_comm" in segments:
            segments.discard("grocery")
        segments = {s for s in segments if _direction_ok(s, debit, credit)}
        fallback = FIRST_LEVEL_SEGMENT.get(_label(row, "First Level Classification"))
        if not segments and fallback and _direction_ok(fallback, debit, credit):
            segments.add(fallback)
        return segments, payee, info.is_bank, purpose

    def _row_flags(self, row: pd.Series) -> dict[str, int]:
        segments, payee, is_bank, purpose = self._segments(row)
        debit_amount = float(row.get("debit_num", 0) or 0)
        level_5, level_3 = _label(row, "Level_5"), _label(row, "Level_3")
        first_level = _label(row, "First Level Classification")
        own_account = "own_account" in segments or level_5 in L5_SEGMENT_MAP["own_account"]
        if level_5 in L5_SEGMENT_MAP["own_account"]:
            segments.add("own_account")
        flags = {column: int(segment in segments) for segment, column in SEGMENT_FLAG.items()}
        if level_5 == "digital services":
            flags["exp_sub"] = int("sub" in segments and debit_amount > 0)

        words = word_variants(strip_noise(_particulars(row)))
        flags["exp_atm"] = int(
            debit_amount > 0
            and "penalty" not in segments
            and (first_level in CASH_WITHDRAWAL_FIRST_LEVEL or (not level_5 and bool(ATM_RE.search(words))))
        )
        flags["exp_micro"] = int(0 < debit_amount < 200 and not own_account)
        flags["exp_luxury"] = int(
            level_3 == "merchant"
            and level_5 in LUXURY_LEVEL_5
            and LUXURY_MIN < debit_amount <= LUXURY_MAX
            and first_level not in LUXURY_EXCLUDED_FIRST_LEVEL
            and not (segments & LUXURY_BLOCKING_SEGMENTS)
            and not is_bank
            and not MEDICAL_RE.search(payee + purpose)
        )
        flags["exp_upi_velocity"] = int(0 < debit_amount < 500 and _label(row, "Level_1") == "upi" and not own_account)
        flags["_own_account"] = int(own_account)
        return flags

    # ------------------------------------------------------------------ whole frame
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return an enriched dataframe with segment flags and scores."""

        output_df = refine_salary(df)  # Salary Flag, Level 3 / 5 and First Level settle before segments
        output_df["debit_num"] = output_df["Debit"].apply(parse_amount)
        output_df["credit_num"] = output_df["Credit"].apply(parse_amount)

        flag_rows = [self._row_flags(row) for _, row in output_df.iterrows()]
        flag_frame = pd.DataFrame(flag_rows, index=output_df.index)
        for column in (
            "exp_food", "exp_grocery", "exp_quick_comm", "exp_ecomm", "exp_entertainment", "exp_sub",
            "exp_micro", "exp_luxury", "exp_saving", "exp_investment", "exp_loan", "exp_creditcard",
            "exp_emi", "exp_utility", "exp_fuel", "exp_transport", "exp_travel", "exp_pharmacy",
            "exp_insurance", "exp_tax", "exp_education", "exp_atm", "exp_penalty", "exp_refund",
            "exp_upi_velocity",
        ):
            output_df[column] = flag_frame[column].astype(int) if len(flag_frame) else 0
        output_df["IsSalaryTxn"] = (
            output_df["IsSalary"].fillna(0).astype(int).eq(1) & output_df["credit_num"].gt(0)
        ).astype(int)

        output_df["_own_account"] = flag_frame["_own_account"] if len(flag_frame) else 0
        output_df["MicroSegment"] = output_df.apply(_assign_microsegment, axis=1)
        output_df["present_bias_score"] = output_df.apply(_compute_bias, axis=1)
        output_df = output_df.drop(columns=["_own_account"])

        logger.info("Enriched %s rows with micro-segmentation flags", len(output_df))
        return output_df