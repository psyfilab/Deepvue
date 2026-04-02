"""Micro-segmentation enrichment for classified bank statements."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

from .statement_classifier import parse_amount

logger = logging.getLogger(__name__)

L5_SEGMENT_MAP = {
    "food": {"dining", "dining/bar", "dining/kirana", "food delivery"},
    "grocery": {"grocery", "groceries", "ecommerce/blinkit"},
    "ecomm": {"retail", "ecommerce"},
    "entertainment": {"entertainment", "entertainment/retail"},
    "subscription": {"subscription"},
    "transport": {"transport", "transport/vehicle"},
    "travel": {"travel"},
    "utility": {"utilities", "mobile recharge", "bill payment"},
    "fuel": {"fuel"},
    "pharmacy": {"pharmacy"},
    "insurance": {"insurance"},
    "investment": {"investment", "financial services"},
    "loan": {"loan repayment"},
    "creditcard": {"credit card"},
    "salary": {"salary"},
    "tax": {"tax"},
    "education": {"education", "education fees", "educational"},
    "refund": {"refund"},
}

SEGMENT_PRIORITY = [
    ("IsSalaryTxn", "Salary"),
    ("exp_food", "Food"),
    ("exp_grocery", "Grocery"),
    ("exp_ecomm", "Ecommerce"),
    ("exp_entertainment", "Entertainment"),
    ("exp_sub", "Subscription"),
    ("exp_investment", "Investment"),
    ("exp_saving", "Saving / Insurance"),
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
    ("exp_penalty", "Penalty / Charges"),
    ("exp_refund", "Refund"),
    ("exp_luxury", "Luxury"),
    ("exp_micro", "Micro-spend"),
    ("exp_upi_velocity", "UPI Velocity"),
]

BIAS_WEIGHTS = {
    "exp_food": 2.0,
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

L5_TO_SEGMENT = {
    value.lower(): segment for segment, values in L5_SEGMENT_MAP.items() for value in values
}
MEANINGFUL_HASH_RE = re.compile(r"^[a-z0-9]{12,}$")
ATM_RE = re.compile(r"\b(atm|atm wdl|cash wd|cash withdrawal|cash withd)\b", re.IGNORECASE)
PENALTY_RE = re.compile(
    r"\b(nach rtn|bounce|penalty|late fee|return chg|sms chrg|alert charge|fine|overdue|"
    r"chrgs?|chgs?|surcharge|markup)\b",
    re.IGNORECASE,
)
EMI_RE = re.compile(
    r"\b(emi|loan emi|intrest emi|loan repay|loan payment)\b",
    re.IGNORECASE,
)
SAVING_RE = re.compile(
    r"\b(fd|fixed deposit|rd|recurring deposit|ppf|nps|mutual fund|sip)\b",
    re.IGNORECASE,
)
INSURANCE_RE = re.compile(
    r"\b(lic|insurance|policy|premium|acko|policybazaar|star health|niva bupa|hdfc ergo)\b",
    re.IGNORECASE,
)
REFUND_RE = re.compile(r"\b(refund|reversal|cashback|reversed)\b", re.IGNORECASE)
SALARY_RE = re.compile(r"\b(salary|salaryxce|payroll|payslip|labour|wages|sal)\b", re.IGNORECASE)


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
    """Build segment-specific keyword vocabularies from the master sheet."""

    master_df = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    master_df.columns = [column.strip() for column in master_df.columns]
    vocab: dict[str, set[str]] = defaultdict(set)

    for _, row in master_df.iterrows():
        level_5 = str(row.get("Level 5", "")).strip().lower()
        segment = L5_TO_SEGMENT.get(level_5)
        keyword = str(row.get("Key Word", "")).strip().lower()
        if segment and is_meaningful_keyword(keyword):
            vocab[segment].add(keyword)

    return {segment: frozenset(values) for segment, values in vocab.items()}


def build_regexes(vocab: dict[str, frozenset[str]]) -> dict[str, re.Pattern[str]]:
    """Compile one regex per segment."""

    regexes: dict[str, re.Pattern[str]] = {}
    for segment, keywords in vocab.items():
        parts = []
        for keyword in sorted(keywords, key=len, reverse=True):
            escaped = re.escape(keyword)
            parts.append(rf"\b{escaped}\b" if len(keyword) <= 5 else escaped)
        if parts:
            regexes[segment] = re.compile("|".join(parts), re.IGNORECASE)
    return regexes


def _label(row: pd.Series, column_name: str) -> str:
    """Read a lowercase label from a dataframe row."""

    return str(row.get(column_name, "") or "").strip().lower()


def _particulars(row: pd.Series) -> str:
    """Read the raw particulars string from a row."""

    return str(row.get("Particulars", "") or "").strip()


def _flag_segment(
    row: pd.Series,
    segment: str,
    regexes: dict[str, re.Pattern[str]],
) -> int:
    """Return 1 when a row matches a segment via labels or regex."""

    if L5_TO_SEGMENT.get(_label(row, "Level_5")) == segment:
        return 1
    pattern = regexes.get(segment)
    if pattern and pattern.search(_particulars(row)):
        return 1
    return 0


def _assign_microsegment(row: pd.Series) -> str:
    """Assign one final micro-segment label."""

    if _label(row, "Level_3") == "p2p":
        return "P2P Transfer"
    if _label(row, "Level_5") == "digital services" and _label(row, "Level_3") != "p2p":
        return "Digital Payment"
    for column_name, label in SEGMENT_PRIORITY:
        if int(row.get(column_name, 0) or 0) == 1:
            return label
    return "Other"


def _compute_bias(row: pd.Series) -> float:
    """Compute a simple present-bias score from segment flags."""

    return round(
        sum(
            weight
            for column_name, weight in BIAS_WEIGHTS.items()
            if int(row.get(column_name, 0) or 0) == 1
        ),
        2,
    )


class Microsegmenter:
    """Enrich classified transactions with consumer micro-segmentation flags."""

    def __init__(self, master_sheet_path: str) -> None:
        """Build regex vocabularies from the local master sheet."""

        self.vocab = build_vocab(master_sheet_path)
        self.regexes = build_regexes(self.vocab)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return an enriched dataframe with segment flags and scores."""

        output_df = df.copy()
        output_df["debit_num"] = output_df["Debit"].apply(parse_amount)
        output_df["credit_num"] = output_df["Credit"].apply(parse_amount)

        output_df["exp_food"] = output_df.apply(
            lambda row: _flag_segment(row, "food", self.regexes), axis=1
        )
        output_df["exp_grocery"] = output_df.apply(
            lambda row: _flag_segment(row, "grocery", self.regexes), axis=1
        )
        output_df["exp_ecomm"] = output_df.apply(
            lambda row: _flag_segment(row, "ecomm", self.regexes), axis=1
        )
        output_df["exp_entertainment"] = output_df.apply(
            lambda row: _flag_segment(row, "entertainment", self.regexes), axis=1
        )
        output_df["exp_sub"] = output_df.apply(
            lambda row: int(
                _flag_segment(row, "subscription", self.regexes) == 1
                or (
                    _label(row, "Level_7") == "recurring"
                    and _label(row, "Level_5") in {"subscription", "digital services"}
                )
            ),
            axis=1,
        )
        output_df["exp_micro"] = output_df["debit_num"].apply(lambda value: int(0 < value < 200))
        output_df["exp_luxury"] = output_df.apply(
            lambda row: int(
                5000 < row["debit_num"] <= 500000
                and _label(row, "Level_3") != "p2p"
                and _label(row, "Level_5")
                not in {
                    "utilities",
                    "mobile recharge",
                    "salary",
                    "p2p transfer",
                    "loan repayment",
                    "credit card",
                    "insurance",
                    "investment",
                    "tax",
                }
            ),
            axis=1,
        )
        output_df["exp_saving"] = output_df.apply(
            lambda row: int(
                row["debit_num"] > 0
                and SAVING_RE.search(_particulars(row)) is not None
            ),
            axis=1,
        )
        output_df["exp_investment"] = output_df.apply(
            lambda row: int(
                row["debit_num"] > 0
                and (
                    _flag_segment(row, "investment", self.regexes) == 1
                    or _label(row, "First Level Classification") == "investment"
                )
            ),
            axis=1,
        )
        output_df["exp_loan"] = output_df.apply(
            lambda row: int(
                _flag_segment(row, "loan", self.regexes) == 1
                or EMI_RE.search(_particulars(row)) is not None
            ),
            axis=1,
        )
        output_df["exp_creditcard"] = output_df.apply(
            lambda row: int(_flag_segment(row, "creditcard", self.regexes) == 1), axis=1
        )
        output_df["exp_emi"] = output_df.apply(
            lambda row: int(
                EMI_RE.search(_particulars(row)) is not None
                or _label(row, "Level_7") == "recurring"
            ),
            axis=1,
        )
        output_df["exp_utility"] = output_df.apply(
            lambda row: int(_flag_segment(row, "utility", self.regexes) == 1), axis=1
        )
        output_df["exp_fuel"] = output_df.apply(
            lambda row: int(_flag_segment(row, "fuel", self.regexes) == 1), axis=1
        )
        output_df["exp_transport"] = output_df.apply(
            lambda row: int(_flag_segment(row, "transport", self.regexes) == 1), axis=1
        )
        output_df["exp_travel"] = output_df.apply(
            lambda row: int(_flag_segment(row, "travel", self.regexes) == 1), axis=1
        )
        output_df["exp_pharmacy"] = output_df.apply(
            lambda row: int(_flag_segment(row, "pharmacy", self.regexes) == 1), axis=1
        )
        output_df["exp_insurance"] = output_df.apply(
            lambda row: int(
                row["debit_num"] > 0
                and (
                    _flag_segment(row, "insurance", self.regexes) == 1
                    or INSURANCE_RE.search(_particulars(row)) is not None
                )
            ),
            axis=1,
        )
        output_df["exp_tax"] = output_df.apply(
            lambda row: int(_flag_segment(row, "tax", self.regexes) == 1), axis=1
        )
        output_df["exp_education"] = output_df.apply(
            lambda row: int(_flag_segment(row, "education", self.regexes) == 1), axis=1
        )
        output_df["exp_atm"] = output_df["Particulars"].astype(str).apply(
            lambda value: int(ATM_RE.search(value) is not None)
        )
        output_df["exp_penalty"] = output_df["Particulars"].astype(str).apply(
            lambda value: int(PENALTY_RE.search(value) is not None)
        )
        output_df["exp_refund"] = output_df.apply(
            lambda row: int(
                row["credit_num"] > 0
                and (
                    _flag_segment(row, "refund", self.regexes) == 1
                    or REFUND_RE.search(_particulars(row)) is not None
                )
            ),
            axis=1,
        )
        output_df["exp_upi_velocity"] = output_df.apply(
            lambda row: int(
                0 < row["debit_num"] < 500 and _label(row, "Level_1") == "upi"
            ),
            axis=1,
        )
        output_df["IsSalaryTxn"] = output_df.apply(
            lambda row: int(
                int(row.get("IsSalary", 0) or 0) == 1
                or (
                    row["credit_num"] > 0
                    and (
                        _label(row, "Level_5") == "salary"
                        or SALARY_RE.search(_particulars(row)) is not None
                    )
                )
            ),
            axis=1,
        )

        output_df["MicroSegment"] = output_df.apply(_assign_microsegment, axis=1)
        output_df["present_bias_score"] = output_df.apply(_compute_bias, axis=1)

        logger.info("Enriched %s rows with micro-segmentation flags", len(output_df))
        return output_df
