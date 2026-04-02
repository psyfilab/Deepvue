"""Transaction-level classification built on top of keyword models."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

from .predict import KeywordClassifier

logger = logging.getLogger(__name__)

PART_COLUMNS = tuple(f"Part_{index}" for index in range(8))
SALARY_KEYWORDS = re.compile(
    r"\b(?:salary|salaryxce|payroll|payslip|labour|wages|sal)\b",
    flags=re.IGNORECASE,
)
SLASH_UPI_RE = re.compile(
    r"\bUPI\b(?:/\d+)?/(?:DR|CR)(?:/\d+)?/([^/\d\-][^/]{1,19})/([A-Z]{2,6})/(\S+)",
    flags=re.IGNORECASE,
)
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


def apply_raw_transaction_overrides(
    particulars: str,
    result: dict[str, str],
) -> dict[str, str]:
    """Override part aggregation when the raw narration structure is explicit."""

    text = str(particulars or "")
    match = SLASH_UPI_RE.search(text.upper())
    if match:
        name = match.group(1).strip().lower()
        bank_code = match.group(2).strip().lower()
        vpa = match.group(3).strip().lower()
        bank_name = BANK_CODE_TO_NAME.get(bank_code, "")
        is_gateway = any(fragment in vpa for fragment in GATEWAY_VPA_FRAGMENTS)

        result["Level_1"] = result["Level_1"] or "upi"
        result["Level_4"] = bank_name or result["Level_4"]
        result["Level_6"] = name
        if is_gateway:
            result["Level_3"] = "merchant"
        else:
            result["Level_3"] = "p2p"
            result["Level_5"] = "p2p transfer"

    if SALARY_KEYWORDS.search(text):
        entity = SALARY_KEYWORDS.sub("", text)
        entity = re.sub(r"[\-/|]+", " ", entity)
        entity = re.sub(r"\s+", " ", entity).strip().lower()
        result["Level_3"] = "institution"
        result["Level_4"] = ""
        result["Level_5"] = "salary"
        result["Level_6"] = entity

    return _apply_transaction_rules(result)


def derive_first_level_classification(row: pd.Series) -> str:
    """Map row-level predictions into a coarse business transaction label."""

    debit_num = parse_amount(row.get("Debit"))
    credit_num = parse_amount(row.get("Credit"))
    level_1 = str(row.get("Level_1", "") or "").strip().lower()
    level_2 = str(row.get("Level_2", "") or "").strip().lower()
    level_5 = str(row.get("Level_5", "") or "").strip().lower()
    particulars = str(row.get("Particulars", "") or "")

    if level_5 == "salary":
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

    explicit_mask = (
        output_df.get("Level_5", pd.Series("", index=output_df.index))
        .astype(str)
        .str.contains("salary", case=False, na=False)
        | output_df.get("First Level Classification", pd.Series("", index=output_df.index))
        .astype(str)
        .str.contains("salary", case=False, na=False)
        | (
            output_df["Particulars"].astype(str).str.contains(SALARY_KEYWORDS, na=False)
            & (credit_num >= 10_000)
        )
    )

    output_df.loc[explicit_mask, "IsSalary"] = 1
    output_df.loc[explicit_mask, "SalaryConfidence"] = 1.0

    dates = pd.to_datetime(output_df.get("Date"), errors="coerce")
    candidates = output_df.index[
        (~explicit_mask) & (credit_num >= 10_000) & dates.notna()
    ]

    if len(candidates) >= 2:
        amount_keys = ((credit_num.loc[candidates] / 1000).round() * 1000).astype(int)
        clusters: dict[int, list[int]] = defaultdict(list)
        for index, amount_key in amount_keys.items():
            clusters[int(amount_key)].append(index)

        for indices in clusters.values():
            if len(indices) < 2:
                continue

            day_std = dates.loc[indices].dt.day.std(ddof=0)
            text_hit = output_df.loc[indices, "Particulars"].astype(str).str.contains(
                SALARY_KEYWORDS,
                na=False,
            ).any()

            score = 0.40
            score += 0.30 if (pd.notna(day_std) and day_std <= 7) else 0.0
            score += 0.30 if text_hit else 0.0

            if score >= 0.60:
                output_df.loc[indices, "IsSalary"] = 1
                output_df.loc[indices, "SalaryConfidence"] = round(score, 2)

    output_df.loc[
        output_df["IsSalary"].eq(1),
        "First Level Classification",
    ] = "Salary"
    output_df.loc[
        output_df["IsSalary"].eq(1) & output_df["Second Level Classification"].eq(""),
        "Second Level Classification",
    ] = "Salary"

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
            row_prediction = aggregate_part_predictions(grouped_predictions.get(row_index, []))
            row_prediction = apply_raw_transaction_overrides(
                str(output_df.loc[row_index, "Particulars"]),
                row_prediction,
            )
            row_level_predictions.append(row_prediction)

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
