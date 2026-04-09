"""Tests for structured statement-level classification overrides."""

from __future__ import annotations

import pandas as pd

from src.statement_classifier import (
    aggregate_part_predictions,
    apply_structured_upi_overrides,
    build_salary_summary,
    derive_direction_from_amounts,
    detect_salary,
)


class _StubKeywordClassifier:
    """Tiny keyword classifier stub for statement tests."""

    def classify(self, keyword: str) -> dict[str, str]:
        normalized = keyword.strip().lower()
        base = {f"Level_{index}": "" for index in range(8)}
        if normalized == "apple med":
            base["Level_5"] = "pharmacy"
            base["Level_6"] = "apple med"
        elif normalized == "tata starbucks privat":
            base["Level_5"] = "dining"
            base["Level_6"] = "tata starbucks privat"
        elif normalized == "urmila shriram rajbha":
            base["Level_3"] = "p2p"
            base["Level_6"] = "urmila shriram rajbha"
        return base


def _blank_prediction() -> dict[str, str]:
    """Return an empty level dict."""

    return {f"Level_{index}": "" for index in range(8)}


def test_structured_p2m_sets_merchant_and_purpose() -> None:
    """P2M rows should use merchant structure and entity purpose."""

    part_predictions = [
        ("UPI", {"Level_1": "upi", **{f"Level_{i}": "" for i in range(8) if i != 1}}),
        ("P2M", _blank_prediction()),
        ("APPLE MED", _blank_prediction()),
        ("HDFC BANK", {"Level_4": "hdfc bank", **{f"Level_{i}": "" for i in range(8) if i != 4}}),
        ("UPI MAND", _blank_prediction()),
        ("P2V", _blank_prediction()),
    ]

    result = aggregate_part_predictions(part_predictions)
    result = apply_structured_upi_overrides(part_predictions, _StubKeywordClassifier(), result)

    assert result["Level_1"] == "upi"
    assert result["Level_3"] == "merchant"
    assert result["Level_4"] == "hdfc bank"
    assert result["Level_5"] == "pharmacy"
    assert result["Level_6"] == "apple med"


def test_structured_p2a_sets_p2p_transfer() -> None:
    """P2A rows should become p2p transfers."""

    part_predictions = [
        ("UPI", {"Level_1": "upi", **{f"Level_{i}": "" for i in range(8) if i != 1}}),
        ("P2A", _blank_prediction()),
        ("URMILA SHRIRAM RAJBHA", _blank_prediction()),
        ("UPI", _blank_prediction()),
        ("UNION BANK OF INDIA", {"Level_4": "union bank of india", **{f"Level_{i}": "" for i in range(8) if i != 4}}),
    ]

    result = aggregate_part_predictions(part_predictions)
    result = apply_structured_upi_overrides(part_predictions, _StubKeywordClassifier(), result)

    assert result["Level_1"] == "upi"
    assert result["Level_3"] == "p2p"
    assert result["Level_5"] == "p2p transfer"
    assert result["Level_6"] == "urmila shriram rajbha"


def test_direction_comes_from_debit_credit_columns() -> None:
    """Direction should be derived directly from amount columns."""

    assert derive_direction_from_amounts(100, 0) == "debit"
    assert derive_direction_from_amounts(0, 250) == "credit"
    assert derive_direction_from_amounts(0, 0) == ""


def test_salary_detection_solidifies_to_salary_level() -> None:
    """Salary rows should be normalized into one stable salary path."""

    df = pd.DataFrame(
        [
            {
                "Date": "2026-03-31",
                "Particulars": "SALARY ACME CORP",
                "Debit": 0,
                "Credit": 45000,
                "Level_0": "credit",
                "Level_1": "",
                "Level_2": "",
                "Level_3": "",
                "Level_4": "",
                "Level_5": "",
                "Level_6": "",
                "Level_7": "",
                "First Level Classification": "",
                "Second Level Classification": "",
            }
        ]
    )

    detected = detect_salary(df)
    salary_summary = build_salary_summary(detected)

    assert int(detected.loc[0, "IsSalary"]) == 1
    assert detected.loc[0, "Level_0"] == "credit"
    assert detected.loc[0, "Level_3"] == "institution"
    assert detected.loc[0, "Level_5"] == "salary"
    assert detected.loc[0, "First Level Classification"] == "Salary"
    assert detected.loc[0, "Second Level Classification"] == ""
    assert len(salary_summary) == 1
