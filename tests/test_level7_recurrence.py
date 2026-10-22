"""Tests for Level 7 (Recurring / One-time) classification."""

from __future__ import annotations

import pandas as pd
# pyrefly: ignore[missing-import]
import pytest

from src.statement_classifier import (
    LEVEL7_ONE_TIME,
    LEVEL7_RECURRING,
    BankStatementClassifier,
    apply_level7_recurrence,
    decide_level7,
    level7_counterparty,
)

R = LEVEL7_RECURRING
O = LEVEL7_ONE_TIME


def frame(rows: list[dict]) -> pd.DataFrame:
    """Build a statement frame. Each row: date, text, and debit or credit."""

    data = []
    for row in rows:
        data.append(
            {
                "Date": row.get("date"),
                "Particulars": row.get("text", ""),
                "Debit": row.get("debit", "-"),
                "Credit": row.get("credit", "-"),
                "Level_1": row.get("l1", ""),
                "Level_3": row.get("l3", ""),
                "Level_5": row.get("l5", ""),
            }
        )
    return pd.DataFrame(data)


def labels(rows: list[dict]) -> list[str]:
    return [label for label, _ in decide_level7(frame(rows))]


def upi(name: str, ref: int, bank: str = "SBIN") -> str:
    return f"TO TRANSFER- UPI/DR/{ref}/{name}/{bank}/MERCHANT/Groc-"


# ---------------------------------------------------------------- completeness


def test_every_valid_row_gets_exactly_one_label_and_non_transactions_stay_blank() -> None:
    df = frame(
        [
            {"date": "2025-01-05", "text": "UPI/DR/1/ARUN/SBIN/x", "debit": 100},
            {"date": "2025-01-06", "text": "UPI/CR/2/ARUN/SBIN/x", "credit": "1,200.50"},
            {"date": None, "text": None, "debit": 50},
            {"date": "2025-01-07", "text": "OPENING BALANCE"},
        ]
    )
    out = apply_level7_recurrence(df)
    assert out["Level_7"].tolist()[:3] != ["", "", ""]
    assert set(out["Level_7"].iloc[:3]) <= {R, O}
    assert out["Level_7"].iloc[3] == ""


def test_label_casing_matches_existing_column_values() -> None:
    assert (R, O) == ("recurring", "one-time")


# ---------------------------------------------------------------- explicit evidence


@pytest.mark.parametrize(
    "text",
    [
        "BIL/Home Loan XX64149 EMI Adimoola",
        "ACH D- HDFC BANK LTD-123456",
        "NACH/DR/HDFC LOAN/987654",
        "STANDING INSTRUCTION TO LIC",
        "AUTOPAY NETFLIX",
        "TO TRANSFER- UPI/DR/123456789012/NETFLIX/HDFC/UPI MANDATE/Mandate",
    ],
)
def test_explicit_emi_or_instruction_debit_is_recurring_without_history(text: str) -> None:
    assert labels([{"date": "2025-03-10", "text": text, "debit": 5000}]) == [R]


@pytest.mark.parametrize(
    "text",
    [
        "TO TRANSFER- UPI/DR/123456789012/PRIYA SI/BARB/9977082052/Payme -",
        "RTGS-CIUBR52025102900407024-SI HOMES-5109090100422",
        "EMI PROCESSING FEE",
        "EMI CONVERSION CHARGES",
        "NACH MANDATE REGISTRATION",
        "TO TRANSFER- UPI/DR/123456789012/RAHUL/HDFC/9977082052/EMI",
        "NEFT-HDFCN12345678-RAHUL-EMI",
        "RTGS-RATN12345678-LANDLORD-RENT",
    ],
)
def test_weak_or_unreliable_evidence_is_one_time(text: str) -> None:
    assert labels([{"date": "2025-03-10", "text": text, "debit": 5000}]) == [O]


def test_explicit_evidence_applies_to_debits_only() -> None:
    assert labels([{"date": "2025-03-10", "text": "INF/INFT/041/Loan Emi /GSS TORQUE E", "credit": 1000000}]) == [O]


def test_nach_credit_is_not_explicit_evidence() -> None:
    assert labels([{"date": "2025-03-10", "text": "ACH C- DIVIDEND ABC LTD", "credit": 120}]) == [O]


# ---------------------------------------------------------------- monthly history & month-end dates


def test_two_monthly_bill_payments_with_variable_amounts_are_recurring() -> None:
    rows = [
        {"date": "2025-01-05", "text": "BBPS/DR/AIRTEL2025010543/MOBILE RECHARGE/", "debit": 499},
        {"date": "2025-02-05", "text": "BBPS/DR/AIRTEL2025020543/MOBILE RECHARGE/", "debit": 549},
    ]
    assert labels(rows) == [R, R]


def test_month_end_date_shift_qualifies_as_recurring() -> None:
    rows = [
        {"date": "2025-01-31", "text": "BBPS/DR/BESCOM20250131/ELECTRICITY/", "debit": 1200},
        {"date": "2025-02-27", "text": "BBPS/DR/BESCOM20250227/ELECTRICITY/", "debit": 1350},
    ]
    assert labels(rows) == [R, R]


def test_missing_month_bimonthly_bill_is_recurring() -> None:
    rows = [
        {"date": "2025-01-10", "text": "BBPS/DR/WATER20250110/WATER BILL/", "debit": 350},
        {"date": "2025-03-10", "text": "BBPS/DR/WATER20250310/WATER BILL/", "debit": 420},
    ]
    assert labels(rows) == [R, R]


def test_two_monthly_generic_merchant_purchases_with_different_amounts_are_one_time() -> None:
    rows = [
        {"date": "2025-01-05", "text": upi("DMART ME", 111111111111), "debit": 504.85},
        {"date": "2025-02-05", "text": upi("DMART ME", 222222222222), "debit": 1250.00},
    ]
    assert labels(rows) == [O, O]


def test_two_monthly_identical_amounts_on_same_day_of_month_are_recurring() -> None:
    rows = [
        {"date": "2025-01-05", "text": upi("GYM FIT", 111111111111), "debit": 1500},
        {"date": "2025-02-04", "text": upi("GYM FIT", 222222222222), "debit": 1500},
    ]
    assert labels(rows) == [R, R]


def test_two_monthly_p2p_transfers_are_one_time() -> None:
    rows = [
        {"date": "2025-01-05", "text": upi("RAHUL K", 111111111111), "debit": 1500, "l3": "p2p"},
        {"date": "2025-02-04", "text": upi("RAHUL K", 222222222222), "debit": 1500, "l3": "p2p"},
    ]
    assert labels(rows) == [O, O]


def test_three_monthly_p2p_transfers_with_same_amount_are_recurring() -> None:
    rows = [
        {"date": d, "text": upi("LANDLORD", 100000000000 + i), "debit": 12000, "l3": "p2p"}
        for i, d in enumerate(["2025-01-02", "2025-02-02", "2025-03-04"])
    ]
    assert labels(rows) == [R, R, R]


def test_three_monthly_p2p_transfers_with_varying_amounts_are_one_time() -> None:
    amounts = [12000, 3000, 8000]
    rows = [
        {"date": d, "text": upi("FRIEND A", 100000000000 + i), "debit": a, "l3": "p2p"}
        for i, (d, a) in enumerate(zip(["2025-01-02", "2025-02-02", "2025-03-04"], amounts))
    ]
    assert labels(rows) == [O, O, O]


@pytest.mark.parametrize(
    ("second_date", "expected"),
    [
        ("2025-02-04", R),  # 30 days
        ("2025-01-30", R),  # 25 days: lower bound
        ("2025-02-09", R),  # 35 days: upper bound
        ("2025-01-29", O),  # 24 days
        ("2025-02-10", O),  # 36 days
    ],
)
def test_monthly_gap_window_bounds(second_date: str, expected: str) -> None:
    rows = [
        {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
        {"date": second_date, "text": "BIL/Electricity BESCOM", "debit": 1100, "l5": "utilities"},
    ]
    assert labels(rows)[0] == expected


def test_only_the_dates_inside_the_chain_are_recurring() -> None:
    rows = [
        {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
        {"date": "2025-02-05", "text": "BIL/Electricity BESCOM", "debit": 950, "l5": "utilities"},
        {"date": "2025-02-12", "text": "BIL/Electricity BESCOM", "debit": 100, "l5": "utilities"},
    ]
    assert labels(rows) == [R, R, O]


# ---------------------------------------------------------------- weekly / fortnightly


def test_three_weekly_fixed_amount_payments_are_recurring_and_two_are_not() -> None:
    dates = ["2025-01-06", "2025-01-13", "2025-01-20"]
    rows = [{"date": d, "text": upi("TUTOR", 100000000000 + i), "debit": 800} for i, d in enumerate(dates)]
    assert labels(rows) == [R, R, R]
    assert labels(rows[:2]) == [O, O]


def test_three_fortnightly_fixed_amount_payments_are_recurring() -> None:
    dates = ["2025-01-01", "2025-01-15", "2025-01-29"]
    rows = [{"date": d, "text": upi("MAID", 100000000000 + i), "debit": 2000} for i, d in enumerate(dates)]
    assert labels(rows) == [R, R, R]


def test_weekly_gap_outside_window_is_one_time() -> None:
    dates = ["2025-01-06", "2025-01-10", "2025-01-14"]
    rows = [{"date": d, "text": upi("TUTOR", 100000000000 + i), "debit": 800} for i, d in enumerate(dates)]
    assert labels(rows) == [O, O, O]


def test_weekly_everyday_purchases_with_changing_amounts_are_one_time() -> None:
    amounts = [120, 460, 215, 90]
    dates = ["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27"]
    rows = [
        {"date": d, "text": upi("DMART ME", 100000000000 + i), "debit": a}
        for i, (d, a) in enumerate(zip(dates, amounts))
    ]
    assert labels(rows) == [O] * 4


def test_atm_cash_withdrawals_are_never_history_recurring() -> None:
    dates = ["2025-01-06", "2025-01-13", "2025-01-20"]
    rows = [
        {"date": d, "text": f"TO ATM WDL- ATM/DR/{130247 + i}/AXIS BANK ATM/MUMBAI", "debit": 3000, "l1": "cash"}
        for i, d in enumerate(dates)
    ]
    assert labels(rows) == [O, O, O]


# ---------------------------------------------------------------- duplicates & identity


def test_same_day_transactions_count_as_one_occurrence_date() -> None:
    rows = [
        {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
        {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
    ]
    assert labels(rows) == [O, O]


def test_duplicate_row_does_not_change_the_label_of_a_real_chain() -> None:
    rows = [
        {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
        {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
        {"date": "2025-02-05", "text": "BIL/Electricity BESCOM", "debit": 950, "l5": "utilities"},
    ]
    assert labels(rows) == [R, R, R]


def test_payment_platform_names_do_not_fragment_counterparty_keys() -> None:
    first = level7_counterparty("PAID TO RAHUL VIA PHONEPE")
    second = level7_counterparty("PAID TO RAHUL VIA GPAY")
    assert first == second == "rahul"


# ---------------------------------------------------------------- refunds and reversals


def test_refund_or_reversal_rows_are_one_time() -> None:
    rows = [
        {"date": "2025-01-05", "text": "REV-UPI/CR/111111111111/NETFLIX/HDFC/Refund", "credit": 649},
        {"date": "2025-01-05", "text": "RTN CHQ 123456 FUNDS INSUFFICIENT", "credit": 5000},
    ]
    assert labels(rows) == [O, O]


def test_reversed_original_is_removed_from_the_chain() -> None:
    rows = [
        {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
        {"date": "2025-02-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
        {"date": "2025-02-07", "text": "REVERSAL BIL/Electricity BESCOM", "credit": 900},
    ]
    assert labels(rows) == [O, O, O]


# ---------------------------------------------------------------- missing and odd data


def test_missing_dates_and_narration_never_raise_and_are_one_time() -> None:
    rows = [
        {"date": None, "text": "BIL/Electricity BESCOM", "debit": 900},
        {"date": "not a date", "text": "BIL/Electricity BESCOM", "debit": 900},
        {"date": "2025-01-05", "text": None, "debit": 900},
        {"date": "2025-02-05", "text": "", "debit": 900},
    ]
    assert labels(rows) == [O, O, O, O]


def test_missing_date_with_explicit_instruction_is_still_recurring() -> None:
    assert labels([{"date": None, "text": "NACH/DR/HDFC LOAN/987654", "debit": 5000}]) == [R]


# ---------------------------------------------------------------- scope & integration


def test_only_level_7_changes_and_input_is_not_mutated() -> None:
    df = frame(
        [
            {"date": "2025-01-05", "text": "BIL/Electricity BESCOM", "debit": 900, "l5": "utilities"},
            {"date": "2025-02-05", "text": "BIL/Electricity BESCOM", "debit": 950, "l5": "utilities"},
        ]
    )
    df["Level_7"] = ["", "recurring"]
    before = df.copy(deep=True)
    out = apply_level7_recurrence(df)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(out.drop(columns="Level_7"), before.drop(columns="Level_7"))
    assert out["Level_7"].tolist() == [R, R]


def test_classifier_pipeline_fills_level_7_for_every_transaction() -> None:
    class FakeKeywordClassifier:
        def classify_batch(self, parts: list[str]) -> list[dict[str, str]]:
            return [{} for _ in parts]

    classifier = BankStatementClassifier.__new__(BankStatementClassifier)
    classifier.keyword_classifier = FakeKeywordClassifier()
    df = frame(
        [
            {"date": "2025-06-15", "text": "BIL/Home Loan XX64149 EMI Adimoola", "debit": 232230},
            {"date": "2025-06-16", "text": "TO TRANSFER- UPI/DR/111111111111/DMART ME/SBIN/MERCHANT/Groc-", "debit": 500},
        ]
    )
    df = df.drop(columns=["Level_1", "Level_3", "Level_5"])
    for index in range(8):
        df[f"Part_{index}"] = ""
    df["Part_0"] = df["Particulars"]
    out = classifier.classify(df)
    assert out["Level_7"].tolist() == [R, O]