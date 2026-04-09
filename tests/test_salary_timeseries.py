"""Tests for salary-window analytics."""

from __future__ import annotations

import pandas as pd

from src.salary_timeseries import build_salary_time_series


def _row(
    date: str,
    particulars: str,
    debit: float,
    credit: float,
    *,
    is_salary: int = 0,
    exp_food: int = 0,
    exp_entertainment: int = 0,
    exp_saving: int = 0,
    exp_investment: int = 0,
    exp_quick_comm: int = 0,
    exp_micro: int = 0,
    exp_sub: int = 0,
    exp_emi: int = 0,
    exp_atm: int = 0,
    exp_creditcard: int = 0,
    exp_ecomm: int = 0,
    exp_luxury: int = 0,
    source_file: str = "",
) -> dict[str, object]:
    """Build one compact test row."""

    return {
        "Date": date,
        "Particulars": particulars,
        "Debit": debit,
        "Credit": credit,
        "IsSalary": is_salary,
        "SalaryConfidence": 1.0 if is_salary else 0.0,
        "Level_6": "acme corp" if is_salary else "",
        "MicroSegment": "",
        "exp_food": exp_food,
        "exp_ecomm": exp_ecomm,
        "exp_entertainment": exp_entertainment,
        "exp_sub": exp_sub,
        "exp_luxury": exp_luxury,
        "exp_micro": exp_micro,
        "exp_quick_comm": exp_quick_comm,
        "exp_emi": exp_emi,
        "exp_atm": exp_atm,
        "exp_creditcard": exp_creditcard,
        "exp_saving": exp_saving,
        "exp_investment": exp_investment,
        "exp_upi_velocity": 0,
        "Source_File": source_file,
    }


def test_salary_window_uses_new_savings_to_expenditure_ratio() -> None:
    """Only configured savings and expenditure buckets should count."""

    df = pd.DataFrame(
        [
            _row("2026-03-01", "SALARY ACME", 0, 50000, is_salary=1),
            _row("2026-03-01", "ZOMATO", 500, 0, exp_food=1),
            _row("2026-03-02", "SIP INVESTMENT", 1200, 0, exp_investment=1),
            _row("2026-03-03", "ELECTRICITY BILL", 1000, 0),
            _row("2026-03-15", "MOVIE", 200, 0, exp_entertainment=1),
        ]
    )

    events_df, txns_df, curve_df = build_salary_time_series(df, window_days=10)

    assert len(events_df) == 1
    assert len(txns_df) == 3
    assert events_df.loc[0, "Savings_Amount"] == 1200.0
    assert events_df.loc[0, "Expenditure_Amount"] == 500.0
    assert events_df.loc[0, "Savings_to_Expenditure_Ratio"] == 2.4

    day_two_ratio = curve_df.loc[
        curve_df["Days_After_Salary"].eq(2),
        "Cumulative_Savings_to_Expenditure_Ratio",
    ].iloc[0]
    assert day_two_ratio == 2.4


def test_salary_windows_are_grouped_by_source_file() -> None:
    """Folder-level analysis should keep files separate."""

    df = pd.DataFrame(
        [
            _row("2026-03-01", "SALARY A", 0, 30000, is_salary=1, source_file="a.xlsx"),
            _row("2026-03-02", "SWIGGY", 300, 0, exp_food=1, source_file="a.xlsx"),
            _row("2026-03-05", "SALARY B", 0, 40000, is_salary=1, source_file="b.xlsx"),
            _row("2026-03-06", "MOVIE", 800, 0, exp_entertainment=1, source_file="b.xlsx"),
        ]
    )

    events_df, txns_df, _ = build_salary_time_series(
        df,
        window_days=10,
        group_columns=("Source_File",),
    )

    assert len(events_df) == 2
    assert sorted(events_df["Expenditure_Amount"].tolist()) == [300.0, 800.0]
    assert sorted(txns_df["Source_File"].unique().tolist()) == ["a.xlsx", "b.xlsx"]
