"""Salary-window analytics and plots for classified statement data."""

from __future__ import annotations

import logging
import re
from pathlib import Path

# pyrefly: ignore[missing-import]
import matplotlib
# pyrefly: ignore[missing-import]
import matplotlib.dates as mdates
import pandas as pd

from .statement_classifier import _level7_parse_date, parse_amount

matplotlib.use("Agg")

# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 10
SAVINGS_FLAG_COLUMNS = ("exp_saving", "exp_investment")
EXPENDITURE_FLAG_COLUMNS = (
    "exp_food",
    "exp_micro",
    "exp_luxury",
    "exp_ecomm",
    "exp_entertainment",
    "exp_quick_comm",
    "exp_sub",
    "exp_emi",
    "exp_atm",
    "exp_creditcard",
)


def _empty_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return empty salary analytics outputs."""

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date formats without breaking ISO dates.

    Text dates are read day-first ("2/3/25" is 2 March 2025), the same way
    statement_classifier parses them; real dates and ISO strings are kept.
    """

    return pd.to_datetime(series.map(_level7_parse_date), errors="coerce")


def _has_any_flag(row: pd.Series, flag_columns: tuple[str, ...]) -> int:
    """Return 1 when any of the supplied flags is active."""

    for column_name in flag_columns:
        if int(row.get(column_name, 0) or 0) == 1:
            return 1
    return 0


SAVINGS_TEXT_RE = re.compile(
    # "FD RL0001520" is a wrapped Federal Bank IFSC (FDRL...), not a deposit.
    r"\b(?:FD(?!\s*RL\d)|FIXED\s*DEPOSIT|RD|RECURRING\s*DEPOSIT|SIP|MUTUAL\s*FUNDS?|PPF|NPS|INVESTMENTS?)\b",
    flags=re.IGNORECASE,
)
RECHARGE_TEXT_RE = re.compile(r"\b(?:RECHARGE|DTH)\b", flags=re.IGNORECASE)
CAB_TEXT_RE = re.compile(r"\b(?:CAB|CABS|OLA|UBER|MERU|RAPIDO)\b", flags=re.IGNORECASE)
CARD_FEE_TEXT_RE = re.compile(r"\bCARD\s+(?:CHARGES?|FEES?)\b", flags=re.IGNORECASE)
BILL_TEXT_RE = re.compile(r"\b(?:BBPS|BILL\s*PAYMENT|BILLPAY)\b", flags=re.IGNORECASE)
# Checked on the narration with spaces and punctuation removed, because
# statements wrap these handles ("GPAY-UTILITY@OK", "GOOGLEBBPSUTILI TY@ICICI").
BILL_HANDLE_COMPACT = ("GPAYUTILITY", "GOOGLEBBPS")
# Failed or rejected rows are never savings, expenditure or transfer.
FAILED_TEXT_RE = re.compile(
    r"\b(?:REJECT\w*|PAYMENT\s+STOPPED|RETURN\w*|REVERS\w*)\b|REJECT:",
    flags=re.IGNORECASE,
)
EXTRA_EXPENDITURE_FLAG_COLUMNS = (
    "exp_utility",
    "exp_transport",
    "exp_grocery",
    "exp_pharmacy",
    "exp_insurance",
)
EXTRA_EXPENDITURE_LEVEL5 = frozenset(
    {
        "utilities",
        "mobile recharge",
        "transport",
        "grocery",
        "pharmacy",
        "health",
        "healthcare",
        "insurance",
        "subscription",
        "dining",
        "food",
    }
)
TRANSFER_LEVEL5 = frozenset({"p2p transfer"})


def _text(row: pd.Series, column_name: str) -> str:
    """Return a cell as stripped text ('' for NaN)."""

    value = row.get(column_name, "")
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()


def _is_failed(row: pd.Series) -> bool:
    """Return True for failed, rejected, returned or reversed rows."""

    return bool(FAILED_TEXT_RE.search(_text(row, "Particulars")))


def _savings_match(row: pd.Series) -> bool:
    """Saving/investment flag, or a savings product named in the narration."""

    if _has_any_flag(row, SAVINGS_FLAG_COLUMNS):
        return True
    return bool(SAVINGS_TEXT_RE.search(_text(row, "Particulars")))


def _expenditure_match(row: pd.Series) -> bool:
    """Spending on merchants, bills, utilities, recharge, transport, food,
    grocery, health, insurance, subscriptions and card fees."""

    if _has_any_flag(row, EXPENDITURE_FLAG_COLUMNS + EXTRA_EXPENDITURE_FLAG_COLUMNS):
        return True
    if _text(row, "Level_3").lower() == "merchant":
        return True
    if _text(row, "Level_5").lower() in EXTRA_EXPENDITURE_LEVEL5:
        return True
    if _text(row, "Level_1").lower() == "bbps":
        return True
    particulars = _text(row, "Particulars")
    compact = re.sub(r"[^A-Z0-9]", "", particulars.upper())
    if any(handle in compact for handle in BILL_HANDLE_COMPACT):
        return True
    return any(
        pattern.search(particulars)
        for pattern in (RECHARGE_TEXT_RE, CAB_TEXT_RE, CARD_FEE_TEXT_RE, BILL_TEXT_RE)
    )


def _transfer_match(row: pd.Series) -> bool:
    """A p2p transfer to a person (UPI, NEFT, RTGS or IMPS)."""

    return (
        _text(row, "Level_3").lower() == "p2p"
        or _text(row, "Level_5").lower() in TRANSFER_LEVEL5
    )


def _window_tag(row: pd.Series) -> str:
    """Return 'savings', 'expenditure', 'transfer' or '' for one row.

    The tags never overlap: Savings, then Expenditure, then Transfer.
    Failed or rejected rows get no tag.
    """

    if _is_failed(row):
        return ""
    if _savings_match(row):
        return "savings"
    if _expenditure_match(row):
        return "expenditure"
    if _transfer_match(row):
        return "transfer"
    return ""


def _is_savings(row: pd.Series) -> int:
    """Return 1 when a debit is a saving or investment."""

    return int(_window_tag(row) == "savings")


def _is_expenditure(row: pd.Series) -> int:
    """Return 1 when a debit is spending (never savings)."""

    return int(_window_tag(row) == "expenditure")


def _is_transfer(row: pd.Series) -> int:
    """Return 1 when a debit is a p2p transfer (never savings or spending)."""

    return int(_window_tag(row) == "transfer")


def _build_daily_curve(
    window_transactions_df: pd.DataFrame,
    salary_events_df: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """Aggregate savings and expenditure by day offset from salary."""

    base_df = pd.DataFrame({"Days_After_Salary": list(range(window_days))})
    if salary_events_df.empty:
        return base_df

    if window_transactions_df.empty:
        base_df["Savings_Amount"] = 0.0
        base_df["Expenditure_Amount"] = 0.0
        base_df["Savings_Txn_Count"] = 0
        base_df["Expenditure_Txn_Count"] = 0
    else:
        grouped = (
            window_transactions_df.groupby("Days_After_Salary", as_index=False)
            .agg(
                Savings_Amount=("Savings_Debit", "sum"),
                Expenditure_Amount=("Expenditure_Debit", "sum"),
                Savings_Txn_Count=("Is_Savings", "sum"),
                Expenditure_Txn_Count=("Is_Expenditure", "sum"),
            )
            .sort_values("Days_After_Salary")
        )
        base_df = base_df.merge(grouped, on="Days_After_Salary", how="left")
        base_df["Savings_Amount"] = base_df["Savings_Amount"].fillna(0.0)
        base_df["Expenditure_Amount"] = base_df["Expenditure_Amount"].fillna(0.0)
        base_df["Savings_Txn_Count"] = base_df["Savings_Txn_Count"].fillna(0).astype(int)
        base_df["Expenditure_Txn_Count"] = base_df["Expenditure_Txn_Count"].fillna(0).astype(int)

    window_count = int(len(salary_events_df))
    base_df["Cumulative_Savings_Amount"] = base_df["Savings_Amount"].cumsum()
    base_df["Cumulative_Expenditure_Amount"] = base_df["Expenditure_Amount"].cumsum()
    base_df["Average_Savings_Per_Window"] = (
        base_df["Savings_Amount"] / window_count if window_count else 0.0
    )
    base_df["Average_Expenditure_Per_Window"] = (
        base_df["Expenditure_Amount"] / window_count if window_count else 0.0
    )
    base_df["Cumulative_Savings_to_Expenditure_Ratio"] = base_df.apply(
        lambda row: round(
            row["Cumulative_Savings_Amount"] / row["Cumulative_Expenditure_Amount"],
            4,
        )
        if row["Cumulative_Expenditure_Amount"] > 0
        else 0.0,
        axis=1,
    )
    return base_df


def _analyze_one_group(
    df: pd.DataFrame,
    window_days: int,
    group_values: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute salary-window savings and expenditure metrics for one slice."""

    if "Date" not in df.columns:
        raise KeyError("Salary time-series analysis requires a 'Date' column.")

    working_df = df.copy()
    working_df["_date"] = _parse_dates(working_df["Date"])
    working_df["debit_num"] = working_df["Debit"].apply(parse_amount)
    working_df["credit_num"] = working_df["Credit"].apply(parse_amount)
    window_tags = working_df.apply(_window_tag, axis=1)
    working_df["Is_Savings"] = window_tags.eq("savings").astype(int)
    working_df["Is_Expenditure"] = window_tags.eq("expenditure").astype(int)
    working_df["Is_Transfer"] = window_tags.eq("transfer").astype(int)

    is_salary_series = working_df.get("IsSalary", pd.Series(0, index=working_df.index))
    salary_df = working_df[
        is_salary_series.astype(int).eq(1)
        & working_df["_date"].notna()
        & working_df["credit_num"].gt(0)
    ].sort_values("_date")

    if salary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_values = group_values or {}
    event_rows: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []

    salary_records = list(salary_df.iterrows())
    for event_number, (salary_index, salary_row) in enumerate(salary_records, start=1):
        salary_date = salary_row["_date"].normalize()
        natural_window_end = salary_date + pd.Timedelta(days=window_days - 1)
        next_salary_date = pd.NaT
        if event_number < len(salary_records):
            next_salary_date = salary_records[event_number][1]["_date"].normalize()

        window_end = natural_window_end
        if pd.notna(next_salary_date):
            clipped_end = next_salary_date - pd.Timedelta(days=1)
            if clipped_end < window_end:
                window_end = clipped_end

        spend_mask = (
            working_df["_date"].dt.normalize().between(salary_date, window_end)
            & working_df["debit_num"].gt(0)
            & (working_df.index != salary_index)
        )
        window_df = working_df.loc[spend_mask].copy()
        window_df["Salary_Event_Id"] = event_number
        window_df["Salary_Date"] = salary_date
        window_df["Window_End_Date"] = window_end
        window_df["Salary_Amount"] = float(salary_row["credit_num"])
        window_df["Days_After_Salary"] = (
            window_df["_date"].dt.normalize().sub(salary_date).dt.days.astype(int)
        )
        window_df["Savings_Debit"] = (
            window_df["debit_num"] * window_df["Is_Savings"].astype(int)
        )
        window_df["Expenditure_Debit"] = (
            window_df["debit_num"] * window_df["Is_Expenditure"].astype(int)
        )
        window_df["Transfer_Debit"] = (
            window_df["debit_num"] * window_df["Is_Transfer"].astype(int)
        )

        savings_amount = float(window_df["Savings_Debit"].sum())
        expenditure_amount = float(window_df["Expenditure_Debit"].sum())
        transfer_amount = float(window_df["Transfer_Debit"].sum())

        event_row = {
            **group_values,
            "Salary_Event_Id": event_number,
            "Salary_Date": salary_date,
            "Window_End_Date": window_end,
            "Window_Days": window_days,
            "Salary_Amount": float(salary_row["credit_num"]),
            "Savings_Transactions": int(window_df["Is_Savings"].sum()),
            "Expenditure_Transactions": int(window_df["Is_Expenditure"].sum()),
            "Transfer_Transactions": int(window_df["Is_Transfer"].sum()),
            "Savings_Amount": round(savings_amount, 2),
            "Expenditure_Amount": round(expenditure_amount, 2),
            "Transfer_Amount": round(transfer_amount, 2),
            # Savings vs spending only; transfers are not in the ratio.
            "Savings_to_Expenditure_Ratio": round(savings_amount / expenditure_amount, 4)
            if expenditure_amount
            else 0.0,
            "Salary_Confidence": salary_row.get("SalaryConfidence", 0.0),
            "Salary_Entity": salary_row.get("Level_6", ""),
            "Salary_Particulars": salary_row.get("Particulars", ""),
        }
        event_rows.append(event_row)

        if not window_df.empty:
            for column_name, column_value in group_values.items():
                window_df[column_name] = column_value
            window_frames.append(window_df)

    salary_events_df = pd.DataFrame(event_rows)
    if not window_frames:
        return salary_events_df, pd.DataFrame()

    window_transactions_df = pd.concat(window_frames, ignore_index=True)
    ordered_columns = [
        *group_values.keys(),
        "Salary_Event_Id",
        "Salary_Date",
        "Window_End_Date",
        "Days_After_Salary",
        "Salary_Amount",
        "Is_Savings",
        "Is_Expenditure",
        "Is_Transfer",
        "Savings_Debit",
        "Expenditure_Debit",
        "Transfer_Debit",
    ]
    existing_prefixed = [column for column in ordered_columns if column in window_transactions_df.columns]
    remaining_columns = [
        column for column in window_transactions_df.columns if column not in existing_prefixed
    ]
    window_transactions_df = window_transactions_df[existing_prefixed + remaining_columns]
    return salary_events_df, window_transactions_df


def build_salary_time_series(
    df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    group_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return salary-window summary, transactions, and daily-curve tables."""

    if window_days <= 0:
        raise ValueError("window_days must be a positive integer.")

    if df.empty:
        return _empty_outputs()

    if group_columns:
        event_frames: list[pd.DataFrame] = []
        window_frames: list[pd.DataFrame] = []
        for group_key, group_df in df.groupby(list(group_columns), dropna=False, sort=True):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = {
                column_name: value for column_name, value in zip(group_columns, group_key)
            }
            group_events_df, group_window_df = _analyze_one_group(
                group_df,
                window_days=window_days,
                group_values=group_values,
            )
            if not group_events_df.empty:
                event_frames.append(group_events_df)
            if not group_window_df.empty:
                window_frames.append(group_window_df)

        if not event_frames:
            return _empty_outputs()

        salary_events_df = pd.concat(event_frames, ignore_index=True)
        window_transactions_df = (
            pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
        )
    else:
        salary_events_df, window_transactions_df = _analyze_one_group(
            df,
            window_days=window_days,
        )
        if salary_events_df.empty:
            return _empty_outputs()

    daily_curve_df = _build_daily_curve(
        window_transactions_df=window_transactions_df,
        salary_events_df=salary_events_df,
        window_days=window_days,
    )
    logger.info(
        "Built salary time-series analytics for %s salary events",
        len(salary_events_df),
    )
    return (
        salary_events_df.reset_index(drop=True),
        window_transactions_df.reset_index(drop=True),
        daily_curve_df.reset_index(drop=True),
    )


def save_salary_plots(
    salary_events_df: pd.DataFrame,
    daily_curve_df: pd.DataFrame,
    output_path: Path,
) -> list[Path]:
    """Save savings-versus-expenditure plots next to the main output file."""

    if salary_events_df.empty or daily_curve_df.empty:
        return []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_spend_path = output_path.with_name(
        f"{output_path.stem}_salary_savings_expenditure_daily.png"
    )
    ratio_curve_path = output_path.with_name(
        f"{output_path.stem}_salary_savings_expenditure_ratio.png"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        daily_curve_df["Days_After_Salary"] - 0.2,
        daily_curve_df["Savings_Amount"],
        width=0.4,
        label="Savings",
        color="#1f77b4",
    )
    ax.bar(
        daily_curve_df["Days_After_Salary"] + 0.2,
        daily_curve_df["Expenditure_Amount"],
        width=0.4,
        label="Expenditure",
        color="#ff7f0e",
    )
    ax.set_xlabel("Days after salary credit")
    ax.set_ylabel("Amount")
    ax.set_title("Daily savings and expenditure after salary credit")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(daily_spend_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        daily_curve_df["Days_After_Salary"],
        daily_curve_df["Cumulative_Savings_to_Expenditure_Ratio"],
        marker="o",
        label="Savings / Expenditure",
        color="#1f77b4",
    )
    ax.set_xlabel("Days after salary credit")
    ax.set_ylabel("Cumulative ratio")
    ax.set_title("Cumulative savings-to-expenditure ratio")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ratio_curve_path, dpi=160)
    plt.close(fig)

    return [daily_spend_path, ratio_curve_path]


def _sanitize_file_stem(value: object) -> str:
    """Return a filesystem-safe file stem."""

    text = str(value or "").strip()
    text = re.sub(r"[^\w.-]+", "_", text)
    text = text.strip("._")
    return text or "statement"


def save_salary_timeline_plots(
    df: pd.DataFrame,
    salary_events_df: pd.DataFrame,
    output_dir: Path,
    *,
    group_columns: tuple[str, ...] | None = None,
    stem_prefix: str = "",
) -> list[Path]:
    """Save one highlighted savings/expenditure timeline plot per group."""

    if df.empty or salary_events_df.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    working_df = df.copy()
    working_df["_date"] = _parse_dates(working_df["Date"])
    working_df["debit_num"] = working_df["Debit"].apply(parse_amount)
    working_df["credit_num"] = working_df["Credit"].apply(parse_amount)
    working_df["Is_Savings"] = working_df.apply(_is_savings, axis=1)
    working_df["Is_Expenditure"] = working_df.apply(_is_expenditure, axis=1)
    working_df["Savings_Debit"] = working_df["debit_num"] * working_df["Is_Savings"].astype(int)
    working_df["Expenditure_Debit"] = (
        working_df["debit_num"] * working_df["Is_Expenditure"].astype(int)
    )
    working_df["Salary_Credit"] = working_df["credit_num"] * working_df.get(
        "IsSalary", pd.Series(0, index=working_df.index)
    ).astype(int)

    valid_df = working_df[working_df["_date"].notna()].copy()
    valid_salary_events_df = salary_events_df.copy()
    if "Salary_Date" in valid_salary_events_df.columns:
        valid_salary_events_df["Salary_Date"] = _parse_dates(valid_salary_events_df["Salary_Date"])
    if "Window_End_Date" in valid_salary_events_df.columns:
        valid_salary_events_df["Window_End_Date"] = _parse_dates(
            valid_salary_events_df["Window_End_Date"]
        )

    if group_columns:
        group_keys = list(group_columns)
        grouped = valid_df.groupby(group_keys, dropna=False, sort=True)
    else:
        grouped = [(("statement",), valid_df)]
        group_keys = []

    plot_paths: list[Path] = []
    for group_key, group_df in grouped:
        if group_df.empty:
            continue

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        if group_keys:
            event_mask = pd.Series(True, index=valid_salary_events_df.index)
            for column_name, value in zip(group_keys, group_key):
                event_mask &= valid_salary_events_df[column_name].eq(value)
            group_events_df = valid_salary_events_df.loc[event_mask].copy()
            stem_value = group_key[0]
        else:
            group_events_df = valid_salary_events_df.copy()
            stem_value = stem_prefix or "statement"

        if group_events_df.empty:
            continue

        daily_df = (
            group_df.assign(Date=group_df["_date"].dt.normalize())
            .groupby("Date", as_index=False)
            .agg(
                Savings_Amount=("Savings_Debit", "sum"),
                Expenditure_Amount=("Expenditure_Debit", "sum"),
                Salary_Credit=("Salary_Credit", "sum"),
            )
            .sort_values("Date")
        )

        plot_stem = _sanitize_file_stem(stem_value)
        if stem_prefix and plot_stem != _sanitize_file_stem(stem_prefix):
            plot_stem = f"{_sanitize_file_stem(stem_prefix)}_{plot_stem}"
        plot_path = output_dir / f"{plot_stem}_salary_timeline.png"

        daily_df["Net_Savings"] = daily_df["Savings_Amount"] - daily_df["Expenditure_Amount"]
        daily_df["Cumulative_Savings"] = daily_df["Savings_Amount"].cumsum()
        daily_df["Cumulative_Expenditure"] = daily_df["Expenditure_Amount"].cumsum()
        daily_df["Cumulative_Ratio"] = daily_df.apply(
            lambda row: row["Cumulative_Savings"] / row["Cumulative_Expenditure"]
            if row["Cumulative_Expenditure"] > 0
            else 0.0,
            axis=1,
        )

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.2]},
        )
        for _, event_row in group_events_df.sort_values("Salary_Date").iterrows():
            salary_date = event_row["Salary_Date"]
            window_end = event_row["Window_End_Date"]
            if pd.notna(salary_date) and pd.notna(window_end):
                ax_top.axvspan(
                    salary_date,
                    window_end + pd.Timedelta(days=1),
                    color="#ffd166",
                    alpha=0.18,
                    lw=0,
                )
                ax_bottom.axvspan(
                    salary_date,
                    window_end + pd.Timedelta(days=1),
                    color="#ffd166",
                    alpha=0.12,
                    lw=0,
                )

        ax_top.bar(
            daily_df["Date"],
            daily_df["Expenditure_Amount"],
            width=0.86,
            color="#d1495b",
            alpha=0.82,
            label="Expenditure",
        )
        ax_top.bar(
            daily_df["Date"],
            daily_df["Savings_Amount"],
            width=0.52,
            color="#2e8b57",
            alpha=0.88,
            label="Savings",
        )
        ax_top.plot(
            daily_df["Date"],
            daily_df["Salary_Credit"],
            color="#2b6cb0",
            marker="o",
            linewidth=2.2,
            label="Salary credit",
        )

        ax_top.fill_between(
            daily_df["Date"],
            0,
            daily_df["Net_Savings"],
            where=daily_df["Net_Savings"] >= 0,
            color="#66c2a5",
            alpha=0.18,
            interpolate=True,
            label="Positive net",
        )
        ax_top.fill_between(
            daily_df["Date"],
            0,
            daily_df["Net_Savings"],
            where=daily_df["Net_Savings"] < 0,
            color="#ef476f",
            alpha=0.14,
            interpolate=True,
            label="Negative net",
        )

        ax_bottom.plot(
            daily_df["Date"],
            daily_df["Cumulative_Savings"],
            color="#2e8b57",
            linewidth=2.2,
            label="Cumulative savings",
        )
        ax_bottom.plot(
            daily_df["Date"],
            daily_df["Cumulative_Expenditure"],
            color="#d1495b",
            linewidth=2.2,
            label="Cumulative expenditure",
        )
        ax_bottom_ratio = ax_bottom.twinx()
        ax_bottom_ratio.plot(
            daily_df["Date"],
            daily_df["Cumulative_Ratio"],
            color="#4c78a8",
            linewidth=1.8,
            linestyle="--",
            label="Savings / expenditure ratio",
        )

        salary_total = float(group_events_df["Salary_Amount"].sum())
        savings_total = float(daily_df["Savings_Amount"].sum())
        expenditure_total = float(daily_df["Expenditure_Amount"].sum())
        ratio_value = savings_total / expenditure_total if expenditure_total else 0.0

        ax_top.set_title(f"Savings vs expenditure timeline: {stem_value}")
        ax_top.set_ylabel("Daily amount")
        ax_top.grid(axis="y", alpha=0.22)
        ax_top.text(
            0.01,
            0.98,
            "\n".join(
                [
                    f"Salary credits: {len(group_events_df)}",
                    f"Salary total: {salary_total:,.0f}",
                    f"Savings total: {savings_total:,.0f}",
                    f"Expenditure total: {expenditure_total:,.0f}",
                    f"S/E ratio: {ratio_value:.2f}",
                ]
            ),
            transform=ax_top.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#dddddd"},
        )

        ax_bottom.set_xlabel("Date")
        ax_bottom.set_ylabel("Cumulative amount")
        ax_bottom_ratio.set_ylabel("Cumulative ratio")
        ax_bottom.grid(axis="y", alpha=0.18)

        top_handles, top_labels = ax_top.get_legend_handles_labels()
        bottom_handles, bottom_labels = ax_bottom.get_legend_handles_labels()
        ratio_handles, ratio_labels = ax_bottom_ratio.get_legend_handles_labels()
        ax_top.legend(top_handles[:3], top_labels[:3], loc="upper right", frameon=False)
        ax_bottom.legend(
            bottom_handles + ratio_handles,
            bottom_labels + ratio_labels,
            loc="upper left",
            frameon=False,
        )

        for axis in (ax_top, ax_bottom):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        ax_bottom_ratio.spines["top"].set_visible(False)

        ax_bottom.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=170)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths