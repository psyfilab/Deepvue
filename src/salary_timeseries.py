"""Salary-window analytics and plots for classified statement data."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

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
# Several salary credits on one day are one pay day: one window, salary = their total. Set False to keep
# one event per credit (their windows are then identical, so summing events double counts spending).
SALARY_EVENT_MERGE_SAME_DAY = True
# Microsegmentation decides the window category first, so every sheet tells the same story.
SEGMENT_WINDOW_TAG = {
    "Own Account Transfer": "transfer",  # money moved between the holder's own accounts or wallets
    "P2P Transfer": "transfer",
}
# Used only when no narration rule decided: debt service, tax, charges and fees are money spent.
SEGMENT_WINDOW_FALLBACK = {
    "Loan": "expenditure", "EMI": "expenditure", "Credit Card Payment": "expenditure", "Tax": "expenditure",
    "Penalty / Charges": "expenditure", "Education": "expenditure",
}
# Columns shown on the Salary Window Transactions sheet (the full row stays on the Classified sheet).
WINDOW_TRANSACTION_COLUMNS = (
    "Salary_Event_Id", "Salary_Date", "Window_End_Date", "Days_After_Salary", "Salary_Amount",
    "Date", "Particulars", "Debit", "First Level Classification", "Second Level Classification",
    "Level_3", "Level_5", "Level_6", "Microsegmentation", "Window_Category", "Classified_Amount",
)
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


# ---------------------------------------------------------------------------------------------
# Narration reading shared with microsegment.py: payee name, noise stripping, whole-word matching.
# ---------------------------------------------------------------------------------------------
# Where the payee name sits for each narration layout (first match wins). The UPI handle, bank code,
# reference and purpose tail that follow the name are never part of the payee.
PAYEE_LAYOUTS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:TO|BY) TRANSFER-\s*UPI/(?:DR|CR|P2A|P2M)/\d+/(?P<name>[^/]+)/",
        r"^UPI/(?:DR|CR|P2A|P2M)/\d+/(?P<name>[^/]+)(?:/|$)",
        r"^UPI-(?P<name>[^-]+)-",
        r"^IMPS-\d+-(?P<name>[^-]+)-",
        r"^(?:(?:BY|TO) )?(?:NEFT-|TRANSFER-)?\s*NEFT/(?:CR|DR)/[^/]+/(?P<name>[^/]+)/",
        r"^(?:(?:BY|TO) )?NEFT(?: CR| DR)?-[A-Z0-9]{6,}-(?P<name>[^-]+)(?:-|$)",
        r"^RTGS(?: CR| DR)?-[A-Z0-9]{6,}-(?P<name>[^-]+)(?:-|$)",
        r"^RTGS:[A-Z0-9]+/[^/]+/(?P<name>.+)$",
        r"^ACH [DC]-\s*(?P<name>[^-]+)-",
        r"^(?:MOBFT|TRFR) FROM:\s*(?P<name>[^/]+)",
        r"^INF/INFT/\d+/(?P<name>.+)$",
        r"^(?:POS|ECOM)\s+\S+\s+(?P<name>.+)$",
        r"^\d{6,}-TPT-(?P<name>.+)$",
    )
)
BANK_NAME_RE = re.compile(
    r"\b(?:indian overseas|state bank(?: of [a-z]+)?|bank of [a-z]+|[a-z]+ bank(?: ltd| limited)?|"
    r"hdfc|icici|sbi|kotak|idfc|pnb|canara|bandhan|indusind|rbl|iob|boi)\b",
    re.IGNORECASE,
)
_VPA_RE = re.compile(r"\S+@\S+")
_IFSC_RE = re.compile(r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b")
_WRAPPED_IFSC_RE = re.compile(r"\b[A-Za-z]{2,4}\s?[A-Za-z]{0,2}\s?RL\s?\d{5,}\b|\bFD\s*RL\d+\b", re.IGNORECASE)
_LONG_DIGITS_RE = re.compile(r"\b\w*\d{4,}\w*\b")
_MASKED_RE = re.compile(r"\b\w*x{3,}\w*\b", re.IGNORECASE)
_PURPOSE_TAIL_RE = re.compile(r"/\s*([A-Za-z]{2,8})\s*-\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _squash(text: object) -> str:
    return _NON_ALNUM_RE.sub(" ", str(text or "").lower()).strip()


def word_variants(text: object) -> str:
    """Lower-case words with spaces on both sides, plus joins of a wrapped word ("bigbaske t" -> "bigbasket").

    Matching ` word ` against this string is whole-word matching.
    """

    tokens = _squash(text).split()
    joined = [a + b for a, b in zip(tokens, tokens[1:]) if min(len(a), len(b)) <= 2]
    return " " + " ".join(tokens + joined) + " "


def strip_noise(text: object) -> str:
    """Remove UPI handles, IFSC codes, masked and numeric references and bank names."""

    value = str(text or "")
    for pattern in (_VPA_RE, _WRAPPED_IFSC_RE, _IFSC_RE, _MASKED_RE, _LONG_DIGITS_RE, BANK_NAME_RE):
        value = pattern.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


class PayeeInfo(NamedTuple):
    name: str  # payee as written in the layout slot ("" when the layout has none)
    text: str  # what keyword rules may read: payee (or the cleaned narration when no layout) + Level 6
    is_bank: bool  # the payee slot holds a bank name


def payee_info(particulars: object, level_6: object = "") -> PayeeInfo:
    """Payee name part of a narration, without handle, bank, reference or purpose tail."""

    narration = str(particulars or "").strip()
    name = None
    for layout in PAYEE_LAYOUTS:
        match = layout.search(narration)
        if match:
            name = match.group("name").strip()
            break
    if name is not None:
        is_bank = bool(BANK_NAME_RE.search(name))
        base = strip_noise(name)
    else:
        is_bank = False
        base = strip_noise(narration)
    extra = strip_noise(level_6) if not is_bank else ""
    return PayeeInfo(name or "", f"{base} {extra}".strip(), is_bank)


def purpose_tail(particulars: object) -> str:
    """The purpose tag a UPI narration ends with ("Groc", "Trvl", "Wall"...), lower case, or ''."""

    match = _PURPOSE_TAIL_RE.search(str(particulars or "").strip())
    return match.group(1).lower() if match else ""


def purpose_text(particulars: object) -> str:
    """The last wording segment of a narration ("... -HAND LOAN" -> "HAND LOAN")."""

    for part in reversed(re.split(r"[-/]", str(particulars or ""))):
        if re.search(r"[A-Za-z]{2}", part) and not _LONG_DIGITS_RE.fullmatch(part.strip()):
            return part.strip()
    return ""


# ---------------------------------------------------------------------------------------------
# Salary rules (table driven). refine_salary() runs before micro-segmentation so that Salary Flag,
# Level 5 and Microsegmentation always agree.
# ---------------------------------------------------------------------------------------------
SALARY_WORDS = re.compile(r" (?:salary|salaries|payroll|payslip|wages) ")
SALARY_MIN_INFERRED = 10_000  # inferred (repeating payer) salary credits below this are not salary
SALARY_MAX_WITHOUT_WORD = 500_000  # larger credits are salary only when the narration says SALARY
SALARY_AMOUNT_TOLERANCE = 0.10
SALARY_MIN_MONTHS = 3
SALARY_DAY_STD_MAX = 7
SALARY_CONFIDENCE_EXPLICIT = 1.0
SALARY_CONFIDENCE_REPEATING = 0.8
SALARY_CONFIDENCE_FLOOR = 0.6  # below this a row is not salary
NEVER_SALARY_RE = re.compile(
    r"\bIB\s*FD\s*PREMAT|\bFD\s*(?:PREMAT|MATUR|CLOS|REDEMPTION)|\bTPT\b|\bOWN\s*ACC|\bINTERNAL\b|\bSELF\b|"
    r"\bREFUND|\bREVERS|\bCASHBACK|\bRETURN|\bREJECT|^\s*DRP/|^\s*CLG/|\bCHQ\b|\bCHEQUE\b|\bCLEARING\b",
    re.IGNORECASE,
)
NEVER_SALARY_LEVEL_5 = frozenset({
    "interest", "dividend", "charges", "wallet load", "wallet unload", "refund", "cashback", "loan disbursement",
})
# Payers that move money for others (wallets, gateways, payment banks) are never an employer.
PAYMENT_PLATFORM_PAYER_WORDS = (
    "paytm", "one97", "phonepe", "razorpay", "payu", "cashfree", "billdesk", "mobikwik", "freecharge",
    "bharatpe", "juspay", "gpay", "google pay", "amazon pay", "wallet", "payments", "payment", "pay",
)
FIRST_LEVEL_NEVER_ON_DEBIT = frozenset({"salary"})
SALARY_WORD_MASK_RE = re.compile(r"\b(?:SA\s?LARY|SALARIES|PAYROLL|PAYSLIP|WAGES)\b", re.IGNORECASE)
COMPANY_WORDS = re.compile(r" (?:pvt|private|ltd|limited|llp|india|services|technologies) ")
PAYER_KEY_LENGTH = 8  # statements truncate names at different lengths; compare the first characters
PAYER_TITLE_RE = re.compile(r"^(?:mr|mrs|ms|miss|shri|smt|sri|dr|m s)(?=[a-z ])", re.IGNORECASE)
SALARY_MIN_ONE_PER_MONTH = 0.8  # share of months in which the payer paid exactly once (salary comes monthly)
SALARY_MAX_REPEATING_PAYERS = 3  # more repeating payers than this is a business / receipts-heavy account


def _level(row: pd.Series, name: str) -> str:
    return str(row.get(name, "") or "").strip().lower()


def _payer_key(info: PayeeInfo) -> str:
    name = PAYER_TITLE_RE.sub("", (info.name or info.text).strip().lower()).strip()
    return re.sub(r"[^a-z0-9]", "", name)[:PAYER_KEY_LENGTH]


def infer_account_holder(df: pd.DataFrame) -> str:
    """The account holder's name, read from the beneficiary slot of NEFT credit narrations.

    "NEFT CR-IFSC-PAYER-BENEFICIARY-REF": the beneficiary is the same person on every row.
    Returns a compact lower-case key, or '' when the statement has no such rows.
    """

    counts: dict[str, int] = defaultdict(int)
    for text in df["Particulars"].astype(str):
        parts = text.split("-")
        if len(parts) >= 5 and re.match(r"\s*NEFT CR\s*$", parts[0], re.IGNORECASE):
            key = re.sub(r"[^a-z0-9]", "", parts[3].lower())
            if len(key) >= 6:
                counts[key] += 1
    repeated = [(count, key) for key, count in counts.items() if count >= 2]
    return max(repeated)[1] if repeated else ""


def _is_account_holder(payer_compact: str, holder: str) -> bool:
    if not holder or len(payer_compact) < 6:
        return False
    return payer_compact.startswith(holder) or holder.startswith(payer_compact)


def _rederive_levels(frame: pd.DataFrame, mask: pd.Series, *, blank_second_level: pd.Series) -> None:
    """Recompute First / Second Level for the masked rows with the classifier's own derivations."""

    if not mask.any():
        return
    try:
        from . import statement_classifier as classifier

        first_level = classifier.derive_first_level_classification
        second_level = classifier.derive_second_level_classification
    except (ImportError, AttributeError):
        return
    rows = frame.loc[mask]
    frame.loc[mask, "First Level Classification"] = rows.apply(first_level, axis=1)
    rows = frame.loc[mask]
    frame.loc[mask, "Second Level Classification"] = rows.apply(second_level, axis=1)
    frame.loc[blank_second_level, "Second Level Classification"] = ""


def refine_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute IsSalary / SalaryConfidence and bring Level 3, 5 and First / Second Level in line.

    A credit is salary when the narration has an explicit salary word (1.0) or the same payer sends a
    similar amount in at least SALARY_MIN_MONTHS different months on a steady day of month (0.8).
    A debit is never salary. Rows that are not salary lose any salary label.
    """

    out = df.copy()
    for column, default in (("IsSalary", 0), ("SalaryConfidence", 0.0), ("Level_3", ""), ("Level_5", ""),
                            ("Level_6", ""), ("Level_2", ""), ("First Level Classification", ""),
                            ("Second Level Classification", "")):
        if column not in out.columns:
            out[column] = default
    credit = out["Credit"].apply(parse_amount)
    debit = out["Debit"].apply(parse_amount) if "Debit" in out.columns else pd.Series(0.0, index=out.index)
    credit_only = credit > 0  # a debit is never salary
    particulars = out["Particulars"].fillna("").astype(str)
    level_6 = out["Level_6"].fillna("").astype(str)
    infos = [payee_info(text, l6) for text, l6 in zip(particulars, level_6)]
    keys = pd.Series([_payer_key(info) for info in infos], index=out.index)
    level_5 = out["Level_5"].fillna("").astype(str).str.strip().str.lower()

    holder = infer_account_holder(out)
    never = pd.Series(
        [bool(NEVER_SALARY_RE.search(text)) or level in NEVER_SALARY_LEVEL_5
         or _is_account_holder(re.sub(r"[^a-z0-9]", "", (info.name or "").lower()), holder)
         for text, level, info in zip(particulars, level_5, infos)],
        index=out.index,
    )
    platform_re = re.compile(r" (?:" + "|".join(re.escape(w) for w in PAYMENT_PLATFORM_PAYER_WORDS) + r") ")
    never |= pd.Series([bool(platform_re.search(word_variants(info.name or info.text))) for info in infos],
                       index=out.index)
    has_word = pd.Series([bool(SALARY_WORDS.search(word_variants(text))) for text in particulars], index=out.index)

    salary = pd.Series(False, index=out.index)
    confidence = pd.Series(0.0, index=out.index)
    explicit = credit_only & has_word  # an explicit salary word on a credit is always salary
    salary[explicit] = True
    confidence[explicit] = SALARY_CONFIDENCE_EXPLICIT

    # Repeating payer: same payer, amounts within the tolerance, enough different months, steady day.
    debit_keys = {key for key, amount in zip(keys, debit) if amount > 0 and key}
    dates = _parse_dates(out["Date"]) if "Date" in out.columns else pd.Series(pd.NaT, index=out.index)
    candidates = out.index[
        credit_only & ~has_word & ~never & dates.notna()
        & (credit >= SALARY_MIN_INFERRED) & (credit <= SALARY_MAX_WITHOUT_WORD)
        & keys.ne("") & ~keys.isin(debit_keys)
    ]
    by_payer: dict[str, list[int]] = defaultdict(list)
    for index in candidates:
        by_payer[keys[index]].append(index)
    accepted: list[list[int]] = []
    for indices in by_payer.values():
        cluster: list[int] = []
        for index in sorted(indices, key=lambda i: credit[i]):
            if cluster and credit[index] > credit[cluster[0]] * (1 + SALARY_AMOUNT_TOLERANCE):
                if _is_salary_cluster(cluster, dates):
                    accepted.append(cluster)
                cluster = []
            cluster.append(index)
        if _is_salary_cluster(cluster, dates):
            accepted.append(cluster)
    if len(accepted) <= SALARY_MAX_REPEATING_PAYERS:  # many repeating payers = business account, not a payroll
        for cluster in accepted:
            salary.loc[cluster] = True
            confidence.loc[cluster] = SALARY_CONFIDENCE_REPEATING

    # Level 3 / Level 5 / First & Second Level follow the decision.
    previous_label = level_5.eq("salary") | out["First Level Classification"].astype(str).str.lower().eq("salary")
    out["IsSalary"] = salary.astype(int)
    out["SalaryConfidence"] = confidence.where(salary, 0.0)
    cleared = previous_label & ~salary
    out.loc[level_5.eq("salary") & ~salary, "Level_5"] = ""
    became = salary & ~level_5.eq("salary")
    if "Level_0" in out.columns:
        out.loc[became, "Level_0"] = "credit"
    out.loc[salary, "Level_5"] = "salary"
    blank_entity = salary & out["Level_6"].fillna("").astype(str).str.strip().eq("")
    entity = pd.Series([info.text.lower() for info in infos], index=out.index)
    out.loc[blank_entity, "Level_6"] = entity[blank_entity]
    company = pd.Series(
        [bool(COMPANY_WORDS.search(word_variants(info.name or info.text))) for info in infos], index=out.index
    ) & salary
    out.loc[company, "Level_3"] = "institution"
    out.loc[company, "Level_2"] = "User-generated"
    _rederive_levels(out, cleared | salary, blank_second_level=salary)
    # A debit never carries First Level "Salary": derive it again without the salary word.
    debit_label = debit.gt(0) & out["First Level Classification"].astype(str).str.strip().str.lower().isin(
        FIRST_LEVEL_NEVER_ON_DEBIT
    )
    if debit_label.any():
        masked = out.loc[debit_label].copy()
        masked["Particulars"] = masked["Particulars"].astype(str).map(lambda t: SALARY_WORD_MASK_RE.sub(" ", t))
        _rederive_levels(masked, pd.Series(True, index=masked.index), blank_second_level=pd.Series(False, index=masked.index))
        out.loc[debit_label, ["First Level Classification", "Second Level Classification"]] = masked[
            ["First Level Classification", "Second Level Classification"]
        ]
    return out


def _is_salary_cluster(cluster: list[int], dates: pd.Series) -> bool:
    """Same payer, similar amount: salary when it recurs monthly, once a month, on a steady day."""

    if len(cluster) < SALARY_MIN_MONTHS:
        return False
    months = dates.loc[cluster].dt.to_period("M")
    if months.nunique() < SALARY_MIN_MONTHS:
        return False
    per_month = months.value_counts()
    if (per_month == 1).sum() / len(per_month) < SALARY_MIN_ONE_PER_MONTH:
        return False
    day_std = dates.loc[cluster].dt.day.std(ddof=0)
    return bool(pd.isna(day_std) or day_std <= SALARY_DAY_STD_MAX)


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
    if SEGMENT_WINDOW_TAG.get(_text(row, "MicroSegment") or _text(row, "Microsegmentation")) == "transfer":
        return "transfer"
    if _savings_match(row):
        return "savings"
    if _expenditure_match(row):
        return "expenditure"
    if _transfer_match(row):
        return "transfer"
    return SEGMENT_WINDOW_FALLBACK.get(_text(row, "MicroSegment") or _text(row, "Microsegmentation"), "")


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
    # A debit with no confident purpose stays visible as unclassified instead of being forced into a bucket.
    failed = working_df.apply(_is_failed, axis=1)
    window_tags = window_tags.mask(window_tags.eq("") & working_df["debit_num"].gt(0) & ~failed, "unclassified")
    working_df["Window_Category"] = window_tags.map(
        {"savings": "Savings", "expenditure": "Expenditure", "transfer": "Transfer", "unclassified": "Unclassified"}
    ).fillna("")
    working_df["Is_Savings"] = window_tags.eq("savings").astype(int)
    working_df["Is_Expenditure"] = window_tags.eq("expenditure").astype(int)
    working_df["Is_Transfer"] = window_tags.eq("transfer").astype(int)
    working_df["Is_Unclassified"] = window_tags.eq("unclassified").astype(int)

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
    salary_days = sorted({record["_date"].normalize() for _, record in salary_records})
    if SALARY_EVENT_MERGE_SAME_DAY:
        by_day: dict[pd.Timestamp, list] = {}
        for record in salary_records:
            by_day.setdefault(record[1]["_date"].normalize(), []).append(record)
        pay_days = [by_day[day] for day in sorted(by_day)]
    else:
        pay_days = [[record] for record in salary_records]
    for event_number, pay_day in enumerate(pay_days, start=1):
        salary_index, salary_row = pay_day[0]
        salary_total = float(sum(record["credit_num"] for _, record in pay_day))
        salary_entities = list(dict.fromkeys(str(r.get("Level_6", "")) for _, r in pay_day if str(r.get("Level_6", ""))))
        salary_date = salary_row["_date"].normalize()
        natural_window_end = salary_date + pd.Timedelta(days=window_days - 1)
        # The next LATER salary date caps the window (events on the same day share one window),
        # and a window never ends before its own salary date.
        later_days = [day for day in salary_days if day > salary_date]
        window_end = natural_window_end
        if later_days:
            window_end = min(window_end, later_days[0] - pd.Timedelta(days=1))
        window_end = max(window_end, salary_date)

        spend_mask = (
            working_df["_date"].dt.normalize().between(salary_date, window_end)
            & working_df["debit_num"].gt(0)
            & (working_df.index != salary_index)
        )
        window_df = working_df.loc[spend_mask].copy()
        window_df["Salary_Event_Id"] = event_number
        window_df["Salary_Date"] = salary_date
        window_df["Window_End_Date"] = window_end
        window_df["Salary_Amount"] = salary_total
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
        window_df["Unclassified_Debit"] = window_df["debit_num"] * window_df["Is_Unclassified"].astype(int)
        window_df["Classified_Amount"] = window_df["debit_num"].where(window_df["Window_Category"].ne(""), 0.0)
        unclassified_amount = float(window_df["Unclassified_Debit"].sum())

        event_row = {
            **group_values,
            "Salary_Event_Id": event_number,
            "Salary_Date": salary_date,
            "Window_End_Date": window_end,
            # Effective length: a window cut short by the next pay day reports the days it really covers.
            "Window_Days": int((window_end - salary_date).days) + 1,
            "Salary_Amount": salary_total,
            "Savings_Transactions": int(window_df["Is_Savings"].sum()),
            "Expenditure_Transactions": int(window_df["Is_Expenditure"].sum()),
            "Transfer_Transactions": int(window_df["Is_Transfer"].sum()),
            "Savings_Amount": round(savings_amount, 2),
            "Expenditure_Amount": round(expenditure_amount, 2),
            "Transfer_Amount": round(transfer_amount, 2),
            "Unclassified_Transactions": int(window_df["Is_Unclassified"].sum()),
            "Unclassified_Amount": round(unclassified_amount, 2),
            # Savings vs spending only; transfers are not in the ratio.
            "Savings_to_Expenditure_Ratio": round(savings_amount / expenditure_amount, 4)
            if expenditure_amount
            else 0.0,
            "Salary_Confidence": min(float(r.get("SalaryConfidence", 0.0) or 0.0) for _, r in pay_day),
            "Salary_Entity": "; ".join(salary_entities),
            "Salary_Particulars": " | ".join(str(r.get("Particulars", "")) for _, r in pay_day),
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
    detailed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return salary-window summary, transactions, and daily-curve tables.

    The transactions table keeps the business columns (WINDOW_TRANSACTION_COLUMNS); pass detailed=True
    for every classified column and flag as well.
    """

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
    if not detailed and not window_transactions_df.empty:
        window_transactions_df = window_transactions_df.copy()
        if "Microsegmentation" not in window_transactions_df.columns and "MicroSegment" in window_transactions_df.columns:
            window_transactions_df["Microsegmentation"] = window_transactions_df["MicroSegment"]
        keep = [*(group_columns or ()), *WINDOW_TRANSACTION_COLUMNS]
        window_transactions_df = window_transactions_df[[c for c in keep if c in window_transactions_df.columns]]
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