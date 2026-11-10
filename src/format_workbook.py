"""Final presentation step: rebuild the pipeline workbook as a clean, readable Excel file.

Runs once, after every result has been written. It reads the written workbook, renames columns for
display only (internal names such as Level_5 or MicroSegment are untouched everywhere else), and writes
the sheets back in a fixed order. No classification logic lives here.

    python -m src.format_workbook reports/1_output.xlsx            # format in place
    python -m src.format_workbook reports/1_output.xlsx out.xlsx    # keep the original
"""

from __future__ import annotations

import math
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

# ----------------------------------------------------------------------------------------- palette
TEAL = "2A7F7F"
TEAL_TINT = "DCEFEF"
HEADER_FILL = "EEF1F4"
HEADER_TEXT = "333333"
HEADER_BORDER = "9AA5B1"
BODY_BORDER = "E3E7EB"
PART_TINT = "F4F6F8"
FLAG_TINT = "F1F7F3"
ZERO_GREY = "B4BBC3"
SCALE_END = "CFE8E8"
GREY_TAB = "A6AEB6"
FONT_NAME, FONT_SIZE = "Calibri", 10

# ----------------------------------------------------------------------------------------- naming
SHEET_ORDER = (
    "Classified", "Scores and Flags", "Salary Summary", "Salary Window Summary",
    "Salary Window Transactions", "Split", "Unique Keywords", "Data Dictionary",
)
SOURCE_SHEETS = {  # written name -> display name
    "Classified": "Classified", "Salary_Summary": "Salary Summary",
    "Salary_Window_Summary": "Salary Window Summary", "Salary_Window_Txns": "Salary Window Transactions",
    "Split": "Split", "Unique_Keywords": "Unique Keywords",
}
HELPER_SHEETS = ("Split", "Unique Keywords")
ACRONYMS = {"Upi": "UPI", "Atm": "ATM", "Emi": "EMI"}
RENAMES = {
    "Particulars_Split": "Particulars Split", "MicroSegment": "Microsegmentation",
    "IsSalary": "Salary Flag", "SalaryConfidence": "Salary Confidence Score",
    "present_bias_score": "Present Bias Score",
    **{f"Part_{n}": f"Part {n}" for n in range(8)}, **{f"Level_{n}": f"Level {n}" for n in range(8)},
}
# Header rows for the salary sheets when a statement has no salary credit, so every workbook has the same sheets.
EMPTY_SALARY_SHEETS = {
    "Salary Summary": ["Date", "Month", "Particulars", "Credit", "SalaryConfidence", "Level_5", "Level_6",
                       "First Level Classification"],
    "Salary Window Summary": ["Salary_Event_Id", "Salary_Date", "Window_End_Date", "Window_Days", "Salary_Amount",
                              "Savings_Transactions", "Expenditure_Transactions", "Transfer_Transactions",
                              "Savings_Amount", "Expenditure_Amount", "Transfer_Amount",
                              "Unclassified_Transactions", "Unclassified_Amount",
                              "Savings_to_Expenditure_Ratio", "Salary_Confidence", "Salary_Entity",
                              "Salary_Particulars"],
    "Salary Window Transactions": ["Salary_Event_Id", "Salary_Date", "Window_End_Date", "Days_After_Salary",
                                   "Salary_Amount", "Date", "Particulars", "Debit", "First Level Classification",
                                   "Second Level Classification", "Level_3", "Level_5", "Level_6",
                                   "Microsegmentation", "Window_Category", "Classified_Amount"],
}
HIDDEN_COLUMNS = ("debit_num", "credit_num", "IsSalaryTxn")  # working columns, never shown
CLASSIFIED_COLUMNS = (
    ["Sl. No.", "Date", "Particulars", "Debit", "Credit", "Balance", "Ref/Chq No.",
     "First Level Classification", "Second Level Classification"]
    + [f"Part {n}" for n in range(8)] + ["Particulars Split"] + [f"Level {n}" for n in range(8)]
    + ["Microsegmentation"]
)
AMOUNT_COLUMNS = ("Debit", "Credit", "Balance")
TEAL_HEADERS = ("Microsegmentation", "Salary Confidence Score", "Present Bias Score")


def flag_display_name(column: str) -> str:
    """exp_food -> Food, exp_upi_velocity -> UPI Velocity."""

    words = column[len("exp_"):].replace("_", " ").title().split()
    return " ".join(ACRONYMS.get(word, word) for word in words)


def display_name(column: str) -> str:
    if column.startswith("exp_"):
        return flag_display_name(column)
    return RENAMES.get(column, column)


# ----------------------------------------------------------------------------------------- values
DATE_FORMATS = ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%Y", "%d-%m-%y", "%d-%b-%y", "%d-%b-%Y", "%d %b %Y",
                "%Y-%m-%d", "%Y-%m-%d %H:%M:%S")


def is_blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or (
        isinstance(value, str) and not value.strip()
    )


def to_date(value: object) -> object:
    """Real date when the text is a date (day first, like the classifier); otherwise unchanged."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        return value
    text = value.strip()
    for pattern in DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return value


def to_number(value: object) -> object:
    """Real number for '6,195.38'; None for blank or '-'; anything unparseable is kept as text."""

    if is_blank(value) or (isinstance(value, str) and value.strip() in {"-", "--"}):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return value


def is_date_column(name: str) -> bool:
    return name == "Date" or name.lower().endswith("date")


def read_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """Every sheet as an object DataFrame, cell values exactly as written."""

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    frames = {}
    for sheet in workbook.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        if rows:
            frames[sheet.title] = pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]], dtype=object)
    workbook.close()
    return frames


# ----------------------------------------------------------------------------------------- styles
THIN_BODY = Side(style="thin", color=BODY_BORDER)
BODY_BORDERS = Border(left=THIN_BODY, right=THIN_BODY, top=THIN_BODY, bottom=THIN_BODY)
HEADER_BORDERS = Border(left=THIN_BODY, right=THIN_BODY, top=THIN_BODY, bottom=Side(style="thin", color=HEADER_BORDER))
BODY_FONT = Font(name=FONT_NAME, size=FONT_SIZE)
ZERO_FONT = Font(name=FONT_NAME, size=FONT_SIZE, color=ZERO_GREY)


def fill(color: str) -> PatternFill:
    return PatternFill("solid", start_color=color, end_color=color)


def style_header(cell, name: str, tint: str | None = None) -> None:
    teal = name in TEAL_HEADERS
    cell.value = name
    cell.font = Font(name=FONT_NAME, size=FONT_SIZE, bold=True, color=TEAL if teal else HEADER_TEXT)
    cell.fill = fill(TEAL_TINT if teal else (tint or HEADER_FILL))
    cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    cell.border = HEADER_BORDERS


def write_table(workbook, title: str, frame: pd.DataFrame, *, tints: dict[str, str] | None = None,
                widths: dict[str, float] | None = None, formats: dict[str, str] | None = None,
                freeze: str = "A2", default_width: float | None = None) -> openpyxl.worksheet.worksheet.Worksheet:
    """One styled sheet: header in row 1, body below, filter, frozen panes, widths and number formats."""

    sheet = workbook.create_sheet(title)
    tints, widths, formats = tints or {}, widths or {}, formats or {}
    for col, name in enumerate(frame.columns, start=1):
        style_header(sheet.cell(1, col), name, tints.get(name))
    sheet.row_dimensions[1].height = 32
    for row_number, values in enumerate(frame.itertuples(index=False), start=2):
        for col, (name, value) in enumerate(zip(frame.columns, values), start=1):
            value = None if is_blank(value) else value
            cell = sheet.cell(row_number, col, value)
            cell.font, cell.border = BODY_FONT, BODY_BORDERS
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            cell.alignment = Alignment(vertical="center", horizontal="right" if numeric else None)
            if name in formats and (numeric or isinstance(value, datetime)):
                cell.number_format = formats[name]
    for col, name in enumerate(frame.columns, start=1):
        if name in widths:
            width = widths[name]
        else:
            longest = max([len(str(name))] + [len(str(v)) for v in frame[name].head(300) if not is_blank(v)])
            width = max(10, min(longest + 2, default_width or 40))
        sheet.column_dimensions[get_column_letter(col)].width = width
    sheet.freeze_panes = freeze
    if len(frame):
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(frame.columns))}{len(frame) + 1}"
    return sheet


def generic_formats(frame: pd.DataFrame) -> dict[str, str]:
    """Salary / helper sheets: dates dd-mmm-yy, decimals #,##0.00, whole numbers 0."""

    formats = {}
    for name in frame.columns:
        values = [v for v in frame[name] if not is_blank(v)]
        if is_date_column(name) and values and all(isinstance(v, datetime) for v in values):
            formats[name] = "dd-mmm-yy"
        elif values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            formats[name] = "0" if all(float(v).is_integer() for v in values) and name not in AMOUNT_COLUMNS else "#,##0.00"
    return formats


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Display names, hidden working columns dropped, dates and amounts converted to real values."""

    frame = frame.drop(columns=[c for c in HIDDEN_COLUMNS if c in frame.columns]).copy()
    frame.columns = [display_name(c) for c in frame.columns]
    for name in frame.columns:
        if is_date_column(name):
            frame[name] = frame[name].map(to_date)
        elif name in AMOUNT_COLUMNS or name in {"Credit"}:
            frame[name] = frame[name].map(to_number)
    if "Salary Confidence Score" in frame.columns:
        frame["Salary Confidence Score"] = frame["Salary Confidence Score"].map(to_number)
    return frame


# ----------------------------------------------------------------------------------------- sheets
def build_classified(workbook, classified: pd.DataFrame) -> None:
    frame = classified[list(CLASSIFIED_COLUMNS)]
    tints = {name: PART_TINT for name in frame.columns if name.startswith("Part ")}
    tints.update({name: HEADER_FILL for name in frame.columns if name.startswith("Level ")})
    widths = {"Sl. No.": 8, "Date": 11, "Particulars": 45, "Ref/Chq No.": 18, "Particulars Split": 40,
              "First Level Classification": 24, "Second Level Classification": 24, "Microsegmentation": 22,
              **{name: 14 for name in AMOUNT_COLUMNS}, **{f"Part {n}": 18 for n in range(8)},
              **dict(zip((f"Level {n}" for n in range(8)), (16, 16, 18, 16, 18, 20, 22, 16)))}
    formats = {"Date": "dd-mmm-yy", **{name: "#,##0.00" for name in AMOUNT_COLUMNS}}
    sheet = write_table(workbook, "Classified", frame, tints=tints, widths=widths, formats=formats, freeze="D2")
    for col, name in enumerate(frame.columns, start=1):  # Particulars never wraps; amounts right aligned
        if name == "Particulars":
            for row in range(2, len(frame) + 2):
                sheet.cell(row, col).alignment = Alignment(wrap_text=False, vertical="center")
        if name in AMOUNT_COLUMNS:
            for row in range(2, len(frame) + 2):
                sheet.cell(row, col).alignment = Alignment(horizontal="right", vertical="center")
    first, last = (get_column_letter(list(frame.columns).index(f"Part {n}") + 1) for n in (0, 7))
    sheet.column_dimensions.group(first, last, outline_level=1, hidden=False)
    sheet.column_dimensions[first].width = 18  # grouping keeps one shared width
    sheet.sheet_properties.tabColor = TEAL
    sheet.page_setup.orientation = "landscape"
    sheet.print_title_rows = "1:1"
    sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    sheet.page_setup.fitToWidth, sheet.page_setup.fitToHeight = 1, 0


def build_scores(workbook, classified: pd.DataFrame, flag_columns: list[str]) -> None:
    columns = ["Sl. No.", "Date", "Particulars"] + flag_columns + [
        "Salary Flag", "Salary Confidence Score", "Present Bias Score"]
    frame = classified[columns].copy()
    for name in flag_columns + ["Salary Flag"]:
        frame[name] = frame[name].map(lambda v: None if is_blank(v) else int(float(v)))
    for name in ("Salary Confidence Score", "Present Bias Score"):
        frame[name] = frame[name].map(lambda v: None if is_blank(v) else float(v))
    tints = {name: FLAG_TINT for name in flag_columns}
    widths = {"Sl. No.": 8, "Date": 11, "Particulars": 45, "Salary Flag": 12,
              "Salary Confidence Score": 18, "Present Bias Score": 16, **{name: 13 for name in flag_columns}}
    formats = {"Date": "dd-mmm-yy", "Salary Flag": "0", "Salary Confidence Score": "0.00",
               "Present Bias Score": "0.00", **{name: "0" for name in flag_columns}}
    sheet = write_table(workbook, "Scores and Flags", frame, tints=tints, widths=widths, formats=formats, freeze="D2")
    last_row = len(frame) + 1
    for col, name in enumerate(frame.columns, start=1):
        if name in flag_columns or name == "Salary Flag":
            for row in range(2, last_row + 1):
                cell = sheet.cell(row, col)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if name in flag_columns and cell.value == 0:
                    cell.font = ZERO_FONT
        if name == "Salary Flag":
            letter = get_column_letter(col)
            sheet.conditional_formatting.add(
                f"{letter}2:{letter}{last_row}", CellIsRule(operator="equal", formula=["1"], fill=fill(TEAL_TINT)))
        if name == "Present Bias Score":
            letter = get_column_letter(col)
            sheet.conditional_formatting.add(
                f"{letter}2:{letter}{last_row}",
                ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color=SCALE_END))


MICROSEGMENT_VALUES = (
    "Salary", "P2P Transfer", "Own Account Transfer", "Inward Transfer", "Food", "Grocery", "Quick Commerce",
    "Ecommerce", "Entertainment", "Subscription", "Investment", "Saving / Insurance", "Loan", "Credit Card Payment",
    "EMI", "Utility", "Fuel", "Transport", "Travel", "Pharmacy", "Tax", "Education", "ATM Withdrawal",
    "Penalty / Charges", "Refund", "Luxury", "Micro-spend", "UPI Velocity", "Digital Payment", "Other",
)
LEVEL_MEANINGS = (
    ("Level 0", "Payment rail or direction: how the money moved, or which way it went (UPI, NEFT, IMPS, NACH, debit, credit)."),
    ("Level 1", "Transaction type: P2P, auto-debit, UPI, NACH, cheque, wallet, cash, card and so on."),
    ("Level 2", "Who initiated the transaction."),
    ("Level 3", "Counterparty type: person (p2p), merchant, business and so on."),
    ("Level 4", "Platform or bank involved in the transaction."),
    ("Level 5", "Purpose of the transaction, for example salary, loan repayment, dining."),
    ("Level 6", "Payee or brand name."),
    ("Level 7", "Frequency: one-time or recurring."),
)
SCORE_MEANINGS = (
    ("Salary Flag", "1 if the row is a salary credit, otherwise 0. Only credits can be salary. The flag, Level 5 and Microsegmentation always agree."),
    ("Salary Confidence Score", "1.0 = the narration has an explicit salary word (salary, payroll, payslip, wages). 0.8 = the same payer sends a similar amount (within 10%) once a month in at least 3 different months on a steady day. Below 0.6 = not salary."),
    ("Present Bias Score", "Weighted total of the spend flags on the row. Higher means more short-term or impulsive "
                           "spending, lower or negative means saving-oriented. Indicative only; the weights are not validated."),
)
FLAG_MEANINGS = {
    "exp_food": "Food delivery, restaurants and cafes (debit).",
    "exp_grocery": "Grocery and daily-needs shops (debit).",
    "exp_quick_comm": "Quick-commerce apps such as Blinkit, Zepto, Instamart (debit).",
    "exp_ecomm": "Online retail and marketplaces (debit).",
    "exp_entertainment": "Movies, events, gaming and leisure (debit).",
    "exp_sub": "Subscriptions and recurring digital services (debit).",
    "exp_micro": "Any debit under 200.",
    "exp_luxury": "Merchant debit above 5,000 up to 5,00,000 where Level 5 is blank, retail or services. Never for interest, charges, loans, financial services, tax, insurance, healthcare, wallet moves, bank names or medical words.",
    "exp_saving": "Savings products named in the narration: FD, RD, PPF, NPS, SIP, mutual fund.",
    "exp_investment": "Investment platforms and mutual-fund payments (debit).",
    "exp_loan": "Loan repayment, or financial services whose payee says EMI, loan, fin, finance, finserv, fintech, kissht or slice (debit).",
    "exp_creditcard": "Credit card bill payments and card apps (debit).",
    "exp_emi": "An EMI or loan word in the narration when Level 5 is blank (debit). A recurring NACH debit without those words is not an EMI.",
    "exp_utility": "Electricity, gas, water, mobile recharge and bill payments (debit).",
    "exp_fuel": "Petrol and fuel stations (debit).",
    "exp_transport": "Cabs, metro, rail, fastag and other transport (debit).",
    "exp_travel": "Flights, hotels and travel bookings (debit).",
    "exp_pharmacy": "Pharmacies and medicines (debit).",
    "exp_insurance": "Insurance premiums and claims (debit or credit).",
    "exp_tax": "GST, income tax, TDS and other taxes (debit).",
    "exp_education": "School, college and course fees (debit).",
    "exp_atm": "ATM and cash withdrawals (debit).",
    "exp_penalty": "Interest and charges: bank charges, penalties, bounce and late-fee entries.",
    "exp_refund": "Refunds, reversals and cashbacks (credit).",
    "exp_upi_velocity": "UPI debit under 500: small, frequent, low-friction spending.",
}


def build_dictionary(workbook, flag_pairs: list[tuple[str, str]]) -> None:
    sheet = workbook.create_sheet("Data Dictionary")
    sheet.sheet_view.showGridLines = False
    style_header(sheet.cell(1, 1), "Item")
    style_header(sheet.cell(1, 2), "Meaning")
    sheet.row_dimensions[1].height = 32
    row = 2

    def section(title: str) -> None:
        nonlocal row
        row += 1 if row > 2 else 0
        cell = sheet.cell(row, 1, title)
        cell.font = Font(name=FONT_NAME, size=FONT_SIZE, bold=True, color=TEAL)
        row += 1

    def line(item: str, meaning: str, swatch: str | None = None) -> None:
        nonlocal row
        first, second = sheet.cell(row, 1, item), sheet.cell(row, 2, meaning)
        for cell in (first, second):
            cell.font, cell.border = BODY_FONT, BODY_BORDERS
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        if swatch:
            first.fill = fill(swatch)
        row += 1

    section("Levels (columns Level 0 to Level 7 on the Classified sheet)")
    for item, meaning in LEVEL_MEANINGS:
        line(item, meaning)
    section("Colour key")
    line("Teal", "Key output columns: Microsegmentation, Salary Confidence Score, Present Bias Score.", TEAL_TINT)
    line("Grey", "Supporting columns: the original statement, the split narration and the Level columns.", HEADER_FILL)
    section("Scores")
    for item, meaning in SCORE_MEANINGS:
        line(item, meaning)
    section("Salary windows (Salary Window Summary and Salary Window Transactions sheets)")
    line("Window", "From the pay day to the day before the next pay day, at most 10 days. Window_Days is the effective "
                   "length, so a window cut short by the next pay day shows fewer than 10. Credits on the same day "
                   "are one pay day: salary is their total.")
    line("Window_Category", "Savings (SIP, FD, RD, mutual fund, PPF, NPS, investment platform), Expenditure (spending, bills, "
                            "loans, tax, charges), Transfer (to other people, or between the holder's own accounts and wallets), "
                            "Unclassified (a debit whose purpose could not be decided; shown, never forced into a bucket). "
                            "Failed or reversed rows are left out.")
    line("Classified_Amount", "The debit counted in the row's Window_Category.")
    section("Microsegmentation values")
    line("Rule", "Level 5 always decides first. Keywords are read only when Level 5 is blank (or generic), in the payee "
                 "name only, as whole words; a UPI handle, bank name, reference or purpose tail never triggers a flag.")
    line("Values", ", ".join(MICROSEGMENT_VALUES))
    section("Spend flags (1 = yes, 0 = no, on the Scores and Flags sheet)")
    for column, name in flag_pairs:
        line(name, FLAG_MEANINGS.get(column, ""))
    sheet.column_dimensions["A"].width = 26
    sheet.column_dimensions["B"].width = 100
    sheet.freeze_panes = "A2"


# ----------------------------------------------------------------------------------------- checks
def norm(value: object) -> str:
    return "" if is_blank(value) else str(value).strip()


def assert_matches_dataframe(classified: pd.DataFrame, source: pd.DataFrame) -> None:
    """The formatted Classified values must equal the pipeline DataFrame, cell for cell."""

    assert len(classified) == len(source), f"row count changed: {len(source)} -> {len(classified)}"
    for column in source.columns:
        name = display_name(column)
        if name not in classified.columns:
            continue
        shown, original = classified[name].tolist(), source[column].tolist()
        for row, (new, old) in enumerate(zip(shown, original), start=2):
            if is_date_column(name):
                same = norm(new) == norm(old) or to_date(old) == new
            elif name in AMOUNT_COLUMNS or name in {"Salary Confidence Score", "Present Bias Score"}:
                a, b = to_number(new), to_number(old)
                same = a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-9)
            else:
                same = norm(new) == norm(old) or (
                    not is_blank(new) and not is_blank(old) and str(new).strip() == str(old).strip())
                if not same:
                    try:
                        same = float(new) == float(old)
                    except (TypeError, ValueError):
                        pass
            assert same, f"Classified row {row}, column {name!r}: {old!r} became {new!r}"


def verify_workbook(path: Path) -> list[str]:
    """Reload the file and check the formatting contract. Returns one line per check; raises on failure."""

    workbook = openpyxl.load_workbook(path)
    results = []
    classified = workbook["Classified"]
    headers = [c.value for c in classified[1]]
    assert classified.freeze_panes == "D2", classified.freeze_panes
    assert headers[-1] == "Microsegmentation", headers[-1]
    assert list(workbook.sheetnames) == [s for s in SHEET_ORDER if s in workbook.sheetnames], workbook.sheetnames
    results.append(f"sheets {workbook.sheetnames}; Classified freeze={classified.freeze_panes}; last column={headers[-1]}")
    banned = re.compile(r"^(exp_|debit_num$|credit_num$|IsSalaryTxn$)")
    bad = [(s.title, c.value) for s in workbook for c in s[1] if isinstance(c.value, str) and banned.match(c.value)]
    assert not bad, bad
    results.append("no exp_*, debit_num, credit_num or IsSalaryTxn header on any sheet")
    navy = []
    for sheet in workbook:
        for row in sheet.iter_rows():
            for cell in row:
                colors = [cell.fill.start_color.rgb if cell.fill and cell.fill.fill_type == "solid" else None,
                          cell.font.color.rgb if cell.font and cell.font.color else None]
                for rgb in colors:
                    if isinstance(rgb, str) and len(rgb) in (6, 8):
                        r, g, b = (int(rgb[-6:][i:i + 2], 16) for i in (0, 2, 4))
                        if max(r, g, b) < 0x90 and b > g and b > r:
                            navy.append((sheet.title, cell.coordinate, rgb))
    assert not navy, navy[:3]
    results.append("no navy fill or font colour anywhere")
    dictionary = workbook["Data Dictionary"]
    assert dictionary.sheet_view.showGridLines is False
    return results


# ----------------------------------------------------------------------------------------- entry
def format_workbook(output_path: str | Path, classified_df: pd.DataFrame | None = None) -> Path:
    """Reformat the written workbook in place. Pass the pipeline DataFrame to also assert the values match."""

    path = Path(output_path)
    raw = {SOURCE_SHEETS[name]: frame for name, frame in read_sheets(path).items() if name in SOURCE_SHEETS}
    flag_columns = [c for c in raw["Classified"].columns if c.startswith("exp_")]
    classified_full = prepare(raw["Classified"])
    classified_full["Date"] = classified_full["Date"].map(to_date)
    if classified_df is not None:
        assert_matches_dataframe(classified_full, classified_df.reset_index(drop=True))
    source_rows = len(raw["Classified"])

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    build_classified(workbook, classified_full)
    flag_pairs = [(c, flag_display_name(c)) for c in flag_columns]
    build_scores(workbook, classified_full, [name for _, name in flag_pairs])
    for title in SHEET_ORDER[2:7]:
        if title in EMPTY_SALARY_SHEETS and (title not in raw or raw[title].empty):
            raw[title] = pd.DataFrame(columns=EMPTY_SALARY_SHEETS[title], dtype=object)
        if title not in raw:
            continue
        frame = prepare(raw[title])
        tints = {name: FLAG_TINT for _, name in flag_pairs if name in frame.columns}
        sheet = write_table(workbook, title, frame, tints=tints, formats=generic_formats(frame),
                            default_width=45 if title in HELPER_SHEETS or "Transactions" in title else 40)
        if title in HELPER_SHEETS:
            sheet.sheet_properties.tabColor = GREY_TAB
    build_dictionary(workbook, flag_pairs)
    assert workbook["Classified"].max_row == source_rows + 1, "Classified row count changed"

    temp = path.with_suffix(".formatting.xlsx")
    workbook.save(temp)
    shutil.move(str(temp), str(path))
    verify_workbook(path)
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    if len(sys.argv) > 2:
        shutil.copy(sys.argv[1], target)
    format_workbook(target)
    for result in verify_workbook(target):
        print(result)