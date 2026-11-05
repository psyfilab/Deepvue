"""Accuracy of our First Level Classification against the reference (Deepvue) First Level.

Reference : synthetic/<n>.xlsx                         -> sheet "Bank Statement", column "First Level Classification"
Ours      : reports/testing/latest_run/<n>_output.xlsx -> sheet "Classified",     column "First Level Classification"

Run from anywhere:
    python scripts/first_level_accuracy.py                 # statements 1..10 (a missing file is reported)
    python scripts/first_level_accuracy.py 1 3 10          # only these statements
    python scripts/first_level_accuracy.py --out some.xlsx # report path (default: next to the outputs)

How it is calculated
    1. Rows are matched on "Sl. No." (not on position). The narrations must agree; rows that do not are
       counted as "unmatched" and left out of the score, and the count is printed.
    2. Rows where the reference label is blank are not scored (reported as "reference blank").
       A row is correct when the two labels are equal after lower-casing and collapsing spaces. A blank
       label on our side counts as wrong (we did not classify it).
    3. Per file:    accuracy = correct rows / scored rows.
    4. Overall:     micro accuracy = all correct rows / all matched rows (big statements weigh more);
                    macro accuracy = mean of the per-file accuracies (every statement weighs the same).
    5. A looser "direction-agnostic" figure ignores the words Payment / Receipt, so a wrong direction is
       separated from a wrong category.
    6. Per label: precision (of rows we gave that label, how many agree) and recall (of rows the
       reference gave that label, how many we found).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, PatternFill

ROOT = Path(__file__).resolve().parents[1]  # project root = parent of scripts/
INPUT_DIR = ROOT / "synthetic"
OUTPUT_DIR = ROOT / "reports" / "testing" / "latest_run"
LABEL = "First Level Classification"
EXPECTED_STATEMENTS = 10  # statements 1..10 are expected; a missing file is reported, not ignored


def normalise(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def direction_free(series: pd.Series) -> pd.Series:
    return series.str.replace(r"\b(?:payment|receipt)\b", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


DETAIL_STATEMENTS = ("1", "2", "3")  # statements that also get a full side-by-side sheet (override with --detail)
HIGHLIGHT = PatternFill("solid", fgColor="FDE9E7")


def read_statement_columns(path: Path) -> pd.DataFrame:
    """The whole Bank Statement sheet (every column except Second Level), label column renamed for the report."""
    raw = pd.read_excel(path, sheet_name="Bank Statement", header=None)
    hit = raw.index[raw.apply(lambda r: r.astype(str).str.contains(LABEL, regex=False).any(), axis=1)][0]
    frame = pd.read_excel(path, sheet_name="Bank Statement", header=hit)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    frame = frame[pd.to_numeric(frame["Sl. No."], errors="coerce").notna()].copy()
    frame["Sl. No."] = pd.to_numeric(frame["Sl. No."]).astype(int)
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith("Second Level")])
    return frame.rename(columns={LABEL: "Deepvue First Level"})


def side_by_side(number: str) -> pd.DataFrame:
    """Full statement with Deepvue's and our First Level next to each other, plus a Result column."""
    full = read_statement_columns(INPUT_DIR / f"{number}.xlsx")
    ours = read_ours(OUTPUT_DIR / f"{number}_output.xlsx")[["Sl. No.", LABEL]].rename(columns={LABEL: "Our First Level"})
    out = full.merge(ours, on="Sl. No.", how="left")
    deepvue, mine = normalise(out["Deepvue First Level"]), normalise(out["Our First Level"])
    out["Result"] = "Different"
    out.loc[deepvue == mine, "Result"] = "Same"
    out.loc[mine == "", "Result"] = "We left it blank"
    out.loc[deepvue == "", "Result"] = "Not scored (Deepvue blank)"
    return out


def read_reference(path: Path) -> pd.DataFrame:
    """Bank Statement sheet: find the header row that holds the First Level column."""
    raw = pd.read_excel(path, sheet_name="Bank Statement", header=None)
    hits = raw.index[raw.apply(lambda r: r.astype(str).str.contains(LABEL, regex=False).any(), axis=1)]
    if not len(hits):
        raise ValueError(f"{path.name}: no '{LABEL}' column in sheet 'Bank Statement'")
    frame = pd.read_excel(path, sheet_name="Bank Statement", header=hits[0])
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    frame = frame[pd.to_numeric(frame["Sl. No."], errors="coerce").notna()].copy()
    frame["Sl. No."] = pd.to_numeric(frame["Sl. No."]).astype(int)
    keep = [c for c in ("Sl. No.", "Date", "Particulars", "Debit", "Credit", LABEL) if c in frame.columns]
    return frame[keep]


def read_ours(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Classified")
    frame = frame[pd.to_numeric(frame["Sl. No."], errors="coerce").notna()].copy()
    frame["Sl. No."] = pd.to_numeric(frame["Sl. No."]).astype(int)
    return frame[["Sl. No.", "Particulars", LABEL]]


def compare(number: str) -> tuple[pd.DataFrame, dict]:
    ref = read_reference(INPUT_DIR / f"{number}.xlsx")
    ours = read_ours(OUTPUT_DIR / f"{number}_output.xlsx")
    both = ref.merge(ours, on="Sl. No.", how="outer", suffixes=("_ref", "_ours"), indicator=True)
    same_text = both["Particulars_ref"].astype(str).str.strip() == both["Particulars_ours"].astype(str).str.strip()
    matched = both[(both["_merge"] == "both") & same_text].copy()
    matched["reference"] = normalise(matched[f"{LABEL}_ref"])
    matched["ours"] = normalise(matched[f"{LABEL}_ours"])
    # Rows the reference left blank cannot be scored: they are counted and reported, not marked wrong.
    blank_reference = int((matched["reference"] == "").sum())
    blank_rows = matched[matched["reference"] == ""].copy()
    blank_rows["file"] = number
    matched = matched[matched["reference"] != ""].copy()
    matched["correct"] = (matched["reference"] == matched["ours"]) & (matched["ours"] != "")
    matched["correct_any_direction"] = direction_free(matched["reference"]) == direction_free(matched["ours"])
    matched["file"] = number
    stats = {
        "file": number,
        "reference rows": int(len(ref)),
        "our rows": int(len(ours)),
        "unmatched rows": int(len(both) - len(matched) - blank_reference),
        "reference blank (not scored)": blank_reference,
        "matched rows": int(len(matched)),
        "correct": int(matched["correct"].sum()),
        "accuracy %": round(100 * matched["correct"].mean(), 2) if len(matched) else float("nan"),
        "accuracy ignoring Payment/Receipt %": round(100 * matched["correct_any_direction"].mean(), 2) if len(matched) else float("nan"),
        "blank on our side": int((matched["ours"] == "").sum()),
    }
    return matched, stats, blank_rows


SUMMARY_HEADERS = {
    "file": "Statement No.", "reference rows": "Rows in Deepvue file", "our rows": "Rows in our output",
    "unmatched rows": "Rows that could not be paired", "reference blank (not scored)": "Rows Deepvue left blank (not scored)",
    "matched rows": "Rows scored", "correct": "Rows where we agree with Deepvue", "accuracy %": "Accuracy %",
    "accuracy ignoring Payment/Receipt %": "Accuracy % (ignoring Payment/Receipt direction)", "blank on our side": "Scored rows where we left it blank",
}
LABEL_HEADERS = {
    "label": "First Level label", "reference rows": "Rows Deepvue gave this label", "our rows": "Rows we gave this label",
    "agree": "Rows both agree on", "precision %": "When we say this label, how often right (%)",
    "recall %": "Of Deepvue's rows with this label, how many we found (%)",
}
MISMATCH_HEADERS = {
    "rows": "Number of Rows", "statement": "Example: Statement No.", "sl_no": "Example: Sl. No.", "first_narration": "Example: full narration",
}
WRONG_HEADERS = {
    "statement": "Statement No.", "Particulars": "Particulars (full narration)", "mismatch": "Deepvue said -> We said",
    "Our First Level": "Our First Level",
}


def per_label(rows: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(set(rows["reference"]) | set(rows["ours"]) - {""})
    out = []
    for label in labels:
        ref_n = int((rows["reference"] == label).sum())
        our_n = int((rows["ours"] == label).sum())
        hit = int(((rows["reference"] == label) & (rows["ours"] == label)).sum())
        out.append({
            "label": label, "reference rows": ref_n, "our rows": our_n, "agree": hit,
            "precision %": round(100 * hit / our_n, 1) if our_n else None,
            "recall %": round(100 * hit / ref_n, 1) if ref_n else None,
        })
    return pd.DataFrame(out).sort_values("reference rows", ascending=False)


def available(requested: list[str]) -> list[str]:
    # By default: statements 1 to 10, plus any other numbered file found. A missing one is reported.
    found = {p.stem for p in INPUT_DIR.glob("*.xlsx") if re.fullmatch(r"\d+", p.stem)}
    numbers = requested or sorted({str(n) for n in range(1, EXPECTED_STATEMENTS + 1)} | found, key=int)
    usable = []
    for number in numbers:
        missing = [str(p) for p in (INPUT_DIR / f"{number}.xlsx", OUTPUT_DIR / f"{number}_output.xlsx") if not p.exists()]
        if missing:
            print(f"skipped {number}: missing {', '.join(missing)}")
        else:
            usable.append(number)
    return usable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("numbers", nargs="*", help="statement numbers, e.g. 1 3 10 (default: all found)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "first_level_accuracy.xlsx")
    parser.add_argument("--detail", nargs="*", default=list(DETAIL_STATEMENTS),
                        help="statements that also get a full side-by-side sheet (default: 1 2 3; use --detail with no numbers for none)")
    args = parser.parse_args(argv)

    numbers = available(args.numbers)
    if not numbers:
        print("nothing to compare")
        return 1
    all_rows, summaries, blank_ref = [], [], []
    for number in numbers:
        # pyrefly: ignore[bad-unpacking]
        rows, stats, blanks = compare(number)
        all_rows.append(rows)
        summaries.append(stats)
        blank_ref.append(blanks)
    rows = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    total = {
        "file": "ALL (micro)", "reference rows": summary["reference rows"].sum(), "our rows": summary["our rows"].sum(),
        "unmatched rows": summary["unmatched rows"].sum(), "reference blank (not scored)": summary["reference blank (not scored)"].sum(), "matched rows": summary["matched rows"].sum(),
        "correct": summary["correct"].sum(), "accuracy %": round(100 * rows["correct"].mean(), 2),
        "accuracy ignoring Payment/Receipt %": round(100 * rows["correct_any_direction"].mean(), 2),
        "blank on our side": summary["blank on our side"].sum(),
    }
    macro = {"file": "MEAN OF FILES (macro)", "accuracy %": round(summary["accuracy %"].mean(), 2),
             "accuracy ignoring Payment/Receipt %": round(summary["accuracy ignoring Payment/Receipt %"].mean(), 2)}
    summary = pd.concat([summary, pd.DataFrame([total, macro])], ignore_index=True)

    count_cols = ["reference rows", "our rows", "unmatched rows", "reference blank (not scored)", "matched rows", "correct", "blank on our side"]
    summary[count_cols] = summary[count_cols].astype("Int64")
    labels = per_label(rows)
    # Every mismatched row with its FULL narration (the reference file's text, never shortened).
    columns = ["file", "Sl. No.", "Date", "Particulars_ref", "Debit", "Credit", "reference", "ours"]
    wrong = rows[~rows["correct"]][[c for c in columns if c in rows.columns]].rename(
        columns={"Particulars_ref": "Particulars", "file": "statement", "reference": "Deepvue First Level", "ours": "Our First Level"}
    )
    wrong["mismatch"] = wrong["Deepvue First Level"] + " -> " + wrong["Our First Level"].replace("", "(blank)")
    wrong = wrong.sort_values(["mismatch", "statement", "Sl. No."]).reset_index(drop=True)
    blank_ref = pd.concat(blank_ref, ignore_index=True)
    blank_ref = blank_ref[[c for c in ("file", "Sl. No.", "Date", "Particulars_ref", "Debit", "Credit", "ours") if c in blank_ref.columns]].rename(
        columns={"Particulars_ref": "Particulars", "file": "statement", "ours": "Our First Level"}
    )
    confusion = (
        wrong.groupby(["Deepvue First Level", "Our First Level"], sort=False)
        .agg(rows=("Particulars", "size"), statement=("statement", "first"), sl_no=("Sl. No.", "first"), first_narration=("Particulars", "first"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    details = {n: side_by_side(n) for n in args.detail if n in numbers}
    detail_names = {n: f"Statement {n} Side by Side" for n in details}
    readme = pd.DataFrame({"Sheet": [
        "Summary by Statement", "Accuracy by Label", "Mismatch Types", "All Wrong Rows", "Not Scored (Deepvue Blank)"] + list(detail_names.values()), "What it means": [
        "One line per statement: how many rows were checked and how often our First Level equals Deepvue's. Last two lines are the overall totals.",
        "For each First Level label: how many rows each side gave it, and how well we match.",
        "Each kind of disagreement (Deepvue said X, we said Y) with the number of rows, sorted most frequent first. The Example columns show ONE real row of that kind (the earliest in statement order) so you can see what it looks like.",
        "Every single row where we disagree with Deepvue, with the full narration, date, debit and credit.",
        "Rows where Deepvue's own First Level cell is empty. They cannot be checked, so they are listed here and left out of the accuracy."]
        + ["The complete bank statement (all columns except Second Level) with Deepvue's First Level and our First Level side by side. "
           "Result = Same / Different / We left it blank / Not scored (Deepvue blank). Different rows are shaded."] * len(detail_names)})
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(summary.to_string(index=False))
    print("\nTop mismatches (reference -> ours):")
    print(confusion.head(15).to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="Read Me", index=False)
        summary.rename(columns=SUMMARY_HEADERS).to_excel(writer, sheet_name="Summary by Statement", index=False)
        labels.rename(columns=LABEL_HEADERS).to_excel(writer, sheet_name="Accuracy by Label", index=False)
        confusion.rename(columns=MISMATCH_HEADERS).to_excel(writer, sheet_name="Mismatch Types", index=False)
        wrong.rename(columns=WRONG_HEADERS).to_excel(writer, sheet_name="All Wrong Rows", index=False)
        if len(blank_ref):
            blank_ref.rename(columns=WRONG_HEADERS).to_excel(writer, sheet_name="Not Scored (Deepvue Blank)", index=False)
        for number, frame in details.items():
            frame.to_excel(writer, sheet_name=detail_names[number], index=False)
            sheet = writer.sheets[detail_names[number]]
            result_col = list(frame.columns).index("Result") + 1
            for row in range(2, len(frame) + 2):
                if sheet.cell(row, result_col).value == "Different":
                    for col in range(1, len(frame.columns) + 1):
                        sheet.cell(row, col).fill = HIGHLIGHT
        for sheet in writer.sheets.values():  # readable: wrapped full narrations, filters, frozen header
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                header = str(column[0].value)
                wide = header in ("Particulars", "Particulars (full narration)", "Example: full narration", "What it means")
                sheet.column_dimensions[column[0].column_letter].width = 95 if wide else 34 if header in ("Deepvue said -> We said", "First Level label") else 22
                for cell in column[1:]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
    print(f"side-by-side sheets: {', '.join(detail_names.values()) or 'none'}")
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())