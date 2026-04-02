"""Run the full bank-statement pipeline from narration split to micro-segmentation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.microsegment import Microsegmenter
    from src.statement_classifier import BankStatementClassifier
    from src.statement_splitter import load_statement, split_statement
else:
    from .microsegment import Microsegmenter
    from .statement_classifier import BankStatementClassifier
    from .statement_splitter import load_statement, split_statement

logger = logging.getLogger(__name__)


def autofit_worksheet(worksheet: object) -> None:
    """Apply a readable width to each worksheet column."""

    for column_cells in worksheet.columns:
        max_len = max((len(str(cell.value)) if cell.value else 0) for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_len + 2, 55)


def write_output(
    output_path: Path,
    split_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    classified_df: pd.DataFrame,
) -> None:
    """Write pipeline outputs to Excel or CSV."""

    if output_path.suffix.lower() == ".csv":
        classified_df.to_csv(output_path, index=False)
        unique_path = output_path.with_name(f"{output_path.stem}_unique_keywords.csv")
        unique_df.to_csv(unique_path, index=False)
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        split_df.to_excel(writer, sheet_name="Split", index=False)
        unique_df.to_excel(writer, sheet_name="Unique_Keywords", index=False)
        classified_df.to_excel(writer, sheet_name="Classified", index=False)

        for sheet_name in ("Split", "Unique_Keywords", "Classified"):
            autofit_worksheet(writer.sheets[sheet_name])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the full bank statement pipeline.")
    parser.add_argument("--input", required=True, help="CSV or XLSX bank statement file")
    parser.add_argument(
        "--output",
        default="reports/bank_statement_output.xlsx",
        help="Output XLSX or CSV path",
    )
    parser.add_argument("--models", default="models/", help="Directory with trained models")
    parser.add_argument(
        "--master",
        default="MASTER_SHEET.csv",
        help="MASTER_SHEET path used for micro-segmentation vocab",
    )
    parser.add_argument("--sheet-name", default="Bank Statement", help="Excel sheet name")
    parser.add_argument(
        "--header-row",
        default=4,
        type=int,
        help="Zero-based header row for Excel bank statements",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    statement_df = load_statement(
        path=args.input,
        sheet_name=args.sheet_name,
        header_row=args.header_row,
    )
    split_df, unique_df = split_statement(statement_df)
    classified_df = BankStatementClassifier(models_dir=args.models).classify(split_df)
    enriched_df = Microsegmenter(master_sheet_path=args.master).run(classified_df)

    write_output(output_path, split_df, unique_df, enriched_df)
    print(f"Output written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
