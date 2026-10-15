"""Run the full bank-statement pipeline from narration split to micro-segmentation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    # Put the project root first so the working-tree ``src`` package is
    # imported, not an older installed or cached copy found earlier on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.microsegment import Microsegmenter
    from src.salary_timeseries import build_salary_time_series, save_salary_timeline_plots
    from src.statement_classifier import BankStatementClassifier, build_salary_summary
    from src.statement_splitter import load_statement, split_statement
else:
    from .microsegment import Microsegmenter
    from .salary_timeseries import build_salary_time_series, save_salary_timeline_plots
    from .statement_classifier import BankStatementClassifier, build_salary_summary
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
    salary_summary_df: pd.DataFrame,
    salary_window_summary_df: pd.DataFrame,
    salary_window_txn_df: pd.DataFrame,
) -> None:
    """Write pipeline outputs to Excel or CSV."""

    if output_path.suffix.lower() == ".csv":
        classified_df.to_csv(output_path, index=False)
        unique_path = output_path.with_name(f"{output_path.stem}_unique_keywords.csv")
        unique_df.to_csv(unique_path, index=False)
        salary_path = output_path.with_name(f"{output_path.stem}_salary_summary.csv")
        salary_summary_df.to_csv(salary_path, index=False)
        salary_window_path = output_path.with_name(
            f"{output_path.stem}_salary_window_summary.csv"
        )
        salary_window_summary_df.to_csv(salary_window_path, index=False)
        salary_txn_path = output_path.with_name(
            f"{output_path.stem}_salary_window_transactions.csv"
        )
        salary_window_txn_df.to_csv(salary_txn_path, index=False)
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        split_df.to_excel(writer, sheet_name="Split", index=False)
        unique_df.to_excel(writer, sheet_name="Unique_Keywords", index=False)
        classified_df.to_excel(writer, sheet_name="Classified", index=False)
        salary_summary_df.to_excel(writer, sheet_name="Salary_Summary", index=False)
        salary_window_summary_df.to_excel(
            writer,
            sheet_name="Salary_Window_Summary",
            index=False,
        )
        salary_window_txn_df.to_excel(
            writer,
            sheet_name="Salary_Window_Txns",
            index=False,
        )

        for sheet_name in (
            "Split",
            "Unique_Keywords",
            "Classified",
            "Salary_Summary",
            "Salary_Window_Summary",
            "Salary_Window_Txns",
        ):
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
    parser.add_argument(
        "--window-days",
        default=10,
        type=int,
        help="Days after each salary credit to include in spend analysis",
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
    salary_summary_df = build_salary_summary(enriched_df)
    salary_window_summary_df, salary_window_txn_df, _ = (
        build_salary_time_series(
            enriched_df,
            window_days=args.window_days,
        )
    )

    write_output(
        output_path,
        split_df,
        unique_df,
        enriched_df,
        salary_summary_df,
        salary_window_summary_df,
        salary_window_txn_df,
    )
    plots_dir = output_path.with_name(f"{output_path.stem}_plots")
    plot_paths = save_salary_timeline_plots(
        enriched_df,
        salary_window_summary_df,
        plots_dir,
        stem_prefix=output_path.stem,
    )
    print(f"Output written to {output_path}")
    if plot_paths:
        print(f"Plots written to {plots_dir}")
        for plot_path in plot_paths:
            print(f"Plot written to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())