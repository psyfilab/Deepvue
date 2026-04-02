"""Run the statement pipeline for every supported file in a folder."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.microsegment import Microsegmenter
    from src.run_statement_pipeline import autofit_worksheet
    from src.statement_classifier import BankStatementClassifier
    from src.statement_splitter import load_statement, split_statement
else:
    from .microsegment import Microsegmenter
    from .run_statement_pipeline import autofit_worksheet
    from .statement_classifier import BankStatementClassifier
    from .statement_splitter import load_statement, split_statement

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".xlsx", ".xls", ".csv"}


def list_statement_files(input_dir: Path) -> list[Path]:
    """Return supported statement files from a folder."""

    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith("~$")
    ]
    return files


def classify_one_file(
    path: Path,
    classifier: BankStatementClassifier,
    microsegmenter: Microsegmenter,
    sheet_name: str,
    header_row: int,
) -> pd.DataFrame:
    """Classify one statement file and append source metadata."""

    statement_df = load_statement(
        path=str(path),
        sheet_name=sheet_name,
        header_row=header_row,
    )
    split_df, _ = split_statement(statement_df)
    classified_df = classifier.classify(split_df)
    enriched_df = microsegmenter.run(classified_df)
    enriched_df.insert(0, "Source_File", path.name)
    enriched_df.insert(1, "Source_Path", str(path))
    return enriched_df


def build_run_summary(
    rows: list[dict[str, object]],
    failures: list[dict[str, str]],
) -> pd.DataFrame:
    """Build a compact summary dataframe for the batch run."""

    summary_rows = rows.copy()
    for failure in failures:
        summary_rows.append(
            {
                "Source_File": failure["Source_File"],
                "Status": "failed",
                "Rows": 0,
                "Message": failure["Message"],
            }
        )
    return pd.DataFrame(summary_rows)


def write_batch_output(
    output_path: Path,
    combined_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Write the combined batch output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        combined_df.to_csv(output_path, index=False)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        combined_df.to_excel(writer, sheet_name="Classified_All", index=False)
        summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)
        autofit_worksheet(writer.sheets["Classified_All"])
        autofit_worksheet(writer.sheets["Run_Summary"])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run the statement pipeline for every file in a folder."
    )
    parser.add_argument(
        "--input-dir",
        default="Raw Data",
        help="Folder containing statement files",
    )
    parser.add_argument(
        "--output",
        default="reports/all_statements_classified.xlsx",
        help="Combined XLSX or CSV output path",
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
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files = list_statement_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No supported statement files found in {input_dir}")

    classifier = BankStatementClassifier(models_dir=args.models)
    microsegmenter = Microsegmenter(master_sheet_path=args.master)

    combined_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for path in files:
        try:
            enriched_df = classify_one_file(
                path=path,
                classifier=classifier,
                microsegmenter=microsegmenter,
                sheet_name=args.sheet_name,
                header_row=args.header_row,
            )
            combined_frames.append(enriched_df)
            summary_rows.append(
                {
                    "Source_File": path.name,
                    "Status": "ok",
                    "Rows": int(len(enriched_df)),
                    "Message": "",
                }
            )
            logger.info("Finished %s with %s rows", path.name, len(enriched_df))
        except Exception as exc:  # pragma: no cover
            failures.append({"Source_File": path.name, "Message": str(exc)})
            logger.exception("Failed to process %s", path.name)

    if not combined_frames:
        raise RuntimeError("No files were processed successfully.")

    combined_df = pd.concat(combined_frames, ignore_index=True)
    summary_df = build_run_summary(summary_rows, failures)
    write_batch_output(Path(args.output), combined_df, summary_df)

    print(f"Processed {len(combined_frames)} files")
    print(f"Combined rows: {len(combined_df)}")
    print(f"Output written to {args.output}")
    if failures:
        print(f"Failed files: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
