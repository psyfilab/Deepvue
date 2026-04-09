"""Run the statement pipeline for every supported file in a folder."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_master
    from src.microsegment import Microsegmenter
    from src.preprocess import KeywordPreprocessor
    from src.run_statement_pipeline import autofit_worksheet
    from src.salary_timeseries import build_salary_time_series, save_salary_timeline_plots
    from src.statement_classifier import BankStatementClassifier, build_salary_summary
    from src.statement_splitter import load_statement, split_statement
else:
    from .data_loader import load_master
    from .microsegment import Microsegmenter
    from .preprocess import KeywordPreprocessor
    from .run_statement_pipeline import autofit_worksheet
    from .salary_timeseries import build_salary_time_series, save_salary_timeline_plots
    from .statement_classifier import BankStatementClassifier, build_salary_summary
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


def build_folder_salary_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Build one combined salary summary sheet for the folder run."""

    return build_salary_summary(combined_df)


def build_folder_salary_time_series(
    combined_df: pd.DataFrame,
    window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build salary-window analytics per source file."""

    group_columns = ("Source_File",)
    if "Source_Path" in combined_df.columns:
        group_columns = ("Source_File", "Source_Path")
    return build_salary_time_series(
        combined_df,
        window_days=window_days,
        group_columns=group_columns,
    )


def build_keyword_review(
    combined_df: pd.DataFrame,
    master_path: str,
    classifier: BankStatementClassifier,
) -> pd.DataFrame:
    """Build a review sheet for keywords missing from the master sheet."""

    part_columns = [column for column in combined_df.columns if column.startswith("Part_")]
    if not part_columns:
        return pd.DataFrame()

    flattened_rows: list[dict[str, str]] = []
    for _, row in combined_df.iterrows():
        for column in part_columns:
            keyword = str(row.get(column, "") or "").strip()
            if keyword:
                flattened_rows.append(
                    {
                        "Source_File": str(row.get("Source_File", "")),
                        "Key Word": keyword,
                    }
                )

    parts_df = pd.DataFrame(flattened_rows)
    if parts_df.empty:
        return pd.DataFrame()

    grouped = (
        parts_df.groupby("Key Word", as_index=False)
        .agg(
            Count=("Key Word", "size"),
            Source_Files=("Source_File", lambda values: ", ".join(sorted(set(values))[:10])),
        )
        .sort_values(["Count", "Key Word"], ascending=[False, True])
        .reset_index(drop=True)
    )

    master_df = load_master(master_path)
    preprocessor = KeywordPreprocessor()
    master_lookup = {
        preprocessor.transform(keyword): keyword
        for keyword in master_df["Key Word"].astype(str).tolist()
    }

    grouped["_normalized"] = grouped["Key Word"].map(preprocessor.transform)
    grouped["In_Master"] = grouped["_normalized"].map(lambda value: int(value in master_lookup))
    grouped["Master_Match"] = grouped["_normalized"].map(lambda value: master_lookup.get(value, ""))

    predictions = classifier.keyword_classifier.classify_batch_with_metadata(
        grouped["Key Word"].tolist()
    )
    predictions_df = pd.DataFrame(predictions)
    review_df = pd.concat([grouped, predictions_df], axis=1)
    review_df = review_df.drop(columns=["_normalized"])
    return review_df


def build_unknown_keyword_review(keyword_review_df: pd.DataFrame) -> pd.DataFrame:
    """Return unseen, weak-confidence keywords for manual review."""

    if keyword_review_df.empty:
        return pd.DataFrame()

    review_df = keyword_review_df.copy()
    unknown_mask = (
        review_df["In_Master"].eq(0)
        & review_df["review_unknown"].astype(int).eq(1)
    )
    unknown_df = review_df.loc[unknown_mask].copy()
    if unknown_df.empty:
        return unknown_df

    unknown_df["Original_Level_0"] = unknown_df["Level_0"]
    unknown_df["Original_Level_1"] = unknown_df["Level_1"]
    unknown_df["Original_Level_2"] = unknown_df["Level_2"]
    unknown_df["Original_Level_3"] = unknown_df["Level_3"]
    unknown_df["Original_Level_4"] = unknown_df["Level_4"]
    unknown_df["Original_Level_5"] = unknown_df["Level_5"]
    unknown_df["Original_Level_6"] = unknown_df["Level_6"]
    unknown_df["Original_Level_7"] = unknown_df["Level_7"]
    unknown_df["Review_Status"] = "unknown"

    unknown_df["Level_0"] = "unknown"
    for column_name in ("Level_1", "Level_2", "Level_3", "Level_4", "Level_5", "Level_6", "Level_7"):
        unknown_df[column_name] = ""

    output_df = unknown_df.rename(columns={"decision_source": "Decision_Status"})
    return output_df[
        [
            "Key Word",
            "Count",
            "Source_Files",
            "Review_Status",
            "Decision_Status",
        ]
    ].reset_index(drop=True)


def write_batch_output(
    output_path: Path,
    combined_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    keyword_review_df: pd.DataFrame,
    unknown_keyword_df: pd.DataFrame,
    salary_summary_df: pd.DataFrame,
    salary_window_summary_df: pd.DataFrame,
    salary_window_txn_df: pd.DataFrame,
) -> None:
    """Write the combined batch output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        combined_df.to_csv(output_path, index=False)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        review_path = output_path.with_name(f"{output_path.stem}_keyword_review.csv")
        keyword_review_df.to_csv(review_path, index=False)
        unknown_path = output_path.with_name(f"{output_path.stem}_unknown_keywords.csv")
        unknown_keyword_df.to_csv(unknown_path, index=False)
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
        combined_df.to_excel(writer, sheet_name="Classified_All", index=False)
        summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)
        keyword_review_df.to_excel(writer, sheet_name="Keyword_Review", index=False)
        unknown_keyword_df.to_excel(writer, sheet_name="Unknown_Keywords", index=False)
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
        autofit_worksheet(writer.sheets["Classified_All"])
        autofit_worksheet(writer.sheets["Run_Summary"])
        autofit_worksheet(writer.sheets["Keyword_Review"])
        autofit_worksheet(writer.sheets["Unknown_Keywords"])
        autofit_worksheet(writer.sheets["Salary_Summary"])
        autofit_worksheet(writer.sheets["Salary_Window_Summary"])
        autofit_worksheet(writer.sheets["Salary_Window_Txns"])


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
    keyword_review_df = build_keyword_review(combined_df, args.master, classifier)
    unknown_keyword_df = build_unknown_keyword_review(keyword_review_df)
    salary_summary_df = build_folder_salary_summary(combined_df)
    salary_window_summary_df, salary_window_txn_df, _ = (
        build_folder_salary_time_series(
            combined_df,
            window_days=args.window_days,
        )
    )
    write_batch_output(
        Path(args.output),
        combined_df,
        summary_df,
        keyword_review_df,
        unknown_keyword_df,
        salary_summary_df,
        salary_window_summary_df,
        salary_window_txn_df,
    )
    plots_dir = Path(args.output).with_name(f"{Path(args.output).stem}_plots")
    plot_paths = save_salary_timeline_plots(
        combined_df,
        salary_window_summary_df,
        plots_dir,
        group_columns=("Source_File", "Source_Path"),
        stem_prefix=Path(args.output).stem,
    )

    print(f"Processed {len(combined_frames)} files")
    print(f"Combined rows: {len(combined_df)}")
    print(f"Keyword review rows: {len(keyword_review_df)}")
    print(f"Unknown keyword rows: {len(unknown_keyword_df)}")
    print(f"Salary summary rows: {len(salary_summary_df)}")
    print(f"Salary window events: {len(salary_window_summary_df)}")
    print(f"Output written to {args.output}")
    if plot_paths:
        print(f"Plots written to {plots_dir}")
        for plot_path in plot_paths:
            print(f"Plot written to {plot_path}")
    if failures:
        print(f"Failed files: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
