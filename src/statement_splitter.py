"""Utilities for splitting bank statement narrations into keyword parts."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

P2_INDICATOR_RE = re.compile(r"^P2[A-Z]$", flags=re.IGNORECASE)


def normalize_part(part: str) -> str:
    """Normalize one narration part while preserving exact P2 markers."""

    value = "" if part is None else str(part).upper()
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""

    compact = re.sub(r"[^A-Z0-9]", "", value)
    if P2_INDICATOR_RE.fullmatch(compact):
        return compact

    value = re.sub(r"\d+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_narration(text: str, max_parts: int = 8) -> list[str]:
    """Split a raw narration into stable uppercase parts."""

    value = "" if text is None else str(text).upper()
    value = re.sub(r"\bTO\s+TRANSFER\s*-", "TO TRANSFER/", value)
    value = re.sub(r"\bBY\s+TRANSFER\s*-", "BY TRANSFER/", value)

    raw_parts = re.split(r"[/\-*]", value)
    parts = [normalize_part(part) for part in raw_parts]
    parts = [part for part in parts if part]
    return parts[:max_parts]


def build_particulars_split(parts: list[str]) -> str:
    """Join split parts into the pipe-delimited format used downstream."""

    return " | ".join(part for part in parts if part)


def load_statement(
    path: str,
    sheet_name: str = "Bank Statement",
    header_row: int = 4,
) -> pd.DataFrame:
    """Load a bank statement from CSV or Excel."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header_row,
            engine="openpyxl",
        )
    else:
        raise ValueError(f"Unsupported statement format: {file_path.suffix}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Particulars" not in df.columns:
        raise ValueError("Statement file must contain a 'Particulars' column")

    df = df.copy()
    df["Particulars"] = df["Particulars"].fillna("").astype(str).str.strip()
    df = df[df["Particulars"].ne("") & df["Particulars"].str.lower().ne("nan")].copy()
    df.reset_index(drop=True, inplace=True)

    logger.info("Loaded %s statement rows from %s", len(df), file_path)
    return df


def split_statement(
    df: pd.DataFrame,
    particulars_column: str = "Particulars",
    max_parts: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append `Part_*` columns and return unique extracted keywords."""

    if particulars_column not in df.columns:
        raise KeyError(f"Column not found: {particulars_column}")

    output_df = df.copy()
    split_series = output_df[particulars_column].apply(
        lambda text: split_narration(text, max_parts=max_parts)
    )

    for part_index in range(max_parts):
        output_df[f"Part_{part_index}"] = split_series.apply(
            lambda parts, idx=part_index: parts[idx] if idx < len(parts) else ""
        )

    output_df["Particulars_Split"] = split_series.apply(build_particulars_split)

    unique_keywords = sorted(
        {
            part
            for parts in split_series
            for part in parts
            if part
        }
    )
    unique_df = pd.DataFrame({"Unique_Words": unique_keywords})

    logger.info("Split narrations into %s unique keywords", len(unique_df))
    return output_df, unique_df
