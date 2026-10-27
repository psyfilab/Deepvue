"""Utilities for splitting bank statement narrations into keyword parts.

Part layout (Pipeline Reference, sections 6.1, 9.4 and 9.8): the narration is
split on ``/ - *`` and digits are stripped, giving ``Part_0`` .. ``Part_7``.
For slash-structured narrations every ``/`` segment keeps its own position, so
the reference layout holds for every row:

    TO TRANSFER | UPI | DR | <NAME> | <BANK CODE> | <UPI APP / q> | <TAG>
       Part_0    Part_1 Part_2 Part_3    Part_4         Part_5       Part_6

* The transaction reference -- the first ``/`` segment that is a reference
  number (``506142753794``, ``N20250320065457``) -- is dropped, as in the
  reference examples.
* Any other ``/`` segment that is empty after stripping digits (a phone
  number in the UPI-app slot, a cheque number) is kept as an empty part, so
  the parts after it do not shift left.
"""

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
    if not re.search(r"[A-Z]", value):
        # Only punctuation left (e.g. "@" from "762075197@", ".." from a date).
        return ""
    return value


def is_reference_code(segment: str) -> bool:
    """Return True for a transaction reference such as 506142753794 or N2025..."""

    compact = re.sub(r"[^A-Z0-9]", "", str(segment or "").upper())
    digits = sum(character.isdigit() for character in compact)
    letters = len(compact) - digits
    return digits >= 6 and letters <= 2


def _reference_slots(segments: list[str]) -> set[int]:
    """Return the index of the transaction reference (the first reference code)."""

    for index, segment in enumerate(segments):
        if is_reference_code(segment):
            return {index}
    return set()


def split_narration(text: str, max_parts: int = 8) -> list[str]:
    """Split a raw narration into position-stable uppercase parts."""

    value = "" if text is None else str(text).upper()
    value = re.sub(r"\bTO\s+TRANSFER\s*-", "TO TRANSFER/", value)
    value = re.sub(r"\bBY\s+TRANSFER\s*-", "BY TRANSFER/", value)

    segments = value.split("/")
    reference_slots = _reference_slots(segments) if len(segments) > 1 else set()

    parts: list[str] = []
    for index, segment in enumerate(segments):
        if index in reference_slots:
            continue
        pieces = [normalize_part(piece) for piece in re.split(r"[\-*]", segment)]
        pieces = [piece for piece in pieces if piece]
        if pieces:
            parts.extend(pieces)
        elif len(segments) > 1:
            parts.append("")  # keep the slot so later parts stay in position

    while parts and not parts[-1]:
        parts.pop()
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