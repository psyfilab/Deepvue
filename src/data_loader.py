"""Load and normalize the MASTER_SHEET codebook."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import (
    COMMON_EMPTY_TOKENS,
    COUNT_COLUMN,
    LEVEL3_ALLOWED,
    LEVEL4_DISALLOWED,
    LEVEL6_DISALLOWED,
    REQUIRED_COLUMNS,
    RECURRING_ALLOWED,
    SOURCE_LEVEL_COLUMNS,
)

logger = logging.getLogger(__name__)


def _strip_frame_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with all string values stripped."""

    cleaned = df.copy()
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return cleaned


def _normalize_level_value(value: str, column_name: str) -> str:
    """Normalize a single level cell to a clean lowercase label."""

    normalized = str(value).strip()
    lowered = normalized.lower()

    if lowered in COMMON_EMPTY_TOKENS:
        return ""

    if lowered == column_name.lower():
        return ""

    if column_name == SOURCE_LEVEL_COLUMNS[3] and lowered not in LEVEL3_ALLOWED:
        return ""

    if column_name == SOURCE_LEVEL_COLUMNS[4] and lowered in LEVEL4_DISALLOWED:
        return ""

    if column_name == SOURCE_LEVEL_COLUMNS[6] and lowered in LEVEL6_DISALLOWED:
        return ""

    if column_name == SOURCE_LEVEL_COLUMNS[7] and lowered not in RECURRING_ALLOWED:
        return ""

    return lowered


def load_master(path: str) -> pd.DataFrame:
    """Load the MASTER_SHEET (CSV or XLSX) and normalize its label columns."""

    master_path = Path(path)
    if master_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(master_path, dtype=str)
    else:
        df = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    df.columns = [column.strip() for column in df.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"MASTER_SHEET is missing required columns: {missing}")

    df = _strip_frame_strings(df)

    for level_column in SOURCE_LEVEL_COLUMNS.values():
        df[level_column] = df[level_column].map(
            lambda value, column=level_column: _normalize_level_value(value, column)
        )

    df[COUNT_COLUMN] = (
        pd.to_numeric(df[COUNT_COLUMN], errors="coerce").fillna(0).astype(int)
    )

    logger.info("Loaded %s rows from %s", len(df), master_path)
    return df


def train_test_split_master(
    df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the codebook into train and test partitions with Level 3 stratification."""

    stratify_labels = df[SOURCE_LEVEL_COLUMNS[3]].replace("", "__empty__")
    label_counts = stratify_labels.value_counts()
    stratify = stratify_labels if (label_counts >= 2).all() else None

    if stratify is None:
        logger.warning(
            "Falling back to an unstratified split because at least one Level 3 "
            "bucket has fewer than two rows."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info("Split data into %s train rows and %s test rows", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
