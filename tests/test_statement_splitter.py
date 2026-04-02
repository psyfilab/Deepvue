"""Tests for statement splitting helpers."""

from src.statement_splitter import build_particulars_split, split_narration


def test_split_narration_preserves_transfer_phrase() -> None:
    """Transfer prefixes should stay grouped before splitting."""

    parts = split_narration("TO TRANSFER-ARUN-SBIN")
    assert parts[0] == "TO TRANSFER"
    assert parts[1] == "ARUN"


def test_build_particulars_split_joins_non_empty_parts() -> None:
    """Split parts should join with pipes."""

    assert build_particulars_split(["UPI", "DR", "ARUN"]) == "UPI | DR | ARUN"
