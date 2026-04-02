"""Deterministic rules for fully explicit transaction levels."""

from __future__ import annotations

import re

from .constants import OUTPUT_LEVEL_COLUMNS


class RuleEngine:
    """Classify deterministic level values using regex rules."""

    def __init__(self) -> None:
        """Compile rule patterns once for reuse."""

        self._level0_rules = (
            (re.compile(r"\b(?:dr|debit|inward)\b", flags=re.IGNORECASE), "debit"),
            (re.compile(r"\b(?:cr|credit|outward)\b", flags=re.IGNORECASE), "credit"),
        )
        self._level1_rules = (
            (re.compile(r"\bupi\b", flags=re.IGNORECASE), "upi"),
            (re.compile(r"\bimps\b", flags=re.IGNORECASE), "imps"),
            (re.compile(r"\bneft\b", flags=re.IGNORECASE), "neft"),
            (re.compile(r"\brtgs\b", flags=re.IGNORECASE), "rtgs"),
            (re.compile(r"\b(?:nach|achdr)\b", flags=re.IGNORECASE), "nach"),
            (re.compile(r"\b(?:cheque|chq)\b", flags=re.IGNORECASE), "cheque"),
            (re.compile(r"\bcash\b", flags=re.IGNORECASE), "cash"),
        )
        self._level2_rule = re.compile(
            r"\b(?:nach|autopay|mandate|scheduled)\b|auto.?debit|standing instruction",
            flags=re.IGNORECASE,
        )
        self._level7_rule = re.compile(
            r"\b(?:emi|si)\b|recurring|monthly|standing instruction",
            flags=re.IGNORECASE,
        )

    def classify(self, keyword: str) -> dict[str, str]:
        """Return deterministic classifications for Levels 0, 1, 2, and 7."""

        text = "" if keyword is None else str(keyword)
        result = {
            OUTPUT_LEVEL_COLUMNS[0]: "",
            OUTPUT_LEVEL_COLUMNS[1]: "",
            OUTPUT_LEVEL_COLUMNS[2]: "",
            OUTPUT_LEVEL_COLUMNS[7]: "",
        }

        for pattern, label in self._level0_rules:
            if pattern.search(text):
                result[OUTPUT_LEVEL_COLUMNS[0]] = label
                break

        for pattern, label in self._level1_rules:
            if pattern.search(text):
                result[OUTPUT_LEVEL_COLUMNS[1]] = label
                break

        if self._level2_rule.search(text):
            result[OUTPUT_LEVEL_COLUMNS[2]] = "auto-debit"

        if self._level7_rule.search(text):
            result[OUTPUT_LEVEL_COLUMNS[7]] = "recurring"

        return result
