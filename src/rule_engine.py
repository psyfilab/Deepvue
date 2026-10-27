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

        # Level 1 is priority-ordered: the first matching rule wins.
        self._level1_rules = (
            # 1. Explicit person-to-person transfers.
            # Lookarounds instead of \b: "_" is a regex word character, so
            # \bp2p\b would miss "P2P_TRANSFER" and "UPI_P2P_RAHUL".
            (
                re.compile(
                    r"(?<![a-z0-9])(?:"
                    r"p2p(?:[\s_-]*transfer)?"
                    r"|person[\s_-]*to[\s_-]*person"
                    r"|peer[\s_-]*to[\s_-]*peer"
                    r")(?![a-z0-9])",
                    flags=re.IGNORECASE,
                ),
                "p2p",
            ),
            # 2. Explicit payment-system markers
            (re.compile(r"\b(?:nach|achdr)\b", re.IGNORECASE), "nach"),
            (re.compile(r"\bupi\b", re.IGNORECASE), "upi"),
            (re.compile(r"\bimps\b", re.IGNORECASE), "imps"),
            (re.compile(r"\bneft\b", re.IGNORECASE), "neft"),
            (re.compile(r"\brtgs\b", re.IGNORECASE), "rtgs"),
            # "CLG" is the cheque-clearing marker (e.g. "CLG/<drawer>/<chq no>/...").
            (re.compile(r"\b(?:cheque|chq|clg)\b", re.IGNORECASE), "cheque"),
            # 3. Automatic debit when no more-specific rail is identified
            (
                re.compile(
                    r"\b(?:autopay|mandate|scheduled|ecs)\b"
                    r"|auto[\s-]*debit"
                    r"|\bstanding[\s-]+instruction\b",
                    flags=re.IGNORECASE,
                ),
                "auto-debit",
            ),
            # 4. Bharat Bill Payment System – match bbps as a substring since
            #    bank narrations frequently concatenate tokens (e.g. "BBPSBP",
            #    "googlebbpsutility", "hdfc bank bbps").
            (
                re.compile(
                    r"bbps"
                    r"|bharat\s+bill\s+pay(?:ment)?"
                    r"|bill\s+pay(?:ment)?\s+system",
                    flags=re.IGNORECASE,
                ),
                "bbps",
            ),
            # 5. Explicit cash transactions
            (re.compile(r"\bcash\b", re.IGNORECASE), "cash"),
        )

        self._level2_rule = re.compile(
            r"\b(?:nach|autopay|mandate|scheduled)\b|auto.?debit|standing instruction",
            flags=re.IGNORECASE,
        )
        self._level7_rule = re.compile(
            r"\b(?:standing\s+instruction|recurring\s+payment|nach|ecs|autodebit|auto[\s-]?debit|autopay|auto[\s-]?pay|upi\s+mandate|si[\s-]?debit)\b|\brecurring\b",
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