"""Project-wide constants for the bank keyword classifier."""

from __future__ import annotations

from typing import Final

SOURCE_KEYWORD_COLUMN: Final[str] = "Key Word"
SOURCE_COLUMN: Final[str] = "Source"
COUNT_COLUMN: Final[str] = "Count"

SOURCE_LEVEL_COLUMNS: Final[dict[int, str]] = {
    0: "Level 0",
    1: "Level 1",
    2: "Level 2",
    3: "Level 3",
    4: "Level 4",
    5: "Level 5",
    6: "Level 6",
    7: "Level 7",
}

OUTPUT_LEVEL_COLUMNS: Final[dict[int, str]] = {
    0: "Level_0",
    1: "Level_1",
    2: "Level_2",
    3: "Level_3",
    4: "Level_4",
    5: "Level_5",
    6: "Level_6",
    7: "Level_7",
}

MODEL_LEVELS: Final[tuple[int, ...]] = (3, 4, 5, 6)
RULE_LEVELS: Final[tuple[int, ...]] = (0, 1, 2, 7)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    SOURCE_COLUMN,
    COUNT_COLUMN,
    SOURCE_KEYWORD_COLUMN,
    *SOURCE_LEVEL_COLUMNS.values(),
)

COMMON_EMPTY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "unknown",
        "other",
        "**purpose**",
        "level 0",
        "level 1",
        "level 2",
        "level 3",
        "level 4",
        "level 5",
        "level 6",
        "level 7",
    }
)

RULE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "debit",
        "credit",
        "upi",
        "imps",
        "neft",
        "rtgs",
        "nach",
        "cheque",
        "cash",
        "auto-debit",
        "recurring",
    }
)

LEVEL3_ALLOWED: Final[frozenset[str]] = frozenset({"p2p", "merchant", "institution"})

GENERIC_PURPOSE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "salary",
        "tax",
        "loan repayment",
        "credit card",
        "mobile recharge",
        "utilities",
        "pharmacy",
        "dining",
        "grocery",
        "subscription",
        "p2p transfer",
        "transport",
        "digital services",
        "retail",
        "services",
        "insurance",
        "financial services",
        "fuel",
        "travel",
        "healthcare",
        "education",
        "refund",
        "bill payment",
        "ecommerce",
    }
)

LEVEL4_DISALLOWED: Final[frozenset[str]] = (
    LEVEL3_ALLOWED
    | RULE_LABELS
    | GENERIC_PURPOSE_LABELS
    | frozenset({"salary", "refund", "place", "duplicate entry", "bank remark", "transfer remark"})
)

LEVEL6_DISALLOWED: Final[frozenset[str]] = (
    LEVEL3_ALLOWED
    | RULE_LABELS
    | GENERIC_PURPOSE_LABELS
    | frozenset({"salary", "refund", "auto-debit", "platform", "system", "income", "expense"})
)

RECURRING_ALLOWED: Final[frozenset[str]] = frozenset({"recurring"})
