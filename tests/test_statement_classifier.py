"""Regression tests for salary-on-debit and ATM/ATW classification."""

from __future__ import annotations

import pandas as pd
# pyrefly: ignore [missing-import]
import pytest

from src.statement_classifier import (
    ATM_WITHDRAWAL_RE,
    BankStatementClassifier,
    apply_raw_transaction_overrides,
    detect_salary,
)

LEVELS = [f"Level_{i}" for i in range(8)]


def blank_levels() -> dict[str, str]:
    return {level: "" for level in LEVELS}


class FakeKeywordClassifier:
    """Returns canned part-level predictions keyed by the part text."""

    def __init__(self, canned: dict[str, dict[str, str]] | None = None) -> None:
        self.canned = canned or {}

    def classify(self, text: str) -> dict[str, str]:
        return {**blank_levels(), **self.canned.get(text, {})}

    def classify_batch(self, texts: list[str]) -> list[dict[str, str]]:
        return [self.classify(text) for text in texts]


def make_classifier(canned: dict[str, dict[str, str]] | None = None) -> BankStatementClassifier:
    classifier = BankStatementClassifier.__new__(BankStatementClassifier)
    classifier.keyword_classifier = FakeKeywordClassifier(canned)
    return classifier


def statement_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a statement frame with the Part_0..Part_7 split columns."""

    frame = pd.DataFrame(rows)
    for index in range(8):
        column = f"Part_{index}"
        if column not in frame.columns:
            frame[column] = ""
    return frame.fillna("")


def labelled_frame(rows: list[dict]) -> pd.DataFrame:
    """Frame shaped like classifier output, for calling detect_salary directly."""

    frame = pd.DataFrame(rows)
    for level in LEVELS:
        if level not in frame.columns:
            frame[level] = ""
    for column in ("First Level Classification", "Second Level Classification"):
        if column not in frame.columns:
            frame[column] = ""
    return frame


# ---------------------------------------------------------------- salary ----


def test_debit_row_with_stale_salary_labels_is_not_salary():
    df = labelled_frame(
        [
            {
                "Date": "2026-03-01",
                "Particulars": "SALARY ADVANCE RECOVERY",
                "Debit": 25000,
                "Credit": "",
                "Level_5": "salary",
                "First Level Classification": "Salary",
            }
        ]
    )

    out = detect_salary(df)

    assert out.loc[0, "IsSalary"] == 0
    assert out.loc[0, "SalaryConfidence"] == 0.0
    assert out.loc[0, "First Level Classification"] != "Salary"
    assert out.loc[0, "Level_5"] != "salary"


def test_debit_with_salary_text_and_large_amount_is_not_salary():
    df = labelled_frame(
        [
            {
                "Date": "2026-03-01",
                "Particulars": "PAYROLL REVERSAL SALARY",
                "Debit": 80000,
                "Credit": "",
            }
        ]
    )

    out = detect_salary(df)

    assert out.loc[0, "IsSalary"] == 0
    assert out.loc[0, "SalaryConfidence"] == 0.0


def test_valid_explicit_salary_credit_is_still_flagged():
    df = labelled_frame(
        [
            {
                "Date": "2026-03-01",
                "Particulars": "NEFT SALARY ACME CORP",
                "Debit": "",
                "Credit": 50000,
            }
        ]
    )

    out = detect_salary(df)

    assert out.loc[0, "IsSalary"] == 1
    assert out.loc[0, "SalaryConfidence"] == 1.0
    assert out.loc[0, "Level_5"] == "salary"
    assert out.loc[0, "First Level Classification"] == "Salary"


def test_recurring_credit_scoring_is_unchanged():
    # Three similar credits on similar days, no salary text: 0.40 + 0.30 = 0.70.
    df = labelled_frame(
        [
            {"Date": "2026-01-01", "Particulars": "NEFT ACME", "Debit": "", "Credit": 30000},
            {"Date": "2026-02-02", "Particulars": "NEFT ACME", "Debit": "", "Credit": 30000},
            {"Date": "2026-03-01", "Particulars": "NEFT ACME", "Debit": "", "Credit": 30000},
        ]
    )

    out = detect_salary(df)

    assert out["IsSalary"].tolist() == [1, 1, 1]
    assert out["SalaryConfidence"].tolist() == [0.7, 0.7, 0.7]


def test_recurring_debits_are_never_salary():
    df = labelled_frame(
        [
            {"Date": "2026-01-01", "Particulars": "RENT", "Debit": 30000, "Credit": ""},
            {"Date": "2026-02-01", "Particulars": "RENT", "Debit": 30000, "Credit": ""},
            {"Date": "2026-03-01", "Particulars": "RENT", "Debit": 30000, "Credit": ""},
        ]
    )

    out = detect_salary(df)

    assert out["IsSalary"].tolist() == [0, 0, 0]
    assert out["SalaryConfidence"].tolist() == [0.0, 0.0, 0.0]


def test_mixed_frame_only_credit_is_flagged():
    df = labelled_frame(
        [
            {"Date": "2026-03-01", "Particulars": "SALARY MARCH ACME", "Debit": "", "Credit": 60000},
            {"Date": "2026-03-02", "Particulars": "SALARY ADVANCE REPAID", "Debit": 20000, "Credit": ""},
        ]
    )

    out = detect_salary(df)

    assert out["IsSalary"].tolist() == [1, 0]
    assert out["SalaryConfidence"].tolist() == [1.0, 0.0]


def test_raw_override_skips_salary_for_debit_but_keeps_it_for_credit():
    text = "SALARY ADVANCE RECOVERY"

    debit = apply_raw_transaction_overrides(text, blank_levels(), None, 25000, "")
    credit = apply_raw_transaction_overrides(text, blank_levels(), None, "", 25000)
    legacy = apply_raw_transaction_overrides(text, blank_levels())

    assert debit["Level_5"] != "salary"
    assert credit["Level_5"] == "salary"
    assert legacy["Level_5"] == "salary"  # callers without amounts behave as before


def test_pipeline_debit_with_salary_keyword_is_not_salary():
    classifier = make_classifier({"SALARY": {"Level_5": "salary", "Level_3": "institution"}})
    df = statement_frame(
        [
            {
                "Date": "2026-03-05",
                "Particulars": "SALARY ADVANCE RECOVERY",
                "Debit": 25000,
                "Credit": "",
                "Part_0": "SALARY",
                "Part_1": "ADVANCE",
            }
        ]
    )

    out = classifier.classify(df)

    assert out.loc[0, "IsSalary"] == 0
    assert out.loc[0, "SalaryConfidence"] == 0.0
    assert out.loc[0, "Level_5"] != "salary"
    assert out.loc[0, "First Level Classification"] != "Salary"
    assert out.loc[0, "Level_0"] == "debit"


def test_pipeline_salary_credit_is_still_classified_as_salary():
    classifier = make_classifier({"SALARY": {"Level_5": "salary", "Level_3": "institution"}})
    df = statement_frame(
        [
            {
                "Date": "2026-03-05",
                "Particulars": "SALARY MARCH ACME CORP",
                "Debit": "",
                "Credit": 55000,
                "Part_0": "SALARY",
                "Part_1": "ACME CORP",
            }
        ]
    )

    out = classifier.classify(df)

    assert out.loc[0, "IsSalary"] == 1
    assert out.loc[0, "SalaryConfidence"] == 1.0
    assert out.loc[0, "Level_5"] == "salary"
    assert out.loc[0, "First Level Classification"] == "Salary"
    assert out.loc[0, "Level_0"] == "credit"
    # Level_3 is deliberately not asserted: salary purpose alone does not
    # establish the counterparty type.


def test_salary_credit_does_not_overwrite_existing_level3():
    classifier = make_classifier(
        {
            "SALARY": {"Level_5": "salary"},
            "RAHUL": {"Level_3": "p2p"},
        }
    )
    df = statement_frame(
        [
            {
                "Date": "2026-03-05",
                "Particulars": "SALARY MARCH RAHUL",
                "Debit": "",
                "Credit": 55000,
                "Part_0": "SALARY",
                "Part_1": "RAHUL",
            }
        ]
    )

    out = classifier.classify(df)

    assert out.loc[0, "IsSalary"] == 1
    assert out.loc[0, "Level_5"] == "salary"
    assert out.loc[0, "Level_3"] == "p2p"


# ------------------------------------------------------------------- ATM ----


@pytest.mark.parametrize(
    "narration",
    [
        "ATW-512345XXXXXX6789-SBIN0001234-HYDERABAD",
        "NFS/ATW/512345XXXXXX6789/HYDERABAD",
        "ATW | 512345XXXXXX6789 | HYDERABAD",
        "atw cash 5000",
        "ATM WDL 512345XXXXXX6789",
        "ATM-WDL KUKATPALLY",
        "ATM/DR/512345",
        "ATM | DR | 512345",
    ],
)
def test_atm_regex_matches_withdrawal_narrations(narration):
    assert ATM_WITHDRAWAL_RE.search(narration)


@pytest.mark.parametrize(
    "narration",
    [
        "UPI/DR/123/ATWOOD STORES/HDFC/atwood@okhdfc",
        "BATWING RESTAURANT",
        "ATWELL SERVICES",
        "NEFT RENT PAYMENT",
    ],
)
def test_atm_regex_ignores_unrelated_narrations(narration):
    assert not ATM_WITHDRAWAL_RE.search(narration)


@pytest.mark.parametrize(
    "narration, parts",
    [
        (
            "ATW-512345XXXXXX6789-SBIN0001234-HYDERABAD",
            ["ATW", "512345XXXXXX6789", "SBIN0001234", "HYDERABAD"],
        ),
        ("NFS/ATW/512345XXXXXX6789/HYDERABAD", ["NFS", "ATW", "512345XXXXXX6789", "HYDERABAD"]),
        ("ATM WDL 512345XXXXXX6789 HYDERABAD", ["ATM WDL", "512345XXXXXX6789", "HYDERABAD"]),
    ],
)
def test_atm_debit_is_cash_withdrawal_and_clears_contradictory_labels(narration, parts):
    canned = {
        # Part-level model guesses that contradict a cash withdrawal.
        parts[-1]: {"Level_3": "p2p", "Level_5": "p2p transfer", "Level_6": "hyderabad"},
        parts[1]: {"Level_3": "merchant"},
    }
    classifier = make_classifier(canned)
    row = {"Date": "2026-04-10", "Particulars": narration, "Debit": 5000, "Credit": ""}
    row.update({f"Part_{i}": part for i, part in enumerate(parts)})

    out = classifier.classify(statement_frame([row]))

    assert out.loc[0, "Level_1"] == "cash"
    assert out.loc[0, "First Level Classification"] == "Cash Withdrawal"
    assert out.loc[0, "Level_3"] == ""
    assert out.loc[0, "Level_5"] == ""
    assert out.loc[0, "Level_0"] == "debit"
    assert out.loc[0, "IsSalary"] == 0


def test_atm_withdrawal_overrides_upi_and_salary_labels():
    classifier = make_classifier(
        {"ATW": {"Level_1": "upi", "Level_5": "salary", "Level_3": "institution"}}
    )
    row = {
        "Date": "2026-04-10",
        "Particulars": "ATW-512345XXXXXX6789-HYDERABAD",
        "Debit": 10000,
        "Credit": "",
        "Part_0": "ATW",
        "Part_1": "512345XXXXXX6789",
    }

    out = classifier.classify(statement_frame([row]))

    assert out.loc[0, "Level_1"] == "cash"
    assert out.loc[0, "Level_3"] == ""
    assert out.loc[0, "Level_5"] == ""
    assert out.loc[0, "First Level Classification"] == "Cash Withdrawal"
    assert out.loc[0, "IsSalary"] == 0


def test_atw_credit_reversal_is_not_a_cash_withdrawal():
    classifier = make_classifier()
    row = {
        "Date": "2026-04-11",
        "Particulars": "REV ATW-512345XXXXXX6789-HYDERABAD",
        "Debit": "",
        "Credit": 5000,
        "Part_0": "REV",
        "Part_1": "ATW",
    }

    out = classifier.classify(statement_frame([row]))

    assert out.loc[0, "Level_1"] != "cash"
    assert out.loc[0, "First Level Classification"] != "Cash Withdrawal"


# ------------------------------------------- retail label on ATM rows -------

from src.statement_classifier import apply_atm_withdrawal_overrides  # noqa: E402


@pytest.mark.parametrize("narration", [
    "ATW-512345XXXXXX6789-SBIN0001234-HYDERABAD",
    "NFS/ATW/512345XXXXXX6789/HYDERABAD",
])
def test_atw_debit_with_p2p_and_retail_labels_is_clean_cash_withdrawal(narration):
    parts = [p for p in narration.replace("/", "-").split("-") if p]
    canned = {
        parts[0]: {"Level_3": "p2p", "Level_5": "p2p transfer"},
        parts[1]: {"Level_3": "merchant", "Level_5": "retail"},
    }
    row = {"Date": "2026-04-10", "Particulars": narration, "Debit": 5000, "Credit": ""}
    row.update({f"Part_{i}": p for i, p in enumerate(parts)})

    out = make_classifier(canned).classify(statement_frame([row]))

    assert out.loc[0, "Level_1"] == "cash"
    assert out.loc[0, "First Level Classification"] == "Cash Withdrawal"
    assert out.loc[0, "Level_3"] == ""
    assert out.loc[0, "Level_5"] == ""
    assert out.loc[0, "Second Level Classification"] == ""


def test_atw_debit_with_only_retail_level5_is_cleared():
    canned = {"ATW": {"Level_5": "retail"}}
    row = {
        "Date": "2026-04-10",
        "Particulars": "ATW-512345XXXXXX6789-HYDERABAD",
        "Debit": 2000,
        "Credit": "",
        "Part_0": "ATW",
        "Part_1": "512345XXXXXX6789",
    }

    out = make_classifier(canned).classify(statement_frame([row]))

    assert out.loc[0, "Level_5"] == ""
    assert out.loc[0, "Level_1"] == "cash"


def test_atm_override_clears_retail_directly_and_leaves_credit_alone():
    result = {**blank_levels(), "Level_5": "retail", "Level_3": "merchant"}
    cleared = apply_atm_withdrawal_overrides("ATW-1234", [("ATW", {})], dict(result), 500, "")
    untouched = apply_atm_withdrawal_overrides("REV ATW-1234", [("ATW", {})], dict(result), "", 500)

    assert cleared["Level_1"] == "cash"
    assert cleared["Level_5"] == "" and cleared["Level_3"] == ""
    assert untouched["Level_5"] == "retail" and untouched["Level_1"] == ""


def test_non_atm_row_keeps_retail_label():
    canned = {"AMAZON": {"Level_3": "merchant", "Level_5": "retail"}}
    row = {
        "Date": "2026-04-12",
        "Particulars": "POS AMAZON",
        "Debit": 999,
        "Credit": "",
        "Part_0": "POS",
        "Part_1": "AMAZON",
    }

    out = make_classifier(canned).classify(statement_frame([row]))

    assert out.loc[0, "Level_5"] == "retail"
    assert out.loc[0, "Level_1"] != "cash"


def test_mixed_statement_atm_debits_clean_and_salary_only_on_credits():
    canned = {
        "ATW": {"Level_3": "p2p", "Level_5": "retail"},
        "SALARY": {"Level_5": "salary", "Level_3": "institution"},
    }
    rows = [
        {"Date": "2026-04-01", "Particulars": "SALARY APRIL ACME", "Debit": "", "Credit": 60000,
         "Part_0": "SALARY", "Part_1": "ACME"},
        {"Date": "2026-04-02", "Particulars": "ATW-512345XXXXXX6789-HYDERABAD", "Debit": 10000, "Credit": "",
         "Part_0": "ATW", "Part_1": "512345XXXXXX6789"},
        {"Date": "2026-04-03", "Particulars": "SALARY ADVANCE REPAID", "Debit": 15000, "Credit": "",
         "Part_0": "SALARY", "Part_1": "ADVANCE"},
    ]

    out = make_classifier(canned).classify(statement_frame(rows))

    assert out["IsSalary"].tolist() == [1, 0, 0]
    assert out["SalaryConfidence"].tolist() == [1.0, 0.0, 0.0]
    assert out.loc[1, "Level_1"] == "cash"
    assert out.loc[1, "First Level Classification"] == "Cash Withdrawal"
    assert out.loc[1, "Level_3"] == "" and out.loc[1, "Level_5"] == ""
    assert out.loc[2, "First Level Classification"] != "Salary"


# ------------------------- hyphen-format UPI merchants (Amazon, Zomato) -------

from src.statement_classifier import (  # noqa: E402
    aggregate_part_predictions,
    apply_strict_level2,
    derive_upi_bank,
    is_payment_intermediary_label,
    KNOWN_MERCHANT_FRAGMENTS,
    PAYMENT_INTERMEDIARY_FRAGMENTS,
    extract_hyphen_upi_merchant,
)

AMAZON_NARRATION = "UPI-YESBANK-AMAZON@YAPL-YESB0APLUPI-4093 68524272-REQUEST FROM AMAZO"
ZOMATO_NARRATION = "UPI-RAZORPAYZOMATO-ZOMATOINDIA@ICICI-ICI C0DC0099-409799883968-PAYVIARAZORPAY"

# Bank-looking parts get institution + a bank, which is what produced the bug.
BANKISH_PARTS = {
    "UPI": {"Level_1": "upi"},
    "YESBANK": {"Level_3": "institution", "Level_4": "yes bank"},
    "YESB0APLUPI": {"Level_4": "yes bank"},
    "RAZORPAYZOMATO": {"Level_3": "institution", "Level_4": "razorpay"},
    "ZOMATOINDIA@ICICI": {"Level_4": "icici bank"},
    "ICI C0DC0099": {"Level_3": "institution", "Level_4": "icici bank"},
}


def upi_row(narration: str, debit=450, credit="") -> dict:
    row = {"Date": "2026-04-01", "Particulars": narration, "Debit": debit, "Credit": credit}
    row.update({f"Part_{i}": part for i, part in enumerate(narration.split("-")[:8])})
    return row


def test_amazon_hyphen_upi_is_retail_merchant_with_level6():
    out = make_classifier(BANKISH_PARTS).classify(statement_frame([upi_row(AMAZON_NARRATION)]))

    assert out.loc[0, "Level_3"] == "merchant"
    assert out.loc[0, "Level_5"] == "retail"
    assert out.loc[0, "Level_6"] == "amazon"
    assert out.loc[0, "Level_4"] == "yes bank"  # valid bank is preserved
    assert out.loc[0, "Level_1"] == "upi"
    assert out.loc[0, "First Level Classification"] == "UPI Payment"
    assert out.loc[0, "IsSalary"] == 0


# The incorrect original: the gateway part is the only one carrying Level_4, and
# no part predicts the real bank, so the correct bank can only come from the
# narration itself (IFSC "ICI C0DC0099" -> ICIC, handle "@ICICI").
ZOMATO_RAZORPAY_ONLY_PARTS = {
    "UPI": {"Level_1": "upi"},
    "RAZORPAYZOMATO": {"Level_3": "institution", "Level_4": "razorpay"},
}


def test_zomato_hyphen_upi_replaces_razorpay_level4_with_the_real_bank():
    classifier = make_classifier(ZOMATO_RAZORPAY_ONLY_PARTS)

    # Reproduce the original incorrect classification before any merchant override.
    part_predictions = [
        (part, classifier.keyword_classifier.classify(part))
        for part in ZOMATO_NARRATION.split("-")
    ]
    original = aggregate_part_predictions(part_predictions)
    assert original["Level_3"] == "institution"
    assert original["Level_4"] == "razorpay"
    assert original["Level_5"] == "" and original["Level_6"] == ""

    out = classifier.classify(statement_frame([upi_row(ZOMATO_NARRATION)]))

    assert out.loc[0, "Level_3"] == "merchant"
    assert out.loc[0, "Level_5"] == "dining"
    assert out.loc[0, "Level_6"] == "zomato"
    assert out.loc[0, "Level_6"] != "razorpay"
    assert out.loc[0, "Level_4"] == "icici bank"  # a bank, derived deliberately
    assert out.loc[0, "Level_4"] != "razorpay"
    assert out.loc[0, "First Level Classification"] == "UPI Payment"


def test_zomato_razorpay_level4_prefers_a_bank_the_model_predicted_on_another_part():
    # BANKISH_PARTS has razorpay on the gateway part and icici bank on the VPA part.
    out = make_classifier(BANKISH_PARTS).classify(statement_frame([upi_row(ZOMATO_NARRATION)]))

    assert out.loc[0, ["Level_3", "Level_4", "Level_5", "Level_6"]].tolist() == [
        "merchant", "icici bank", "dining", "zomato",
    ]


def test_valid_model_bank_is_preserved_even_if_narration_names_another_bank():
    canned = {**ZOMATO_RAZORPAY_ONLY_PARTS, "RAZORPAYZOMATO": {"Level_3": "institution", "Level_4": "state bank of india"}}
    out = make_classifier(canned).classify(statement_frame([upi_row(ZOMATO_NARRATION)]))

    assert out.loc[0, "Level_4"] == "state bank of india"
    assert out.loc[0, "Level_6"] == "zomato"


def test_razorpay_level4_is_cleared_when_no_bank_can_be_identified():
    narration = "UPI-RAZORPAYSWIGGY-SWIGGY@OKAXIS-123456-NOTE"  # no IFSC, handle is not a bank code
    canned = {"UPI": {"Level_1": "upi"}, "RAZORPAYSWIGGY": {"Level_3": "institution", "Level_4": "razorpay"}}
    out = make_classifier(canned).classify(statement_frame([upi_row(narration)]))

    assert out.loc[0, "Level_4"] == ""
    assert out.loc[0, ["Level_3", "Level_5", "Level_6"]].tolist() == ["merchant", "dining", "swiggy"]


@pytest.mark.parametrize(
    "narration, bank",
    [
        (ZOMATO_NARRATION, "icici bank"),  # IFSC split with a space: ICI C0DC0099
        (AMAZON_NARRATION, "yes bank"),  # handle @YAPL is not a bank, IFSC YESB is
        ("UPI-X-Y@HDFCBANK-123456", "hdfc bank"),  # handle only, prefix match
        ("UPI-X-Y@OKAXIS-UTIB0000001-123456", "axis bank"),  # IFSC only
        ("UPI-X-Y@YBL-123456", ""),
        ("", ""),
    ],
)
def test_derive_upi_bank(narration, bank):
    assert derive_upi_bank(narration) == bank


@pytest.mark.parametrize(
    "label, expected",
    [
        ("razorpay", True),
        ("Google Pay", True),
        ("phonepe", True),
        ("paytm", False),  # a bank value in BANK_CODE_TO_NAME
        ("icici bank", False),
        ("yes bank", False),
        ("", False),
    ],
)
def test_is_payment_intermediary_label(label, expected):
    assert is_payment_intermediary_label(label) is expected


def test_keyword_model_purpose_for_the_merchant_name_wins_over_default():
    canned = {**BANKISH_PARTS, "amazon": {"Level_5": "shopping"}}
    out = make_classifier(canned).classify(statement_frame([upi_row(AMAZON_NARRATION)]))

    assert out.loc[0, "Level_5"] == "shopping"
    assert out.loc[0, "Level_6"] == "amazon"


@pytest.mark.parametrize(
    "narration, merchant",
    [
        (AMAZON_NARRATION, "amazon"),
        (ZOMATO_NARRATION, "zomato"),
        ("UPI-SWIGGY-SWIGGY.STORES@AXISBANK-UTIB0000001-123456-ORDER", "swiggy"),
        ("UPI-FLIPKART@YBL-YESB0YBLUPI-123456-PAYMENT", "flipkart"),  # VPA only, no payee segment
        ("UPI-RAZORPAYSWIGGY-RAZORPAY@ICICI-ICIC0000001-123456-NOTE", "swiggy"),  # merchant in payee
        ("upi-zomato-zomato@hdfcbank-hdfc0000001-123456-note", "zomato"),  # lowercase
    ],
)
def test_extract_hyphen_upi_merchant_finds_actual_merchant(narration, merchant):
    assert extract_hyphen_upi_merchant(narration) == merchant


@pytest.mark.parametrize(
    "narration",
    [
        "UPI-RAZORPAY-RAZORPAY@ICICI-ICIC0000001-123456-NOTE",  # intermediary only
        "UPI-RAHUL KUMAR-RAHUL@OKAXIS-UTIB0000001-123456-PAYVIARAZORPAY ZOMATO",  # brand only in note
        "UPI-P2A-AMAZON@YAPL-YESB0APLUPI-123456-NOTE",  # explicit P2P marker owns the row
        "UPI/DR/123456/AMAZON/YESB/amazon@yapl",  # slash format has its own rule
        "NEFT-AMAZON@YAPL-123456",  # not UPI
        "UPI-YESBANK-AMAZON-YESB0APLUPI-123456",  # no VPA handle to read
        "",
    ],
)
def test_extract_hyphen_upi_merchant_ignores_non_merchant_narrations(narration):
    assert extract_hyphen_upi_merchant(narration) == ""


def test_intermediaries_are_never_in_the_merchant_set():
    assert not (PAYMENT_INTERMEDIARY_FRAGMENTS & KNOWN_MERCHANT_FRAGMENTS)
    assert {"amazon", "zomato", "swiggy", "flipkart"} <= KNOWN_MERCHANT_FRAGMENTS
    assert {"razorpay", "paytm", "phonepe", "payu"} <= PAYMENT_INTERMEDIARY_FRAGMENTS


def test_person_to_person_hyphen_upi_is_left_unchanged():
    narration = "UPI-RAHUL KUMAR-RAHUL@OKAXIS-UTIB0000001-123456-DINNER"
    canned = {"RAHUL KUMAR": {"Level_3": "p2p", "Level_5": "p2p transfer", "Level_6": "rahul kumar"}}
    out = make_classifier(canned).classify(statement_frame([upi_row(narration)]))

    assert out.loc[0, "Level_3"] == "p2p"
    assert out.loc[0, "Level_5"] == "p2p transfer"
    assert out.loc[0, "Level_6"] == "rahul kumar"


def test_slash_format_upi_merchant_behaviour_is_unchanged():
    narration = "UPI/DR/123456/AMAZON/YESB/amazon@yapl"
    row = {"Date": "2026-04-01", "Particulars": narration, "Debit": 450, "Credit": "",
           "Part_0": "UPI", "Part_1": "DR"}
    out = make_classifier().classify(statement_frame([row]))

    assert out.loc[0, "Level_3"] == "merchant"
    assert out.loc[0, "Level_6"] == "amazon"
    assert out.loc[0, "Level_4"] == "yes bank"


def test_merchant_survives_later_rules_in_a_mixed_statement():
    rows = [
        upi_row(AMAZON_NARRATION),
        upi_row(ZOMATO_NARRATION, debit=300),
        {"Date": "2026-04-02", "Particulars": "ATW-512345XXXXXX6789-HYDERABAD", "Debit": 5000, "Credit": "",
         "Part_0": "ATW", "Part_1": "512345XXXXXX6789"},
        {"Date": "2026-04-03", "Particulars": "SALARY APRIL ACME", "Debit": "", "Credit": 60000,
         "Part_0": "SALARY", "Part_1": "ACME"},
    ]
    canned = {**BANKISH_PARTS, "SALARY": {"Level_5": "salary", "Level_3": "institution"}}
    out = make_classifier(canned).classify(statement_frame(rows))

    assert out.loc[0, ["Level_3", "Level_5", "Level_6"]].tolist() == ["merchant", "retail", "amazon"]
    assert out.loc[1, ["Level_3", "Level_5", "Level_6"]].tolist() == ["merchant", "dining", "zomato"]
    assert out.loc[2, "Level_1"] == "cash" and out.loc[2, "First Level Classification"] == "Cash Withdrawal"
    assert out["IsSalary"].tolist() == [0, 0, 0, 1]


def test_amazon_credit_refund_is_merchant_and_not_salary():
    out = make_classifier(BANKISH_PARTS).classify(
        statement_frame([upi_row(AMAZON_NARRATION, debit="", credit=15000)])
    )

    assert out.loc[0, "Level_3"] == "merchant"
    assert out.loc[0, "Level_6"] == "amazon"
    assert out.loc[0, "Level_0"] == "credit"
    assert out.loc[0, "IsSalary"] == 0


# --------------------- slash-UPI purpose, ACH, autopay, strict Level_2 -------


def test_slash_upi_preserves_merchant_purpose_instead_of_forcing_p2p():
    result = blank_levels()
    result.update({
        "Level_1": "upi",
        "Level_3": "p2p",
        "Level_5": "digital services",
    })

    result = apply_raw_transaction_overrides(
        "UPI/DR/123/KE TTO/HDFC/merchant@unknown",
        result,
        debit_value=500,
        credit_value=0,
    )

    assert result["Level_3"] == "merchant"
    assert result["Level_5"] == "digital services"


def test_ambiguous_slash_upi_is_not_confidently_labelled_p2p():
    result = blank_levels()
    result.update({
        "Level_1": "upi",
        "Level_3": "p2p",
        "Level_5": "p2p transfer",
    })

    result = apply_raw_transaction_overrides(
        "UPI/DR/123/UNKNOWN/HDFC/person@unknown",
        result,
        debit_value=200,
        credit_value=0,
    )

    assert result["Level_3"] == ""
    assert result["Level_5"] == ""


def test_ach_credit_does_not_force_institution_counterparty():
    result = blank_levels()
    result.update({
        "Level_3": "p2p",
        "Level_5": "p2p transfer",
    })

    result = apply_raw_transaction_overrides(
        "ACH C-HCL DIVIDEND",
        result,
        debit_value=0,
        credit_value=1000,
    )

    # ACH alone must not overwrite the counterparty type.
    assert result["Level_3"] == "p2p"
    assert result["Level_5"] == "dividend"


def test_autopay_debit_does_not_force_institution_counterparty():
    result = blank_levels()
    result.update({
        "Level_3": "p2p",
        "Level_5": "p2p transfer",
    })

    result = apply_raw_transaction_overrides(
        "CC XXXX9920 AUTOPAY SI-MAD",
        result,
        debit_value=2500,
        credit_value=0,
    )

    # Automation alone must not overwrite the counterparty type.
    assert result["Level_3"] == "p2p"
    assert result["Level_5"] == ""


def test_strict_level2_recognizes_ach_and_autopay():
    df = pd.DataFrame({
        "Particulars": [
            "ACH C-HCL DIVIDEND",
            "CC XXXX9920 AUTOPAY SI-MAD",
        ]
    })

    result = apply_strict_level2(df)

    assert result.loc[0, "Level_2"] == "System generated"
    assert result.loc[1, "Level_2"] == "System generated"


def test_strict_level2_defaults_ordinary_transactions_to_user_generated():
    df = pd.DataFrame({
        "Particulars": [
            "LOCAL SHOP PAYMENT",
            "NEFT TRANSFER TO FRIEND",
            "ATM CASH WITHDRAWAL",
        ],
        "IsSalary": [0, 0, 0],
    })

    result = apply_strict_level2(df)

    assert result["Level_2"].tolist() == [
        "User-generated",
        "User-generated",
        "User-generated",
    ]


# =============================================================================
# Level 5 category classification – regression tests
# =============================================================================

from src.statement_classifier import (  # noqa: E402
    NARRATION_LEVEL5_PATTERNS,
    apply_narration_purpose_override,
)


def _level5(particulars: str, debit=0, credit=0, preset_level5="") -> str:
    """Helper: run apply_narration_purpose_override and return Level 5."""
    result = blank_levels()
    result["Level_5"] = preset_level5
    out = apply_narration_purpose_override(
        particulars,
        result,
        debit_value=debit if debit else None,
        credit_value=credit if credit else None,
    )
    return out["Level_5"]


# ─── narration_purpose_override never overwrites an existing Level 5 ─────────


def test_existing_level5_is_preserved():
    """apply_narration_purpose_override must not overwrite an already-set value."""
    assert _level5("ZOMATO PAYMENT", debit=500, preset_level5="retail") == "retail"
    assert _level5("HOTEL SELECT PAYMENT", debit=3000, preset_level5="services") == "services"


# ─── ATM cash rows are untouched ──────────────────────────────────────────────


def test_atm_row_level5_not_set_by_narration_override():
    """apply_narration_purpose_override skips cash (ATM) rows."""
    result = blank_levels()
    result["Level_1"] = "cash"
    out = apply_narration_purpose_override("ATW-512345XXXXXX6789-HYDERABAD", result, debit_value=5000)
    assert out["Level_5"] == ""


# ─── Dining ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "ZOMATO FOOD DELIVERY",
        "SWIGGY ORDER PAYMENT",
        "HOTEL SELECT DINING",
        "SARAVANA BHAVA HOTEL",
        "ATITHI DHABA AND CATERERS",
        "PESHAWAR RESTAURANT PAYMENT",
        "BURGER KING ORDER",
        "FROZEN BOTTLE PAYMENT",
        "DYU ART CAFE",
        "UPI/DR/123/PIZZAHUT/HDFC/pizza@hdfc",
    ],
)
def test_narration_purpose_dining(narration):
    assert _level5(narration, debit=500) == "dining"


# ─── Grocery ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "BLINKIT GROCERY ORDER",
        "ZEPTO FRESH DELIVERY",
        "BIGBASKET WEEKLY GROCERIES",
        "JIOMART KIRANA PURCHASE",
        "SWIGGY INSTAMART PAYMENT",
    ],
)
def test_narration_purpose_grocery(narration):
    assert _level5(narration, debit=800) == "grocery"


# ─── Fuel ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "HPCL PETROL PUMP PAYMENT",
        "IOCL FUEL STATION",
        "BPCL PETROL PURCHASE",
        "UPI/DR/123/PETROL PUMP/HDFC/fuel@hdfc",
    ],
)
def test_narration_purpose_fuel(narration):
    assert _level5(narration, debit=2000) == "fuel"


# ─── Travel ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "MAKEMYTRIP FLIGHT BOOKING",
        "GOIBIBO HOTEL BOOKING",
        "OYO ROOMS HOTEL STAY",
        "CLEARTRIP AIRLINE TICKET",
        "YATRA TOUR PACKAGE",
    ],
)
def test_narration_purpose_travel(narration):
    assert _level5(narration, debit=5000) == "travel"


# ─── Transport ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "UBER INDIA RIDES PAYMENT",
        "OLA CABS TRIP",
        "RAPIDO BIKE TAXI",
        "IRCTC RAIL TICKET BOOKING",
        "WTI CABS PAYMENT",
    ],
)
def test_narration_purpose_transport(narration):
    assert _level5(narration, debit=300) == "transport"


# ─── Pharmacy ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "PHARMEASY MEDICINE ORDER",
        "MEDPLUS PHARMACY",
        "NETMEDS ONLINE PHARMACY",
        "APOLLO PHARMACY PURCHASE",
        "AYURVEDA KENDRA PAYMENT",
        "PREET MEDICOS GENERAL STORE",
        "SUPRABHO MEDI STORE",
        "ARUL MEDICALS",
    ],
)
def test_narration_purpose_pharmacy(narration):
    assert _level5(narration, debit=500) in {"pharmacy", "healthcare"}


# ─── Healthcare ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "APOLLO HOSPITAL PAYMENT",
        "DELFINIUM FERTILITY CLINIC",
        "PATH LAB DIAGNOSTIC CENTRE",
        "AGS MEDICARE PRIVATE",
        "MS SON X MEDICAL CENTER",
    ],
)
def test_narration_purpose_healthcare(narration):
    assert _level5(narration, debit=1500) in {"healthcare", "pharmacy"}


# ─── Loan repayment ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "HDFC BANK HOME LOAN EMI",
        "SBI PERSONAL LOAN REPAYMENT",
        "ICICI BANK CAR LOAN EMI",
        "EDUCATION LOAN INSTALLMENT",
        "LOAN RECOVERY DEBIT",
    ],
)
def test_narration_purpose_loan_repayment(narration):
    assert _level5(narration, debit=15000) == "loan repayment"


# ─── Credit card payment ──────────────────────────────────────────────────────


def test_narration_purpose_credit_card_debit():
    """A credit-card bill payment is a debit."""
    assert _level5("HDFC CREDIT CARD PAYMENT", debit=12000) == "credit card"


def test_narration_purpose_credit_card_credit_ignored():
    """A credit from a credit-card issuer is not a bill payment."""
    assert _level5("ICICI CREDIT CARD CASHBACK", credit=500) != "credit card"


# ─── Insurance ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "LIC PREMIUM PAYMENT",
        "ACKO HEALTH INSURANCE",
        "STAR HEALTH INSURANCE PREMIUM",
        "BAJAJ ALLIANZ POLICY PREMIUM",
        "HDFC ERGO INSURANCE",
    ],
)
def test_narration_purpose_insurance(narration):
    assert _level5(narration, debit=5000) == "insurance"


# ─── Utilities ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "MSEB ELECTRICITY BILL PAYMENT",
        "BESCOM ELECTRIC BILL",
        "MAHANAGAR GAS BILL",
        "BROADBAND INTERNET BILL PAYMENT",
        "DTH BILL PAYMENT",
    ],
)
def test_narration_purpose_utilities(narration):
    assert _level5(narration, debit=1200) == "utilities"


# ─── Mobile recharge ──────────────────────────────────────────────────────────


def test_narration_purpose_mobile_recharge():
    assert _level5("AIRTEL PREPAID RECHARGE", debit=299) == "mobile recharge"
    assert _level5("JIO TOPUP PAYMENT", debit=199) == "mobile recharge"


# ─── Subscription / OTT ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "NETFLIX MONTHLY SUBSCRIPTION",
        "SPOTIFY PREMIUM PLAN",
        "HOTSTAR ANNUAL PLAN",
        "BOOKMYSHOW SUBSCRIPTION",
    ],
)
def test_narration_purpose_subscription(narration):
    assert _level5(narration, debit=199) == "subscription"


# ─── Education ────────────────────────────────────────────────────────────────


def test_narration_purpose_education():
    assert _level5("BYJU'S COURSE FEE PAYMENT", debit=2000) == "education"
    assert _level5("SCHOOL FEE PAYMENT", debit=3500) == "education"
    assert _level5("COLLEGE TUITION FEE", debit=25000) == "education"


# ─── Financial services / investments ────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "GROWW SIP PAYMENT",
        "ZERODHA MUTUAL FUND INVESTMENT",
        "UPSTOX DEMAT ACCOUNT",
        "NPS CONTRIBUTION PAYMENT",
    ],
)
def test_narration_purpose_financial_services(narration):
    assert _level5(narration, debit=5000) == "financial services"


# ─── Salary direction guard ───────────────────────────────────────────────────


def test_narration_salary_not_set_on_debit():
    """Salary is income — a debit narration with 'SALARY' must not get salary."""
    assert _level5("SALARY ADVANCE RECOVERY", debit=25000) != "salary"


def test_narration_salary_set_on_credit():
    """An explicit credit salary narration should get salary purpose."""
    assert _level5("NEFT SALARY ACME CORP", credit=50000) == "salary"


# ─── Retail ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "narration",
    [
        "AMAZON PURCHASE PAYMENT",
        "FLIPKART ORDER",
        "MYNTRA FASHION ORDER",
        "NYKAA BEAUTY PRODUCTS",
        "DECATHLON SPORTS STORE",
    ],
)
def test_narration_purpose_retail(narration):
    assert _level5(narration, debit=999) == "retail"


# ─── Ambiguous tokens do not generate false positives ─────────────────────────


def test_ambiguous_narration_stays_blank():
    """Generic payment narrations without category evidence stay blank."""
    assert _level5("NEFT TRANSFER TO FRIEND", debit=5000) == ""
    assert _level5("UPI PAYMENT", debit=100) == ""


def test_gas_station_narration_is_fuel_not_utilities():
    """Gas station payment should be fuel, not utilities (ordered priority)."""
    assert _level5("IOCL GAS STATION PAYMENT", debit=1500) == "fuel"


# ─── Pipeline integration ─────────────────────────────────────────────────────


def test_pipeline_dining_merchant_gets_level5():
    """A merchant-classified restaurant row should get dining as Level 5."""
    rows = [
        {
            "Date": "2026-04-01",
            "Particulars": "HOTEL SELECT DINING PAYMENT",
            "Debit": 800,
            "Credit": "",
            "Part_0": "HOTEL",
            "Part_1": "SELECT",
        }
    ]
    out = make_classifier().classify(statement_frame(rows))
    assert out.loc[0, "Level_5"] == "dining"


def test_pipeline_grocery_merchant_gets_level5():
    """Blinkit row should get grocery as Level 5."""
    rows = [
        {
            "Date": "2026-04-01",
            "Particulars": "BLINKIT GROCERY DELIVERY PAYMENT",
            "Debit": 450,
            "Credit": "",
            "Part_0": "BLINKIT",
            "Part_1": "GROCERY",
        }
    ]
    out = make_classifier().classify(statement_frame(rows))
    assert out.loc[0, "Level_5"] == "grocery"


def test_pipeline_loan_emi_narration_gets_level5():
    """Explicit loan EMI narration should get loan repayment Level 5."""
    rows = [
        {
            "Date": "2026-04-01",
            "Particulars": "HDFC BANK HOME LOAN EMI",
            "Debit": 25000,
            "Credit": "",
            "Part_0": "HDFC",
            "Part_1": "BANK",
        }
    ]
    out = make_classifier().classify(statement_frame(rows))
    assert out.loc[0, "Level_5"] == "loan repayment"


def test_pipeline_keyword_level5_beats_narration_override():
    """If keyword model already set Level 5, narration override must not overwrite it."""
    canned = {"UBER": {"Level_5": "digital services", "Level_3": "merchant"}}
    rows = [
        {
            "Date": "2026-04-01",
            "Particulars": "UBER INDIA RIDES PAYMENT",
            "Debit": 350,
            "Credit": "",
            "Part_0": "UBER",
        }
    ]
    out = make_classifier(canned).classify(statement_frame(rows))
    # Keyword model said "digital services"; narration says "transport".
    # Keyword model wins.
    assert out.loc[0, "Level_5"] == "digital services"


def test_pipeline_existing_level5_unchanged_without_narration_evidence():
    """Rows with no purpose evidence in the narration keep Level 5 blank."""
    rows = [
        {
            "Date": "2026-04-01",
            "Particulars": "IMPS PAYMENT REF 123456",
            "Debit": 5000,
            "Credit": "",
            "Part_0": "IMPS",
            "Part_1": "PAYMENT",
        }
    ]
    out = make_classifier().classify(statement_frame(rows))
    # No specific category evidence — Level 5 should remain blank.
    assert out.loc[0, "Level_5"] == ""


# =============================================================================
# Regressions: Date-less salary detection, Level_4 gateways, slash-UPI P2P
# =============================================================================


def test_detect_salary_without_date_column_still_flags_explicit_salary():
    df = labelled_frame([{"Particulars": "NEFT SALARY ACME CORP", "Debit": "", "Credit": 50000}])
    assert "Date" not in df.columns

    out = detect_salary(df)

    assert out.loc[0, "IsSalary"] == 1
    assert out.loc[0, "Level_5"] == "salary"


def test_detect_salary_without_date_column_skips_recurring_inference():
    df = labelled_frame(
        [
            {"Particulars": "NEFT ACME", "Debit": "", "Credit": 30000},
            {"Particulars": "NEFT ACME", "Debit": "", "Credit": 30000},
        ]
    )

    out = detect_salary(df)

    assert out["IsSalary"].tolist() == [0, 0]


def test_slash_upi_with_named_payee_is_still_p2p():
    out = apply_raw_transaction_overrides(
        "UPI/DR/123/ARUN/SBIN/arun@sbi", blank_levels(), debit_value=200, credit_value=0
    )

    assert out["Level_3"] == "p2p"
    assert out["Level_5"] == "p2p transfer"
    assert out["Level_6"] == "arun"


def test_hyphen_merchant_prefers_model_bank_over_gateway_when_narration_has_no_bank():
    narration = "UPI-RAZORPAYSWIGGY-SWIGGY@OKAXIS-123456-NOTE"
    canned = {
        "UPI": {"Level_1": "upi"},
        "RAZORPAYSWIGGY": {"Level_3": "institution", "Level_4": "razorpay"},
        "NOTE": {"Level_4": "hdfc bank"},
    }

    out = make_classifier(canned).classify(statement_frame([upi_row(narration)]))

    assert out.loc[0, "Level_4"] == "hdfc bank"
    assert out.loc[0, "Level_6"] == "swiggy"


def test_non_hyphen_row_still_takes_detected_gateway_level4():
    out = apply_raw_transaction_overrides(
        "RAZORPAY SOFTWARE PVT LTD", blank_levels(), debit_value=100, credit_value=0
    )

    assert out["Level_4"] == "razorpay"