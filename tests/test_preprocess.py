"""Tests for keyword preprocessing."""

from src.preprocess import KeywordPreprocessor


def test_numtoken_replacement_works() -> None:
    """Digits should be replaced with the placeholder token."""

    preprocessor = KeywordPreprocessor()
    assert preprocessor.transform("UPI123PAY") == "upi NUMTOKEN pay"


def test_uppercase_input_is_lowercased() -> None:
    """Alphabetic content should normalize to lowercase."""

    preprocessor = KeywordPreprocessor()
    assert preprocessor.transform("SALARY ACME CORP") == "salary acme corp"


def test_special_characters_are_stripped() -> None:
    """Non-alphanumeric separators should be removed."""

    preprocessor = KeywordPreprocessor()
    assert preprocessor.transform("UPI/DR/*ARUN@SBI") == "upi dr arun sbi"


def test_transform_is_idempotent() -> None:
    """Applying the preprocessor twice should not change the output."""

    preprocessor = KeywordPreprocessor()
    once = preprocessor.transform("UPI/123/ARUN")
    twice = preprocessor.transform(once)
    assert once == twice
