"""Tests for deterministic rule classification."""

from src.rule_engine import RuleEngine


def test_upi_dr_keyword_maps_to_rail_and_direction() -> None:
    """UPI debit strings should trigger Level 1 and Level 0."""

    result = RuleEngine().classify("UPI/DR/123/ARUN/SBIN/arun@sbi")
    assert result["Level_1"] == "upi"
    assert result["Level_0"] == "debit"


def test_nach_emi_keyword_maps_to_autodebit_and_recurring() -> None:
    """Recurring NACH strings should trigger Level 2 and Level 7."""

    result = RuleEngine().classify("NACH EMI HDFC")
    assert result["Level_2"] == "auto-debit"
    assert result["Level_7"] == "recurring"


def test_cr_maps_only_to_credit() -> None:
    """A plain credit token should only set Level 0."""

    result = RuleEngine().classify("CR")
    assert result["Level_0"] == "credit"
    assert result["Level_1"] == ""
    assert result["Level_2"] == ""
    assert result["Level_7"] == ""


def test_salary_is_not_a_rule_level_field() -> None:
    """Salary should not trigger any deterministic rule outputs."""

    result = RuleEngine().classify("SALARY")
    assert result == {
        "Level_0": "",
        "Level_1": "",
        "Level_2": "",
        "Level_7": "",
    }
