"""Tests for micro-segmentation flags."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.microsegment import Microsegmenter


def test_quick_comm_is_flagged_separately(tmp_path: Path) -> None:
    """Blinkit-style rows should set exp_quick_comm and get the right segment."""

    master_path = tmp_path / "master.csv"
    pd.DataFrame(
        [
            {"Key Word": "blinkit", "Level 5": "grocery"},
            {"Key Word": "zomato", "Level 5": "dining"},
        ]
    ).to_csv(master_path, index=False)

    df = pd.DataFrame(
        [
            {
                "Particulars": "UPI BLINKIT ORDER",
                "Debit": 450,
                "Credit": 0,
                "Level_1": "upi",
                "Level_3": "merchant",
                "Level_5": "grocery",
                "Level_6": "blinkit",
                "Level_7": "",
                "First Level Classification": "",
                "IsSalary": 0,
            }
        ]
    )

    result = Microsegmenter(str(master_path)).run(df)

    assert int(result.loc[0, "exp_quick_comm"]) == 1
    assert result.loc[0, "MicroSegment"] == "Quick Commerce"
