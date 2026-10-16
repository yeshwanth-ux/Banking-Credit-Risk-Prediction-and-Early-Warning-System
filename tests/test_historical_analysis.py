from __future__ import annotations

import pandas as pd

from src.historical_analysis import (
    default_rate_by_quantile,
    extreme_value_counts,
    find_redundant_pairs,
)


def test_default_rate_by_quantile_preserves_all_valid_rows() -> None:
    data = pd.DataFrame(
        {"amount": range(1, 21), "defaulted": [0] * 10 + [1] * 10}
    )

    result = default_rate_by_quantile(data, "amount", "defaulted", bins=4)

    assert result["records"].sum() == 20
    assert result.iloc[0]["default_rate"] == 0
    assert result.iloc[-1]["default_rate"] == 1


def test_extreme_value_count_uses_iqr_rule() -> None:
    data = pd.DataFrame({"amount": [10, 11, 12, 13, 1000]})

    assert extreme_value_counts(data, ["amount"]) == {"amount": 1}


def test_redundant_pairs_are_unique_and_thresholded() -> None:
    data = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 0, 1, 0]}
    )

    result = find_redundant_pairs(data, ["a", "b", "c"])

    assert len(result) == 1
    assert result[0][0:2] == ("a", "b")
    assert result[0][2] == 1.0
