from __future__ import annotations

import pandas as pd
import pytest

from data_io import _validate, to_wide_close


def _make_valid_long_df() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = []
    for ticker, base in (("A", 100.0), ("B", 50.0)):
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": base,
                    "high": base + 1.0,
                    "low": base - 1.0,
                    "close": base + 0.5,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def test_validate_passes_on_clean_data() -> None:
    _validate(_make_valid_long_df())  # should not raise


def test_validate_rejects_high_below_close() -> None:
    df = _make_valid_long_df()
    df.loc[0, "high"] = df.loc[0, "close"] - 10.0
    with pytest.raises(ValueError, match="high below"):
        _validate(df)


def test_validate_rejects_duplicates() -> None:
    df = _make_valid_long_df()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _validate(dup)


def test_to_wide_close_shape_and_values() -> None:
    df = _make_valid_long_df()
    wide = to_wide_close(df)
    assert list(wide.columns) == ["A", "B"]
    assert len(wide) == 5
    assert wide["A"].iloc[0] == pytest.approx(100.5)


def test_to_wide_close_raises_on_gaps() -> None:
    df = _make_valid_long_df()
    df = df.drop(index=0)  # remove one (date, ticker) cell for "A"
    with pytest.raises(ValueError, match="gaps"):
        to_wide_close(df)
