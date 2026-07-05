from __future__ import annotations

import numpy as np
import pandas as pd

from returns import cumulative_returns, log_returns, simple_returns, summary_stats


def test_log_returns_roundtrip_to_price_ratio(wide_prices: pd.DataFrame) -> None:
    lr = log_returns(wide_prices)
    reconstructed = wide_prices.iloc[0] * np.exp(lr.cumsum())
    pd.testing.assert_frame_equal(reconstructed, wide_prices.iloc[1:], check_exact=False, rtol=1e-8)


def test_simple_returns_roundtrip_to_price_ratio(wide_prices: pd.DataFrame) -> None:
    sr = simple_returns(wide_prices)
    reconstructed = wide_prices.iloc[0] * (1.0 + sr).cumprod()
    pd.testing.assert_frame_equal(reconstructed, wide_prices.iloc[1:], check_exact=False, rtol=1e-8)


def test_cumulative_returns_log_matches_price_ratio(wide_prices: pd.DataFrame) -> None:
    lr = log_returns(wide_prices)
    cum = cumulative_returns(lr, log=True)
    expected = wide_prices.iloc[1:] / wide_prices.iloc[0]
    pd.testing.assert_frame_equal(cum, expected, check_exact=False, rtol=1e-8)


def test_cumulative_returns_simple_matches_price_ratio(wide_prices: pd.DataFrame) -> None:
    sr = simple_returns(wide_prices)
    cum = cumulative_returns(sr, log=False)
    expected = wide_prices.iloc[1:] / wide_prices.iloc[0]
    pd.testing.assert_frame_equal(cum, expected, check_exact=False, rtol=1e-8)


def test_summary_stats_shape_and_sanity(log_rets: pd.DataFrame) -> None:
    stats = summary_stats(log_rets, log=True)
    assert list(stats.index) == list(log_rets.columns)
    assert (stats["ann_vol"] > 0).all()
    assert np.isfinite(stats["sharpe"]).all()
    # Annualised vol should be in a sane ballpark for the fixture's daily sigmas.
    assert stats["ann_vol"].between(0.05, 1.0).all()
