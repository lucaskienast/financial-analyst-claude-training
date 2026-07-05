"""Shared pytest fixtures: small deterministic synthetic price/return panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def wide_prices() -> pd.DataFrame:
    """Deterministic 3-asset wide close-price matrix, 500 trading days."""
    rng = np.random.default_rng(7)
    n_days, tickers = 500, ["A", "B", "C"]
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    daily_mu = np.array([0.0003, 0.0002, 0.0001])
    daily_sigma = np.array([0.015, 0.02, 0.01])
    log_rets = rng.normal(daily_mu, daily_sigma, size=(n_days, 3))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def log_rets(wide_prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(wide_prices / wide_prices.shift(1)).dropna(how="all")


@pytest.fixture
def simple_rets(wide_prices: pd.DataFrame) -> pd.DataFrame:
    return wide_prices.pct_change().dropna(how="all")


@pytest.fixture
def equal_weights(simple_rets: pd.DataFrame) -> pd.Series:
    n = simple_rets.shape[1]
    return pd.Series(1.0 / n, index=simple_rets.columns)
