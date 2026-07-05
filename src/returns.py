"""Return calculations and summary statistics.

Log returns are used throughout for statistics, volatility, and Monte Carlo
(they time-aggregate additively: sum of daily log returns = period log return).
Simple returns are used when aggregating across assets into a portfolio
(portfolio simple return = weighted sum of asset simple returns, which does
not hold for log returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(P_t / P_{t-1}). First row is dropped (no prior price)."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: P_t / P_{t-1} - 1. First row is dropped (no prior price)."""
    return prices.pct_change().dropna(how="all")


def cumulative_returns(returns: pd.DataFrame, *, log: bool) -> pd.DataFrame:
    """Cumulative growth of $1, rebased to 1.0 at the first return date.

    Args:
        returns: log or simple returns.
        log: True if `returns` are log returns, False if simple returns.
            Must match how `returns` was computed — the compounding formula
            differs (cumsum+exp for log, cumprod for simple).
    """
    if log:
        return np.exp(returns.cumsum())
    return (1.0 + returns).cumprod()


def summary_stats(
    returns: pd.DataFrame,
    *,
    log: bool,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Per-asset summary statistics table.

    Args:
        returns: log or simple daily returns (one column per asset).
        log: whether `returns` are log returns (changes annualisation of mean
            return: log means sum/compound additively, simple means don't).
        periods_per_year: annualisation factor (default 252 trading days).

    Returns:
        DataFrame indexed by asset with columns:
        mean_daily, ann_return, ann_vol, sharpe, skew, kurtosis.
        Sharpe assumes a 0% risk-free rate.
    """
    mean_daily = returns.mean()
    vol_daily = returns.std(ddof=1)

    if log:
        ann_return = mean_daily * periods_per_year
    else:
        ann_return = (1.0 + mean_daily) ** periods_per_year - 1.0

    ann_vol = vol_daily * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol

    out = pd.DataFrame(
        {
            "mean_daily": mean_daily,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "skew": returns.skew(),
            "kurtosis": returns.kurt(),
        }
    )
    out.index.name = "ticker"
    return out
