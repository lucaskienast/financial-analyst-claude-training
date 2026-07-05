"""Volatility, correlation, and portfolio Value-at-Risk / Conditional VaR.

VaR/CVaR are reported as **positive loss magnitudes** (in both return-% and $
terms) for a 1-day horizon at a given confidence level (e.g. 0.95, 0.99).
Three independent methods are provided so results can be cross-checked:

- historical: empirical quantile of realised portfolio returns (no
  distributional assumption, but limited by sample size / history length).
- parametric: variance-covariance method assuming portfolio returns are
  normally distributed (closed-form, fast, understates fat tails).
- monte_carlo: empirical quantile of simulated correlated portfolio returns
  (captures the covariance structure; distributional assumption lives in the
  simulation, not in the VaR estimator itself).

Portfolio returns are combined using **simple returns** (weighted sum), since
simple returns aggregate correctly across assets on a given day; log returns
do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Volatility & correlation
# --------------------------------------------------------------------------- #


def rolling_volatility(
    returns: pd.DataFrame,
    window: int,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Rolling annualised volatility (per asset) over a trailing window of days."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def ewma_volatility(
    returns: pd.DataFrame,
    *,
    lambda_: float = 0.94,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Exponentially-weighted annualised volatility (RiskMetrics-style, lambda=0.94).

    var_t = lambda * var_{t-1} + (1 - lambda) * r_{t-1}^2, seeded with the
    full-sample variance.
    """
    var0 = returns.var(ddof=1)
    ewm_var = returns.pow(2).ewm(alpha=(1 - lambda_), adjust=False).mean()
    ewm_var.iloc[0] = var0
    return np.sqrt(ewm_var * periods_per_year)


def covariance_matrix(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Sample covariance matrix of returns, optionally annualised."""
    cov = returns.cov()
    return cov * periods_per_year if annualize else cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample Pearson correlation matrix of returns."""
    return returns.corr()


# --------------------------------------------------------------------------- #
# Portfolio returns
# --------------------------------------------------------------------------- #


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted-sum portfolio return series from per-asset simple returns.

    Args:
        returns: simple daily returns, one column per asset.
        weights: portfolio weights indexed by the same asset labels, summing to 1.
    """
    weights = weights.reindex(returns.columns)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"weights must sum to 1.0, got {weights.sum():.6f}")
    return returns.mul(weights, axis=1).sum(axis=1)


# --------------------------------------------------------------------------- #
# VaR / CVaR
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VaRResult:
    """1-day VaR/CVaR at a given confidence level, as positive loss magnitudes."""

    method: str
    confidence: float
    var_pct: float
    cvar_pct: float
    portfolio_value: float

    @property
    def var_dollar(self) -> float:
        return self.var_pct * self.portfolio_value

    @property
    def cvar_dollar(self) -> float:
        return self.cvar_pct * self.portfolio_value


def historical_var_cvar(
    port_returns: pd.Series,
    confidence: float,
    portfolio_value: float,
) -> VaRResult:
    """Empirical (historical simulation) VaR/CVaR from realised portfolio returns."""
    losses = -port_returns.to_numpy()
    var_pct = float(np.quantile(losses, confidence))
    tail = losses[losses >= var_pct]
    cvar_pct = float(tail.mean()) if len(tail) else var_pct
    return VaRResult("historical", confidence, var_pct, cvar_pct, portfolio_value)


def parametric_var_cvar(
    port_returns: pd.Series,
    confidence: float,
    portfolio_value: float,
) -> VaRResult:
    """Variance-covariance (delta-normal) VaR/CVaR assuming normal portfolio returns."""
    mean = port_returns.mean()
    vol = port_returns.std(ddof=1)
    z = stats.norm.ppf(confidence)
    var_pct = float(-mean + vol * z)
    cvar_pct = float(-mean + vol * stats.norm.pdf(z) / (1.0 - confidence))
    return VaRResult("parametric", confidence, var_pct, cvar_pct, portfolio_value)


def monte_carlo_var_cvar(
    simulated_port_returns: np.ndarray,
    confidence: float,
    portfolio_value: float,
) -> VaRResult:
    """Empirical VaR/CVaR from simulated 1-day correlated portfolio returns."""
    losses = -np.asarray(simulated_port_returns)
    var_pct = float(np.quantile(losses, confidence))
    tail = losses[losses >= var_pct]
    cvar_pct = float(tail.mean()) if len(tail) else var_pct
    return VaRResult("monte_carlo", confidence, var_pct, cvar_pct, portfolio_value)
