"""Monte Carlo simulation under Geometric Brownian Motion (GBM).

Two simulation modes:
- Single-asset price paths (independent), used for per-stock fan charts.
- Correlated multivariate simulation via Cholesky decomposition of the
  covariance matrix, used both for correlated multi-year price paths and for
  the 1-day portfolio VaR/CVaR simulation (so the joint tail risk reflects
  how the stocks actually move together, not as if they were independent).

All functions take a `seed` for reproducibility (per CLAUDE.md conventions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simulate_gbm_paths(
    s0: float,
    mu_annual: float,
    sigma_annual: float,
    n_days: int,
    n_paths: int,
    seed: int,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Simulate independent single-asset GBM price paths.

    Args:
        s0: starting price.
        mu_annual: annualised expected log-return (drift).
        sigma_annual: annualised volatility.
        n_days: number of trading days to simulate forward.
        n_paths: number of simulated paths.
        seed: RNG seed.
        periods_per_year: trading days per year used to de-annualise mu/sigma.

    Returns:
        Array of shape (n_days + 1, n_paths); row 0 is `s0` for every path.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    daily_mu = (mu_annual - 0.5 * sigma_annual**2) * dt
    daily_sigma = sigma_annual * np.sqrt(dt)

    shocks = rng.normal(loc=daily_mu, scale=daily_sigma, size=(n_days, n_paths))
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(shocks, axis=0)])
    return s0 * np.exp(log_paths)


def simulate_correlated_gbm_paths(
    s0: pd.Series,
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    n_days: int,
    n_paths: int,
    seed: int,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, np.ndarray]:
    """Simulate correlated multi-asset GBM price paths via Cholesky decomposition.

    Args:
        s0: starting prices, indexed by ticker.
        mu_annual: annualised expected log-returns, indexed by ticker (same order as s0).
        cov_annual: annualised covariance matrix of log-returns (tickers x tickers).
        n_days: number of trading days to simulate forward.
        n_paths: number of simulated paths.
        seed: RNG seed.
        periods_per_year: trading days per year used to de-annualise mu/cov.

    Returns:
        Dict mapping ticker -> price-path array of shape (n_days + 1, n_paths).
    """
    tickers = list(s0.index)
    cov_annual = cov_annual.loc[tickers, tickers]
    n_assets = len(tickers)

    dt = 1.0 / periods_per_year
    daily_cov = cov_annual.to_numpy() * dt
    daily_sigma = np.sqrt(np.diag(daily_cov))
    daily_mu = (mu_annual.loc[tickers].to_numpy() - 0.5 * daily_sigma**2) * dt

    chol = np.linalg.cholesky(daily_cov)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_days, n_paths, n_assets))
    correlated_shocks = z @ chol.T + daily_mu

    log_paths = np.concatenate(
        [np.zeros((1, n_paths, n_assets)), np.cumsum(correlated_shocks, axis=0)],
        axis=0,
    )
    prices = s0.loc[tickers].to_numpy()[None, None, :] * np.exp(log_paths)

    return {ticker: prices[:, :, i] for i, ticker in enumerate(tickers)}


def simulate_correlated_daily_returns(
    mean_daily: pd.Series,
    cov_daily: pd.DataFrame,
    n_sims: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate one day of correlated simple returns via multivariate normal.

    Used for the Monte Carlo portfolio VaR/CVaR: draws `n_sims` joint 1-day
    return scenarios that respect the historical covariance structure between
    assets, rather than simulating each asset independently.

    Args:
        mean_daily: mean daily simple return per asset, indexed by ticker.
        cov_daily: daily covariance matrix of simple returns (tickers x tickers).
        n_sims: number of simulated scenarios.
        seed: RNG seed.

    Returns:
        DataFrame of shape (n_sims, n_assets), columns matching `mean_daily.index`.
    """
    tickers = list(mean_daily.index)
    cov_daily = cov_daily.loc[tickers, tickers]

    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(
        mean=mean_daily.loc[tickers].to_numpy(),
        cov=cov_daily.to_numpy(),
        size=n_sims,
    )
    return pd.DataFrame(sims, columns=tickers)
