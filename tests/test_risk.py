from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk import (
    correlation_matrix,
    covariance_matrix,
    historical_var_cvar,
    parametric_var_cvar,
    portfolio_returns,
    rolling_volatility,
)


def test_portfolio_returns_rejects_bad_weights(simple_rets: pd.DataFrame) -> None:
    bad_weights = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5})
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_returns(simple_rets, bad_weights)


def test_portfolio_returns_equal_weight_matches_manual_average(
    simple_rets: pd.DataFrame, equal_weights: pd.Series
) -> None:
    port = portfolio_returns(simple_rets, equal_weights)
    manual = simple_rets.mean(axis=1)
    pd.testing.assert_series_equal(port, manual, check_exact=False, rtol=1e-10)


def test_correlation_matrix_diagonal_is_one(log_rets: pd.DataFrame) -> None:
    corr = correlation_matrix(log_rets)
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_covariance_matrix_annualized_scales_by_periods(log_rets: pd.DataFrame) -> None:
    daily = covariance_matrix(log_rets, annualize=False)
    annual = covariance_matrix(log_rets, annualize=True, periods_per_year=252)
    np.testing.assert_allclose(annual.to_numpy(), daily.to_numpy() * 252)


def test_rolling_volatility_nan_before_window(log_rets: pd.DataFrame) -> None:
    vol = rolling_volatility(log_rets, window=21)
    assert vol.iloc[:20].isna().all().all()
    assert vol.iloc[21:].notna().all().all()


@pytest.mark.parametrize("method_fn", [historical_var_cvar, parametric_var_cvar])
def test_cvar_at_least_var(method_fn, simple_rets: pd.DataFrame, equal_weights: pd.Series) -> None:
    port = portfolio_returns(simple_rets, equal_weights)
    result = method_fn(port, confidence=0.95, portfolio_value=1_000_000.0)
    assert result.cvar_pct >= result.var_pct
    assert result.cvar_dollar >= result.var_dollar


@pytest.mark.parametrize("method_fn", [historical_var_cvar, parametric_var_cvar])
def test_var_increases_with_confidence(
    method_fn, simple_rets: pd.DataFrame, equal_weights: pd.Series
) -> None:
    port = portfolio_returns(simple_rets, equal_weights)
    var_95 = method_fn(port, confidence=0.95, portfolio_value=1_000_000.0).var_pct
    var_99 = method_fn(port, confidence=0.99, portfolio_value=1_000_000.0).var_pct
    assert var_99 > var_95


def test_var_dollar_scales_with_portfolio_value(
    simple_rets: pd.DataFrame, equal_weights: pd.Series
) -> None:
    port = portfolio_returns(simple_rets, equal_weights)
    small = historical_var_cvar(port, 0.95, 1_000_000.0)
    large = historical_var_cvar(port, 0.95, 2_000_000.0)
    assert large.var_dollar == pytest.approx(2 * small.var_dollar)
