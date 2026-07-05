from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte_carlo import (
    simulate_correlated_daily_returns,
    simulate_correlated_gbm_paths,
    simulate_gbm_paths,
)


def test_simulate_gbm_paths_shape_and_start_price() -> None:
    paths = simulate_gbm_paths(
        s0=100.0, mu_annual=0.08, sigma_annual=0.20, n_days=252, n_paths=1000, seed=1
    )
    assert paths.shape == (253, 1000)
    np.testing.assert_allclose(paths[0], 100.0)
    assert (paths > 0).all()


def test_simulate_gbm_paths_terminal_mean_near_drift() -> None:
    s0, mu, sigma = 100.0, 0.08, 0.20
    paths = simulate_gbm_paths(
        s0=s0, mu_annual=mu, sigma_annual=sigma, n_days=252, n_paths=50_000, seed=2
    )
    terminal = paths[-1]
    # E[S_T] under GBM = S0 * exp(mu * T); loose tolerance for MC noise.
    expected = s0 * np.exp(mu * 1.0)
    assert terminal.mean() == pytest.approx(expected, rel=0.05)


def test_simulate_correlated_gbm_paths_reproduces_correlation() -> None:
    s0 = pd.Series({"A": 100.0, "B": 50.0})
    mu = pd.Series({"A": 0.05, "B": 0.03})
    target_corr = 0.6
    sigma = np.array([0.20, 0.30])
    cov = np.outer(sigma, sigma) * np.array([[1.0, target_corr], [target_corr, 1.0]])
    cov_annual = pd.DataFrame(cov, index=["A", "B"], columns=["A", "B"])

    paths = simulate_correlated_gbm_paths(s0, mu, cov_annual, n_days=1, n_paths=200_000, seed=3)
    ret_a = np.log(paths["A"][1] / paths["A"][0])
    ret_b = np.log(paths["B"][1] / paths["B"][0])
    sample_corr = np.corrcoef(ret_a, ret_b)[0, 1]
    assert sample_corr == pytest.approx(target_corr, abs=0.02)


def test_simulate_correlated_daily_returns_matches_moments() -> None:
    mean_daily = pd.Series({"A": 0.0005, "B": -0.0002})
    sigma = np.array([0.015, 0.025])
    target_corr = -0.4
    cov = np.outer(sigma, sigma) * np.array([[1.0, target_corr], [target_corr, 1.0]])
    cov_daily = pd.DataFrame(cov, index=["A", "B"], columns=["A", "B"])

    sims = simulate_correlated_daily_returns(mean_daily, cov_daily, n_sims=200_000, seed=4)

    np.testing.assert_allclose(sims.mean().to_numpy(), mean_daily.to_numpy(), atol=2e-4)
    sample_cov = sims.cov().to_numpy()
    np.testing.assert_allclose(sample_cov, cov, atol=5e-5)
