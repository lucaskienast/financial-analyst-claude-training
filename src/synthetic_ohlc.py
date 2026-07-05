"""Generate synthetic but sensible daily OHLC(V) trading data.

Prices follow geometric Brownian motion (GBM) at the daily close-to-close level.
Around each daily close we build an open (with an overnight gap), and an intraday
high/low that always respect the OHLC invariants:

    low <= min(open, close) <= max(open, close) <= high

Volume is drawn to be positively correlated with the day's realised range, which
loosely mirrors the empirical volume/volatility relationship.

The generator is deterministic given a seed so datasets are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class StockSpec:
    """Parameters describing one synthetic stock."""

    ticker: str
    name: str
    start_price: float
    annual_drift: float  # expected annual log-return (mu)
    annual_volatility: float  # annualised volatility (sigma)
    avg_daily_volume: float  # baseline shares traded per day


# Five stocks with deliberately different risk/return profiles.
DEFAULT_STOCKS: tuple[StockSpec, ...] = (
    StockSpec("AHT", "Alpha Health Tech", 142.00, 0.14, 0.28, 3_200_000),
    StockSpec("BRVE", "Bravo Energy", 58.50, 0.06, 0.35, 5_800_000),
    StockSpec("CDRA", "Cedar Industrials", 96.20, 0.09, 0.22, 1_900_000),
    StockSpec("DLTA", "Delta Financial Group", 33.75, 0.11, 0.30, 8_400_000),
    StockSpec("EPSN", "Epsilon Consumer", 210.40, 0.04, 0.18, 950_000),
)


def _generate_one(
    spec: StockSpec,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate an OHLCV frame for a single stock over the given trading dates."""
    n = len(dates)
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    # --- Daily closes via GBM ------------------------------------------------
    daily_mu = (spec.annual_drift - 0.5 * spec.annual_volatility**2) * dt
    daily_sigma = spec.annual_volatility * np.sqrt(dt)
    log_returns = rng.normal(loc=daily_mu, scale=daily_sigma, size=n)
    close = spec.start_price * np.exp(np.cumsum(log_returns))

    # --- Open: previous close plus a small overnight gap ---------------------
    prev_close = np.empty(n)
    prev_close[0] = spec.start_price
    prev_close[1:] = close[:-1]
    overnight_gap = rng.normal(loc=0.0, scale=daily_sigma * 0.5, size=n)
    open_ = prev_close * np.exp(overnight_gap)

    # --- Intraday high/low bracketing open and close -------------------------
    hi_lo_base = np.maximum(open_, close)
    lo_hi_base = np.minimum(open_, close)
    # Extra wick beyond the open/close body, as a fraction of price.
    up_wick = np.abs(rng.normal(0.0, daily_sigma * 0.7, size=n))
    dn_wick = np.abs(rng.normal(0.0, daily_sigma * 0.7, size=n))
    high = hi_lo_base * (1.0 + up_wick)
    low = lo_hi_base * (1.0 - dn_wick)

    # --- Volume: correlated with the day's realised range --------------------
    daily_range = (high - low) / close
    range_z = (daily_range - daily_range.mean()) / (daily_range.std() + 1e-12)
    volume_factor = np.exp(0.35 * range_z + rng.normal(0.0, 0.20, size=n))
    volume = np.round(spec.avg_daily_volume * volume_factor).astype("int64")

    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": spec.ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Round money columns to cents; enforce invariants after rounding.
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].round(2)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    df["ticker"] = df["ticker"].astype("string")
    return df


def generate_ohlc(
    stocks: tuple[StockSpec, ...] = DEFAULT_STOCKS,
    *,
    end_date: str | pd.Timestamp = "2026-07-03",
    n_days: int = TRADING_DAYS_PER_YEAR,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a tidy long-format OHLCV DataFrame for all stocks.

    Args:
        stocks: specifications of the stocks to generate.
        end_date: last trading day (inclusive). Weekends are excluded.
        n_days: number of business days to generate (default one trading year).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns [date, ticker, open, high, low, close, volume],
        sorted by ticker then date.
    """
    dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n_days, name="date")
    rng = np.random.default_rng(seed)

    frames = [_generate_one(spec, dates, rng) for spec in stocks]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    _validate(out)
    return out


def _validate(df: pd.DataFrame) -> None:
    """Fail loudly if OHLC invariants or basic sanity checks are violated."""
    assert not df.isna().any().any(), "unexpected NaNs in generated data"
    assert (df["high"] >= df["low"]).all(), "high < low"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), "high below body"
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), "low above body"
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), "non-positive price"
    assert (df["volume"] > 0).all(), "non-positive volume"
