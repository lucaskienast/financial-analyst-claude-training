"""Generate synthetic OHLC data and write it to an Excel workbook.

Usage:
    uv run python scripts/generate_ohlc.py

Output:
    data/raw/ohlc_5stocks_2025-2026.xlsx
        - one sheet per ticker (chronological OHLCV)
        - an "ALL" sheet with the tidy long-format combined data
        - an "info" sheet documenting the schema and generation parameters
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_ohlc import DEFAULT_STOCKS, generate_ohlc  # noqa: E402

OUT_PATH = Path("data/raw/ohlc_5stocks_2025-2026.xlsx")


def main() -> None:
    df = generate_ohlc()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    info = pd.DataFrame(
        {
            "ticker": [s.ticker for s in DEFAULT_STOCKS],
            "name": [s.name for s in DEFAULT_STOCKS],
            "start_price": [s.start_price for s in DEFAULT_STOCKS],
            "annual_drift": [s.annual_drift for s in DEFAULT_STOCKS],
            "annual_volatility": [s.annual_volatility for s in DEFAULT_STOCKS],
            "avg_daily_volume": [s.avg_daily_volume for s in DEFAULT_STOCKS],
        }
    )

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl", datetime_format="yyyy-mm-dd") as xl:
        info.to_excel(xl, sheet_name="info", index=False)
        df.to_excel(xl, sheet_name="ALL", index=False)
        for ticker, g in df.groupby("ticker", sort=True):
            g.drop(columns="ticker").to_excel(xl, sheet_name=str(ticker), index=False)

    n_days = df["date"].nunique()
    print(f"Wrote {OUT_PATH}")
    print(f"  {df['ticker'].nunique()} tickers x {n_days} trading days = {len(df)} rows")
    print(f"  date range: {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print("\nPer-ticker summary:")
    summary = df.groupby("ticker").agg(
        first_close=("close", "first"),
        last_close=("close", "last"),
        min_low=("low", "min"),
        max_high=("high", "max"),
        avg_volume=("volume", "mean"),
    )
    summary["total_return_%"] = (
        (summary["last_close"] / summary["first_close"] - 1.0) * 100
    ).round(1)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(summary)


if __name__ == "__main__":
    main()
