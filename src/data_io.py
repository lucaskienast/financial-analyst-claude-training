"""Load and validate OHLCV data, and reshape it for returns/risk analysis.

Reads the tidy long-format "ALL" sheet produced by scripts/generate_ohlc.py,
validates the schema before anything downstream trusts it, and pivots the
close price to a wide (date x ticker) matrix — the shape returns/risk/Monte
Carlo code operates on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")


def load_ohlc_excel(path: str | Path, sheet_name: str = "ALL") -> pd.DataFrame:
    """Load the tidy long-format OHLCV sheet and validate it.

    Args:
        path: path to the .xlsx workbook.
        sheet_name: sheet containing the combined long-format data.

    Returns:
        DataFrame with columns [date, ticker, open, high, low, close, volume],
        dtypes enforced, sorted by ticker then date.

    Raises:
        ValueError: if the schema, dtypes, or basic data-quality checks fail.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, parse_dates=["date"])

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {sorted(missing)}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["ticker"] = df["ticker"].astype("string")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].astype("int64")

    _validate(df)

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _validate(df: pd.DataFrame) -> None:
    """Fail loudly on schema/quality issues rather than silently analysing bad data."""
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"unexpected NaNs in columns: {bad_cols}")

    dupes = df.duplicated(subset=["ticker", "date"]).sum()
    if dupes:
        raise ValueError(f"found {dupes} duplicate (ticker, date) rows")

    if not (df["high"] >= df[["open", "close"]].max(axis=1)).all():
        raise ValueError("OHLC invariant violated: high below open/close")
    if not (df["low"] <= df[["open", "close"]].min(axis=1)).all():
        raise ValueError("OHLC invariant violated: low above open/close")
    if not (df[["open", "high", "low", "close"]] > 0).all().all():
        raise ValueError("non-positive prices found")
    if not (df["volume"] > 0).all():
        raise ValueError("non-positive volume found")

    for ticker, g in df.groupby("ticker", observed=True):
        n_dates = g["date"].nunique()
        n_rows = len(g)
        if n_dates != n_rows:
            raise ValueError(f"{ticker}: {n_rows} rows but only {n_dates} unique dates")


def to_wide_close(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format OHLCV data to a wide close-price matrix.

    Args:
        df: long-format DataFrame as returned by `load_ohlc_excel`.

    Returns:
        DataFrame indexed by date, one column per ticker, sorted by date,
        containing the close price. Raises if any (date, ticker) cell is
        missing (i.e. tickers don't share a common trading calendar).
    """
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    wide.columns.name = None
    if wide.isna().any().any():
        raise ValueError("wide close-price matrix has gaps; tickers do not share a calendar")
    return wide
