# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

This is a **Python-based quantitative analysis project for traders**. Its purpose is to
ingest, clean, and analyse financial and market data — primarily from **CSV and Excel
files** — using **Python scripts** and **Jupyter notebooks**. Typical work includes
exploratory data analysis, building indicators/signals, statistical modelling,
backtesting, and producing charts and reports.

The audience is quantitative traders and analysts, so **correctness, reproducibility, and
numerical rigour matter more than clever abstractions**. Prefer clear, verifiable code
over premature generalisation.

> This project may be greenfield or partially built. Treat the structure and commands
> below as the intended conventions. If a file or directory referenced here does not yet
> exist, create it following these conventions rather than inventing a different layout.

## Tech Stack & Environment

- **Python 3.11+**
- **Environment & packages: [uv](https://docs.astral.sh/uv/)** — the single source of
  truth for dependencies. Do not use bare `pip install` into the global interpreter.
- **Core libraries:**
  - `pandas` — tabular data (the workhorse)
  - `numpy` — numerical arrays / vectorised math
  - `openpyxl` — reading/writing `.xlsx` Excel files (pandas uses it under the hood)
  - `pyarrow` — fast Parquet I/O for processed data
  - `matplotlib` / `plotly` — charting
  - `scipy`, `statsmodels` — statistics, regressions, time-series models
  - `jupyterlab` — notebooks
- **Optional / suggested** (add only when needed, not by default): `scikit-learn`
  (ML), `TA-Lib` or `pandas-ta` (technical indicators), `vectorbt` or `backtrader`
  (backtesting), `yfinance` (data pulls).

## Development Commands

```bash
# Create the virtual environment (once)
uv venv

# Add / remove dependencies (updates pyproject.toml + uv.lock)
uv add pandas numpy openpyxl
uv remove <package>

# Sync the env to the lockfile (e.g. after pulling changes)
uv sync

# Run a script inside the managed environment
uv run python scripts/analyse_returns.py

# Launch Jupyter
uv run jupyter lab

# Tests and linting
uv run pytest
uv run ruff check .
uv run ruff format .
```

Always run Python via `uv run …` so the correct, locked environment is used.

## Suggested Project Structure

```
data/
  raw/         # immutable source CSV/Excel files — never modified in place
  processed/   # cleaned / derived datasets (Parquet preferred)
notebooks/     # exploratory Jupyter notebooks
src/           # reusable, importable modules (data loaders, indicators, backtests)
scripts/       # runnable entrypoints (thin wrappers around src/)
tests/         # pytest tests for code in src/
outputs/       # generated charts, reports, exports
```

`data/`, `outputs/`, and notebook checkpoints should be **git-ignored**. Reusable logic
belongs in `src/` (importable and testable), not buried inside notebooks.

## Coding Conventions

- Follow **PEP 8**; format and lint with **ruff**.
- Use **type hints** and concise **docstrings** on public functions.
- Prefer **small, pure functions** in `src/` that take and return DataFrames/arrays, so
  they can be unit-tested and reused across notebooks and scripts.
- **Vectorise** with pandas/numpy; avoid Python `for` loops over rows (`iterrows`) in hot
  paths.
- Be **explicit about dtypes, indexes, and dates** — do not rely on pandas' inference for
  anything that affects results.
- Keep functions deterministic where possible; isolate I/O and randomness.

## Data Handling Rules

- **Read data explicitly.** When loading CSV/Excel, specify `dtype`, `parse_dates`, and
  the relevant sheet/columns rather than relying on defaults:
  ```python
  df = pd.read_csv("data/raw/prices.csv", parse_dates=["date"], dtype={"ticker": "string"})
  df = pd.read_excel("data/raw/book.xlsx", sheet_name="Positions", parse_dates=["asof"])
  ```
- **Validate on load** — check expected columns, dtypes, date ranges, duplicates, and NaN
  counts before analysis. Fail loudly on unexpected schema.
- **Raw data is immutable.** Never overwrite files in `data/raw/`. Write cleaned/derived
  outputs to `data/processed/` (Parquet).
- **Never commit data or secrets** (API keys, credentials, position files) to the repo.
- Document the expected format/schema of each input file (in code or a short README).

## Quant / Financial Correctness

These are the mistakes that silently corrupt quant results — guard against them:

- **No look-ahead bias / future leakage.** At each timestamp use only information
  available up to that point. `shift()` signals appropriately; never compute a signal from
  the same bar you trade on unless that is explicitly intended and documented.
- **Time & calendars.** Handle timezones explicitly and align to trading days/market
  calendars. Be deliberate with `resample`, forward/back-fill, and how missing sessions
  are treated.
- **Corporate actions.** Use split/dividend-**adjusted** prices for return calculations;
  state which price series (raw vs adjusted) is being used.
- **Returns.** Be consistent about simple vs log returns; annualise correctly (state the
  periods-per-year assumption); compound rather than sum simple returns.
- **Backtest realism.** Include transaction costs and slippage; avoid survivorship bias;
  keep a clear split between in-sample and out-of-sample data.
- **Numerical care.** Watch floating-point precision for money; guard ratios against
  divide-by-zero; be explicit about NaN handling in aggregations.
- **Reproducibility.** Set random seeds, pin dependencies via `uv.lock`, and ensure any
  analysis can be re-run end-to-end from raw data.

When results look surprisingly good, suspect a data/methodology bug (leakage, bias) before
believing the edge is real.

## Notebook Conventions

- Notebooks must run **top-to-bottom** cleanly. Before sharing or committing, do
  **Restart Kernel → Run All**.
- Factor reusable logic out of notebooks into `src/`; import it back in. Notebooks are for
  exploration and presentation, not as the home of core logic.
- Keep heavy cell output (large tables, images) out of committed notebooks where possible
  to keep diffs readable.
- Name notebooks descriptively (e.g. `2026-07-05_momentum_signal_eda.ipynb`).
