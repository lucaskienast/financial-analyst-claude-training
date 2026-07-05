---
name: analysis-notebook
description: Use when creating a new exploratory analysis Jupyter notebook in this quant project — EDA on a dataset, a returns/volatility/risk study, a new signal or indicator investigation, or any notebook that loads market data and produces charts. Scaffolds a dated notebook wired to the repo's src/ modules (data_io + plotting), styled with the validated dataviz palette, that runs top-to-bottom cleanly. Triggers on: "new notebook", "analysis notebook", "EDA", "explore this data", "chart this / plot this in a notebook", "returns study", "risk notebook", "signal/indicator study".
---

# Analysis Notebook Scaffold

Build a new professional analysis notebook the way `notebooks/2026-07-05_risk_analytics.ipynb`
was built: reusable logic imported from `src/`, charts in the shared validated palette,
guaranteed to run top-to-bottom. This procedure makes that repeatable.

Read `CLAUDE.md` ("Notebook Conventions", "Coding Conventions", "Data Handling Rules",
"Quant / Financial Correctness") — those rules govern; this skill operationalises them.

## Procedure

### 1. Name and locate the notebook
`notebooks/YYYY-MM-DD_descriptive_name.ipynb` — today's date, snake_case topic
(e.g. `2026-07-05_momentum_signal_eda.ipynb`). Confirm the topic with the user if unclear.

### 2. Build it programmatically with `nbformat`
Do **not** hand-author raw `.ipynb` JSON. Write a one-off builder script in the
scratchpad directory, run it with `uv run python`, then execute + verify (step 5).
This is the proven, reliable loop. Skeleton of the builder:

```python
import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))
# ... md(...) / code(...) calls in order (see cell skeleton below) ...
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
with open("notebooks/<name>.ipynb", "w") as f:
    nbf.write(nb, f)
```

### 3. Standard cell skeleton (in order)

1. **Overview** (markdown): objective; data provenance (which file, raw vs adjusted);
   assumptions; methodology; a note to "Restart Kernel → Run All" before sharing.
2. **`%matplotlib inline`** — in its **own** code cell, alone. (Keeping it separate from
   the imports avoids a ruff E402 false-positive.)
3. **Imports** (code): wire up `src/` and import the shared helpers:
   ```python
   import sys
   from pathlib import Path
   REPO_ROOT = Path.cwd().parent
   sys.path.insert(0, str(REPO_ROOT / "src"))

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from IPython.display import display

   from data_io import load_ohlc_excel, to_wide_close
   from plotting import new_figure, style_axes, ticker_colors, CATEGORICAL, SURFACE
   # add from returns / risk / monte_carlo as needed

   plt.rcParams["figure.facecolor"] = SURFACE
   plt.rcParams["axes.facecolor"] = SURFACE
   pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
   ```
4. **Load + validate** (code): `long_df = load_ohlc_excel(DATA_PATH)` then
   `close = to_wide_close(long_df)`. Print shape / date range; `display(close.head())`.
5. **Analysis sections**: one markdown header + code cell(s) each.
6. **Summary** (markdown + a print of key figures), plus caveats.

### 4. Reuse, do not reinvent
- **Loading/validation** → `load_ohlc_excel`, `to_wide_close` (`src/data_io.py`). Never
  re-implement Excel/CSV loading; if a new input schema is needed, extend `src/data_io.py`.
- **Charts** → `new_figure()`, `style_axes()`, `ticker_colors()` and the palette
  constants in `src/plotting.py`. Assign colours per ticker via `ticker_colors` so a
  ticker keeps its colour across every chart. Add direct end-of-line labels on
  multi-series line charts (the aqua/yellow slots are sub-3:1 contrast — relief rule).
- **Returns / volatility / risk** → `src/returns.py`, `src/risk.py` (and
  `src/monte_carlo.py`). Honour log-vs-simple returns and 252-day annualisation.
- **Any new chart type not covered by `src/plotting.py`** → load the **dataviz** skill
  first, and validate any new palette with its `validate_palette.js`.
- If a cell grows into reusable logic, move it into `src/` (with a pytest) and import it
  back — notebooks are for exploration/presentation, not the home of core logic.

### 5. Execute and verify (required before done)
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
uv run ruff check .
```
Then programmatically assert there are **no error outputs**:
```python
import nbformat
nb = nbformat.read("notebooks/<name>.ipynb", as_version=4)
errs = sum(1 for c in nb.cells if c.cell_type=="code"
           for o in c.get("outputs", []) if o.get("output_type")=="error")
assert errs == 0, f"{errs} cell errors"
```
If any cell errored, read the traceback, fix the builder script, rebuild, re-run.
Optionally extract 1–2 chart PNGs and eyeball them for layout/overflow before finishing.

## Reference
`notebooks/2026-07-05_risk_analytics.ipynb` is the canonical example of this structure.
