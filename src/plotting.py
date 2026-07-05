"""Shared matplotlib styling: a validated, colorblind-safe categorical palette,
a diverging blue/red scale for signed magnitudes (correlation, P&L), and a
light, recessive chart chrome. Keeps every chart in the notebook visually
consistent instead of relying on matplotlib defaults per cell.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

# Fixed-order categorical hues (validated: worst adjacent CVD delta-E 24.2).
# Assigned by position, never re-cycled when a filter changes series count.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#4a3aa7",  # 4 violet
    "#e34948",  # 5 red (reserved as slot 5 here, not a status color)
]

# Diverging poles for signed magnitudes (correlation, returns/P&L).
DIVERGING_NEG = "#e34948"  # red
DIVERGING_MID = "#f0efec"  # neutral gray midpoint
DIVERGING_POS = "#2a78d6"  # blue

# Status colors — reserved for state (thresholds), never reused as series colors.
STATUS_CRITICAL = "#d03b3b"
STATUS_WARNING = "#fab219"
STATUS_GOOD = "#0ca30c"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue (positive) <-> gray (zero) <-> red (negative) colormap for correlation/P&L."""
    return LinearSegmentedColormap.from_list(
        "diverging_blue_red", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS]
    )


def ticker_colors(tickers: Sequence[str]) -> dict[str, str]:
    """Assign each ticker a fixed-order categorical hue (stable across charts)."""
    if len(tickers) > len(CATEGORICAL):
        raise ValueError(f"only {len(CATEGORICAL)} categorical slots defined")
    return dict(zip(tickers, CATEGORICAL))


def style_axes(ax: Axes, *, grid_axis: str = "y") -> None:
    """Apply recessive chrome: light gridlines, muted axes, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(TEXT_PRIMARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def new_figure(figsize: tuple[float, float] = (10, 5)) -> tuple[plt.Figure, Axes]:
    """Figure/axes pair with the shared surface color and chrome already applied."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    style_axes(ax)
    return fig, ax
