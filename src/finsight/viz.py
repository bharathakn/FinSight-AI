"""Shared matplotlib style for every report figure (validated reference palette, light mode).

Categorical slots are assigned in fixed order and follow the *entity* across all figures:
LEGITIMATE = slot 1 (blue), FRAUDULENT = slot 2 (orange), BORDERLINE = slot 3 (aqua).
Only the first three slots are used where all pairs are on screen (scatter, maps) — they are the
ones that pass the all-pairs CVD check. Aqua is < 3:1 on the surface, so every figure ships a
legend and/or direct labels (relief rule).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from finsight import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LABEL_COLORS = {"LEGITIMATE": SERIES[0], "FRAUDULENT": SERIES[1], "BORDERLINE": SERIES[2]}
LABEL_ORDER = ["LEGITIMATE", "FRAUDULENT", "BORDERLINE"]
VERSION_COLORS = {"v1.0": SERIES[0], "v1.1": SERIES[1]}

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("finsight_seq", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("finsight_div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19b9a", "#b42d2d"])

FIG_DIR = config.REPORTS_DIR / "figures"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK_2, "text.color": INK,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def subtitle(ax, text: str) -> None:
    """Secondary line under a left-aligned title."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def save(fig, name: str, directory: Path | None = None) -> Path:
    d = Path(directory or FIG_DIR)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path)
    return path
