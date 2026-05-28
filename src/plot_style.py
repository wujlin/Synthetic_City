"""
Shared publication-style plotting defaults.

Goals:
- use consistent fonts, sizes, line widths, and colors;
- keep export parameters stable across LaTeX composites;
- provide robust panel labels for multi-panel figures.

Conventions:
- avoid a seaborn dependency;
- expose global rcParams plus an optional rc_context wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm


# Okabe-Ito colorblind-friendly palette.
OKABE_ITO: Dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#777777",
}


@dataclass(frozen=True)
class PaperStyle:
    # These defaults are calibrated for readability after LaTeX embedding.
    # Typical use:
    # - half-width panels: inserted near 0.48\\linewidth, about 3.1in wide;
    # - full-width panels: inserted near \\linewidth, about 6.5in wide.
    # Keep script-side figsize close to the final embedded width.
    # 
    # 2024-12 adjustment: smaller labels reduce crowding in half-width panels.
    font_size: float = 11.0
    axes_labelsize: float = 12.0
    axes_titlesize: float = 11.0
    tick_labelsize: float = 10.0
    legend_fontsize: float = 9.0
    axes_linewidth: float = 1.2
    lines_linewidth: float = 2.4
    lines_markersize: float = 5.5
    figure_dpi: int = 150
    savefig_dpi: int = 300


FIGSIZE_FULL: tuple[float, float] = (6.5, 4.0)
"""Figure size for full-width panels, close to LaTeX \\linewidth."""

FIGSIZE_HALF: tuple[float, float] = (3.2, 2.45)
"""Figure size for half-width panels, close to 0.48\\linewidth."""


def _resolve_times_font() -> tuple[str, list[str]]:
    """
    Prefer Times New Roman when available; otherwise fall back to STIXGeneral.
    """

    times_paths = [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ]
    if any(p.exists() for p in times_paths):
        for p in times_paths:
            if p.exists():
                fm.fontManager.addfont(str(p))
        return "Times New Roman", ["Times New Roman"]
    return "STIXGeneral", ["STIXGeneral", "DejaVu Serif"]


def paper_rcparams(style: PaperStyle | None = None) -> Dict[str, object]:
    style = style or PaperStyle()
    font_family, serif_fallback = _resolve_times_font()

    # Limit BLAS threads to reduce noise in WSL/container environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    return {
        "font.family": font_family,
        "font.serif": serif_fallback,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.grid": False,
        "axes.linewidth": style.axes_linewidth,
        "lines.linewidth": style.lines_linewidth,
        "lines.markersize": style.lines_markersize,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "font.size": style.font_size,
        "axes.titlesize": style.axes_titlesize,
        "axes.labelsize": style.axes_labelsize,
        "xtick.labelsize": style.tick_labelsize,
        "ytick.labelsize": style.tick_labelsize,
        "legend.fontsize": style.legend_fontsize,
        # Avoid Type 3 fonts for cleaner LaTeX/printing output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": style.figure_dpi,
        "savefig.dpi": style.savefig_dpi,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }


@contextmanager
def paper_style(style: PaperStyle | None = None) -> Iterator[None]:
    """Apply the paper style inside a context manager."""

    with mpl.rc_context(paper_rcparams(style=style)):
        yield


def save_figure(
    fig: mpl.figure.Figure,
    out_path: str | Path,
    *,
    dpi: Optional[int] = None,
) -> None:
    """
    Save figures without bbox_inches='tight' so panel extents remain stable
    across LaTeX composites.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)


def despine(ax: mpl.axes.Axes) -> None:
    """Remove top and right spines."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_paper_style(style: PaperStyle | None = None) -> None:
    """Apply the style to global rcParams."""

    mpl.rcParams.update(paper_rcparams(style=style))


def add_panel_label(
    ax: mpl.axes.Axes,
    label: str,
    *,
    # Anchor to the upper-left axes corner and use fixed point offsets.
    x: float = 0.0,
    y: float = 1.0,
    dx: float = -42.0,
    dy: float = 4.0,
    fontsize: float | None = None,
) -> mpl.text.Text:
    """
    Add a panel label outside the upper-left corner of an axes.

    Layout rules:
    - use axes-fraction coordinates for the anchor;
    - use point offsets so long tick labels do not shift the label.
    """

    label_text = str(label)
    if not (label_text.startswith("(") and label_text.endswith(")")):
        label_text = f"({label_text})"

    t = ax.annotate(
        label_text,
        xy=(x, y),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=fontsize or float(mpl.rcParams.get("axes.labelsize", 12.0)),
        color="black",
        annotation_clip=False,
    )
    t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
    return t
