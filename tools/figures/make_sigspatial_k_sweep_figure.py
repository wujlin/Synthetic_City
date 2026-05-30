#!/usr/bin/env python3
"""Draw the supplementary K-sweep figure for the SIGSPATIAL manuscript."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthpop.plot_style import OKABE_ITO, PaperStyle, despine, paper_style, save_figure


SOURCE_CSV = REPO_ROOT / "SigSpatial2026_spop" / "figure_source_data" / "k_sweep_poilodes_20260525.csv"
OUT_PDF = REPO_ROOT / "SigSpatial2026_spop" / "figures" / "fig_s05_k_sweep_poilodes.pdf"
OUT_PNG = REPO_ROOT / "SigSpatial2026_spop" / "figures" / "fig_s05_k_sweep_poilodes.png"


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    df["seed_sd"] = pd.to_numeric(df["seed_sd"], errors="coerce")

    x = df["k"].to_numpy()
    y = df["mean_tvd"].to_numpy()
    yerr = df["seed_sd"].to_numpy()
    has_sd = ~np.isnan(yerr)

    style = PaperStyle(
        font_size=10.0,
        axes_labelsize=11.0,
        tick_labelsize=9.5,
        legend_fontsize=9.0,
        axes_linewidth=1.05,
        lines_linewidth=1.7,
        lines_markersize=4.8,
    )
    with paper_style(style):
        fig, ax = plt.subplots(figsize=(4.9, 2.9))

        line_color = OKABE_ITO["blue"]
        selected_color = OKABE_ITO["orange"]
        ax.plot(
            x,
            y,
            color=line_color,
            marker="o",
            markersize=4.8,
            linewidth=1.7,
            label="Mean TVD",
            zorder=2,
        )
        ax.errorbar(
            x[has_sd],
            y[has_sd],
            yerr=yerr[has_sd],
            fmt="none",
            ecolor=line_color,
            elinewidth=0.85,
            capsize=2.4,
            alpha=0.75,
            zorder=1,
        )

        selected_k = 960
        selected_y = float(df.loc[df["k"].eq(selected_k), "mean_tvd"].iloc[0])
        ax.axvline(selected_k, color="#666666", linestyle="--", linewidth=0.8, alpha=0.55, zorder=0)
        ax.scatter(
            [selected_k],
            [selected_y],
            s=44,
            color=selected_color,
            edgecolor="black",
            linewidth=0.55,
            label=r"Selected $K=960$",
            zorder=3,
        )

        ax.set_xlabel(r"Coarse state-space size $K$")
        ax.set_ylabel("Held-out TVD")
        ax.set_xticks(x)
        ax.set_xlim(240, 2480)
        ax.set_ylim(0.1182, 0.1240)
        ax.grid(False)
        ax.legend(frameon=False, loc="upper right", handlelength=1.5, borderaxespad=0.2)
        despine(ax)
        fig.tight_layout(pad=0.35)

        save_figure(fig, OUT_PDF)
        save_figure(fig, OUT_PNG, dpi=300)


if __name__ == "__main__":
    main()
