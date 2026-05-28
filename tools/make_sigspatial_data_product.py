#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_style import OKABE_ITO, PaperStyle, add_panel_label, despine, paper_style, save_figure


def _fmt_count(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def _pct(x: float, _pos=None) -> str:
    return f"{100 * x:.1f}%"


def _card(ax, xy, width, height, title: str, value: str, subtitle: str, color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.0,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.045, y + height - 0.055, title, transform=ax.transAxes, fontsize=8.5, color="#333333", va="top")
    ax.text(x + 0.045, y + height * 0.47, value, transform=ax.transAxes, fontsize=18, fontweight="bold", color=color, va="center")
    ax.text(x + 0.045, y + 0.055, subtitle, transform=ax.transAxes, fontsize=7.6, color="#555555", va="bottom")


def build_figure(*, qc_json: Path, out_pdf: Path, out_png: Path) -> None:
    qc = json.loads(qc_json.read_text(encoding="utf-8"))
    n_people = int(qc["national_total_synthetic_persons"])
    n_pumas = int(qc["number_of_pumas_completed"])
    n_states = int(qc["number_of_states_completed"])
    workers = int(qc["work_eligible_persons"])
    home_rate = float(qc["home_assignment_rate"])
    work_rate = float(qc["work_coordinate_assignment_rate_among_workers"])
    home_missing = int(qc["missing_home_coordinate_count"])
    work_missing = int(qc["missing_work_coordinate_count_among_workers"])
    assigned_home = n_people - home_missing
    assigned_work = workers - work_missing

    style = PaperStyle(font_size=8.5, axes_labelsize=9.5, axes_titlesize=10.0, tick_labelsize=8.3, legend_fontsize=8.0)
    with paper_style(style):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.15, 1.10, 1.0]})
        ax_a, ax_b, ax_c = axes

        ax_a.set_axis_off()
        _card(ax_a, (0.02, 0.56), 0.94, 0.36, "Synthetic individuals", f"{_fmt_count(n_people)}", "national total", OKABE_ITO["blue"])
        _card(ax_a, (0.02, 0.10), 0.44, 0.34, "PUMAs", f"{n_pumas:,}", "completed units", OKABE_ITO["bluish_green"])
        _card(ax_a, (0.52, 0.10), 0.44, 0.34, "Jurisdictions", f"{n_states}", "50 states + D.C.", OKABE_ITO["orange"])
        add_panel_label(ax_a, "a", dx=-20)

        labels = ["Home coordinates", "Work coordinates\n(workers only)"]
        rates = [home_rate, work_rate]
        ax_b.barh([1, 0], rates, color=[OKABE_ITO["bluish_green"], OKABE_ITO["orange"]], edgecolor="#333333", linewidth=0.7)
        ax_b.barh([1, 0], [1 - home_rate, 1 - work_rate], left=rates, color="#E5E5E5", edgecolor="#AAAAAA", linewidth=0.5)
        ax_b.set_xlim(0.95, 1.0)
        ax_b.set_yticks([1, 0], labels)
        ax_b.xaxis.set_major_formatter(FuncFormatter(_pct))
        ax_b.set_xlabel("Assignment rate")
        ax_b.set_title("Coordinate assignment")
        for y, r in zip([1, 0], rates):
            ax_b.text(r - 0.0007, y, f"{100*r:.3f}%", va="center", ha="right", fontsize=9, color="#222222")
        despine(ax_b)
        add_panel_label(ax_b, "b", dx=-26)

        ax_c.bar(
            [0, 1],
            [assigned_home / 1e6, assigned_work / 1e6],
            color=[OKABE_ITO["bluish_green"], OKABE_ITO["orange"]],
            edgecolor="#333333",
            linewidth=0.7,
        )
        ax_c.set_xticks([0, 1], ["Home\npoints", "Work\npoints"])
        ax_c.set_ylabel("Assigned coordinates (millions)")
        ax_c.set_title("Point-location records")
        for x, value in zip([0, 1], [assigned_home, assigned_work]):
            ax_c.text(x, value / 1e6 + 8, _fmt_count(value), ha="center", va="bottom", fontsize=9)
        despine(ax_c)
        add_panel_label(ax_c, "c", dx=-26)

        fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.25, wspace=0.42)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qc-json",
        type=Path,
        default=REPO_ROOT / "outputs/sigspatial_result_assets_20260517/national_qc_summary.json",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=REPO_ROOT / "SigSpatial2026_spop/figures/fig_06_national_data_product.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=REPO_ROOT / "SigSpatial2026_spop/figures/fig_06_national_data_product.png",
    )
    args = parser.parse_args()
    build_figure(qc_json=args.qc_json, out_pdf=args.out_pdf, out_png=args.out_png)


if __name__ == "__main__":
    main()
