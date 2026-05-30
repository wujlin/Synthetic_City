#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthpop.plot_style import OKABE_ITO, add_panel_label, despine, paper_style, save_figure


VARIABLE_LABELS = {
    "AGEP_bin": "Age",
    "SEX": "Gender",
    "SCHL_allpop": "Education",
    "ESR_allpop": "Employment",
    "EARN_16p_bin": "Income",
}

GROUP_LABELS = {
    "children_0_17": "Children",
    "female": "Female",
    "employed": "Employed",
    "bachelor_plus": "Bachelor's+",
    "income_100k_plus": "$100k+",
}

GROUP_ORDER = ["children_0_17", "female", "employed", "bachelor_plus", "income_100k_plus"]


def _pct(x: float, _pos=None) -> str:
    return f"{100 * x:.2f}%"


def _variable_order(df: pd.DataFrame) -> list[str]:
    return [v for v in VARIABLE_LABELS if v in set(df["variable"])]


def build_figure(
    *,
    puma_csv: Path,
    tract_summary_csv: Path,
    tract_detail_csv: Path,
    out_pdf: Path,
    out_png: Path,
) -> None:
    puma = pd.read_csv(puma_csv)
    tract_summary = pd.read_csv(tract_summary_csv)
    tract_detail = pd.read_csv(tract_detail_csv)

    var_order = _variable_order(puma)
    puma["variable_label"] = puma["variable"].map(VARIABLE_LABELS)
    tract_summary["group_label"] = tract_summary["attribute_group"].map(GROUP_LABELS)
    tract_detail["group_label"] = tract_detail["attribute_group"].map(GROUP_LABELS)

    colors = {
        "puma": OKABE_ITO["blue"],
        "tract": OKABE_ITO["bluish_green"],
        "soft": OKABE_ITO["orange"],
        "gray": "#6F6F6F",
    }

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.9))
        ax_a, ax_b, ax_c, ax_d = axes.ravel()

        # (a) PUMA-level census-data gaps by variable.
        box_data = [puma.loc[puma["variable"] == v, "abs_share_gap"].to_numpy(dtype=float) for v in var_order]
        bp = ax_a.boxplot(
            box_data,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            medianprops={"color": "white", "linewidth": 1.2},
            boxprops={"linewidth": 0.9, "color": "#2F5F8F"},
            whiskerprops={"linewidth": 0.9, "color": "#2F5F8F"},
            capprops={"linewidth": 0.9, "color": "#2F5F8F"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#8CB7D8")
            patch.set_alpha(0.85)
        ax_a.set_xticks(range(1, len(var_order) + 1), [VARIABLE_LABELS[v] for v in var_order], rotation=25, ha="right")
        ax_a.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax_a.set_ylabel("Absolute share gap")
        ax_a.set_title("PUMA-level census data")
        ax_a.text(
            0.02,
            0.95,
            "mean gap = $6.65\\times10^{-5}$\nmax gap = $4.20\\times10^{-4}$",
            transform=ax_a.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#333333",
        )
        despine(ax_a)
        add_panel_label(ax_a, "a", dx=-28)

        # (b) PUMA-level maximum gaps by variable.
        max_gap = puma.groupby("variable", observed=True)["abs_share_gap"].max().reindex(var_order)
        ax_b.bar(
            range(len(var_order)),
            max_gap.to_numpy(dtype=float),
            color="#8CB7D8",
            edgecolor="#2F5F8F",
            linewidth=0.8,
        )
        ax_b.set_xticks(range(len(var_order)), [VARIABLE_LABELS[v] for v in var_order], rotation=25, ha="right")
        ax_b.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax_b.set_ylabel("Maximum share gap")
        ax_b.set_title("Largest PUMA-level deviations")
        despine(ax_b)
        add_panel_label(ax_b, "b", dx=-28)

        # (c) Tract-level census-data agreement.
        tract_summary = tract_summary.set_index("attribute_group").reindex(GROUP_ORDER).reset_index()
        bar_colors = [
            colors["tract"],
            colors["tract"],
            colors["soft"],
            colors["soft"],
            colors["soft"],
        ]
        ax_c.bar(
            range(len(tract_summary)),
            tract_summary["share_spearman"].to_numpy(dtype=float),
            color=bar_colors,
            edgecolor="#3F3F3F",
            linewidth=0.7,
        )
        ax_c.set_ylim(0, 1.04)
        ax_c.set_xticks(range(len(tract_summary)), tract_summary["group_label"], rotation=20, ha="right")
        ax_c.set_ylabel("Spearman correlation")
        ax_c.set_title("Tract-level census data")
        ax_c.axhline(0.9, color="#999999", linestyle=":", linewidth=1.0)
        despine(ax_c)
        add_panel_label(ax_c, "c", dx=-28)

        # (d) Tract-level weighted share error.
        ax_d.bar(
            range(len(tract_summary)),
            tract_summary["share_weighted_mae"].to_numpy(dtype=float),
            color=bar_colors,
            edgecolor="#3F3F3F",
            linewidth=0.7,
        )
        ax_d.set_xticks(range(len(tract_summary)), tract_summary["group_label"], rotation=20, ha="right")
        ax_d.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{100*x:.1f}%"))
        ax_d.set_ylabel("Weighted absolute share gap")
        ax_d.set_title("Tract-level share error")
        despine(ax_d)
        add_panel_label(ax_d, "d", dx=-28)

        fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.15, wspace=0.30, hspace=0.52)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--puma-csv",
        type=Path,
        default=REPO_ROOT / "outputs/sigspatial_result_assets_20260517/michigan_puma_condition_consistency.csv",
    )
    parser.add_argument(
        "--tract-summary-csv",
        type=Path,
        default=REPO_ROOT / "outputs/sigspatial_result_assets_20260517/michigan_tract_acs_consistency_summary.csv",
    )
    parser.add_argument(
        "--tract-detail-csv",
        type=Path,
        default=REPO_ROOT / "outputs/sigspatial_result_assets_20260517/michigan_tract_acs_consistency_detail.csv",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=REPO_ROOT / "SigSpatial2026_spop/figures/fig_04_input_consistency.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=REPO_ROOT / "SigSpatial2026_spop/figures/fig_04_input_consistency.png",
    )
    args = parser.parse_args()
    build_figure(
        puma_csv=args.puma_csv,
        tract_summary_csv=args.tract_summary_csv,
        tract_detail_csv=args.tract_detail_csv,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()
