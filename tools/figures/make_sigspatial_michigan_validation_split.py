#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthpop.plot_style import PaperStyle, despine, paper_style, save_figure


PAIR_LABELS = {
    "AGEP_bin__SEX": "Age-Gender",
    "AGEP_bin__SCHL_allpop": "Age-Edu",
    "AGEP_bin__ESR_allpop": "Age-Emp",
    "AGEP_bin__EARN_16p_bin": "Age-Inc",
    "SEX__SCHL_allpop": "Gender-Edu",
    "SEX__ESR_allpop": "Gender-Emp",
    "SEX__EARN_16p_bin": "Gender-Inc",
    "SCHL_allpop__ESR_allpop": "Edu-Emp",
    "SCHL_allpop__EARN_16p_bin": "Edu-Inc",
    "ESR_allpop__EARN_16p_bin": "Emp-Inc",
}


def _plot_method_comparison(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    methods = [
        ("Proposed framework", "pipeline_tvd_mean", "#4d908e"),
        ("IPF", "ipf_tvd", "#b07243"),
        ("CO", "tvd_co_national", "#8a6f9e"),
        ("One-stage\nDDPM", "one_shot_tvd_mean", "#d99b5d"),
    ]
    data = [df[col].to_numpy(dtype=float) for _, col, _ in methods]

    style = PaperStyle(font_size=8.5, axes_labelsize=9.0, axes_titlesize=9.0, tick_labelsize=8.0, legend_fontsize=8.0)
    with paper_style(style):
        fig, ax = plt.subplots(figsize=(3.45, 3.0))
        bp = ax.boxplot(
            data,
            positions=np.arange(1, len(methods) + 1),
            widths=0.50,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.1},
            whiskerprops={"color": "#666666", "linewidth": 0.9},
            capprops={"color": "#666666", "linewidth": 0.9},
        )
        rng = np.random.default_rng(20260429)
        for i, ((_, _, color), vals) in enumerate(zip(methods, data), start=1):
            bp["boxes"][i - 1].set_facecolor(color)
            bp["boxes"][i - 1].set_alpha(0.55)
            bp["boxes"][i - 1].set_edgecolor("none")
            jitter = rng.normal(0.0, 0.035, size=vals.size)
            ax.scatter(np.full(vals.size, i) + jitter, vals, s=8.0, color=color, alpha=0.34, linewidth=0)
            ax.scatter(i, np.mean(vals), marker="D", s=30, color="#222222", zorder=5)

        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(["Proposed\nframework", "IPF", "CO", "One-stage\nDDPM"], rotation=0, ha="center")
        ax.set_ylabel("TVD")
        ax.set_ylim(0.088, 0.19)
        despine(ax)
        fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.20)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def _plot_pairwise_decomposition(pair_df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    models = [
        ("IPF", "ipf", "#b07243"),
        ("CO", "co_national", "#8a6f9e"),
        ("One-stage DDPM", "one_shot_ddpm", "#d99b5d"),
    ]
    metric = "gap_vs_hierarchical_seed_mean"
    plot_df = pair_df.loc[pair_df["model"].isin([model for _, model, _ in models])].copy()
    if plot_df.empty or metric not in plot_df.columns:
        raise ValueError(f"Pairwise data must contain model rows and {metric}.")

    order = (
        plot_df.loc[plot_df["model"].eq("ipf")]
        .sort_values(metric, ascending=False)["pair"]
        .astype(str)
        .tolist()
    )
    if not order:
        order = sorted(plot_df["pair"].astype(str).unique().tolist())

    x = np.arange(len(order), dtype=float)
    width = 0.23
    offsets = [-width, 0.0, width]
    all_vals: list[float] = []

    style = PaperStyle(font_size=8.5, axes_labelsize=9.0, axes_titlesize=9.0, tick_labelsize=7.7, legend_fontsize=8.0)
    with paper_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 2.85))
        for (label, model, color), offset in zip(models, offsets):
            vals = (
                plot_df.loc[plot_df["model"].eq(model)]
                .set_index("pair")
                .reindex(order)[metric]
                .astype(float)
                .to_numpy()
            )
            all_vals.extend(float(v) for v in vals if np.isfinite(v))
            ax.bar(x + offset, vals, width=width, color=color, alpha=0.86, label=label)

        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([PAIR_LABELS.get(pair, pair) for pair in order], rotation=35, ha="right")
        ax.set_ylabel(r"$\Delta$TVD (baseline - proposed)")
        if all_vals:
            ymin = min(0.0, min(all_vals))
            ymax = max(0.0, max(all_vals))
            pad = max((ymax - ymin) * 0.13, 0.001)
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.legend(frameon=False, loc="upper right", ncol=3)
        despine(ax)
        fig.subplots_adjust(left=0.115, right=0.99, top=0.92, bottom=0.32)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--co-by-puma-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs/_paper1_CO_baselines_michigan_20260429T000000Z/metrics/co_baselines_by_puma.csv",
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs/_paper1_michigan_pairwise_model_comparison_20260501T010827Z/metrics/michigan_pairwise_tvd_gap_vs_hierarchical_model_mean.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "SigSpatial2026_spop/figures")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "SigSpatial2026_spop/figure_source_data",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)

    by_puma = pd.read_csv(args.co_by_puma_csv)
    pairwise = pd.read_csv(args.pairwise_csv)
    by_puma.to_csv(args.source_dir / "michigan_method_comparison_by_puma.csv", index=False)
    pairwise.to_csv(args.source_dir / "michigan_pairwise_decomposition.csv", index=False)

    _plot_method_comparison(
        by_puma,
        args.out_dir / "fig_04_michigan_method_comparison.pdf",
        args.out_dir / "fig_04_michigan_method_comparison.png",
    )
    _plot_pairwise_decomposition(
        pairwise,
        args.out_dir / "fig_05_michigan_pairwise_decomposition.pdf",
        args.out_dir / "fig_05_michigan_pairwise_decomposition.png",
    )


if __name__ == "__main__":
    main()
