"""
Supplementary Figure: Person-level diffusion fails to recover copula.

Shows 5-fold cross-validation results comparing person-level conditional
diffusion (copula TVD) against the training-set average baseline.

Usage:
    python tools/essay/plot_si_person_level.py

Output:
    Essay/figures/fig_si_person_level.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from plot_style import OKABE_ITO, paper_style, save_figure, despine, add_panel_label

OUT_DIR = REPO / "Essay" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_DIFF = OKABE_ITO["blue"]
C_BASE = OKABE_ITO["vermillion"]
C_GRAY = OKABE_ITO["gray"]


def load_data():
    p = (REPO / "outputs" / "_exp2_attrdiff_mi_puma20_20260218T095102Z"
         / "copula_baseline_demo_race_puma.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def plot():
    data = load_data()
    folds = sorted(data["by_fold"].keys(), key=int)

    # Per-fold mean copula TVD
    baseline_means = []
    diffusion_means = []
    for fold_id in folds:
        pumas = data["by_fold"][fold_id]["by_puma"]
        b_vals = [v["baseline_copula_tvd_age_income"] for v in pumas.values()]
        d_vals = [v["diffusion_copula_tvd_age_income"] for v in pumas.values()]
        baseline_means.append(np.mean(b_vals))
        diffusion_means.append(np.mean(d_vals))

    # Per-PUMA scatter (all folds pooled)
    all_baseline = []
    all_diffusion = []
    for fold_id in folds:
        pumas = data["by_fold"][fold_id]["by_puma"]
        for v in pumas.values():
            all_baseline.append(v["baseline_copula_tvd_age_income"])
            all_diffusion.append(v["diffusion_copula_tvd_age_income"])

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
        fig.subplots_adjust(wspace=0.38)

        # (a) Paired bar chart: per-fold mean copula TVD
        ax = axes[0]
        x = np.arange(len(folds))
        w = 0.35
        ax.bar(x - w / 2, baseline_means, w, color=C_BASE, label="Training-set average",
               edgecolor="white", linewidth=0.5)
        ax.bar(x + w / 2, diffusion_means, w, color=C_DIFF, label="Person-level diffusion",
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {int(f)}" for f in folds], fontsize=8)
        ax.set_ylabel("Mean copula TVD")
        ax.legend(frameon=False, fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        despine(ax)
        add_panel_label(ax, "a")

        # (b) Per-PUMA scatter: baseline vs diffusion
        ax = axes[1]
        ax.scatter(all_baseline, all_diffusion, s=12, alpha=0.5, color=C_DIFF,
                   edgecolors="white", linewidth=0.3, rasterized=True)
        lim = [0, max(max(all_baseline), max(all_diffusion)) * 1.08]
        ax.plot(lim, lim, color=C_GRAY, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Baseline copula TVD\n(training-set average)")
        ax.set_ylabel("Person-level diffusion\ncopula TVD")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")

        n_above = sum(1 for b, d in zip(all_baseline, all_diffusion) if d > b)
        n_total = len(all_baseline)
        ax.text(0.05, 0.95, f"{n_above}/{n_total} PUMAs\nabove diagonal",
                transform=ax.transAxes, fontsize=7, va="top", ha="left",
                color=C_DIFF)
        despine(ax)
        add_panel_label(ax, "b")

        save_figure(fig, OUT_DIR / "fig_si_person_level.pdf")
        fig.savefig(OUT_DIR / "fig_si_person_level.png", dpi=200)
        print(f"  → {OUT_DIR / 'fig_si_person_level.pdf'}")
        print(f"  → {OUT_DIR / 'fig_si_person_level.png'}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating SI person-level figure...")
    plot()
    print("Done.")
