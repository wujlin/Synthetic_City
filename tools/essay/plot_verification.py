#!/usr/bin/env python3
from __future__ import annotations

"""
Plot fig_verification.pdf (1x3 panels):
(a) raw marginal consistency
(b) convergence vs epoch
(c) MC stability vs draw count
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.plot_style import OKABE_ITO, add_panel_label, despine, paper_style, save_figure


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="plot_verification")
    ap.add_argument("--raw_metrics_json", required=True, help="internal_acs_holdout.json")
    ap.add_argument("--convergence_json", required=True, help="exp2 convergence json")
    ap.add_argument("--mc_json", required=True, help="exp3 mc json")
    ap.add_argument("--condition", default="pairwise")
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_png", default="")
    args = ap.parse_args()

    raw = _load_json(pathlib.Path(args.raw_metrics_json).expanduser().resolve())
    conv = _load_json(pathlib.Path(args.convergence_json).expanduser().resolve())
    mc = _load_json(pathlib.Path(args.mc_json).expanduser().resolve())

    cond = str(args.condition)
    by_cond = raw.get("by_condition", {})
    if cond not in by_cond:
        raise SystemExit(f"condition not found in raw_metrics_json: {cond}")
    overall = by_cond[cond]["overall"]

    attrs = ["age", "sex", "income", "schl", "esr"]
    y_vals = [float(overall[f"tvd_{a}_raw"]["mean"]) for a in attrs]

    rows_conv = sorted(conv.get("rows", []), key=lambda x: int(x["epoch"]))
    if not rows_conv:
        raise SystemExit("convergence_json has no rows.")
    x_epoch = np.array([int(r["epoch"]) for r in rows_conv], dtype=float)
    y_conv = np.array([float(r["tvd_joint"]["mean"]) for r in rows_conv], dtype=float)
    y_conv_std = np.array([float(r["tvd_joint"]["std"]) for r in rows_conv], dtype=float)

    rows_mc = sorted(mc.get("rows", []), key=lambda x: int(x["n_draws"]))
    if not rows_mc:
        raise SystemExit("mc_json has no rows.")
    x_mc = np.array([int(r["n_draws"]) for r in rows_mc], dtype=float)
    y_mc = np.array([float(r["mean_over_seeds"]) for r in rows_mc], dtype=float)
    y_mc_std = np.array([float(r["std_over_seeds"]) for r in rows_mc], dtype=float)

    C1 = OKABE_ITO["blue"]
    C2 = OKABE_ITO["vermillion"]
    C3 = OKABE_ITO["bluish_green"]

    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
        fig.subplots_adjust(wspace=0.30)

        # (a) marginal consistency
        ax = axes[0]
        ax.bar(np.arange(len(attrs)), y_vals, color=C1, alpha=0.80, edgecolor="white", linewidth=0.6)
        ax.axhline(0.002, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(np.arange(len(attrs)))
        ax.set_xticklabels(attrs)
        ax.set_ylabel("TVD")
        ax.set_title("Marginal consistency (raw)")
        despine(ax)
        add_panel_label(ax, "a", dx=-18)

        # (b) convergence
        ax = axes[1]
        ax.plot(x_epoch, y_conv, color=C2, linewidth=2.0)
        ax.fill_between(x_epoch, y_conv - y_conv_std, y_conv + y_conv_std, color=C2, alpha=0.18)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("TVD")
        ax.set_title("Training convergence")
        despine(ax)
        add_panel_label(ax, "b", dx=-18)

        # (c) MC stability
        ax = axes[2]
        ax.plot(x_mc, y_mc, color=C3, linewidth=2.0)
        ax.fill_between(x_mc, y_mc - y_mc_std, y_mc + y_mc_std, color=C3, alpha=0.20)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Draw count")
        ax.set_ylabel("TVD")
        ax.set_title("MC stability")
        despine(ax)
        add_panel_label(ax, "c", dx=-18)

        out_pdf = pathlib.Path(args.out_pdf).expanduser().resolve()
        save_figure(fig, out_pdf)
        if str(args.out_png).strip():
            fig.savefig(pathlib.Path(args.out_png).expanduser().resolve(), dpi=220)
        plt.close(fig)

    print(f"[ok] wrote: {pathlib.Path(args.out_pdf).expanduser().resolve()}")
    if str(args.out_png).strip():
        print(f"[ok] wrote: {pathlib.Path(args.out_png).expanduser().resolve()}")


if __name__ == "__main__":
    main()
