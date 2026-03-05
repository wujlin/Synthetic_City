#!/usr/bin/env python3
from __future__ import annotations

"""
Experiment 1: visualize output examples for representative Michigan PUMAs.

Outputs:
- figures/fig_output_examples.pdf
- metrics/output_examples.json
"""

import argparse
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.plot_style import OKABE_ITO, add_panel_label, paper_style, save_figure, despine
from src.synthpop.model.diffusion_tabular import DiffusionTabularModel
from _eval_5var_common import (
    _marginal_from_joint,
    _summ,
    _tvd,
    infer_one_region,
    load_eval_data,
    write_json,
)


def _select_representatives(
    *,
    data: Any,
    n_examples: int,
    quantiles: list[float],
) -> list[int]:
    test = data.test_idx
    train = data.train_idx
    w = np.maximum(data.totals[train], 0.0)
    if float(w.sum()) <= 0:
        p_global = np.mean(data.p_joint[train], axis=0)
    else:
        p_global = np.sum(data.p_joint[train] * w.reshape(-1, 1), axis=0) / float(w.sum())
    p_global = p_global / max(float(p_global.sum()), 1e-12)

    tvd_test = np.array([_tvd(data.p_joint[i], p_global) for i in test], dtype=float)
    order = np.argsort(tvd_test)
    picked: list[int] = []
    for q in quantiles[:n_examples]:
        target = float(np.quantile(tvd_test, q))
        cand = order[np.argmin(np.abs(tvd_test[order] - target))]
        idx = int(test[cand])
        if idx not in picked:
            picked.append(idx)
    while len(picked) < int(n_examples):
        idx = int(test[order[len(picked)]])
        if idx not in picked:
            picked.append(idx)
    return picked[: int(n_examples)]


def _profile_text(data: Any, idx: int) -> str:
    p = data.p_joint[idx]
    age = _marginal_from_joint(p, shape=data.shape, axis=0)
    income = _marginal_from_joint(p, shape=data.shape, axis=2)
    schl = _marginal_from_joint(p, shape=data.shape, axis=3)
    esr = _marginal_from_joint(p, shape=data.shape, axis=4)
    return (
        f"age_bin0={age[0]:.2f}, income_bin0={income[0]:.2f}\n"
        f"schl_bin0={schl[0]:.2f}, esr_bin0={esr[0]:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp1_output_examples")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--condition", choices=["none", "marginal", "pairwise", "marginal_pairwise"], default="pairwise")
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "mi_kfold"], default="leave_mi_out")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--posthoc_ipf", action="store_true")
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    out_pdf = pathlib.Path(args.out_pdf).expanduser().resolve()
    out_json = pathlib.Path(args.out_json).expanduser().resolve()
    if not joint_csv.exists():
        raise SystemExit(f"joint_wide_csv not found: {joint_csv}")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    data = load_eval_data(
        joint_wide_csv=joint_csv,
        condition_names=[str(args.condition)],
        eval_mode=str(args.eval_mode),
        n_folds=int(args.n_folds),
        fold_index=int(args.fold_index),
        seed=int(args.seed),
    )

    model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))
    model.load(ckpt)
    quantiles = [0.90, 0.75, 0.50, 0.25]
    rep_idx = _select_representatives(data=data, n_examples=int(args.n_examples), quantiles=quantiles)

    C_TRUE = OKABE_ITO["blue"]
    C_PRED = OKABE_ITO["vermillion"]
    C_RES = OKABE_ITO["gray"]

    records: list[dict[str, Any]] = []
    with paper_style():
        fig, axes = plt.subplots(2, int(args.n_examples), figsize=(12.5, 4.8), sharex=False)
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        for j, idx in enumerate(rep_idx):
            p_true = data.p_joint[idx]
            p_hat_raw, p_hat_eval = infer_one_region(
                model=model,
                data=data,
                row_idx=int(idx),
                condition=str(args.condition),
                n_eval_joint_samples=int(args.n_eval_joint_samples),
                device=None,
                posthoc_ipf=bool(args.posthoc_ipf),
                ipf_iters=int(args.ipf_iters),
            )
            order = np.argsort(-p_true)
            x = np.arange(p_true.size)
            ax0 = axes[0, j]
            ax1 = axes[1, j]
            w = 0.42
            ax0.bar(x - w / 2, p_true[order], width=w, color=C_TRUE, alpha=0.85, label="true")
            ax0.bar(x + w / 2, p_hat_eval[order], width=w, color=C_PRED, alpha=0.75, label="generated")
            ax0.set_title(
                f"PUMA {data.ids[idx]} (q~{quantiles[j]:.2f})\nTVD={_tvd(p_hat_eval, p_true):.3f}",
                fontsize=8,
            )
            if j == 0:
                ax0.legend(frameon=False, fontsize=7)
                ax0.set_ylabel("Probability")
            ax0.text(
                0.02,
                0.98,
                _profile_text(data, idx),
                transform=ax0.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
                color="black",
            )
            despine(ax0)
            add_panel_label(ax0, chr(ord("a") + j), dx=-18)

            res = (p_hat_eval - p_true)[order]
            ax1.axhline(0.0, color="black", linewidth=0.8)
            ax1.axhline(0.01, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
            ax1.axhline(-0.01, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
            ax1.bar(x, res, width=0.8, color=C_RES, alpha=0.75)
            if j == 0:
                ax1.set_ylabel("Residual")
            ax1.set_xlabel("Cells (sorted by true prob.)")
            despine(ax1)
            add_panel_label(ax1, chr(ord("e") + j), dx=-18)

            records.append(
                {
                    "puma_uid": data.ids[idx],
                    "tvd": float(_tvd(p_hat_eval, p_true)),
                    "tvd_raw": float(_tvd(p_hat_raw, p_true)),
                    "profile": _profile_text(data, idx),
                    "quantile_anchor": float(quantiles[j]),
                }
            )

        save_figure(fig, out_pdf)
        plt.close(fig)

    write_json(
        out_json,
        {
            "checkpoint": str(ckpt),
            "joint_wide_csv": str(joint_csv),
            "condition": str(args.condition),
            "eval_mode": str(args.eval_mode),
            "n_eval_joint_samples": int(args.n_eval_joint_samples),
            "posthoc_ipf": bool(args.posthoc_ipf),
            "examples": records,
            "summary_tvd": _summ([float(r["tvd"]) for r in records]),
        },
    )
    print(f"[ok] wrote: {out_pdf}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
