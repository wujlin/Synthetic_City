#!/usr/bin/env python3
from __future__ import annotations

"""
Experiment 1: product-style output examples for representative Michigan PUMAs.

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
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

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


PROFILE_SPECS: list[tuple[str, str, str, int]] = [
    ("Age", "Young", "Old", 0),
    ("Sex", "Male", "Female", 1),
    ("Income", "Low", "High", 2),
    ("Education", "Low", "High", 3),
    ("Employment", "Employed", "Not-Empl", 4),
]

PROFILE_LABELS = {
    "Age": "Age (Y/O)",
    "Sex": "Sex (M/F)",
    "Income": "Income (L/H)",
    "Education": "Educ. (L/H)",
    "Employment": "Employ. (E/N)",
}

DEFAULT_PUMA_UIDS = ["2602903", "2601100"]
DEFAULT_REGION_LABELS = {
    "2602903": "University Town",
    "2601100": "Typical Suburb",
}


def _parse_csv_list(spec: str) -> list[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def _global_reference(data: Any) -> np.ndarray:
    train = data.train_idx
    w = np.maximum(data.totals[train], 0.0)
    if float(w.sum()) <= 0:
        p_global = np.mean(data.p_joint[train], axis=0)
    else:
        p_global = np.sum(data.p_joint[train] * w.reshape(-1, 1), axis=0) / float(w.sum())
    p_global = p_global / max(float(p_global.sum()), 1e-12)
    return p_global.astype(np.float64)


def _select_fixed_representatives(*, data: Any, puma_uids: list[str]) -> list[int]:
    id_to_idx = {str(pid): int(i) for i, pid in enumerate(data.ids)}
    missing = [pid for pid in puma_uids if pid not in id_to_idx]
    if missing:
        raise SystemExit(f"Requested PUMA(s) not found in eval data: {missing}")
    picked = [id_to_idx[pid] for pid in puma_uids]
    bad = [data.ids[i] for i in picked if int(i) not in set(map(int, data.test_idx.tolist()))]
    if bad:
        raise SystemExit(f"Requested PUMA(s) are not in the test split: {bad}")
    return picked


def _format_count(x: float) -> str:
    return f"{int(round(float(x))):,}"


def _profile_marginals(p: np.ndarray, shape: tuple[int, ...]) -> list[np.ndarray]:
    return [_marginal_from_joint(p, shape=shape, axis=axis) for axis in range(len(shape))]


def _cross_tab_income_education(p: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    tab = np.asarray(p, dtype=float).reshape(shape)
    out = tab.sum(axis=(0, 1, 4))  # keep income(axis=2) and schl(axis=3)
    out = out / max(float(out.sum()), 1e-12)
    return out.astype(np.float64)


def _mix_with_white(color: str, alpha: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple((1.0 - alpha) * np.ones(3) + alpha * rgb)


def _cell_label(idx: int, shape: tuple[int, ...]) -> str:
    a, s, inc, edu, emp = np.unravel_index(int(idx), shape)
    age_lab = ["Y", "O"][int(a)]
    sex_lab = ["M", "F"][int(s)]
    inc_lab = ["L", "H"][int(inc)]
    edu_lab = ["L", "H"][int(edu)]
    emp_lab = ["E", "N"][int(emp)]
    return f"{age_lab}/{sex_lab}/{inc_lab}/{edu_lab}/{emp_lab}"


def _draw_profile_panel(
    *,
    ax: Any,
    p_true: np.ndarray,
    p_gen: np.ndarray,
    total_pop: float,
    region_title: str,
    puma_uid: str,
    shape: tuple[int, ...],
    show_xlabel: bool = True,
) -> None:
    marg_true = _profile_marginals(p_true, shape=shape)
    marg_gen = _profile_marginals(p_gen, shape=shape)
    y_base = np.arange(len(PROFILE_SPECS))[::-1].astype(float)
    h = 0.28

    c_true0 = _mix_with_white(OKABE_ITO["blue"], 0.92)
    c_true1 = _mix_with_white(OKABE_ITO["sky_blue"], 0.92)
    c_gen0 = _mix_with_white(OKABE_ITO["vermillion"], 0.88)
    c_gen1 = _mix_with_white(OKABE_ITO["orange"], 0.88)

    for i, (attr, bin0, bin1, axis) in enumerate(PROFILE_SPECS):
        y = y_base[i]
        t = marg_true[axis] * float(total_pop)
        g = marg_gen[axis] * float(total_pop)

        ax.barh(y + h / 2, t[0], height=h, color=c_true0, edgecolor="white", linewidth=0.8)
        ax.barh(y + h / 2, t[1], left=t[0], height=h, color=c_true1, edgecolor="white", linewidth=0.8)
        ax.barh(y - h / 2, g[0], height=h, color=c_gen0, edgecolor="white", linewidth=0.8)
        ax.barh(y - h / 2, g[1], left=g[0], height=h, color=c_gen1, edgecolor="white", linewidth=0.8)

    ax.set_xlim(0, float(total_pop))
    ax.set_ylim(-0.8, len(PROFILE_SPECS) - 0.2)
    ax.set_yticks(y_base, labels=[PROFILE_LABELS[attr] for attr, _, _, _ in PROFILE_SPECS])
    ax.tick_params(axis="y", labelsize=7)
    if show_xlabel:
        ax.set_xlabel("Person count")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_title(region_title, fontsize=9, pad=2)
    ax.text(
        0.02,
        1.01,
        f"PUMA {puma_uid} | N = {_format_count(total_pop)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    despine(ax)


def _draw_heatmap_panel(
    *,
    parent_spec: Any,
    fig: Any,
    cross_true: np.ndarray,
    cross_gen: np.ndarray,
    total_pop: float,
    show_xlabel: bool = True,
) -> list[Any]:
    sub = parent_spec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.18)
    axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
    cax = fig.add_subplot(sub[0, 2])
    vmax = max(float(cross_true.max()), float(cross_gen.max()), 1e-6)
    panels = [("True", cross_true), ("Generated", cross_gen)]
    cmap = "YlOrRd"
    im = None
    for ax, (title, arr) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_xticks([0, 1], labels=["Low", "High"])
        ax.set_yticks([0, 1], labels=["Low", "High"])
        if show_xlabel:
            ax.set_xlabel("Income")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if ax is axes[0]:
            ax.set_ylabel("Education")
        ax.set_title(title, fontsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Share", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return axes + [cax]


def _draw_joint_panel(
    *,
    ax: Any,
    p_true: np.ndarray,
    p_gen: np.ndarray,
    total_pop: float,
    shape: tuple[int, ...],
    show_xlabel: bool = True,
) -> None:
    order = np.argsort(-p_true)
    x = np.arange(p_true.size)
    cnt_true = p_true[order] * float(total_pop)
    cnt_gen = p_gen[order] * float(total_pop)
    w = 0.42
    ax.bar(x - w / 2, cnt_true, width=w, color=OKABE_ITO["blue"], alpha=0.85, label="True")
    ax.bar(x + w / 2, cnt_gen, width=w, color=OKABE_ITO["vermillion"], alpha=0.75, label="Generated")
    if show_xlabel:
        ax.set_xlabel("Joint cell (sorted by true mass)")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Person count")
    ax.set_xticks([0, 7, 15, 23, 31], labels=["1", "8", "16", "24", "32"])
    ax.text(
        0.98,
        0.98,
        f"TVD = {_tvd(p_gen, p_true):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    despine(ax)


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
    ap.add_argument("--n_examples", type=int, default=2)
    ap.add_argument("--puma_uids", default="2602903,2601100")
    ap.add_argument("--region_labels", default="University Town,Typical Suburb")
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_png", default="")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    out_pdf = pathlib.Path(args.out_pdf).expanduser().resolve()
    out_png = pathlib.Path(args.out_png).expanduser().resolve() if str(args.out_png).strip() else None
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
    puma_uids = _parse_csv_list(str(args.puma_uids))
    if not puma_uids:
        puma_uids = list(DEFAULT_PUMA_UIDS)
    region_labels = _parse_csv_list(str(args.region_labels))
    if not region_labels:
        region_labels = [DEFAULT_REGION_LABELS.get(pid, f"PUMA {pid}") for pid in puma_uids]
    if len(region_labels) != len(puma_uids):
        raise SystemExit("--region_labels must have the same length as --puma_uids")
    rep_idx = _select_fixed_representatives(data=data, puma_uids=puma_uids)

    p_global = _global_reference(data)

    records: list[dict[str, Any]] = []
    with paper_style():
        fig = plt.figure(figsize=(7.2, 4.8))
        outer = fig.add_gridspec(2, 3, width_ratios=[1.30, 1.08, 1.52], wspace=0.40, hspace=0.52)
        profile_axes = []
        joint_axes = []
        heat_axes: list[list[Any]] = []
        for row, idx in enumerate(rep_idx):
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
            total_pop = float(data.totals[idx])
            cross_true = _cross_tab_income_education(p_true, shape=data.shape)
            cross_gen = _cross_tab_income_education(p_hat_eval, shape=data.shape)
            profile_ax = fig.add_subplot(outer[row, 0])
            joint_ax = fig.add_subplot(outer[row, 2])
            region_title = region_labels[row]

            _draw_profile_panel(
                ax=profile_ax,
                p_true=p_true,
                p_gen=p_hat_eval,
                total_pop=total_pop,
                region_title=region_title,
                puma_uid=data.ids[idx],
                shape=data.shape,
                show_xlabel=(row == len(rep_idx) - 1),
            )
            heat_pair = _draw_heatmap_panel(
                parent_spec=outer[row, 1],
                fig=fig,
                cross_true=cross_true,
                cross_gen=cross_gen,
                total_pop=total_pop,
                show_xlabel=(row == len(rep_idx) - 1),
            )
            _draw_joint_panel(
                ax=joint_ax,
                p_true=p_true,
                p_gen=p_hat_eval,
                total_pop=total_pop,
                shape=data.shape,
                show_xlabel=(row == len(rep_idx) - 1),
            )
            profile_axes.append(profile_ax)
            joint_axes.append(joint_ax)
            heat_axes.append(heat_pair)

            records.append(
                {
                    "puma_uid": data.ids[idx],
                    "region_label": region_title,
                    "tvd": float(_tvd(p_hat_eval, p_true)),
                    "tvd_raw": float(_tvd(p_hat_raw, p_true)),
                    "tvd_to_global": float(_tvd(p_true, p_global)),
                    "total_population": total_pop,
                    "marginals_true": {
                        spec[0].lower(): _marginal_from_joint(p_true, shape=data.shape, axis=spec[3]).astype(float).tolist()
                        for spec in PROFILE_SPECS
                    },
                    "marginals_generated": {
                        spec[0].lower(): _marginal_from_joint(p_hat_eval, shape=data.shape, axis=spec[3]).astype(float).tolist()
                        for spec in PROFILE_SPECS
                    },
                    "cross_tab_true": cross_true.astype(float).tolist(),
                    "cross_tab_gen": cross_gen.astype(float).tolist(),
                    "cell_counts_true": (p_true * total_pop).astype(float).tolist(),
                    "cell_counts_gen": (p_hat_eval * total_pop).astype(float).tolist(),
                }
            )

        add_panel_label(profile_axes[0], "a", dx=-24, dy=6)
        add_panel_label(heat_axes[0][0], "b", dx=-20, dy=6)
        add_panel_label(joint_axes[0], "c", dx=-22, dy=6)
        add_panel_label(profile_axes[1], "d", dx=-24, dy=6)
        add_panel_label(heat_axes[1][0], "e", dx=-20, dy=6)
        add_panel_label(joint_axes[1], "f", dx=-22, dy=6)

        fig.text(0.19, 0.97, "Demographic profile", ha="center", va="top", fontsize=9)
        fig.text(0.50, 0.97, "Education × Income", ha="center", va="top", fontsize=9)
        fig.text(0.82, 0.97, "Full 32-cell joint", ha="center", va="top", fontsize=9)

        fig.legend(
            handles=[
                Patch(facecolor=OKABE_ITO["blue"], edgecolor="none", alpha=0.85, label="True"),
                Patch(facecolor=OKABE_ITO["vermillion"], edgecolor="none", alpha=0.75, label="Generated"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=2,
            frameon=False,
            fontsize=8,
        )

        save_figure(fig, out_pdf)
        if out_png is not None:
            fig.savefig(out_png, dpi=220)
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
    if out_png is not None:
        print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
