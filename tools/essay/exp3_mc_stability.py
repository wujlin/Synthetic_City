#!/usr/bin/env python3
from __future__ import annotations

"""
Experiment 3: Monte Carlo stability over draw counts and random seeds.
"""

import argparse
import pathlib
import random
import sys
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.synthpop.model.diffusion_tabular import DiffusionTabularModel
from _eval_5var_common import _summ, _tvd, infer_one_region, load_eval_data, write_json


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for tok in [x.strip() for x in str(spec).split(",") if x.strip()]:
        out.append(int(tok))
    return out


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch  # type: ignore

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp3_mc_stability")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--condition", choices=["none", "marginal", "pairwise", "marginal_pairwise"], default="pairwise")
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "mi_kfold"], default="leave_mi_out")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_index", type=int, default=0)
    ap.add_argument("--draw_counts", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--posthoc_ipf", action="store_true")
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    out_json = pathlib.Path(args.out_json).expanduser().resolve()
    if not joint_csv.exists():
        raise SystemExit(f"joint_wide_csv not found: {joint_csv}")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    draw_counts = _parse_int_list(str(args.draw_counts))
    seeds = _parse_int_list(str(args.seeds))
    if not draw_counts:
        raise SystemExit("--draw_counts cannot be empty")
    if not seeds:
        raise SystemExit("--seeds cannot be empty")

    data = load_eval_data(
        joint_wide_csv=joint_csv,
        condition_names=[str(args.condition)],
        eval_mode=str(args.eval_mode),
        n_folds=int(args.n_folds),
        fold_index=int(args.fold_index),
        seed=0,
    )
    model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=0)
    model.load(ckpt)

    rows: list[dict[str, Any]] = []
    for n_draw in draw_counts:
        run_means: list[float] = []
        for sd in seeds:
            _set_seed(int(sd))
            tvd_vals: list[float] = []
            for idx in data.test_idx:
                p_true = data.p_joint[int(idx)]
                _, p_hat = infer_one_region(
                    model=model,
                    data=data,
                    row_idx=int(idx),
                    condition=str(args.condition),
                    n_eval_joint_samples=int(n_draw),
                    device=None,
                    posthoc_ipf=bool(args.posthoc_ipf),
                    ipf_iters=int(args.ipf_iters),
                )
                tvd_vals.append(_tvd(p_hat, p_true))
            run_means.append(float(np.mean(tvd_vals)))
        rows.append(
            {
                "n_draws": int(n_draw),
                "seed_means": [float(x) for x in run_means],
                "mean_over_seeds": float(np.mean(run_means)),
                "std_over_seeds": float(np.std(run_means, ddof=0)),
            }
        )
        print(
            f"[ok] n_draws={n_draw} mean={float(np.mean(run_means)):.6f} "
            f"std={float(np.std(run_means, ddof=0)):.6f}"
        )

    write_json(
        out_json,
        {
            "joint_wide_csv": str(joint_csv),
            "checkpoint": str(ckpt),
            "condition": str(args.condition),
            "eval_mode": str(args.eval_mode),
            "draw_counts": draw_counts,
            "seeds": seeds,
            "posthoc_ipf": bool(args.posthoc_ipf),
            "rows": rows,
            "summary_mean_curve": _summ([float(r["mean_over_seeds"]) for r in rows]),
            "summary_std_curve": _summ([float(r["std_over_seeds"]) for r in rows]),
        },
    )
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
