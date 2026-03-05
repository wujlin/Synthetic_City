#!/usr/bin/env python3
from __future__ import annotations

"""
Experiment 2: evaluate convergence curve from saved checkpoints.

Input checkpoints are expected under:
  <ckpt_root>/epoch_XXXXX.pt
"""

import argparse
import pathlib
import re
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


def _discover_ckpts(ckpt_root: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    out: list[tuple[int, pathlib.Path]] = []
    pat = re.compile(r"epoch_(\d{5})\.pt$")
    for p in sorted(ckpt_root.glob("epoch_*.pt")):
        m = pat.search(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    return out


def _parse_epochs(spec: str) -> list[int]:
    xs: list[int] = []
    for t in [x.strip() for x in str(spec).split(",") if x.strip()]:
        xs.append(int(t))
    return xs


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp2_convergence_curve")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--ckpt_root", required=True, help="Directory containing epoch_XXXXX.pt")
    ap.add_argument("--condition", choices=["none", "marginal", "pairwise", "marginal_pairwise"], default="pairwise")
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "mi_kfold"], default="leave_mi_out")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--posthoc_ipf", action="store_true")
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--epochs", default="", help="Optional comma-separated epoch filter.")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    ckpt_root = pathlib.Path(args.ckpt_root).expanduser().resolve()
    out_json = pathlib.Path(args.out_json).expanduser().resolve()
    if not joint_csv.exists():
        raise SystemExit(f"joint_wide_csv not found: {joint_csv}")
    if not ckpt_root.exists():
        raise SystemExit(f"ckpt_root not found: {ckpt_root}")

    data = load_eval_data(
        joint_wide_csv=joint_csv,
        condition_names=[str(args.condition)],
        eval_mode=str(args.eval_mode),
        n_folds=int(args.n_folds),
        fold_index=int(args.fold_index),
        seed=int(args.seed),
    )

    ckpts = _discover_ckpts(ckpt_root)
    if not ckpts:
        raise SystemExit(f"No epoch checkpoints found under: {ckpt_root}")
    if str(args.epochs).strip():
        keep = set(_parse_epochs(str(args.epochs)))
        ckpts = [(ep, p) for ep, p in ckpts if ep in keep]
        if not ckpts:
            raise SystemExit("No checkpoints matched --epochs filter.")

    rows: list[dict[str, Any]] = []
    for ep, ckpt in ckpts:
        model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))
        model.load(ckpt)
        tvd_vals: list[float] = []
        for idx in data.test_idx:
            p_true = data.p_joint[int(idx)]
            _, p_hat = infer_one_region(
                model=model,
                data=data,
                row_idx=int(idx),
                condition=str(args.condition),
                n_eval_joint_samples=int(args.n_eval_joint_samples),
                device=None,
                posthoc_ipf=bool(args.posthoc_ipf),
                ipf_iters=int(args.ipf_iters),
            )
            tvd_vals.append(_tvd(p_hat, p_true))
        rows.append(
            {
                "epoch": int(ep),
                "checkpoint": str(ckpt),
                "tvd_joint": _summ(tvd_vals),
            }
        )
        print(f"[ok] epoch={ep} tvd_mean={float(np.mean(tvd_vals)):.6f} ckpt={ckpt}")

    rows = sorted(rows, key=lambda x: int(x["epoch"]))
    write_json(
        out_json,
        {
            "joint_wide_csv": str(joint_csv),
            "ckpt_root": str(ckpt_root),
            "condition": str(args.condition),
            "eval_mode": str(args.eval_mode),
            "n_eval_joint_samples": int(args.n_eval_joint_samples),
            "posthoc_ipf": bool(args.posthoc_ipf),
            "rows": rows,
        },
    )
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
