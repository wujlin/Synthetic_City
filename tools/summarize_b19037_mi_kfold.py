#!/usr/bin/env python3
from __future__ import annotations

"""
Summarize MI k-fold robustness for US-PUMA B19037 diffusion vs IPF baseline.

Input:
  <run_dir>/metrics/internal_acs_holdout.json
  <run_dir>/metrics/baselines_internal.json

Output:
  <run_dir>/metrics/mi_kfold_significance.json
"""

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np


def _load_json(p: pathlib.Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _exact_sign_test_pvalue(*, wins: int, n: int) -> float:
    """
    One-sided exact sign test:
      H0: P(diffusion < ipf) = 0.5
      H1: P(diffusion < ipf) > 0.5
    p = sum_{k=wins..n} C(n,k) / 2^n
    """
    if n <= 0:
        return float("nan")
    s = 0.0
    for k in range(int(wins), int(n) + 1):
        s += math.comb(int(n), int(k)) * (0.5 ** int(n))
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_b19037_mi_kfold")
    ap.add_argument("--run_dir", required=True, help="Run directory of train_us_puma_b19037_diffusion.py")
    ap.add_argument("--condition", default="marginal", choices=["none", "marginal"])
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    internal_path = run_dir / "metrics" / "internal_acs_holdout.json"
    baseline_path = run_dir / "metrics" / "baselines_internal.json"
    if not internal_path.exists():
        raise SystemExit(f"missing file: {internal_path}")
    if not baseline_path.exists():
        raise SystemExit(f"missing file: {baseline_path}")

    internal = _load_json(internal_path)
    baseline = _load_json(baseline_path)

    cond = str(args.condition)
    by_fold_cond = internal["by_condition"][cond]["by_fold"]
    by_fold_ipf = baseline["by_baseline"]["ipf_train_seed"]["by_fold"]

    fold_names = sorted(set(by_fold_cond.keys()) & set(by_fold_ipf.keys()))
    if not fold_names:
        raise SystemExit("no overlapping folds between condition metrics and ipf baseline")

    rows: list[dict[str, Any]] = []
    diffs: list[float] = []
    wins = 0
    ties = 0
    for f in fold_names:
        d = float(by_fold_cond[f]["tvd_joint"]["mean"])
        i = float(by_fold_ipf[f]["tvd_joint"]["mean"])
        diff = d - i
        if diff < 0:
            wins += 1
        elif diff == 0:
            ties += 1
        diffs.append(diff)
        rows.append(
            {
                "fold": f,
                "diffusion_tvd_joint": d,
                "ipf_seed_tvd_joint": i,
                "diff_minus_ipf": diff,
                "diffusion_better": bool(diff < 0),
            }
        )

    arr = np.asarray(diffs, dtype=float)
    n = int(arr.size)
    p_sign = _exact_sign_test_pvalue(wins=wins, n=n)
    out = {
        "run_dir": str(run_dir),
        "condition": cond,
        "n_folds": n,
        "wins_diffusion_vs_ipf": int(wins),
        "ties": int(ties),
        "losses_diffusion_vs_ipf": int(n - wins - ties),
        "diff_minus_ipf_summary": {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "p90": float(np.quantile(arr, 0.9)),
            "p10": float(np.quantile(arr, 0.1)),
        },
        "one_sided_sign_test_pvalue": float(p_sign),
        "folds": rows,
    }

    out_path = run_dir / "metrics" / "mi_kfold_significance.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    print(
        f"[summary] condition={cond} folds={n} wins={wins} ties={ties} "
        f"mean(diffusion-ipf)={out['diff_minus_ipf_summary']['mean']:.6f} p_sign={p_sign:.6g}"
    )


if __name__ == "__main__":
    main()
