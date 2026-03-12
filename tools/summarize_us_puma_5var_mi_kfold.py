#!/usr/bin/env python3
from __future__ import annotations

"""
Summarize Michigan 5-fold robustness for the US-PUMA 5-var diffusion model
against the IPF(train-seed) baseline.

Input:
  <run_dir>/metrics/internal_acs_holdout.json
  <run_dir>/metrics/baselines_internal.json

Output:
  <run_dir>/metrics/mi_kfold_significance.json
  <run_dir>/metrics/table3_mi_kfold.csv
"""

import argparse
import csv
import json
import math
import pathlib
from typing import Any

import numpy as np


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _exact_sign_test_pvalue(*, wins: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    s = 0.0
    for k in range(int(wins), int(n) + 1):
        s += math.comb(int(n), int(k)) * (0.5 ** int(n))
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_us_puma_5var_mi_kfold")
    ap.add_argument("--run_dir", required=True, help="Run directory of train_us_puma_5var_diffusion.py")
    ap.add_argument(
        "--condition",
        default="marginal",
        choices=[
            "none",
            "marginal",
            "pairwise",
            "marginal_pairwise",
            "spatial",
            "marginal_spatial",
            "pairwise_spatial",
            "marginal_pairwise_spatial",
        ],
    )
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    internal_path = run_dir / "metrics" / "internal_acs_holdout.json"
    baseline_path = run_dir / "metrics" / "baselines_internal.json"
    if not internal_path.exists():
        raise SystemExit(f"missing file: {internal_path}")
    if not baseline_path.exists():
        raise SystemExit(f"missing file: {baseline_path}")

    internal = _load_json(internal_path)
    baselines = _load_json(baseline_path)

    cond = str(args.condition)
    by_fold_cond = internal["by_condition"][cond]["by_fold"]
    by_fold_ipf = baselines["by_baseline"]["ipf_train_seed"]["by_fold"]
    fold_names = sorted(set(by_fold_cond.keys()) & set(by_fold_ipf.keys()))
    if not fold_names:
        raise SystemExit("no overlapping folds between condition metrics and ipf baseline")

    rows: list[dict[str, Any]] = []
    diffs: list[float] = []
    wins = 0
    ties = 0

    for fold_name in fold_names:
        diff_tvd = float(by_fold_cond[fold_name]["tvd_joint"]["mean"])
        ipf_tvd = float(by_fold_ipf[fold_name]["tvd_joint"]["mean"])
        delta = diff_tvd - ipf_tvd
        if delta < 0:
            wins += 1
        elif delta == 0:
            ties += 1
        diffs.append(delta)
        rows.append(
            {
                "fold": fold_name,
                "diffusion_tvd_joint": diff_tvd,
                "ipf_seed_tvd_joint": ipf_tvd,
                "diff_minus_ipf": delta,
                "diffusion_better": bool(delta < 0),
            }
        )

    diff_arr = np.asarray(diffs, dtype=float)
    diff_vals = np.asarray([r["diffusion_tvd_joint"] for r in rows], dtype=float)
    ipf_vals = np.asarray([r["ipf_seed_tvd_joint"] for r in rows], dtype=float)
    n = int(diff_arr.size)
    p_sign = _exact_sign_test_pvalue(wins=wins, n=n)

    out = {
        "run_dir": str(run_dir),
        "condition": cond,
        "n_folds": n,
        "wins_diffusion_vs_ipf": int(wins),
        "ties": int(ties),
        "losses_diffusion_vs_ipf": int(n - wins - ties),
        "diff_minus_ipf_summary": {
            "mean": float(np.mean(diff_arr)),
            "std": float(np.std(diff_arr, ddof=0)),
            "max": float(np.max(diff_arr)),
            "min": float(np.min(diff_arr)),
            "p90": float(np.quantile(diff_arr, 0.9)),
            "p10": float(np.quantile(diff_arr, 0.1)),
        },
        "mean_row": {
            "fold": "Mean",
            "diffusion_tvd_joint": float(np.mean(diff_vals)),
            "ipf_seed_tvd_joint": float(np.mean(ipf_vals)),
            "diff_minus_ipf": float(np.mean(diff_arr)),
        },
        "one_sided_sign_test_pvalue": float(p_sign),
        "folds": rows,
    }

    json_path = run_dir / "metrics" / "mi_kfold_significance.json"
    csv_path = run_dir / "metrics" / "table3_mi_kfold.csv"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Diffusion TVD", "IPF TVD", "Δ"])
        for row in rows:
            writer.writerow(
                [
                    row["fold"],
                    f"{row['diffusion_tvd_joint']:.6f}",
                    f"{row['ipf_seed_tvd_joint']:.6f}",
                    f"{row['diff_minus_ipf']:.6f}",
                ]
            )
        writer.writerow(
            [
                "Mean",
                f"{out['mean_row']['diffusion_tvd_joint']:.6f}",
                f"{out['mean_row']['ipf_seed_tvd_joint']:.6f}",
                f"{out['mean_row']['diff_minus_ipf']:.6f}",
            ]
        )

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {csv_path}")
    print(
        f"[summary] condition={cond} folds={n} wins={wins} ties={ties} "
        f"mean(diffusion-ipf)={out['diff_minus_ipf_summary']['mean']:.6f} p_sign={p_sign:.6g}"
    )


if __name__ == "__main__":
    main()
