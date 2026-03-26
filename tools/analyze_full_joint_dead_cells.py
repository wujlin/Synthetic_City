#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summ(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="analyze_full_joint_dead_cells")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    joint_wide_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    out_json = pathlib.Path(args.out_json).expanduser().resolve()
    if not joint_wide_csv.exists():
        raise SystemExit(f"path not found: {joint_wide_csv}")

    df = pd.read_csv(joint_wide_csv, low_memory=False)
    p_cols = [c for c in df.columns if c.startswith("p_joint_")]
    if not p_cols:
        raise SystemExit("joint_wide_csv missing p_joint_* columns")

    p = df[p_cols].to_numpy(dtype=np.float64)
    eps = float(args.eps)
    nonzero_mask = p > eps

    n_regions, fine_k = p.shape
    nonzero_counts_per_region = nonzero_mask.sum(axis=1)
    zero_counts_per_region = fine_k - nonzero_counts_per_region
    nonzero_frac_per_region = nonzero_counts_per_region / fine_k
    zero_frac_per_region = zero_counts_per_region / fine_k

    active_region_count_per_cell = nonzero_mask.sum(axis=0)
    total_mass_per_cell = p.sum(axis=0)
    mean_mass_per_cell = total_mass_per_cell / max(n_regions, 1)

    bins = {
        "zero_regions": int(np.sum(active_region_count_per_cell == 0)),
        "one_region": int(np.sum(active_region_count_per_cell == 1)),
        "two_to_five_regions": int(np.sum((active_region_count_per_cell >= 2) & (active_region_count_per_cell <= 5))),
        "six_to_twenty_regions": int(np.sum((active_region_count_per_cell >= 6) & (active_region_count_per_cell <= 20))),
        "twentyone_to_hundred_regions": int(np.sum((active_region_count_per_cell >= 21) & (active_region_count_per_cell <= 100))),
        "gt_hundred_regions": int(np.sum(active_region_count_per_cell > 100)),
    }

    top_cells_idx = np.argsort(-active_region_count_per_cell)[:10]
    top_cells = []
    for idx in top_cells_idx.tolist():
        top_cells.append(
            {
                "cell_index": int(idx),
                "active_region_count": int(active_region_count_per_cell[idx]),
                "mean_mass": float(mean_mass_per_cell[idx]),
                "total_mass": float(total_mass_per_cell[idx]),
            }
        )

    summary = {
        "joint_wide_csv": str(joint_wide_csv),
        "n_regions": int(n_regions),
        "fine_k": int(fine_k),
        "eps": eps,
        "region_nonzero_count": _summ(nonzero_counts_per_region),
        "region_zero_count": _summ(zero_counts_per_region),
        "region_nonzero_fraction": _summ(nonzero_frac_per_region),
        "region_zero_fraction": _summ(zero_frac_per_region),
        "cell_active_region_count": _summ(active_region_count_per_cell),
        "cell_mean_mass": _summ(mean_mass_per_cell),
        "cell_support_bins": bins,
        "top_supported_cells": top_cells,
    }
    _write_json(out_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
