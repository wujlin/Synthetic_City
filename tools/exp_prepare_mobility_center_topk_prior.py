#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import ensure_dir, project_root


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_prepare_mobility_center_topk_prior")
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--label", default="prepare_mobility_center_topk_prior")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--rank_col", default="S000")
    ap.add_argument("--distance_col", default="distance_km")
    ap.add_argument("--mobility_count_col", default="mobility_od_count")
    ap.add_argument("--mobility_residual_col", default="mobility_od_residual")
    args = ap.parse_args()

    if int(args.top_k) <= 0:
        raise SystemExit("--top_k must be positive")

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    if not tract_od_path.exists():
        raise SystemExit(f"input not found: {tract_od_path}")

    od = pd.read_csv(tract_od_path, low_memory=False)
    required = [
        "home_tract_geoid",
        "work_tract_geoid",
        "work_center_geoid",
        "S000",
        str(args.mobility_count_col),
        str(args.mobility_residual_col),
    ]
    missing = [c for c in required if c not in od.columns]
    if missing:
        raise SystemExit(f"tract_od missing columns: {missing}")

    od["home_tract_geoid"] = od["home_tract_geoid"].astype(str)
    od["work_tract_geoid"] = od["work_tract_geoid"].astype(str)
    od["work_center_geoid"] = od["work_center_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    od["S000"] = pd.to_numeric(od["S000"], errors="coerce").fillna(0.0)
    od[str(args.mobility_count_col)] = pd.to_numeric(od[str(args.mobility_count_col)], errors="coerce").fillna(0.0)
    od[str(args.mobility_residual_col)] = pd.to_numeric(od[str(args.mobility_residual_col)], errors="coerce").fillna(0.0)
    if str(args.rank_col) not in od.columns:
        raise SystemExit(f"rank column not found: {args.rank_col}")
    od[str(args.rank_col)] = pd.to_numeric(od[str(args.rank_col)], errors="coerce").fillna(0.0)
    if str(args.distance_col) in od.columns:
        od[str(args.distance_col)] = pd.to_numeric(od[str(args.distance_col)], errors="coerce").fillna(np.inf)
    else:
        od[str(args.distance_col)] = np.inf
    od = od[od["S000"] > 0.0].copy()

    od = od.sort_values(
        by=["home_tract_geoid", "work_center_geoid", str(args.rank_col), str(args.distance_col), "work_tract_geoid"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    od["rank_within_home_center"] = (
        od.groupby(["home_tract_geoid", "work_center_geoid"], sort=False).cumcount() + 1
    ).astype(int)
    od["is_topk_within_home_center"] = (od["rank_within_home_center"] <= int(args.top_k)).astype(int)
    od["mobility_pair_present"] = (od[str(args.mobility_count_col)] > 0).astype(int)
    od["is_topk_hit_pair"] = ((od["is_topk_within_home_center"] > 0) & (od["mobility_pair_present"] > 0)).astype(int)

    bonus_col = f"mobility_center_topk_bonus_k{int(args.top_k)}"
    od[bonus_col] = 1.0
    bonus_mask = od["is_topk_hit_pair"] > 0
    od.loc[bonus_mask, bonus_col] = np.clip(
        od.loc[bonus_mask, str(args.mobility_residual_col)].to_numpy(dtype=float),
        1.0,
        None,
    )

    topk_pairs = od[od["is_topk_within_home_center"] > 0].copy()
    hit_pairs = od[od["mobility_pair_present"] > 0].copy()
    topk_hit_pairs = od[od["is_topk_hit_pair"] > 0].copy()

    home_any_hit = od.groupby("home_tract_geoid", sort=False)["is_topk_hit_pair"].max().astype(int)
    center_any_hit = od.groupby(["home_tract_geoid", "work_center_geoid"], sort=False)["is_topk_hit_pair"].max().astype(int)

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "tract_od_path": str(tract_od_path),
        "output_tract_od_path": str(run_dir / "tract_od_with_mobility_center_topk.csv"),
        "top_k": int(args.top_k),
        "rank_col": str(args.rank_col),
        "distance_tiebreak_col": str(args.distance_col),
        "mobility_count_col": str(args.mobility_count_col),
        "mobility_residual_col": str(args.mobility_residual_col),
        "generated_columns": [
            "rank_within_home_center",
            "is_topk_within_home_center",
            "mobility_pair_present",
            "is_topk_hit_pair",
            bonus_col,
        ],
        "n_candidate_pairs": int(od.shape[0]),
        "n_candidate_origins": int(od["home_tract_geoid"].nunique()),
        "n_candidate_home_center_groups": int(od.groupby(["home_tract_geoid", "work_center_geoid"], sort=False).ngroups),
        "n_topk_pairs": int(topk_pairs.shape[0]),
        "share_topk_pairs": float(topk_pairs.shape[0] / max(int(od.shape[0]), 1)),
        "n_hit_pairs": int(hit_pairs.shape[0]),
        "share_hit_pairs": float(hit_pairs.shape[0] / max(int(od.shape[0]), 1)),
        "n_topk_hit_pairs": int(topk_hit_pairs.shape[0]),
        "share_topk_hit_within_all_pairs": float(topk_hit_pairs.shape[0] / max(int(od.shape[0]), 1)),
        "share_topk_hit_within_topk_pairs": float(topk_hit_pairs.shape[0] / max(int(topk_pairs.shape[0]), 1)),
        "share_topk_hit_within_hit_pairs": float(topk_hit_pairs.shape[0] / max(int(hit_pairs.shape[0]), 1)),
        "topk_lodes_mass_share_of_total": float(topk_pairs["S000"].sum() / max(float(od["S000"].sum()), 1e-12)),
        "topk_hit_lodes_mass_share_of_total": float(topk_hit_pairs["S000"].sum() / max(float(od["S000"].sum()), 1e-12)),
        "origins_with_any_topk_hit": int(home_any_hit.sum()),
        "share_origins_with_any_topk_hit": float(home_any_hit.mean()),
        "home_center_groups_with_any_topk_hit": int(center_any_hit.sum()),
        "share_home_center_groups_with_any_topk_hit": float(center_any_hit.mean()),
        "bonus_col_mean_all_pairs": float(pd.to_numeric(od[bonus_col], errors="coerce").fillna(1.0).mean()),
        "bonus_col_mean_topk_pairs": float(pd.to_numeric(topk_pairs[bonus_col], errors="coerce").fillna(1.0).mean()),
        "bonus_col_mean_topk_hit_pairs": float(pd.to_numeric(topk_hit_pairs[bonus_col], errors="coerce").fillna(1.0).mean()) if not topk_hit_pairs.empty else 1.0,
    }

    out_path = run_dir / "tract_od_with_mobility_center_topk.csv"
    od.to_csv(out_path, index=False)
    topk_pairs.to_csv(metrics_dir / "topk_pairs.csv", index=False)
    topk_hit_pairs.to_csv(metrics_dir / "topk_hit_pairs.csv", index=False)
    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
