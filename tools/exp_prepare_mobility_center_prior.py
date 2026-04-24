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
from src.synthpop.validation.mobility_anchor import compare_share_frames


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_prepare_mobility_center_prior")
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--label", default="prepare_mobility_center_prior")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    if not tract_od_path.exists():
        raise SystemExit(f"input not found: {tract_od_path}")

    od = pd.read_csv(tract_od_path, low_memory=False)
    required = ["home_tract_geoid", "work_tract_geoid", "work_center_geoid", "S000", "mobility_od_count"]
    missing = [c for c in required if c not in od.columns]
    if missing:
        raise SystemExit(f"tract_od missing columns: {missing}")

    od["home_tract_geoid"] = od["home_tract_geoid"].astype(str)
    od["work_tract_geoid"] = od["work_tract_geoid"].astype(str)
    od["work_center_geoid"] = od["work_center_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    od["S000"] = pd.to_numeric(od["S000"], errors="coerce").fillna(0.0)
    od["mobility_od_count"] = pd.to_numeric(od["mobility_od_count"], errors="coerce").fillna(0.0)
    od = od[od["S000"] > 0.0].copy()

    center = (
        od.groupby(["home_tract_geoid", "work_center_geoid"], as_index=False, sort=False)[["S000", "mobility_od_count"]]
        .sum()
    )
    center["n_candidate_centers"] = center.groupby("home_tract_geoid", sort=False)["work_center_geoid"].transform("size").astype(float)

    lodes_origin_total = center.groupby("home_tract_geoid", sort=False)["S000"].transform("sum")
    mobility_origin_total = center.groupby("home_tract_geoid", sort=False)["mobility_od_count"].transform("sum")
    alpha = float(args.alpha)

    center["lodes_work_center_share"] = center["S000"] / np.clip(lodes_origin_total, 1e-12, None)
    center["mobility_work_center_share_smoothed"] = (
        center["mobility_od_count"] + alpha
    ) / np.clip(mobility_origin_total + alpha * center["n_candidate_centers"], 1e-12, None)
    center["mobility_work_center_residual"] = (
        center["mobility_work_center_share_smoothed"] / np.clip(center["lodes_work_center_share"], 1e-12, None)
    )
    center["mobility_work_center_present"] = (center["mobility_od_count"] > 0).astype(int)

    center_comp, center_summary = compare_share_frames(
        left=center.loc[:, ["home_tract_geoid", "work_center_geoid", "S000"]].rename(columns={"S000": "lodes_count"}),
        right=center.loc[:, ["home_tract_geoid", "work_center_geoid", "mobility_od_count"]].rename(columns={"mobility_od_count": "mobility_count"}),
        key_cols=["home_tract_geoid", "work_center_geoid"],
        left_value_col="lodes_count",
        right_value_col="mobility_count",
    )

    out = od.merge(
        center.loc[
            :,
            [
                "home_tract_geoid",
                "work_center_geoid",
                "lodes_work_center_share",
                "mobility_work_center_share_smoothed",
                "mobility_work_center_residual",
                "mobility_work_center_present",
            ],
        ],
        on=["home_tract_geoid", "work_center_geoid"],
        how="left",
    )

    out_path = run_dir / "tract_od_with_mobility_center.csv"
    out.to_csv(out_path, index=False)
    center.to_csv(metrics_dir / "home_work_center_counts.csv", index=False)
    center_comp.to_csv(metrics_dir / "home_work_center_alignment.csv", index=False)

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "tract_od_path": str(tract_od_path),
        "output_tract_od_path": str(out_path),
        "alpha": alpha,
        "n_candidate_pairs": int(od.shape[0]),
        "n_candidate_center_pairs": int(center.shape[0]),
        "n_origins": int(center["home_tract_geoid"].nunique()),
        "n_hit_center_pairs": int((center["mobility_od_count"] > 0).sum()),
        "share_hit_center_pairs": float((center["mobility_od_count"] > 0).mean()),
        "center_alignment": center_summary,
        "generated_columns": [
            "lodes_work_center_share",
            "mobility_work_center_share_smoothed",
            "mobility_work_center_residual",
            "mobility_work_center_present",
        ],
    }
    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
