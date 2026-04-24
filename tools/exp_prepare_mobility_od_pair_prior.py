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
from src.synthpop.validation.mobility_anchor import (
    AnchorSpec,
    compare_share_frames,
    load_bg_units,
    load_events_in_bbox,
    select_device_anchors,
    spatial_join_points_to_bg,
)


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _agg_od_counts(df: pd.DataFrame, *, home_col: str, work_col: str, count_col: str = "count") -> pd.DataFrame:
    out = (
        df.dropna(subset=[home_col, work_col])
        .groupby([home_col, work_col], as_index=False, sort=False)[count_col]
        .sum()
        .rename(columns={home_col: "home_tract_geoid", work_col: "work_tract_geoid", count_col: "mobility_od_count"})
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_prepare_mobility_od_pair_prior")
    ap.add_argument("--mobility_csv", required=True)
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--tiger_bg_zip", required=True)
    ap.add_argument("--label", default="prepare_mobility_od_pair_prior")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--chunksize", type=int, default=500000)
    ap.add_argument("--min_home_secs", type=int, default=6 * 3600)
    ap.add_argument("--min_work_secs", type=int, default=3 * 3600)
    ap.add_argument("--min_home_work_distance_m", type=float, default=500.0)
    ap.add_argument("--bbox_margin_deg", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    if not tract_od_path.exists():
        raise SystemExit(f"input not found: {tract_od_path}")
    tract_od = pd.read_csv(tract_od_path, low_memory=False)
    required = ["home_tract_geoid", "work_tract_geoid", "S000"]
    missing = [c for c in required if c not in tract_od.columns]
    if missing:
        raise SystemExit(f"tract_od missing columns: {missing}")
    tract_od["home_tract_geoid"] = tract_od["home_tract_geoid"].astype(str)
    tract_od["work_tract_geoid"] = tract_od["work_tract_geoid"].astype(str)
    tract_od["S000"] = pd.to_numeric(tract_od["S000"], errors="coerce").fillna(0.0)
    tract_od = tract_od[tract_od["S000"] > 0.0].copy()

    tract_set = set(tract_od["home_tract_geoid"].tolist()) | set(tract_od["work_tract_geoid"].tolist())
    bg_units = load_bg_units(tiger_bg_zip=args.tiger_bg_zip, allowed_tracts=tract_set)
    bounds = bg_units.total_bounds
    margin = float(args.bbox_margin_deg)
    bbox = (float(bounds[0] - margin), float(bounds[1] - margin), float(bounds[2] + margin), float(bounds[3] + margin))

    spec = AnchorSpec(
        min_home_secs=int(args.min_home_secs),
        min_work_secs=int(args.min_work_secs),
        min_home_work_distance_m=float(args.min_home_work_distance_m),
    )
    events = load_events_in_bbox(path=args.mobility_csv, bbox=bbox, chunksize=int(args.chunksize))
    mobility_home, mobility_work, anchor_summary = select_device_anchors(events, spec=spec)

    mob_work_home = spatial_join_points_to_bg(
        points=mobility_work,
        x_col="home_longitude",
        y_col="home_latitude",
        bg_units=bg_units,
        keep_cols=["ad_id", "work_longitude", "work_latitude", "home_longitude", "home_latitude"],
    ).rename(columns={"tract_geoid": "home_tract_geoid", "bg_geoid": "home_bg_geoid"})
    mob_work_full = spatial_join_points_to_bg(
        points=mobility_work,
        x_col="work_longitude",
        y_col="work_latitude",
        bg_units=bg_units,
        keep_cols=["ad_id", "home_longitude", "home_latitude", "work_longitude", "work_latitude"],
    ).rename(columns={"tract_geoid": "work_tract_geoid", "bg_geoid": "work_bg_geoid"})
    mob_work_full = mob_work_full.merge(
        mob_work_home.loc[:, ["ad_id", "home_tract_geoid", "home_bg_geoid"]],
        on="ad_id",
        how="left",
    )
    mobility_od = _agg_od_counts(
        mob_work_full.assign(count=1),
        home_col="home_tract_geoid",
        work_col="work_tract_geoid",
        count_col="count",
    )

    candidate = tract_od.copy()
    lodes_origin_total = candidate.groupby("home_tract_geoid", sort=False)["S000"].transform("sum")
    candidate["lodes_od_share"] = candidate["S000"] / np.clip(lodes_origin_total, 1e-12, None)

    out = candidate.merge(
        mobility_od,
        on=["home_tract_geoid", "work_tract_geoid"],
        how="left",
    )
    out["mobility_od_count"] = pd.to_numeric(out["mobility_od_count"], errors="coerce").fillna(0.0)
    candidate_pair_count = out.groupby("home_tract_geoid", sort=False)["work_tract_geoid"].transform("size").astype(float)
    mobility_origin_total = out.groupby("home_tract_geoid", sort=False)["mobility_od_count"].transform("sum")
    alpha = float(args.alpha)
    out["mobility_od_share_smoothed"] = (
        out["mobility_od_count"] + alpha
    ) / np.clip(mobility_origin_total + alpha * candidate_pair_count, 1e-12, None)
    out["mobility_od_residual"] = out["mobility_od_share_smoothed"] / np.clip(out["lodes_od_share"], 1e-12, None)
    out["mobility_od_present"] = (out["mobility_od_count"] > 0).astype(int)

    pair_comp, pair_summary = compare_share_frames(
        left=out.loc[:, ["home_tract_geoid", "work_tract_geoid", "S000"]].rename(columns={"S000": "lodes_count"}),
        right=out.loc[:, ["home_tract_geoid", "work_tract_geoid", "mobility_od_count"]],
        key_cols=["home_tract_geoid", "work_tract_geoid"],
        left_value_col="lodes_count",
        right_value_col="mobility_od_count",
    )

    tract_out_path = run_dir / "tract_od_with_mobility.csv"
    out.to_csv(tract_out_path, index=False)
    pair_comp.to_csv(metrics_dir / "lodes_vs_mobility_od_comparison.csv", index=False)
    mobility_od.to_csv(metrics_dir / "mobility_od_counts.csv", index=False)

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "tract_od_path": str(tract_od_path),
        "output_tract_od_path": str(tract_out_path),
        "mobility_csv": str(pathlib.Path(args.mobility_csv).expanduser().resolve()),
        "tiger_bg_zip": str(pathlib.Path(args.tiger_bg_zip).expanduser().resolve()),
        "bbox": {
            "minx": bbox[0],
            "miny": bbox[1],
            "maxx": bbox[2],
            "maxy": bbox[3],
        },
        "anchor_spec": {
            "min_home_secs": spec.min_home_secs,
            "min_work_secs": spec.min_work_secs,
            "min_home_work_distance_m": spec.min_home_work_distance_m,
        },
        "alpha": alpha,
        "anchor_summary": anchor_summary,
        "n_candidate_pairs": int(out.shape[0]),
        "n_candidate_home_tracts": int(out["home_tract_geoid"].nunique()),
        "n_candidate_work_tracts": int(out["work_tract_geoid"].nunique()),
        "n_mobility_pairs_nonzero_within_candidates": int((out["mobility_od_count"] > 0).sum()),
        "share_mobility_pairs_nonzero_within_candidates": float((out["mobility_od_count"] > 0).mean()),
        "n_home_tracts_with_any_mobility_obs": int((out.groupby("home_tract_geoid", sort=False)["mobility_od_count"].sum() > 0).sum()),
        "od_alignment_summary_raw": pair_summary,
        "generated_columns": [
            "lodes_od_share",
            "mobility_od_count",
            "mobility_od_share_smoothed",
            "mobility_od_residual",
            "mobility_od_present",
        ],
    }
    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
