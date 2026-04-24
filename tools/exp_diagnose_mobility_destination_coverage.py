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


def _coverage_stats(df: pd.DataFrame, value_col: str = "mobility_od_count") -> dict[str, Any]:
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work["S000"] = pd.to_numeric(work["S000"], errors="coerce").fillna(0.0)
    hit = work[value_col] > 0.0
    by_origin = work.assign(_hit=hit.astype(int)).groupby("home_tract_geoid", sort=False)
    return {
        "n_units": int(work.shape[0]),
        "n_hit_units": int(hit.sum()),
        "share_hit_units": float(hit.mean()) if len(work) else float("nan"),
        "lodes_mass_covered_share": float(work.loc[hit, "S000"].sum() / max(float(work["S000"].sum()), 1e-12)),
        "n_origins": int(work["home_tract_geoid"].nunique()),
        "n_origins_with_any_hit": int((by_origin["_hit"].sum() > 0).sum()),
        "share_origins_with_any_hit": float(((by_origin["_hit"].sum() > 0).mean()) if work["home_tract_geoid"].nunique() else float("nan")),
    }


def _topk_coverage(df: pd.DataFrame, topk_values: list[int]) -> pd.DataFrame:
    work = df.copy()
    work["S000"] = pd.to_numeric(work["S000"], errors="coerce").fillna(0.0)
    work["mobility_od_count"] = pd.to_numeric(work["mobility_od_count"], errors="coerce").fillna(0.0)
    work = work.sort_values(["home_tract_geoid", "S000", "work_tract_geoid"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    work["rank_within_home"] = work.groupby("home_tract_geoid", sort=False).cumcount() + 1
    rows: list[dict[str, Any]] = []
    for k in topk_values:
        sub = work[work["rank_within_home"] <= int(k)].copy()
        stats = _coverage_stats(sub)
        stats["topk"] = int(k)
        rows.append(stats)
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, as_index=False, sort=False)[["S000", "mobility_od_count"]]
        .sum()
    )
    return out


def _alignment(left_right: pd.DataFrame, key_cols: list[str], left_name: str, right_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    comp, summary = compare_share_frames(
        left=left_right.loc[:, key_cols + ["S000"]].rename(columns={"S000": left_name}),
        right=left_right.loc[:, key_cols + ["mobility_od_count"]].rename(columns={"mobility_od_count": right_name}),
        key_cols=key_cols,
        left_value_col=left_name,
        right_value_col=right_name,
    )
    return comp, summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_diagnose_mobility_destination_coverage")
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--label", default="diagnose_mobility_destination_coverage")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--topk_values", default="5,10,20,50")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    if not tract_od_path.exists():
        raise SystemExit(f"input not found: {tract_od_path}")
    usecols = [
        "home_tract_geoid",
        "work_tract_geoid",
        "S000",
        "mobility_od_count",
        "work_center_geoid",
        "work_center_county_geoid",
    ]
    df = pd.read_csv(tract_od_path, usecols=usecols, low_memory=False)
    df["home_tract_geoid"] = df["home_tract_geoid"].astype(str)
    df["work_tract_geoid"] = df["work_tract_geoid"].astype(str)
    df["work_center_geoid"] = df["work_center_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["work_county_geoid"] = df["work_tract_geoid"].str.slice(0, 5)
    df["work_center_county_geoid"] = df["work_center_county_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["S000"] = pd.to_numeric(df["S000"], errors="coerce").fillna(0.0)
    df["mobility_od_count"] = pd.to_numeric(df["mobility_od_count"], errors="coerce").fillna(0.0)
    df = df[df["S000"] > 0.0].copy()

    topk_values = [int(x.strip()) for x in str(args.topk_values).split(",") if x.strip()]

    pair_cov = _coverage_stats(df)
    topk_cov = _topk_coverage(df, topk_values)

    county_df = _aggregate(df.loc[:, ["home_tract_geoid", "work_county_geoid", "S000", "mobility_od_count"]], ["home_tract_geoid", "work_county_geoid"])
    county_comp, county_align = _alignment(county_df, ["home_tract_geoid", "work_county_geoid"], "lodes_count", "mobility_count")
    county_cov = _coverage_stats(county_df)

    center_df = _aggregate(df.loc[:, ["home_tract_geoid", "work_center_geoid", "S000", "mobility_od_count"]], ["home_tract_geoid", "work_center_geoid"])
    center_comp, center_align = _alignment(center_df, ["home_tract_geoid", "work_center_geoid"], "lodes_count", "mobility_count")
    center_cov = _coverage_stats(center_df)

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "tract_od_path": str(tract_od_path),
        "topk_values": topk_values,
        "pair_coverage": pair_cov,
        "pair_alignment_previous_reference": {
            "note": "pair-level alignment is already reported in prepare_mobility_od_pair_prior summary if needed"
        },
        "county_coverage": county_cov,
        "county_alignment": county_align,
        "center_coverage": center_cov,
        "center_alignment": center_align,
        "recommended_state_rule": "prefer the finest state that materially improves coverage and alignment over tract-OD without collapsing to county-level triviality",
    }

    topk_cov.to_csv(metrics_dir / "topk_coverage.csv", index=False)
    county_df.to_csv(metrics_dir / "county_aggregated_counts.csv", index=False)
    county_comp.to_csv(metrics_dir / "county_alignment.csv", index=False)
    center_df.to_csv(metrics_dir / "center_aggregated_counts.csv", index=False)
    center_comp.to_csv(metrics_dir / "center_alignment.csv", index=False)
    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
