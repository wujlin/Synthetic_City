#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import ensure_dir, project_root
from src.synthpop.validation.mobility_anchor import (
    compare_share_frames,
    load_bg_units,
    spatial_join_points_to_bg,
    within_tract_bg_spearman,
)


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_person_locations(path: pathlib.Path) -> pd.DataFrame:
    usecols = [
        "person_id",
        "tract_geoid",
        "home_candidate_id",
        "home_x",
        "home_y",
    ]
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _prepare_synthetic_home_points(persons: pd.DataFrame) -> pd.DataFrame:
    homes = persons.dropna(subset=["home_candidate_id", "home_x", "home_y"]).copy()
    homes["tract_geoid"] = homes["tract_geoid"].astype(str)
    grouped = (
        homes.groupby(["home_candidate_id", "home_x", "home_y", "tract_geoid"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return grouped


def _prepare_teacher_home_points(path: pathlib.Path) -> pd.DataFrame:
    ref = pd.read_parquet(path)
    out = ref.loc[:, ["ad_id", "home_tract_geoid", "latitude", "longitude"]].copy()
    out = out.rename(
        columns={
            "home_tract_geoid": "tract_geoid",
            "latitude": "y",
            "longitude": "x",
        }
    )
    out["tract_geoid"] = out["tract_geoid"].astype(str)
    out["count"] = 1
    return out


def _agg_bg_counts(df: pd.DataFrame, *, count_col: str = "count") -> pd.DataFrame:
    out = (
        df.dropna(subset=["bg_geoid", "tract_geoid"])
        .groupby(["tract_geoid", "bg_geoid"], as_index=False, sort=False)[count_col]
        .sum()
        .rename(columns={count_col: "count"})
    )
    return out


def _agg_tract_counts(df: pd.DataFrame, *, tract_col: str = "tract_geoid", count_col: str = "count") -> pd.DataFrame:
    out = (
        df.dropna(subset=[tract_col])
        .groupby([tract_col], as_index=False, sort=False)[count_col]
        .sum()
        .rename(columns={tract_col: "tract_geoid", count_col: "count"})
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="prepare_detroit_teacher_home_metrics")
    ap.add_argument("--synthetic_person_locations", required=True)
    ap.add_argument("--teacher_home_devices", required=True)
    ap.add_argument("--tiger_bg_zip", required=True)
    ap.add_argument("--label", default="phase3_validate_teacher_home")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--min_teacher_total", type=int, default=20)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics_detailed")

    synthetic_path = pathlib.Path(args.synthetic_person_locations).expanduser().resolve()
    teacher_path = pathlib.Path(args.teacher_home_devices).expanduser().resolve()
    bg_zip = pathlib.Path(args.tiger_bg_zip).expanduser().resolve()

    persons = _read_person_locations(synthetic_path)
    tract_set = set(persons["tract_geoid"].astype(str).unique().tolist())
    bg_units = load_bg_units(tiger_bg_zip=bg_zip, allowed_tracts=tract_set)

    syn_home_points = _prepare_synthetic_home_points(persons)
    syn_home_bg = spatial_join_points_to_bg(
        points=syn_home_points,
        x_col="home_x",
        y_col="home_y",
        bg_units=bg_units,
        keep_cols=["home_candidate_id", "count"],
    )
    syn_home_bg["count"] = pd.to_numeric(syn_home_bg["count"], errors="coerce").fillna(0).astype(int)
    syn_home_bg_counts = _agg_bg_counts(syn_home_bg)
    syn_home_tract_counts = _agg_tract_counts(syn_home_bg)

    teacher_home_points = _prepare_teacher_home_points(teacher_path)
    teacher_home_bg = spatial_join_points_to_bg(
        points=teacher_home_points,
        x_col="x",
        y_col="y",
        bg_units=bg_units,
        keep_cols=["ad_id", "count"],
    )
    teacher_home_bg["count"] = 1
    teacher_home_bg_counts = _agg_bg_counts(teacher_home_bg)
    teacher_home_tract_counts = _agg_tract_counts(teacher_home_bg)

    home_tract_comp, home_tract_summary = compare_share_frames(
        left=syn_home_tract_counts.rename(columns={"count": "synthetic_count"}),
        right=teacher_home_tract_counts.rename(columns={"count": "teacher_count"}),
        key_cols=["tract_geoid"],
        left_value_col="synthetic_count",
        right_value_col="teacher_count",
    )
    home_bg_spearman, home_bg_summary = within_tract_bg_spearman(
        synthetic_bg_counts=syn_home_bg_counts,
        mobility_bg_counts=teacher_home_bg_counts,
        min_mobility_total=int(args.min_teacher_total),
    )

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "synthetic_person_locations": str(synthetic_path),
        "teacher_home_devices": str(teacher_path),
        "tiger_bg_zip": str(bg_zip),
        "teacher_home_summary": {
            "devices": int(teacher_home_points["ad_id"].nunique()),
            "joined_bg_rows": int(len(teacher_home_bg)),
            "tracts_with_reference": int(teacher_home_tract_counts["tract_geoid"].nunique()),
        },
        "home_tract_validation": home_tract_summary,
        "home_bg_within_tract_validation": home_bg_summary,
    }

    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)
    home_tract_comp.to_csv(metrics_dir / "home_tract_comparison.csv", index=False)
    home_bg_spearman.to_csv(metrics_dir / "home_bg_spearman_by_tract.csv", index=False)
    teacher_home_tract_counts.rename(columns={"count": "teacher_count"}).to_csv(
        metrics_dir / "teacher_home_tract_counts.csv",
        index=False,
    )
    teacher_home_bg_counts.rename(columns={"count": "teacher_count"}).to_csv(
        metrics_dir / "teacher_home_bg_counts.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
