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
    AnchorSpec,
    compare_share_frames,
    haversine_m,
    load_bg_units,
    load_events_in_bbox,
    select_device_anchors,
    spatial_join_points_to_bg,
    summarize_distance_distribution,
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
        "work_candidate_id",
        "home_candidate_id",
        "home_x",
        "home_y",
        "work_x",
        "work_y",
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


def _prepare_synthetic_work_points(persons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    workers = persons.dropna(subset=["work_candidate_id", "work_x", "work_y"]).copy()
    workers["tract_geoid"] = workers["tract_geoid"].astype(str)
    unique_points = (
        workers.groupby(["work_candidate_id", "work_x", "work_y"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "count"})
    )
    worker_flows = workers.loc[:, ["person_id", "tract_geoid", "work_candidate_id", "home_x", "home_y", "work_x", "work_y"]].copy()
    return unique_points, worker_flows


def _prepare_mobility_anchor_points(home: pd.DataFrame, work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    home_points = home.rename(columns={"home_longitude": "x", "home_latitude": "y"}).loc[:, ["ad_id", "x", "y"]].copy()
    home_points["count"] = 1

    work_points = work.rename(
        columns={
            "home_longitude": "home_x",
            "home_latitude": "home_y",
            "work_longitude": "work_x",
            "work_latitude": "work_y",
        }
    ).copy()
    return home_points, work_points


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


def _agg_od_counts(df: pd.DataFrame, *, home_col: str, work_col: str, count_col: str = "count") -> pd.DataFrame:
    out = (
        df.dropna(subset=[home_col, work_col])
        .groupby([home_col, work_col], as_index=False, sort=False)[count_col]
        .sum()
        .rename(columns={home_col: "home_tract_geoid", work_col: "work_tract_geoid", count_col: "count"})
    )
    return out


def _join_synthetic_and_mobility(
    *,
    persons: pd.DataFrame,
    bg_units: Any,
    mobility_home: pd.DataFrame,
    mobility_work: pd.DataFrame,
) -> dict[str, Any]:
    syn_home_points = _prepare_synthetic_home_points(persons)
    syn_work_points, syn_worker_flows = _prepare_synthetic_work_points(persons)
    mob_home_points, mob_work_points = _prepare_mobility_anchor_points(mobility_home, mobility_work)

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

    syn_work_bg = spatial_join_points_to_bg(
        points=syn_work_points,
        x_col="work_x",
        y_col="work_y",
        bg_units=bg_units,
        keep_cols=["work_candidate_id", "count"],
    )
    syn_work_bg["count"] = pd.to_numeric(syn_work_bg["count"], errors="coerce").fillna(0).astype(int)
    syn_work_tract_by_candidate = syn_work_bg.loc[:, ["work_candidate_id", "tract_geoid"]].dropna().drop_duplicates("work_candidate_id")
    syn_workers = syn_worker_flows.merge(
        syn_work_tract_by_candidate.rename(columns={"tract_geoid": "work_tract_geoid"}),
        on="work_candidate_id",
        how="left",
    )
    syn_work_tract_counts = _agg_tract_counts(
        syn_workers.assign(count=1),
        tract_col="work_tract_geoid",
        count_col="count",
    )
    syn_od_counts = _agg_od_counts(
        syn_workers.assign(count=1),
        home_col="tract_geoid",
        work_col="work_tract_geoid",
        count_col="count",
    )
    syn_commute_distance_m = pd.Series(
        haversine_m(syn_workers["home_x"], syn_workers["home_y"], syn_workers["work_x"], syn_workers["work_y"]),
        name="distance_m",
    )

    mob_home_bg = spatial_join_points_to_bg(
        points=mob_home_points,
        x_col="x",
        y_col="y",
        bg_units=bg_units,
        keep_cols=["ad_id", "count"],
    )
    mob_home_bg["count"] = 1
    mob_home_bg_counts = _agg_bg_counts(mob_home_bg)
    mob_home_tract_counts = _agg_tract_counts(mob_home_bg)

    mob_work_home = spatial_join_points_to_bg(
        points=mob_work_points,
        x_col="home_x",
        y_col="home_y",
        bg_units=bg_units,
        keep_cols=["ad_id", "work_x", "work_y", "home_x", "home_y"],
    )
    mob_work_home = mob_work_home.rename(columns={"tract_geoid": "home_tract_geoid", "bg_geoid": "home_bg_geoid"})
    mob_work_full = spatial_join_points_to_bg(
        points=mob_work_points,
        x_col="work_x",
        y_col="work_y",
        bg_units=bg_units,
        keep_cols=["ad_id", "home_x", "home_y", "work_x", "work_y"],
    )
    mob_work_full = mob_work_full.rename(columns={"tract_geoid": "work_tract_geoid", "bg_geoid": "work_bg_geoid"})
    mob_work_full = mob_work_full.merge(
        mob_work_home.loc[:, ["ad_id", "home_tract_geoid", "home_bg_geoid"]],
        on="ad_id",
        how="left",
    )
    mob_work_tract_counts = _agg_tract_counts(
        mob_work_full.assign(count=1),
        tract_col="work_tract_geoid",
        count_col="count",
    )
    mob_od_counts = _agg_od_counts(
        mob_work_full.assign(count=1),
        home_col="home_tract_geoid",
        work_col="work_tract_geoid",
        count_col="count",
    )
    mob_commute_distance_m = pd.Series(
        haversine_m(mob_work_full["home_x"], mob_work_full["home_y"], mob_work_full["work_x"], mob_work_full["work_y"]),
        name="distance_m",
    )

    return {
        "syn_home_bg_counts": syn_home_bg_counts,
        "syn_home_tract_counts": syn_home_tract_counts,
        "syn_work_tract_counts": syn_work_tract_counts,
        "syn_od_counts": syn_od_counts,
        "syn_commute_distance_m": syn_commute_distance_m,
        "mob_home_bg_counts": mob_home_bg_counts,
        "mob_home_tract_counts": mob_home_tract_counts,
        "mob_work_tract_counts": mob_work_tract_counts,
        "mob_od_counts": mob_od_counts,
        "mob_commute_distance_m": mob_commute_distance_m,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase3_validate_mobility_anchor")
    ap.add_argument("--mobility_csv", required=True)
    ap.add_argument("--synthetic_person_locations", required=True)
    ap.add_argument("--tiger_bg_zip", required=True)
    ap.add_argument("--label", default="phase3_validate_mobility_anchor")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--chunksize", type=int, default=500000)
    ap.add_argument("--min_home_secs", type=int, default=6 * 3600)
    ap.add_argument("--min_work_secs", type=int, default=3 * 3600)
    ap.add_argument("--min_home_work_distance_m", type=float, default=500.0)
    ap.add_argument("--min_bg_mobility_total", type=int, default=20)
    ap.add_argument("--bbox_margin_deg", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    synthetic_path = pathlib.Path(args.synthetic_person_locations).expanduser().resolve()
    persons = _read_person_locations(synthetic_path)
    tract_set = set(persons["tract_geoid"].astype(str).unique().tolist())
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

    joined = _join_synthetic_and_mobility(
        persons=persons,
        bg_units=bg_units,
        mobility_home=mobility_home,
        mobility_work=mobility_work,
    )

    home_tract_comp, home_tract_summary = compare_share_frames(
        left=joined["syn_home_tract_counts"].rename(columns={"count": "synthetic_count"}),
        right=joined["mob_home_tract_counts"].rename(columns={"count": "mobility_count"}),
        key_cols=["tract_geoid"],
        left_value_col="synthetic_count",
        right_value_col="mobility_count",
    )
    home_bg_spearman, home_bg_summary = within_tract_bg_spearman(
        synthetic_bg_counts=joined["syn_home_bg_counts"],
        mobility_bg_counts=joined["mob_home_bg_counts"],
        min_mobility_total=int(args.min_bg_mobility_total),
    )
    work_tract_comp, work_tract_summary = compare_share_frames(
        left=joined["syn_work_tract_counts"].rename(columns={"count": "synthetic_count"}),
        right=joined["mob_work_tract_counts"].rename(columns={"count": "mobility_count"}),
        key_cols=["tract_geoid"],
        left_value_col="synthetic_count",
        right_value_col="mobility_count",
    )
    od_comp, od_summary = compare_share_frames(
        left=joined["syn_od_counts"].rename(columns={"count": "synthetic_count"}),
        right=joined["mob_od_counts"].rename(columns={"count": "mobility_count"}),
        key_cols=["home_tract_geoid", "work_tract_geoid"],
        left_value_col="synthetic_count",
        right_value_col="mobility_count",
    )
    dist_bins, dist_summary = summarize_distance_distribution(
        synthetic_distance_m=joined["syn_commute_distance_m"],
        mobility_distance_m=joined["mob_commute_distance_m"],
    )

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "mobility_csv": str(pathlib.Path(args.mobility_csv).expanduser().resolve()),
        "synthetic_person_locations": str(synthetic_path),
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
        "anchor_summary": anchor_summary,
        "home_tract_validation": home_tract_summary,
        "home_bg_within_tract_validation": home_bg_summary,
        "work_tract_validation": work_tract_summary,
        "work_od_validation": od_summary,
        "commute_distance_validation": dist_summary,
    }

    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)
    home_tract_comp.to_csv(metrics_dir / "home_tract_comparison.csv", index=False)
    home_bg_spearman.to_csv(metrics_dir / "home_bg_spearman_by_tract.csv", index=False)
    work_tract_comp.to_csv(metrics_dir / "work_tract_comparison.csv", index=False)
    od_comp.to_csv(metrics_dir / "work_od_comparison.csv", index=False)
    dist_bins.to_csv(metrics_dir / "commute_distance_bins.csv", index=False)


if __name__ == "__main__":
    main()
