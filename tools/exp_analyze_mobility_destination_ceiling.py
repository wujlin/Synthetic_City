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


def _resolve_work_center_map(tract_od: pd.DataFrame) -> pd.DataFrame:
    use = tract_od.loc[:, ["work_tract_geoid", "work_center_geoid"]].dropna().copy()
    use["work_tract_geoid"] = use["work_tract_geoid"].astype(str)
    use["work_center_geoid"] = use["work_center_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    counts = (
        use.groupby(["work_tract_geoid", "work_center_geoid"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "n"})
    )
    counts = counts.sort_values(
        by=["work_tract_geoid", "n", "work_center_geoid"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return counts.drop_duplicates("work_tract_geoid", keep="first").loc[:, ["work_tract_geoid", "work_center_geoid"]].copy()


def _agg_counts(df: pd.DataFrame, group_cols: list[str], count_col: str = "count") -> pd.DataFrame:
    out = (
        df.dropna(subset=group_cols)
        .groupby(group_cols, as_index=False, sort=False)[count_col]
        .sum()
    )
    return out


def _compare_level(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: list[str],
    left_value_col: str,
    right_value_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged, summary = compare_share_frames(
        left=left,
        right=right,
        key_cols=key_cols,
        left_value_col=left_value_col,
        right_value_col=right_value_col,
    )
    summary.update(_support_overlap_metrics(merged))
    return merged, summary


def _support_overlap_metrics(merged: pd.DataFrame) -> dict[str, Any]:
    left_pos = pd.to_numeric(merged["left_share"], errors="coerce").fillna(0.0) > 0.0
    right_pos = pd.to_numeric(merged["right_share"], errors="coerce").fillna(0.0) > 0.0
    union = left_pos | right_pos
    inter = left_pos & right_pos

    union_n = int(union.sum())
    inter_n = int(inter.sum())
    if union_n <= 0:
        return {
            "positive_union_n_units": 0,
            "positive_intersection_n_units": 0,
            "support_jaccard": float("nan"),
            "positive_union_spearman_share": float("nan"),
            "positive_union_cosine_share": float("nan"),
            "positive_union_tvd_share": float("nan"),
        }

    pos = merged.loc[union, ["left_share", "right_share"]].copy()
    pos["left_share"] = pd.to_numeric(pos["left_share"], errors="coerce").fillna(0.0)
    pos["right_share"] = pd.to_numeric(pos["right_share"], errors="coerce").fillna(0.0)

    if len(pos) >= 2 and pos["left_share"].nunique() > 1 and pos["right_share"].nunique() > 1:
        pos_spearman = float(pos["left_share"].corr(pos["right_share"], method="spearman"))
    else:
        pos_spearman = float("nan")

    left_vec = pos["left_share"].to_numpy(dtype=float)
    right_vec = pos["right_share"].to_numpy(dtype=float)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    pos_cosine = float(np.dot(left_vec, right_vec) / denom) if denom > 0.0 else float("nan")
    pos_tvd = float(0.5 * np.abs(left_vec - right_vec).sum())

    return {
        "positive_union_n_units": union_n,
        "positive_intersection_n_units": inter_n,
        "support_jaccard": float(inter_n / union_n),
        "positive_union_spearman_share": pos_spearman,
        "positive_union_cosine_share": pos_cosine,
        "positive_union_tvd_share": pos_tvd,
    }


def _split_half_metrics(
    *,
    flows: pd.DataFrame,
    level_name: str,
    key_cols: list[str],
    n_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    device_ids = pd.Index(flows["ad_id"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for rep in range(int(n_repeats)):
        rng = np.random.default_rng(int(seed) + rep)
        perm = device_ids.to_numpy(dtype=object).copy()
        rng.shuffle(perm)
        cut = int(len(perm) // 2)
        left_ids = set(perm[:cut].tolist())
        right_ids = set(perm[cut:].tolist())
        left = _agg_counts(flows[flows["ad_id"].astype(str).isin(left_ids)].assign(count=1), key_cols, "count")
        right = _agg_counts(flows[flows["ad_id"].astype(str).isin(right_ids)].assign(count=1), key_cols, "count")
        _, summary = _compare_level(
            left=left.rename(columns={"count": "left_count"}),
            right=right.rename(columns={"count": "right_count"}),
            key_cols=key_cols,
            left_value_col="left_count",
            right_value_col="right_count",
        )
        rows.append(
            {
                "repeat": rep,
                "level": level_name,
                "spearman_share": summary["spearman_share"],
                "cosine_share": summary["cosine_share"],
                "tvd_share": summary["tvd_share"],
                "top_k_overlap": summary["top_k_overlap"],
                "support_jaccard": summary["support_jaccard"],
                "positive_union_spearman_share": summary["positive_union_spearman_share"],
                "positive_union_cosine_share": summary["positive_union_cosine_share"],
                "positive_union_tvd_share": summary["positive_union_tvd_share"],
                "left_total": summary["left_total"],
                "right_total": summary["right_total"],
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "n_repeats": int(len(out)),
        "mean_spearman_share": float(pd.to_numeric(out["spearman_share"], errors="coerce").mean()),
        "median_spearman_share": float(pd.to_numeric(out["spearman_share"], errors="coerce").median()),
        "mean_cosine_share": float(pd.to_numeric(out["cosine_share"], errors="coerce").mean()),
        "median_cosine_share": float(pd.to_numeric(out["cosine_share"], errors="coerce").median()),
        "mean_tvd_share": float(pd.to_numeric(out["tvd_share"], errors="coerce").mean()),
        "median_tvd_share": float(pd.to_numeric(out["tvd_share"], errors="coerce").median()),
        "mean_top_k_overlap": float(pd.to_numeric(out["top_k_overlap"], errors="coerce").mean()),
        "median_top_k_overlap": float(pd.to_numeric(out["top_k_overlap"], errors="coerce").median()),
        "mean_support_jaccard": float(pd.to_numeric(out["support_jaccard"], errors="coerce").mean()),
        "median_support_jaccard": float(pd.to_numeric(out["support_jaccard"], errors="coerce").median()),
        "mean_positive_union_spearman_share": float(pd.to_numeric(out["positive_union_spearman_share"], errors="coerce").mean()),
        "median_positive_union_spearman_share": float(pd.to_numeric(out["positive_union_spearman_share"], errors="coerce").median()),
        "mean_positive_union_cosine_share": float(pd.to_numeric(out["positive_union_cosine_share"], errors="coerce").mean()),
        "median_positive_union_cosine_share": float(pd.to_numeric(out["positive_union_cosine_share"], errors="coerce").median()),
        "mean_positive_union_tvd_share": float(pd.to_numeric(out["positive_union_tvd_share"], errors="coerce").mean()),
        "median_positive_union_tvd_share": float(pd.to_numeric(out["positive_union_tvd_share"], errors="coerce").median()),
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_analyze_mobility_destination_ceiling")
    ap.add_argument("--mobility_csv", required=True)
    ap.add_argument("--tiger_bg_zip", required=True)
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--label", default="analyze_mobility_destination_ceiling")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--chunksize", type=int, default=500000)
    ap.add_argument("--min_home_secs", type=int, default=6 * 3600)
    ap.add_argument("--min_work_secs", type=int, default=3 * 3600)
    ap.add_argument("--min_home_work_distance_m", type=float, default=500.0)
    ap.add_argument("--bbox_margin_deg", type=float, default=0.05)
    ap.add_argument("--n_repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    tract_od = pd.read_csv(
        tract_od_path,
        usecols=["home_tract_geoid", "work_tract_geoid", "work_center_geoid", "S000"],
        low_memory=False,
    )
    tract_od["home_tract_geoid"] = tract_od["home_tract_geoid"].astype(str)
    tract_od["work_tract_geoid"] = tract_od["work_tract_geoid"].astype(str)
    tract_od["S000"] = pd.to_numeric(tract_od["S000"], errors="coerce").fillna(0.0)
    tract_od = tract_od[tract_od["S000"] > 0.0].copy()
    tract_od["work_county_geoid"] = tract_od["work_tract_geoid"].str.slice(0, 5)
    center_map = _resolve_work_center_map(tract_od)
    tract_od = tract_od.drop(columns=["work_center_geoid"], errors="ignore").merge(center_map, on="work_tract_geoid", how="left")
    tract_od["work_center_geoid"] = tract_od["work_center_geoid"].fillna("missing_center").astype(str)

    allowed_tracts = sorted(set(tract_od["home_tract_geoid"].tolist()) | set(tract_od["work_tract_geoid"].tolist()))
    bg_units = load_bg_units(tiger_bg_zip=args.tiger_bg_zip, allowed_tracts=set(allowed_tracts))
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

    work_home = spatial_join_points_to_bg(
        points=mobility_work,
        x_col="home_longitude",
        y_col="home_latitude",
        bg_units=bg_units,
        keep_cols=["ad_id", "work_longitude", "work_latitude"],
    ).rename(columns={"tract_geoid": "home_tract_geoid", "bg_geoid": "home_bg_geoid"})
    work_full = spatial_join_points_to_bg(
        points=mobility_work,
        x_col="work_longitude",
        y_col="work_latitude",
        bg_units=bg_units,
        keep_cols=["ad_id", "home_longitude", "home_latitude", "work_longitude", "work_latitude"],
    ).rename(columns={"tract_geoid": "work_tract_geoid", "bg_geoid": "work_bg_geoid"})
    work_full = work_full.merge(work_home.loc[:, ["ad_id", "home_tract_geoid", "home_bg_geoid"]], on="ad_id", how="left")
    work_full["home_tract_geoid"] = work_full["home_tract_geoid"].astype(str)
    work_full["work_tract_geoid"] = work_full["work_tract_geoid"].astype(str)
    work_full["work_county_geoid"] = work_full["work_tract_geoid"].str.slice(0, 5)
    work_full = work_full.merge(center_map, on="work_tract_geoid", how="left")
    work_full["work_center_geoid"] = work_full["work_center_geoid"].fillna("missing_center").astype(str)

    lodes_tract = tract_od.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)["S000"].sum()
    lodes_county = tract_od.groupby(["home_tract_geoid", "work_county_geoid"], as_index=False, sort=False)["S000"].sum()
    lodes_center = tract_od.groupby(["home_tract_geoid", "work_center_geoid"], as_index=False, sort=False)["S000"].sum()

    mob_tract = _agg_counts(work_full.assign(count=1), ["home_tract_geoid", "work_tract_geoid"], "count")
    mob_county = _agg_counts(work_full.assign(count=1), ["home_tract_geoid", "work_county_geoid"], "count")
    mob_center = _agg_counts(work_full.assign(count=1), ["home_tract_geoid", "work_center_geoid"], "count")

    tract_comp, tract_summary = _compare_level(
        left=lodes_tract.rename(columns={"S000": "lodes_count"}),
        right=mob_tract.rename(columns={"count": "mobility_count"}),
        key_cols=["home_tract_geoid", "work_tract_geoid"],
        left_value_col="lodes_count",
        right_value_col="mobility_count",
    )
    county_comp, county_summary = _compare_level(
        left=lodes_county.rename(columns={"S000": "lodes_count"}),
        right=mob_county.rename(columns={"count": "mobility_count"}),
        key_cols=["home_tract_geoid", "work_county_geoid"],
        left_value_col="lodes_count",
        right_value_col="mobility_count",
    )
    center_comp, center_summary = _compare_level(
        left=lodes_center.rename(columns={"S000": "lodes_count"}),
        right=mob_center.rename(columns={"count": "mobility_count"}),
        key_cols=["home_tract_geoid", "work_center_geoid"],
        left_value_col="lodes_count",
        right_value_col="mobility_count",
    )

    tract_split, tract_split_summary = _split_half_metrics(
        flows=work_full.loc[:, ["ad_id", "home_tract_geoid", "work_tract_geoid"]].copy(),
        level_name="tract_od",
        key_cols=["home_tract_geoid", "work_tract_geoid"],
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
    )
    county_split, county_split_summary = _split_half_metrics(
        flows=work_full.loc[:, ["ad_id", "home_tract_geoid", "work_county_geoid"]].copy(),
        level_name="county_od",
        key_cols=["home_tract_geoid", "work_county_geoid"],
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
    )
    center_split, center_split_summary = _split_half_metrics(
        flows=work_full.loc[:, ["ad_id", "home_tract_geoid", "work_center_geoid"]].copy(),
        level_name="center_od",
        key_cols=["home_tract_geoid", "work_center_geoid"],
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
    )

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "mobility_csv": str(pathlib.Path(args.mobility_csv).expanduser().resolve()),
        "tiger_bg_zip": str(pathlib.Path(args.tiger_bg_zip).expanduser().resolve()),
        "tract_od_path": str(tract_od_path),
        "bbox": {"minx": bbox[0], "miny": bbox[1], "maxx": bbox[2], "maxy": bbox[3]},
        "anchor_spec": {
            "min_home_secs": spec.min_home_secs,
            "min_work_secs": spec.min_work_secs,
            "min_home_work_distance_m": spec.min_home_work_distance_m,
        },
        "anchor_summary": anchor_summary,
        "cross_source_compatibility": {
            "tract_od": tract_summary,
            "county_od": county_summary,
            "center_od": center_summary,
        },
        "split_half_reliability": {
            "tract_od": tract_split_summary,
            "county_od": county_split_summary,
            "center_od": center_split_summary,
        },
    }

    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)
    tract_comp.to_csv(metrics_dir / "tract_od_cross_source.csv", index=False)
    county_comp.to_csv(metrics_dir / "county_od_cross_source.csv", index=False)
    center_comp.to_csv(metrics_dir / "center_od_cross_source.csv", index=False)
    pd.concat([tract_split, county_split, center_split], ignore_index=True).to_csv(metrics_dir / "split_half_reliability.csv", index=False)


if __name__ == "__main__":
    main()
