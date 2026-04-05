#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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


def _read_geodata(path: pathlib.Path):
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_poi_visit_attractiveness requires geopandas.") from e

    suffix = path.suffix.lower()
    if suffix == ".zip":
        return gpd.read_file(f"zip://{path}")
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _parse_literal(value: Any) -> Any:
    text = str(value).strip()
    if not text or text in {"nan", "None", "null", "{}", "[]"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def _weekday_share(value: Any) -> float:
    obj = _parse_literal(value)
    if not isinstance(obj, dict) or not obj:
        return 1.0
    weekday_keys = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    try:
        total = float(sum(float(v) for v in obj.values()))
        weekday = float(sum(float(obj.get(k, 0.0)) for k in weekday_keys))
    except Exception:
        return 1.0
    if total <= 0.0:
        return 1.0
    return float(np.clip(weekday / total, 0.0, 1.0))


def _workhour_share(value: Any) -> float:
    obj = _parse_literal(value)
    if not isinstance(obj, (list, tuple)) or len(obj) != 24:
        return 1.0
    try:
        arr = np.asarray([float(x) for x in obj], dtype=float)
    except Exception:
        return 1.0
    total = float(arr.sum())
    if total <= 0.0:
        return 1.0
    # 09:00-17:59 local active daytime window.
    work = float(arr[9:18].sum())
    return float(np.clip(work / total, 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(prog="prepare_detroit_poi_visit_attractiveness")
    ap.add_argument("--areas_path", required=True)
    ap.add_argument("--areas_group_col", default="tract_geoid")
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--merged_poi_path", required=True)
    ap.add_argument("--region_filter", default="MI")
    ap.add_argument("--bbox_margin_deg", type=float, default=0.05)
    ap.add_argument("--chunksize", type=int, default=100000)
    ap.add_argument("--base_access_col", default="work_access_jobs_gravity")
    ap.add_argument("--visit_blend_power", type=float, default=0.5)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="prepare_detroit_poi_visit_attractiveness")
    args = ap.parse_args()

    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_poi_visit_attractiveness requires geopandas.") from e

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    areas_path = pathlib.Path(args.areas_path).expanduser().resolve()
    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    merged_poi_path = pathlib.Path(args.merged_poi_path).expanduser().resolve()
    for p in [areas_path, tract_od_path, merged_poi_path]:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")

    areas = _read_geodata(areas_path)
    group_col = str(args.areas_group_col)
    if group_col not in areas.columns:
        if "GEOID" in areas.columns:
            areas[group_col] = areas["GEOID"].astype(str)
        else:
            raise SystemExit(f"areas missing group column: {group_col}")
    areas[group_col] = areas[group_col].astype(str)

    tract_od = pd.read_csv(tract_od_path, low_memory=False)
    tract_od["work_tract_geoid"] = tract_od["work_tract_geoid"].astype(str)
    study_tracts = set(tract_od["work_tract_geoid"].unique().tolist())
    areas = areas[areas[group_col].isin(sorted(study_tracts))].copy()
    if areas.empty:
        raise SystemExit("No study tracts remain after intersecting tract_od with areas")

    areas_4326 = areas.to_crs(4326)
    minx, miny, maxx, maxy = areas_4326.total_bounds
    margin = float(args.bbox_margin_deg)
    bbox = (float(minx - margin), float(miny - margin), float(maxx + margin), float(maxy + margin))

    total_by_tract: dict[str, float] = {}
    workhour_by_tract: dict[str, float] = {}
    weekday_by_tract: dict[str, float] = {}
    n_rows_total = 0
    n_rows_bbox = 0
    n_rows_joined = 0

    usecols = [
        "latitude",
        "longitude",
        "region",
        "raw_visit_counts",
        "raw_visitor_counts",
        "popularity_by_hour",
        "popularity_by_day",
        "naics_code",
    ]

    for chunk in pd.read_csv(merged_poi_path, usecols=usecols, low_memory=False, chunksize=int(args.chunksize)):
        n_rows_total += int(len(chunk))
        chunk["region"] = chunk["region"].astype(str).str.upper()
        chunk = chunk[chunk["region"] == str(args.region_filter).strip().upper()].copy()
        if chunk.empty:
            continue
        chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
        chunk["raw_visit_counts"] = pd.to_numeric(chunk["raw_visit_counts"], errors="coerce").fillna(0.0)
        chunk["raw_visitor_counts"] = pd.to_numeric(chunk["raw_visitor_counts"], errors="coerce").fillna(0.0)
        chunk = chunk.dropna(subset=["latitude", "longitude"]).copy()
        chunk = chunk[
            chunk["longitude"].between(bbox[0], bbox[2], inclusive="both")
            & chunk["latitude"].between(bbox[1], bbox[3], inclusive="both")
        ].copy()
        chunk = chunk[chunk["raw_visit_counts"] > 0].copy()
        if chunk.empty:
            continue
        n_rows_bbox += int(len(chunk))

        chunk["weekday_share"] = chunk["popularity_by_day"].map(_weekday_share).astype(float)
        chunk["workhour_share"] = chunk["popularity_by_hour"].map(_workhour_share).astype(float)
        chunk["visit_weight_workhours"] = (
            chunk["raw_visit_counts"].astype(float)
            * chunk["weekday_share"].astype(float)
            * chunk["workhour_share"].astype(float)
        )
        chunk["visit_weight_weekday"] = chunk["raw_visit_counts"].astype(float) * chunk["weekday_share"].astype(float)

        points = gpd.GeoDataFrame(
            chunk.loc[:, ["raw_visit_counts", "visit_weight_workhours", "visit_weight_weekday"]].copy(),
            geometry=gpd.points_from_xy(chunk["longitude"], chunk["latitude"]),
            crs=4326,
        )
        joined = gpd.sjoin(
            points,
            areas_4326.loc[:, [group_col, "geometry"]].copy(),
            how="inner",
            predicate="within",
        )
        if joined.empty:
            continue
        n_rows_joined += int(len(joined))
        agg = (
            joined.groupby(group_col, as_index=False, sort=False)[
                ["raw_visit_counts", "visit_weight_workhours", "visit_weight_weekday"]
            ]
            .sum()
        )
        for row in agg.itertuples(index=False):
            gid = str(getattr(row, group_col))
            total_by_tract[gid] = total_by_tract.get(gid, 0.0) + float(row.raw_visit_counts)
            workhour_by_tract[gid] = workhour_by_tract.get(gid, 0.0) + float(row.visit_weight_workhours)
            weekday_by_tract[gid] = weekday_by_tract.get(gid, 0.0) + float(row.visit_weight_weekday)

    tract_visit = pd.DataFrame({"work_tract_geoid": sorted(study_tracts)})
    tract_visit["work_poi_visit_total"] = tract_visit["work_tract_geoid"].map(total_by_tract).fillna(0.0)
    tract_visit["work_poi_visit_weekday"] = tract_visit["work_tract_geoid"].map(weekday_by_tract).fillna(0.0)
    tract_visit["work_poi_visit_workhours"] = tract_visit["work_tract_geoid"].map(workhour_by_tract).fillna(0.0)
    tract_visit["work_poi_visit_log1p"] = np.log1p(tract_visit["work_poi_visit_workhours"].astype(float))
    tract_visit["work_poi_visit_strength"] = tract_visit["work_poi_visit_log1p"] + 1.0
    mean_strength = max(float(tract_visit["work_poi_visit_strength"].mean()), 1e-6)
    tract_visit["work_poi_visit_workhours_rel"] = tract_visit["work_poi_visit_strength"] / mean_strength

    tract_od_enriched = tract_od.merge(tract_visit, on="work_tract_geoid", how="left")
    for col in [
        "work_poi_visit_total",
        "work_poi_visit_weekday",
        "work_poi_visit_workhours",
        "work_poi_visit_log1p",
        "work_poi_visit_strength",
        "work_poi_visit_workhours_rel",
    ]:
        tract_od_enriched[col] = pd.to_numeric(tract_od_enriched[col], errors="coerce").fillna(0.0)

    base_access_col = str(args.base_access_col).strip()
    if base_access_col:
        if base_access_col not in tract_od_enriched.columns:
            raise SystemExit(f"base_access_col not found in tract_od: {base_access_col}")
        tract_od_enriched[base_access_col] = pd.to_numeric(
            tract_od_enriched[base_access_col], errors="coerce"
        ).fillna(0.0)
        tract_od_enriched["work_access_jobs_poi_visit_blend"] = (
            tract_od_enriched[base_access_col].astype(float)
            * np.power(
                np.clip(tract_od_enriched["work_poi_visit_workhours_rel"].astype(float), 1e-6, None),
                float(args.visit_blend_power),
            )
        )
    else:
        tract_od_enriched["work_access_jobs_poi_visit_blend"] = np.power(
            np.clip(tract_od_enriched["work_poi_visit_workhours_rel"].astype(float), 1e-6, None),
            float(args.visit_blend_power),
        )

    tract_visit.to_csv(metrics_dir / "tract_poi_visit_attractiveness.csv", index=False)
    tract_od_enriched.to_csv(run_dir / "tract_od.csv", index=False)

    payload = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "areas_path": str(areas_path),
        "areas_group_col": group_col,
        "tract_od_path": str(tract_od_path),
        "merged_poi_path": str(merged_poi_path),
        "region_filter": str(args.region_filter),
        "bbox": {"minx": bbox[0], "miny": bbox[1], "maxx": bbox[2], "maxy": bbox[3]},
        "base_access_col": (base_access_col or None),
        "visit_blend_power": float(args.visit_blend_power),
        "rows": {
            "n_rows_total": int(n_rows_total),
            "n_rows_bbox": int(n_rows_bbox),
            "n_rows_joined": int(n_rows_joined),
        },
        "tract_summary": {
            "n_work_tracts": int(len(tract_visit)),
            "n_positive_visit_tracts": int((tract_visit["work_poi_visit_workhours"] > 0).sum()),
            "mean_workhours_visit": float(tract_visit["work_poi_visit_workhours"].mean()),
            "median_workhours_visit": float(tract_visit["work_poi_visit_workhours"].median()),
            "p90_workhours_visit": float(tract_visit["work_poi_visit_workhours"].quantile(0.9)),
        },
        "artifacts": {
            "tract_od_csv": str(run_dir / "tract_od.csv"),
            "tract_poi_visit_attractiveness_csv": str(metrics_dir / "tract_poi_visit_attractiveness.csv"),
        },
        "new_columns": [
            "work_poi_visit_total",
            "work_poi_visit_weekday",
            "work_poi_visit_workhours",
            "work_poi_visit_log1p",
            "work_poi_visit_strength",
            "work_poi_visit_workhours_rel",
            "work_access_jobs_poi_visit_blend",
        ],
    }
    _write_json(run_dir / "run_summary.json", payload)
    _write_json(metrics_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
