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


def _read_geodata(path: pathlib.Path):
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_poi_attractiveness requires geopandas.") from e

    suffix = path.suffix.lower()
    if suffix == ".zip":
        return gpd.read_file(f"zip://{path}")
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _nonempty(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return (~series.isna()) & (~text.isin({"", "nan", "None", "null"}))


def main() -> None:
    ap = argparse.ArgumentParser(prog="prepare_detroit_poi_attractiveness")
    ap.add_argument("--areas_path", required=True)
    ap.add_argument("--areas_group_col", default="tract_geoid")
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--safegraph_unzip_dir", required=True)
    ap.add_argument("--poi_glob", default="Global_Places_POI_Data*.csv")
    ap.add_argument("--region_filter", default="MI")
    ap.add_argument("--bbox_margin_deg", type=float, default=0.05)
    ap.add_argument("--chunksize", type=int, default=200000)
    ap.add_argument("--base_access_col", default="work_access_jobs_gravity")
    ap.add_argument("--poi_blend_power", type=float, default=0.5)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="prepare_detroit_poi_attractiveness")
    args = ap.parse_args()

    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_poi_attractiveness requires geopandas.") from e

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    areas_path = pathlib.Path(args.areas_path).expanduser().resolve()
    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    safegraph_dir = pathlib.Path(args.safegraph_unzip_dir).expanduser().resolve()
    for p in [areas_path, tract_od_path, safegraph_dir]:
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
    if "work_tract_geoid" not in tract_od.columns:
        raise SystemExit("tract_od_path missing work_tract_geoid column")
    tract_od["work_tract_geoid"] = tract_od["work_tract_geoid"].astype(str)
    study_tracts = set(tract_od["work_tract_geoid"].astype(str).unique().tolist())
    areas = areas[areas[group_col].isin(sorted(study_tracts))].copy()
    if areas.empty:
        raise SystemExit("No study tracts remain after intersecting tract_od with areas")

    areas_4326 = areas.to_crs(4326)
    areas_metric = areas.to_crs(5070)
    tract_area = pd.DataFrame(
        {
            group_col: areas[group_col].astype(str).tolist(),
            "tract_area_km2": (areas_metric.geometry.area / 1_000_000.0).astype(float).tolist(),
        }
    )

    minx, miny, maxx, maxy = areas_4326.total_bounds
    margin = float(args.bbox_margin_deg)
    bbox = (float(minx - margin), float(miny - margin), float(maxx + margin), float(maxy + margin))

    poi_files = sorted(safegraph_dir.glob(str(args.poi_glob)))
    if not poi_files:
        raise SystemExit(f"No POI files matched {args.poi_glob} in {safegraph_dir}")

    total_by_tract: dict[str, int] = {}
    worklike_by_tract: dict[str, int] = {}
    n_files = 0
    n_rows_total = 0
    n_rows_bbox = 0
    n_rows_joined = 0

    usecols = [
        "LATITUDE",
        "LONGITUDE",
        "REGION",
        "NAICS_CODE",
        "CLOSED_ON",
        "TRACKING_CLOSED_SINCE",
    ]

    for poi_path in poi_files:
        n_files += 1
        for chunk in pd.read_csv(poi_path, usecols=usecols, low_memory=False, chunksize=int(args.chunksize)):
            n_rows_total += int(len(chunk))
            chunk = chunk.rename(
                columns={
                    "LATITUDE": "latitude",
                    "LONGITUDE": "longitude",
                    "REGION": "region",
                    "NAICS_CODE": "naics_code",
                    "CLOSED_ON": "closed_on",
                    "TRACKING_CLOSED_SINCE": "tracking_closed_since",
                }
            )
            chunk["region"] = chunk["region"].astype(str).str.upper()
            chunk = chunk[chunk["region"] == str(args.region_filter).strip().upper()].copy()
            if chunk.empty:
                continue
            chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
            chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
            chunk = chunk.dropna(subset=["latitude", "longitude"]).copy()
            chunk = chunk[
                chunk["longitude"].between(bbox[0], bbox[2], inclusive="both")
                & chunk["latitude"].between(bbox[1], bbox[3], inclusive="both")
            ].copy()
            if chunk.empty:
                continue
            n_rows_bbox += int(len(chunk))

            active_mask = ~_nonempty(chunk["closed_on"]) & ~_nonempty(chunk["tracking_closed_since"])
            chunk = chunk[active_mask].copy()
            if chunk.empty:
                continue

            chunk["has_naics"] = _nonempty(chunk["naics_code"]).astype(int)
            points = gpd.GeoDataFrame(
                chunk.loc[:, ["has_naics"]].copy(),
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
            joined[group_col] = joined[group_col].astype(str)
            total_counts = joined.groupby(group_col, sort=False).size()
            work_counts = joined.groupby(group_col, sort=False)["has_naics"].sum()
            for tract_geoid, val in total_counts.items():
                total_by_tract[str(tract_geoid)] = total_by_tract.get(str(tract_geoid), 0) + int(val)
            for tract_geoid, val in work_counts.items():
                worklike_by_tract[str(tract_geoid)] = worklike_by_tract.get(str(tract_geoid), 0) + int(val)

    tract_poi = tract_area.copy()
    tract_poi["poi_count_all"] = tract_poi[group_col].map(total_by_tract).fillna(0).astype(int)
    tract_poi["poi_count_worklike"] = tract_poi[group_col].map(worklike_by_tract).fillna(0).astype(int)
    tract_poi["poi_density_worklike_km2"] = (
        tract_poi["poi_count_worklike"].astype(float) / tract_poi["tract_area_km2"].clip(lower=1e-6)
    )
    tract_poi["poi_log1p_worklike"] = np.log1p(tract_poi["poi_count_worklike"].astype(float))
    tract_poi["poi_strength_worklike"] = tract_poi["poi_log1p_worklike"] + 1.0
    mean_strength = max(float(tract_poi["poi_strength_worklike"].mean()), 1e-6)
    tract_poi["poi_strength_worklike_rel"] = tract_poi["poi_strength_worklike"] / mean_strength

    tract_poi = tract_poi.rename(
        columns={
            group_col: "work_tract_geoid",
            "poi_count_all": "work_poi_count_all",
            "poi_count_worklike": "work_poi_count_worklike",
            "poi_density_worklike_km2": "work_poi_density_worklike_km2",
            "poi_log1p_worklike": "work_poi_log1p_worklike",
            "poi_strength_worklike": "work_poi_strength_worklike",
            "poi_strength_worklike_rel": "work_poi_strength_worklike_rel",
        }
    )

    tract_od_enriched = tract_od.merge(tract_poi, on="work_tract_geoid", how="left")
    poi_cols = [
        "tract_area_km2",
        "work_poi_count_all",
        "work_poi_count_worklike",
        "work_poi_density_worklike_km2",
        "work_poi_log1p_worklike",
        "work_poi_strength_worklike",
        "work_poi_strength_worklike_rel",
    ]
    for col in poi_cols:
        tract_od_enriched[col] = pd.to_numeric(tract_od_enriched[col], errors="coerce").fillna(0.0)

    base_access_col = str(args.base_access_col).strip()
    if base_access_col:
        if base_access_col not in tract_od_enriched.columns:
            raise SystemExit(f"base_access_col not found in tract_od: {base_access_col}")
        tract_od_enriched[base_access_col] = pd.to_numeric(
            tract_od_enriched[base_access_col], errors="coerce"
        ).fillna(0.0)
        tract_od_enriched["work_access_jobs_poi_blend"] = (
            tract_od_enriched[base_access_col].astype(float)
            * np.power(
                np.clip(tract_od_enriched["work_poi_strength_worklike_rel"].astype(float), 1e-6, None),
                float(args.poi_blend_power),
            )
        )
    else:
        tract_od_enriched["work_access_jobs_poi_blend"] = np.power(
            np.clip(tract_od_enriched["work_poi_strength_worklike_rel"].astype(float), 1e-6, None),
            float(args.poi_blend_power),
        )

    tract_poi.to_csv(metrics_dir / "tract_poi_attractiveness.csv", index=False)
    tract_od_enriched.to_csv(run_dir / "tract_od.csv", index=False)

    payload = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "areas_path": str(areas_path),
        "areas_group_col": str(args.areas_group_col),
        "tract_od_path": str(tract_od_path),
        "safegraph_unzip_dir": str(safegraph_dir),
        "poi_glob": str(args.poi_glob),
        "region_filter": str(args.region_filter),
        "bbox": {
            "minx": bbox[0],
            "miny": bbox[1],
            "maxx": bbox[2],
            "maxy": bbox[3],
        },
        "base_access_col": (base_access_col or None),
        "poi_blend_power": float(args.poi_blend_power),
        "rows": {
            "n_files": int(n_files),
            "n_rows_total": int(n_rows_total),
            "n_rows_bbox": int(n_rows_bbox),
            "n_rows_joined": int(n_rows_joined),
        },
        "tract_summary": {
            "n_work_tracts": int(len(tract_poi)),
            "n_positive_worklike_poi_tracts": int((tract_poi["work_poi_count_worklike"] > 0).sum()),
            "mean_worklike_poi_count": float(tract_poi["work_poi_count_worklike"].mean()),
            "median_worklike_poi_count": float(tract_poi["work_poi_count_worklike"].median()),
            "p90_worklike_poi_count": float(tract_poi["work_poi_count_worklike"].quantile(0.9)),
        },
        "artifacts": {
            "tract_od_csv": str(run_dir / "tract_od.csv"),
            "tract_poi_attractiveness_csv": str(metrics_dir / "tract_poi_attractiveness.csv"),
        },
        "new_columns": poi_cols + ["work_access_jobs_poi_blend"],
    }
    _write_json(run_dir / "run_summary.json", payload)
    _write_json(metrics_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
