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
        raise SystemExit("prepare_detroit_teacher_reference_od requires geopandas.") from e

    suffix = path.suffix.lower()
    if suffix == ".zip":
        return gpd.read_file(f"zip://{path}")
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _bbox_from_areas(areas_4326, *, margin_deg: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = areas_4326.total_bounds
    m = float(margin_deg)
    return (float(minx - m), float(miny - m), float(maxx + m), float(maxy + m))


def _collapse_home_reference(
    *,
    home_df: pd.DataFrame,
    areas_4326,
    group_col: str,
    bbox: tuple[float, float, float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_teacher_reference_od requires geopandas.") from e

    work = home_df.loc[
        :,
        ["ad_id", "date", "geohash_7", "latitude", "longitude", "nighttime_time_spent", "income", "rent"],
    ].copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["nighttime_time_spent"] = pd.to_numeric(work["nighttime_time_spent"], errors="coerce").fillna(0.0)
    work["income"] = pd.to_numeric(work["income"], errors="coerce")
    work["rent"] = pd.to_numeric(work["rent"], errors="coerce")
    work = work.dropna(subset=["ad_id", "latitude", "longitude"]).copy()
    rows_raw = int(len(work))
    work = work[
        work["longitude"].between(bbox[0], bbox[2], inclusive="both")
        & work["latitude"].between(bbox[1], bbox[3], inclusive="both")
    ].copy()
    rows_bbox = int(len(work))
    if work.empty:
        return pd.DataFrame(), {
            "rows_raw": rows_raw,
            "rows_bbox": rows_bbox,
            "rows_joined": 0,
            "devices_bbox": 0,
            "devices_collapsed": 0,
        }

    points = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work["longitude"], work["latitude"]),
        crs=4326,
    )
    joined = gpd.sjoin(
        points,
        areas_4326.loc[:, [group_col, "geometry"]].copy(),
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    if joined.empty:
        return pd.DataFrame(), {
            "rows_raw": rows_raw,
            "rows_bbox": rows_bbox,
            "rows_joined": 0,
            "devices_bbox": int(work["ad_id"].nunique()),
            "devices_collapsed": 0,
        }

    agg = (
        joined.groupby(["ad_id", group_col, "geohash_7"], as_index=False, sort=False)
        .agg(
            home_days=("date", "nunique"),
            nighttime_time_spent_total=("nighttime_time_spent", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            income=("income", "median"),
            rent=("rent", "median"),
        )
        .sort_values(
            ["ad_id", "home_days", "nighttime_time_spent_total", "geohash_7"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates(subset=["ad_id"], keep="first")
        .reset_index(drop=True)
        .rename(columns={group_col: "home_tract_geoid"})
    )
    agg["home_tract_geoid"] = agg["home_tract_geoid"].astype(str)
    return agg, {
        "rows_raw": rows_raw,
        "rows_bbox": rows_bbox,
        "rows_joined": int(len(joined)),
        "devices_bbox": int(work["ad_id"].nunique()),
        "devices_collapsed": int(len(agg)),
    }


def _collapse_work_reference(
    *,
    work_path: pathlib.Path,
    areas_4326,
    group_col: str,
    bbox: tuple[float, float, float, float],
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_teacher_reference_od requires geopandas.") from e

    rows_raw = 0
    rows_bbox = 0
    rows_joined = 0
    bbox_devices: set[str] = set()
    joined_parts: list[pd.DataFrame] = []

    usecols = [
        "ad_id",
        "geohash_7",
        "work_latitude",
        "work_longitude",
        "validated_days",
        "total_work_time",
    ]
    for chunk in pd.read_csv(work_path, usecols=usecols, low_memory=False, chunksize=int(chunksize)):
        rows_raw += int(len(chunk))
        chunk["work_latitude"] = pd.to_numeric(chunk["work_latitude"], errors="coerce")
        chunk["work_longitude"] = pd.to_numeric(chunk["work_longitude"], errors="coerce")
        chunk["validated_days"] = pd.to_numeric(chunk["validated_days"], errors="coerce").fillna(0.0)
        chunk["total_work_time"] = pd.to_numeric(chunk["total_work_time"], errors="coerce").fillna(0.0)
        chunk = chunk.dropna(subset=["ad_id", "work_latitude", "work_longitude"]).copy()
        chunk = chunk[
            chunk["work_longitude"].between(bbox[0], bbox[2], inclusive="both")
            & chunk["work_latitude"].between(bbox[1], bbox[3], inclusive="both")
        ].copy()
        if chunk.empty:
            continue
        rows_bbox += int(len(chunk))
        bbox_devices.update(chunk["ad_id"].astype(str).tolist())
        points = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk["work_longitude"], chunk["work_latitude"]),
            crs=4326,
        )
        joined = gpd.sjoin(
            points,
            areas_4326.loc[:, [group_col, "geometry"]].copy(),
            how="inner",
            predicate="within",
        ).drop(columns=["index_right"], errors="ignore")
        if joined.empty:
            continue
        rows_joined += int(len(joined))
        joined_parts.append(joined.drop(columns=["geometry"], errors="ignore").copy())

    if not joined_parts:
        return pd.DataFrame(), {
            "rows_raw": rows_raw,
            "rows_bbox": rows_bbox,
            "rows_joined": rows_joined,
            "devices_bbox": len(bbox_devices),
            "devices_collapsed": 0,
        }

    joined_all = pd.concat(joined_parts, ignore_index=True)
    agg = (
        joined_all.groupby(["ad_id", group_col, "geohash_7"], as_index=False, sort=False)
        .agg(
            work_latitude=("work_latitude", "mean"),
            work_longitude=("work_longitude", "mean"),
            validated_days=("validated_days", "max"),
            total_work_time=("total_work_time", "max"),
        )
        .sort_values(
            ["ad_id", "validated_days", "total_work_time", "geohash_7"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates(subset=["ad_id"], keep="first")
        .reset_index(drop=True)
        .rename(columns={group_col: "work_tract_geoid"})
    )
    agg["work_tract_geoid"] = agg["work_tract_geoid"].astype(str)
    return agg, {
        "rows_raw": rows_raw,
        "rows_bbox": rows_bbox,
        "rows_joined": rows_joined,
        "devices_bbox": len(bbox_devices),
        "devices_collapsed": int(len(agg)),
    }


def _collapse_counts(df: pd.DataFrame, tract_col: str, out_col: str) -> pd.DataFrame:
    out = (
        df.groupby(str(tract_col), as_index=False, sort=False)["ad_id"]
        .nunique()
        .rename(columns={"ad_id": str(out_col)})
        .sort_values(str(tract_col), kind="stable")
        .reset_index(drop=True)
    )
    out[str(tract_col)] = out[str(tract_col)].astype(str)
    out[str(out_col)] = pd.to_numeric(out[str(out_col)], errors="coerce").fillna(0).astype(int)
    return out


def _od_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)["ad_id"]
        .nunique()
        .rename(columns={"ad_id": "S000"})
        .sort_values(["home_tract_geoid", "work_tract_geoid"], kind="stable")
        .reset_index(drop=True)
    )
    out["S000"] = pd.to_numeric(out["S000"], errors="coerce").fillna(0).astype(int)
    return out


def _county_od_counts(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["home_county_geoid"] = work["home_tract_geoid"].astype(str).str.slice(0, 5)
    work["work_county_geoid"] = work["work_tract_geoid"].astype(str).str.slice(0, 5)
    out = (
        work.groupby(["home_county_geoid", "work_county_geoid"], as_index=False, sort=False)["ad_id"]
        .nunique()
        .rename(columns={"ad_id": "S000"})
        .sort_values(["home_county_geoid", "work_county_geoid"], kind="stable")
        .reset_index(drop=True)
    )
    out["S000"] = pd.to_numeric(out["S000"], errors="coerce").fillna(0).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="prepare_detroit_teacher_reference_od")
    ap.add_argument("--areas_path", required=True)
    ap.add_argument("--areas_group_col", default="tract_geoid")
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--teacher_home_path", required=True)
    ap.add_argument("--teacher_work_path", required=True)
    ap.add_argument("--bbox_margin_deg", type=float, default=0.05)
    ap.add_argument("--work_chunksize", type=int, default=500000)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="prepare_detroit_teacher_reference_od")
    args = ap.parse_args()

    try:
        import joblib
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_teacher_reference_od requires joblib.") from e

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    areas_path = pathlib.Path(args.areas_path).expanduser().resolve()
    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    teacher_home_path = pathlib.Path(args.teacher_home_path).expanduser().resolve()
    teacher_work_path = pathlib.Path(args.teacher_work_path).expanduser().resolve()
    for p in [areas_path, tract_od_path, teacher_home_path, teacher_work_path]:
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
    tract_od["home_tract_geoid"] = tract_od["home_tract_geoid"].astype(str)
    tract_od["work_tract_geoid"] = tract_od["work_tract_geoid"].astype(str)
    study_home_tracts = set(tract_od["home_tract_geoid"].unique().tolist())
    study_work_tracts = set(tract_od["work_tract_geoid"].unique().tolist())
    study_tracts = study_home_tracts | study_work_tracts

    areas = areas[areas[group_col].isin(sorted(study_tracts))].copy()
    if areas.empty:
        raise SystemExit("No study tracts remain after intersecting tract_od with areas")
    areas_4326 = areas.to_crs(4326)
    bbox = _bbox_from_areas(areas_4326, margin_deg=float(args.bbox_margin_deg))

    home_areas = areas_4326[areas_4326[group_col].isin(sorted(study_home_tracts))].copy()
    work_areas = areas_4326[areas_4326[group_col].isin(sorted(study_work_tracts))].copy()

    home_df = joblib.load(teacher_home_path)
    collapsed_home, home_summary = _collapse_home_reference(
        home_df=home_df,
        areas_4326=home_areas,
        group_col=group_col,
        bbox=bbox,
    )
    del home_df

    collapsed_work, work_summary = _collapse_work_reference(
        work_path=teacher_work_path,
        areas_4326=work_areas,
        group_col=group_col,
        bbox=bbox,
        chunksize=int(args.work_chunksize),
    )

    if collapsed_home.empty or collapsed_work.empty:
        raise SystemExit("Collapsed home/work reference is empty after Detroit filtering")

    paired = collapsed_home.merge(collapsed_work, on="ad_id", how="inner")
    paired["home_tract_geoid"] = paired["home_tract_geoid"].astype(str)
    paired["work_tract_geoid"] = paired["work_tract_geoid"].astype(str)
    paired["same_tract"] = paired["home_tract_geoid"] == paired["work_tract_geoid"]

    home_tract = _collapse_counts(paired, "home_tract_geoid", "teacher_home_devices")
    work_tract = _collapse_counts(paired, "work_tract_geoid", "teacher_work_devices")
    tract_od_ref = _od_counts(paired)
    county_od_ref = _county_od_counts(paired)

    collapsed_home.to_parquet(run_dir / "teacher_home_devices.parquet", index=False)
    collapsed_work.to_parquet(run_dir / "teacher_work_devices.parquet", index=False)
    paired.to_parquet(run_dir / "teacher_paired_devices.parquet", index=False)
    home_tract.to_csv(metrics_dir / "teacher_home_tract.csv", index=False)
    work_tract.to_csv(metrics_dir / "teacher_work_tract.csv", index=False)
    tract_od_ref.to_csv(run_dir / "teacher_tract_od.csv", index=False)
    county_od_ref.to_csv(metrics_dir / "teacher_county_od.csv", index=False)

    payload = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "areas_path": str(areas_path),
        "areas_group_col": group_col,
        "tract_od_path": str(tract_od_path),
        "teacher_home_path": str(teacher_home_path),
        "teacher_work_path": str(teacher_work_path),
        "bbox": {
            "min_lon": bbox[0],
            "min_lat": bbox[1],
            "max_lon": bbox[2],
            "max_lat": bbox[3],
        },
        "study": {
            "n_home_tracts": len(study_home_tracts),
            "n_work_tracts": len(study_work_tracts),
            "n_union_tracts": len(study_tracts),
        },
        "home_summary": home_summary,
        "work_summary": work_summary,
        "paired_summary": {
            "paired_devices": int(len(paired)),
            "same_tract_share": float(pd.to_numeric(paired["same_tract"], errors="coerce").fillna(False).mean()),
            "unique_home_tracts": int(paired["home_tract_geoid"].nunique()),
            "unique_work_tracts": int(paired["work_tract_geoid"].nunique()),
            "unique_tract_od_pairs": int(len(tract_od_ref)),
            "validated_days_mean": float(pd.to_numeric(paired["validated_days"], errors="coerce").mean()),
            "validated_days_median": float(pd.to_numeric(paired["validated_days"], errors="coerce").median()),
            "total_work_time_mean": float(pd.to_numeric(paired["total_work_time"], errors="coerce").mean()),
            "total_work_time_median": float(pd.to_numeric(paired["total_work_time"], errors="coerce").median()),
        },
        "artifacts": {
            "teacher_home_devices_parquet": str(run_dir / "teacher_home_devices.parquet"),
            "teacher_work_devices_parquet": str(run_dir / "teacher_work_devices.parquet"),
            "teacher_paired_devices_parquet": str(run_dir / "teacher_paired_devices.parquet"),
            "teacher_home_tract_csv": str(metrics_dir / "teacher_home_tract.csv"),
            "teacher_work_tract_csv": str(metrics_dir / "teacher_work_tract.csv"),
            "teacher_tract_od_csv": str(run_dir / "teacher_tract_od.csv"),
            "teacher_county_od_csv": str(metrics_dir / "teacher_county_od.csv"),
        },
    }
    _write_json(run_dir / "run_summary.json", payload)
    _write_json(metrics_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
