#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.data.lodes import (
    assign_job_center_membership,
    aggregate_lodes_to_tract_od,
    aggregate_lodes_wac_to_tract,
    build_tract_centroid_table,
    compute_gravity_accessibility,
    compute_job_center_accessibility,
    ensure_lodes_od_file,
    enrich_tract_od_with_geometry_and_wac,
    load_lodes_rac_or_wac,
    load_lodes_od,
    prepare_internal_study_tract_od,
)
from src.synthpop.paths import ensure_dir, project_root


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_geodata(path: pathlib.Path):
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("prepare_detroit_lodes_tract_od requires geopandas.") from e

    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def main() -> None:
    ap = argparse.ArgumentParser(prog="prepare_detroit_lodes_tract_od")
    ap.add_argument("--areas_path", required=True)
    ap.add_argument("--areas_group_col", default="tract_geoid")
    ap.add_argument("--study_persons_path", default="")
    ap.add_argument("--study_persons_group_col", default="tract_geoid")
    ap.add_argument("--state_postal", default="mi")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--raw_dir", default="")
    ap.add_argument("--main_path", default="")
    ap.add_argument("--aux_path", default="")
    ap.add_argument("--wac_path", default="")
    ap.add_argument("--accessibility_beta", type=float, default=0.1)
    ap.add_argument("--job_center_beta", type=float, default=0.12)
    ap.add_argument("--job_center_top_quantile", type=float, default=0.95)
    ap.add_argument("--job_center_min_centers", type=int, default=10)
    ap.add_argument("--job_center_min_centers_per_county", type=int, default=3)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="prepare_detroit_lodes_tract_od")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    areas_path = pathlib.Path(args.areas_path).expanduser().resolve()
    if not areas_path.exists():
        raise SystemExit(f"areas_path not found: {areas_path}")
    areas = _read_geodata(areas_path)
    group_col = str(args.areas_group_col)
    if group_col not in areas.columns:
        if "GEOID" in areas.columns:
            areas[group_col] = areas["GEOID"].astype(str)
        else:
            raise SystemExit(f"areas missing group column: {group_col}")
    if args.study_persons_path:
        persons_path = pathlib.Path(args.study_persons_path).expanduser().resolve()
        if not persons_path.exists():
            raise SystemExit(f"study_persons_path not found: {persons_path}")
        if persons_path.suffix.lower() in {".parquet", ".pq"}:
            study_df = pd.read_parquet(persons_path, columns=[str(args.study_persons_group_col)])
        else:
            study_df = pd.read_csv(persons_path, usecols=[str(args.study_persons_group_col)], low_memory=False)
        study_tracts = set(study_df[str(args.study_persons_group_col)].astype(str).unique().tolist())
    else:
        study_tracts = set(areas[group_col].astype(str).unique().tolist())

    raw_dir = pathlib.Path(args.raw_dir).expanduser().resolve() if args.raw_dir else (project_root() / "dataset" / "lodes")
    if args.main_path:
        main_path = pathlib.Path(args.main_path).expanduser().resolve()
    else:
        main_path = ensure_lodes_od_file(state_postal=args.state_postal, year=int(args.year), part="main", out_dir=raw_dir)
    if args.aux_path:
        aux_path = pathlib.Path(args.aux_path).expanduser().resolve()
    else:
        aux_path = ensure_lodes_od_file(state_postal=args.state_postal, year=int(args.year), part="aux", out_dir=raw_dir)

    od_block = load_lodes_od(main_path=main_path, aux_path=aux_path)
    tract_od = aggregate_lodes_to_tract_od(od_block)
    internal_od, origin_stats, summary = prepare_internal_study_tract_od(tract_od=tract_od, study_tracts=study_tracts)

    tract_centroids = build_tract_centroid_table(areas=areas, group_col=group_col)
    tract_centroids = tract_centroids[tract_centroids[group_col].isin(sorted(study_tracts))].copy()

    tract_wac = None
    wac_path = None
    if args.wac_path:
        wac_path = pathlib.Path(args.wac_path).expanduser().resolve()
    else:
        cand = raw_dir / f"{str(args.state_postal).strip().lower()}_wac_S000_JT00_{int(args.year)}.csv.gz"
        if cand.exists():
            wac_path = cand
        else:
            repo_cand = project_root() / "dataset" / "lodes" / cand.name
            if repo_cand.exists():
                wac_path = repo_cand
    if wac_path is not None and wac_path.exists():
        wac_block = load_lodes_rac_or_wac(
            path=wac_path,
            geocode_col="w_geocode",
            usecols=[
                "w_geocode",
                "C000",
                "CA01",
                "CA02",
                "CA03",
                "CE01",
                "CE02",
                "CE03",
                *[f"CNS{i:02d}" for i in range(1, 21)],
            ],
        )
        tract_wac = aggregate_lodes_wac_to_tract(wac_block)
        tract_wac = tract_wac[tract_wac["tract_geoid"].isin(sorted(study_tracts))].copy()
        if not tract_wac.empty and float(args.accessibility_beta) > 0.0:
            access = compute_gravity_accessibility(
                tract_centroids=tract_centroids.rename(columns={group_col: "tract_geoid"}),
                tract_mass=tract_wac,
                tract_col="tract_geoid",
                mass_col="C000",
                distance_beta=float(args.accessibility_beta),
                out_col="access_jobs_gravity",
            )
            tract_wac = tract_wac.merge(access, on="tract_geoid", how="left")
            tract_wac["access_jobs_gravity"] = pd.to_numeric(
                tract_wac["access_jobs_gravity"], errors="coerce"
            ).fillna(0.0)
        if not tract_wac.empty and float(args.job_center_beta) > 0.0:
            center_access = compute_job_center_accessibility(
                tract_centroids=tract_centroids.rename(columns={group_col: "tract_geoid"}),
                tract_mass=tract_wac,
                tract_col="tract_geoid",
                mass_col="C000",
                distance_beta=float(args.job_center_beta),
                top_quantile=float(args.job_center_top_quantile),
                min_centers=int(args.job_center_min_centers),
                out_col="access_job_centers_gravity",
            )
            tract_wac = tract_wac.merge(center_access, on="tract_geoid", how="left")
            tract_wac["access_job_centers_gravity"] = pd.to_numeric(
                tract_wac["access_job_centers_gravity"], errors="coerce"
            ).fillna(0.0)
        if not tract_wac.empty:
            center_membership = assign_job_center_membership(
                tract_centroids=tract_centroids.rename(columns={group_col: "tract_geoid"}),
                tract_mass=tract_wac,
                tract_col="tract_geoid",
                mass_col="C000",
                county_col="county_geoid",
                top_quantile=float(args.job_center_top_quantile),
                min_centers_per_county=int(args.job_center_min_centers_per_county),
            )
            tract_wac = tract_wac.merge(center_membership, on="tract_geoid", how="left")
            tract_wac["center_geoid"] = tract_wac["center_geoid"].fillna(tract_wac["tract_geoid"]).astype(str)
            tract_wac["center_county_geoid"] = tract_wac["center_county_geoid"].fillna(
                tract_wac["tract_geoid"].astype(str).str.slice(0, 5)
            ).astype(str)
            tract_wac["center_distance_km"] = pd.to_numeric(
                tract_wac["center_distance_km"], errors="coerce"
            ).fillna(0.0)
            tract_wac["center_mass"] = pd.to_numeric(
                tract_wac["center_mass"], errors="coerce"
            ).fillna(0.0)

    internal_od = enrich_tract_od_with_geometry_and_wac(
        tract_od=internal_od,
        tract_centroids=tract_centroids.rename(columns={group_col: "tract_geoid"}),
        tract_wac=tract_wac,
    )

    internal_od.to_csv(run_dir / "tract_od.csv", index=False)
    origin_stats.to_csv(metrics_dir / "origin_stats.csv", index=False)
    tract_centroids.to_csv(metrics_dir / "tract_centroids.csv", index=False)
    if tract_wac is not None:
        tract_wac.to_csv(metrics_dir / "tract_wac.csv", index=False)
    payload = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "areas_path": str(areas_path),
        "areas_group_col": group_col,
        "study_persons_path": (str(pathlib.Path(args.study_persons_path).expanduser().resolve()) if args.study_persons_path else None),
        "study_persons_group_col": str(args.study_persons_group_col),
        "state_postal": str(args.state_postal),
        "year": int(args.year),
        "main_path": str(main_path),
        "aux_path": str(aux_path),
        "wac_path": (str(wac_path) if wac_path is not None and pathlib.Path(wac_path).exists() else None),
        "accessibility_beta": float(args.accessibility_beta),
        "job_center_beta": float(args.job_center_beta),
        "job_center_top_quantile": float(args.job_center_top_quantile),
        "job_center_min_centers": int(args.job_center_min_centers),
        "job_center_min_centers_per_county": int(args.job_center_min_centers_per_county),
        "summary": summary,
        "enrichment": {
            "has_distance_km": bool("distance_km" in internal_od.columns),
            "has_wac_features": bool(tract_wac is not None and not tract_wac.empty),
            "has_accessibility_feature": bool("work_access_jobs_gravity" in internal_od.columns),
            "has_job_center_feature": bool("work_access_job_centers_gravity" in internal_od.columns),
            "has_center_membership": bool("work_center_geoid" in internal_od.columns),
            "tract_od_columns": internal_od.columns.astype(str).tolist(),
        },
    }
    _write_json(run_dir / "run_summary.json", payload)
    _write_json(metrics_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
