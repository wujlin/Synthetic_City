#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import ensure_dir, project_root
from src.synthpop.spatial.road_location_allocation import (
    assign_home_work_locations,
    build_road_location_candidates,
)


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _read_persons(path: pathlib.Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise SystemExit(f"unsupported persons file format: {path}")


def _read_geodata(path: pathlib.Path) -> Any:
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("exp_phase3_road_locations requires geopandas.") from e

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    if suffix == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _points_to_frame(gdf: Any) -> pd.DataFrame:
    if gdf is None:
        return pd.DataFrame()
    if int(getattr(gdf, "shape", [0])[0]) == 0:
        out = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
        out["x"] = pd.Series(dtype=float)
        out["y"] = pd.Series(dtype=float)
        out["wkt"] = pd.Series(dtype=str)
        return out
    out = gdf.copy()
    out["x"] = out.geometry.x
    out["y"] = out.geometry.y
    out["wkt"] = out.geometry.to_wkt()
    return pd.DataFrame(out.drop(columns=["geometry"]))


def _persons_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["home_geometry", "work_geometry"]:
        if col in out.columns:
            out[f"{col}_wkt"] = [
                (geom.wkt if geom is not None and not getattr(geom, "is_empty", True) else None)
                for geom in out[col].tolist()
            ]
            out = out.drop(columns=[col])
    return out


def _candidate_group_counts(cands: pd.DataFrame, *, group_col: str, role: str) -> pd.DataFrame:
    if cands.empty:
        return pd.DataFrame(columns=[group_col, "candidate_role", "source_stage", "n_candidates"])
    out = (
        cands.groupby([str(group_col), "candidate_role", "source_stage"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "n_candidates"})
    )
    out["candidate_role"] = str(role)
    return out


def _group_diagnostics(
    *,
    assigned: pd.DataFrame,
    home_frame: pd.DataFrame,
    work_frame: pd.DataFrame,
    group_col: str,
    work_group_col: str,
) -> pd.DataFrame:
    tmp = assigned.copy()
    tmp[str(group_col)] = tmp[str(group_col)].astype(str)
    tmp[str(work_group_col)] = tmp[str(work_group_col)].astype(str)
    home_tmp = home_frame.copy()
    work_tmp = work_frame.copy()
    if str(group_col) in home_tmp.columns:
        home_tmp[str(group_col)] = home_tmp[str(group_col)].astype(str)
    if str(group_col) in work_tmp.columns:
        work_tmp[str(group_col)] = work_tmp[str(group_col)].astype(str)

    groups = pd.DataFrame(
        {
            str(group_col): sorted(
                set(tmp[str(group_col)].unique().tolist()) | set(tmp[str(work_group_col)].unique().tolist())
            )
        }
    )
    tmp["_is_worker"] = tmp["work_assignment_mode"].astype(str) != "ineligible"
    tmp["_is_home_assigned"] = pd.notna(tmp["home_candidate_id"])
    tmp["_is_work_assigned"] = pd.notna(tmp["work_candidate_id"])
    tmp["_is_work_unassigned"] = tmp["_is_worker"] & (~tmp["_is_work_assigned"])

    home_person_counts = (
        tmp.groupby(str(group_col), as_index=False, sort=False)
        .agg(
            n_persons=("person_id", "size"),
            n_home_assigned=("_is_home_assigned", "sum"),
            n_home_unassigned=("_is_home_assigned", lambda s: int((~s.astype(bool)).sum())),
        )
    )
    work_person_counts = (
        tmp.groupby(str(work_group_col), as_index=False, sort=False)
        .agg(
            n_workers=("_is_worker", "sum"),
            n_work_assigned=("_is_work_assigned", "sum"),
            n_work_unassigned=("_is_work_unassigned", "sum"),
        )
        .rename(columns={str(work_group_col): str(group_col)})
    )

    def _role_counts(frame: pd.DataFrame, role: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=[str(group_col), f"n_{role}_candidates", f"{role}_source_stage"])
        counts = (
            frame.groupby(str(group_col), as_index=False, sort=False)
            .agg(
                **{f"n_{role}_candidates": ("candidate_id", "size")},
                **{f"{role}_source_stage": ("source_stage", lambda s: "|".join(sorted({str(v) for v in s.astype(str).tolist()})))},
            )
        )
        return counts

    home_counts = _role_counts(home_tmp, "home")
    work_counts = _role_counts(work_tmp, "work")

    out = groups.merge(home_person_counts, on=str(group_col), how="left")
    out = out.merge(work_person_counts, on=str(group_col), how="left")
    out = out.merge(home_counts, on=str(group_col), how="left")
    out = out.merge(work_counts, on=str(group_col), how="left")
    for col in ["n_persons", "n_home_assigned", "n_home_unassigned", "n_workers", "n_work_assigned", "n_work_unassigned", "n_home_candidates", "n_work_candidates"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    out["home_has_candidates"] = out["n_home_candidates"] > 0
    out["work_has_candidates"] = out["n_work_candidates"] > 0
    out["home_candidates_per_person"] = out["n_home_candidates"] / out["n_persons"].clip(lower=1)
    out["work_candidates_per_worker"] = out["n_work_candidates"] / out["n_workers"].clip(lower=1)
    out["work_assignment_pressure"] = out["n_workers"] / out["n_work_candidates"].clip(lower=1)
    out["home_assignment_pressure"] = out["n_persons"] / out["n_home_candidates"].clip(lower=1)
    return out.sort_values([str(group_col)], kind="stable").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase3_road_locations")
    ap.add_argument("--persons_path", required=True)
    ap.add_argument("--areas_path", required=True)
    ap.add_argument("--roads_path", required=True)
    ap.add_argument("--group_col", default="tract_geoid")
    ap.add_argument("--work_group_col", default="")
    ap.add_argument("--areas_group_col", default="")
    ap.add_argument("--person_id_col", default="person_id")
    ap.add_argument("--household_col", default="household_id")
    ap.add_argument("--work_eligible_col", default="")
    ap.add_argument("--work_eligible_values", default="")
    ap.add_argument("--road_mtfcc_col", default="MTFCC")
    ap.add_argument("--road_component_col", default="component")
    ap.add_argument("--home_mode", default="conservative")
    ap.add_argument("--work_mtfcc_values", default="S1100,S1200")
    ap.add_argument("--work_gap_exception_mtfcc_values", default="S1400")
    ap.add_argument("--allow_home_fallback", action="store_true")
    ap.add_argument("--allow_work_fallback", action="store_true")
    ap.add_argument("--legalization_fraction", type=float, default=1e-6)
    ap.add_argument("--home_interpolation_density", type=float, default=0.0005)
    ap.add_argument("--work_interpolation_density", type=float, default=0.0002)
    ap.add_argument("--dedupe_precision", type=int, default=6)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--parallel_chunksize", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="phase3_road_locations")
    args = ap.parse_args()

    persons_path = pathlib.Path(args.persons_path).expanduser().resolve()
    areas_path = pathlib.Path(args.areas_path).expanduser().resolve()
    roads_path = pathlib.Path(args.roads_path).expanduser().resolve()
    for p in [persons_path, areas_path, roads_path]:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    synthetic_dir = ensure_dir(run_dir / "synthetic")
    metrics_dir = ensure_dir(run_dir / "metrics")

    persons = _read_persons(persons_path)
    persons[str(args.group_col)] = persons[str(args.group_col)].astype(str)
    work_group_col = str(args.work_group_col).strip() or str(args.group_col)
    if work_group_col not in persons.columns:
        raise SystemExit(f"persons missing work_group_col: {work_group_col}")
    persons[work_group_col] = persons[work_group_col].astype(str)
    areas = _read_geodata(areas_path)
    roads = _read_geodata(roads_path)
    areas_group_col = str(args.areas_group_col).strip() or str(args.group_col)
    if areas_group_col not in areas.columns:
        raise SystemExit(f"areas missing group_col: {areas_group_col}")
    if areas_group_col != str(args.group_col):
        areas = areas.rename(columns={areas_group_col: str(args.group_col)}).copy()
    person_groups = set(persons[str(args.group_col)].astype(str).tolist()) | set(persons[work_group_col].astype(str).tolist())
    areas[str(args.group_col)] = areas[str(args.group_col)].astype(str)
    areas = areas[areas[str(args.group_col)].isin(sorted(person_groups))].copy()

    t_build0 = time.perf_counter()
    home_candidates, work_candidates, candidate_meta = build_road_location_candidates(
        areas=areas,
        roads=roads,
        group_col=str(args.group_col),
        road_mtfcc_col=str(args.road_mtfcc_col),
        road_component_col=(str(args.road_component_col) if args.road_component_col else None),
        home_mode=str(args.home_mode),
        work_mtfcc_values=_parse_csv_list(args.work_mtfcc_values),
        work_gap_exception_mtfcc_values=_parse_csv_list(args.work_gap_exception_mtfcc_values),
        allow_home_fallback=bool(args.allow_home_fallback),
        allow_work_fallback=bool(args.allow_work_fallback),
        legalization_fraction=float(args.legalization_fraction),
        home_interpolation_density=float(args.home_interpolation_density),
        work_interpolation_density=float(args.work_interpolation_density),
        dedupe_precision=int(args.dedupe_precision),
        n_jobs=int(args.n_jobs),
        parallel_chunksize=int(args.parallel_chunksize),
    )
    candidate_build_seconds = float(time.perf_counter() - t_build0)

    t_assign0 = time.perf_counter()
    assigned, assignment_meta = assign_home_work_locations(
        persons=persons,
        home_candidates=home_candidates,
        work_candidates=work_candidates,
        group_col=str(args.group_col),
        work_group_col=work_group_col,
        person_id_col=str(args.person_id_col),
        household_col=(str(args.household_col) if args.household_col else None),
        work_eligible_col=(str(args.work_eligible_col) if args.work_eligible_col else None),
        work_eligible_values=_parse_csv_list(args.work_eligible_values),
        seed=int(args.seed),
    )
    assignment_seconds = float(time.perf_counter() - t_assign0)

    home_frame = _points_to_frame(home_candidates)
    work_frame = _points_to_frame(work_candidates)
    person_frame = _persons_to_frame(assigned)
    group_counts = pd.concat(
        [
            _candidate_group_counts(home_frame, group_col=str(args.group_col), role="home"),
            _candidate_group_counts(work_frame, group_col=str(args.group_col), role="work"),
        ],
        axis=0,
        ignore_index=True,
    )
    group_diag = _group_diagnostics(
        assigned=assigned,
        home_frame=home_frame,
        work_frame=work_frame,
        group_col=str(args.group_col),
        work_group_col=work_group_col,
    )
    no_candidate_groups = group_diag[
        (~group_diag["home_has_candidates"]) | (~group_diag["work_has_candidates"]) | (group_diag["n_work_unassigned"] > 0)
    ].copy()

    home_csv = synthetic_dir / "home_candidates.csv"
    work_csv = synthetic_dir / "work_candidates.csv"
    persons_csv = synthetic_dir / "person_locations.csv"
    group_csv = metrics_dir / "candidate_group_counts.csv"
    group_diag_csv = metrics_dir / "group_diagnostics.csv"
    no_candidate_csv = metrics_dir / "no_candidate_groups.csv"
    home_frame.to_csv(home_csv, index=False)
    work_frame.to_csv(work_csv, index=False)
    person_frame.to_csv(persons_csv, index=False)
    group_counts.to_csv(group_csv, index=False)
    group_diag.to_csv(group_diag_csv, index=False)
    no_candidate_groups.to_csv(no_candidate_csv, index=False)

    n_person_groups = int(persons[str(args.group_col)].astype(str).nunique())
    area_groups = set(areas[str(args.group_col)].astype(str).tolist())
    summary = {
        "group_col": str(args.group_col),
        "areas_group_col": areas_group_col,
        "person_id_col": str(args.person_id_col),
        "household_col": (str(args.household_col) if args.household_col else None),
        "work_eligible_col": (str(args.work_eligible_col) if args.work_eligible_col else None),
        "home_mode": str(args.home_mode),
        "work_mtfcc_values": _parse_csv_list(args.work_mtfcc_values),
        "work_gap_exception_mtfcc_values": _parse_csv_list(args.work_gap_exception_mtfcc_values),
        "allow_home_fallback": bool(args.allow_home_fallback),
        "allow_work_fallback": bool(args.allow_work_fallback),
        "legalization_fraction": float(args.legalization_fraction),
        "n_jobs": int(args.n_jobs),
        "parallel_chunksize": int(args.parallel_chunksize),
        "input_paths": {
            "persons_path": str(persons_path),
            "areas_path": str(areas_path),
            "roads_path": str(roads_path),
        },
        "candidate_meta": candidate_meta,
        "assignment_meta": assignment_meta,
        "timing_seconds": {
            "candidate_build": candidate_build_seconds,
            "assignment": assignment_seconds,
            "total_core": float(candidate_build_seconds + assignment_seconds),
        },
        "coverage": {
            "n_person_groups": n_person_groups,
            "n_area_groups": int(len(area_groups)),
            "person_groups_missing_area": int(len(person_groups - area_groups)),
            "area_groups_without_persons": int(len(area_groups - person_groups)),
        },
        "artifacts": {
            "home_candidates_csv": str(home_csv),
            "work_candidates_csv": str(work_csv),
            "person_locations_csv": str(persons_csv),
            "candidate_group_counts_csv": str(group_csv),
            "group_diagnostics_csv": str(group_diag_csv),
            "no_candidate_groups_csv": str(no_candidate_csv),
        },
    }

    run_summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "summary_json": str(metrics_dir / "summary.json"),
        "seed": int(args.seed),
        "home_mode": str(args.home_mode),
        "work_mtfcc_values": _parse_csv_list(args.work_mtfcc_values),
        "work_gap_exception_mtfcc_values": _parse_csv_list(args.work_gap_exception_mtfcc_values),
        "allow_home_fallback": bool(args.allow_home_fallback),
        "allow_work_fallback": bool(args.allow_work_fallback),
        "legalization_fraction": float(args.legalization_fraction),
        "n_jobs": int(args.n_jobs),
        "parallel_chunksize": int(args.parallel_chunksize),
        "home_interpolation_density": float(args.home_interpolation_density),
        "work_interpolation_density": float(args.work_interpolation_density),
        "candidate_meta": candidate_meta,
        "assignment_meta": assignment_meta,
        "timing_seconds": {
            "candidate_build": candidate_build_seconds,
            "assignment": assignment_seconds,
            "total_core": float(candidate_build_seconds + assignment_seconds),
        },
    }
    _write_json(metrics_dir / "summary.json", summary)
    _write_json(run_dir / "run_summary.json", run_summary)
    print(f"[ok] wrote run summary: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
