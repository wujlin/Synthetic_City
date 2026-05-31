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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synthpop.paths import ensure_dir, project_root
from synthpop.spatial.road_location_allocation import (
    _build_candidates_for_area,
    _home_codes,
    _resolve_work_eligible_mask,
    _road_candidate_state,
    _stage_uses_fallback,
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


def _normalize_geoid_text(s: pd.Series, *, width: int = 11) -> pd.Series:
    out = s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    missing = out.isna() | out.str.lower().isin({"", "nan", "none", "<na>"})
    out = out.str.zfill(int(width))
    out[missing] = pd.NA
    return out


def _load_asset_inventory(path: pathlib.Path) -> pd.DataFrame:
    inv = pd.read_csv(path, dtype={"statefp": str}, low_memory=False)
    inv["statefp"] = inv["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    if "status" in inv.columns:
        inv = inv[inv["status"].astype(str).str.lower() == "ready"].copy()
    return inv


def _append_cross_state_support(
    *,
    areas: Any,
    roads: Any,
    persons: pd.DataFrame,
    group_col: str,
    work_group_col: str,
    areas_group_col: str,
    asset_inventory_csv: pathlib.Path,
) -> tuple[Any, Any, dict[str, Any]]:
    """Append destination-state tract/road support for cross-state work tracts."""
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("cross-state road support requires geopandas.") from e

    inv = _load_asset_inventory(asset_inventory_csv)
    inv_by_state = {str(r["statefp"]).zfill(2): r for r in inv.to_dict("records")}

    home_states = set(_normalize_geoid_text(persons[str(group_col)], width=11).str.slice(0, 2).dropna().tolist())
    work_series = _normalize_geoid_text(persons[str(work_group_col)], width=11)
    work_states = set(work_series.dropna().str.slice(0, 2).tolist())
    extra_states = sorted(s for s in work_states - home_states if s and s.lower() != "na")
    if not extra_states:
        return areas, roads, {
            "enabled": True,
            "asset_inventory_csv": str(asset_inventory_csv),
            "home_states": sorted(home_states),
            "work_states": sorted(work_states),
            "extra_statefps_loaded": [],
            "missing_statefps": [],
        }

    needed_groups = set(work_series[work_series.str.slice(0, 2).isin(extra_states)].tolist())
    extra_area_frames = []
    extra_road_frames = []
    missing = []
    for statefp in extra_states:
        row = inv_by_state.get(statefp)
        if row is None:
            missing.append(f"{statefp}:inventory")
            continue
        tract_path = pathlib.Path(str(row.get("tract_zip", ""))).expanduser()
        roads_path = pathlib.Path(str(row.get("roads_path", ""))).expanduser()
        if not tract_path.exists():
            missing.append(f"{statefp}:tract_zip")
            continue
        if not roads_path.exists():
            missing.append(f"{statefp}:roads_path")
            continue
        state_areas = _read_geodata(tract_path)
        if areas_group_col not in state_areas.columns:
            if "GEOID" in state_areas.columns:
                state_areas = state_areas.rename(columns={"GEOID": str(group_col)}).copy()
            else:
                missing.append(f"{statefp}:areas_group_col")
                continue
        elif areas_group_col != str(group_col):
            state_areas = state_areas.rename(columns={areas_group_col: str(group_col)}).copy()
        state_areas[str(group_col)] = _normalize_geoid_text(state_areas[str(group_col)], width=11)
        state_areas = state_areas[state_areas[str(group_col)].isin(needed_groups)].copy()
        if state_areas.empty:
            continue
        if getattr(state_areas, "crs", None) is not None and getattr(areas, "crs", None) is not None and state_areas.crs != areas.crs:
            state_areas = state_areas.to_crs(areas.crs)
        state_roads = _read_geodata(roads_path)
        if getattr(state_roads, "crs", None) is not None and getattr(roads, "crs", None) is not None and state_roads.crs != roads.crs:
            state_roads = state_roads.to_crs(roads.crs)
        extra_area_frames.append(state_areas)
        extra_road_frames.append(state_roads)

    if extra_area_frames:
        areas = gpd.GeoDataFrame(
            pd.concat([areas, *extra_area_frames], ignore_index=True),
            geometry="geometry",
            crs=getattr(areas, "crs", None),
        )
    if extra_road_frames:
        roads = gpd.GeoDataFrame(
            pd.concat([roads, *extra_road_frames], ignore_index=True),
            geometry="geometry",
            crs=getattr(roads, "crs", None),
        )

    return areas, roads, {
        "enabled": True,
        "asset_inventory_csv": str(asset_inventory_csv),
        "home_states": sorted(home_states),
        "work_states": sorted(work_states),
        "extra_statefps_requested": extra_states,
        "extra_statefps_loaded": sorted({str(g[str(group_col)].iloc[0])[:2] for g in extra_area_frames if not g.empty}),
        "missing_statefps": missing,
        "n_extra_area_rows": int(sum(int(g.shape[0]) for g in extra_area_frames)),
        "n_extra_road_rows": int(sum(int(g.shape[0]) for g in extra_road_frames)),
    }


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


def _assign_locations_low_memory(
    *,
    persons: pd.DataFrame,
    areas: Any,
    roads: Any,
    group_col: str,
    work_group_col: str,
    person_id_col: str,
    household_col: str | None,
    work_eligible_col: str | None,
    work_eligible_values: list[str],
    road_mtfcc_col: str,
    road_component_col: str | None,
    home_mode: str,
    work_mtfcc_values: list[str],
    work_gap_exception_mtfcc_values: list[str],
    allow_home_fallback: bool,
    allow_work_fallback: bool,
    legalization_fraction: float,
    home_interpolation_density: float,
    work_interpolation_density: float,
    dedupe_precision: int,
    seed: int,
) -> dict[str, Any]:
    """Assign locations without materializing statewide candidate point tables."""
    import numpy as np

    if not isinstance(persons, pd.DataFrame):
        raise TypeError("persons must be a pandas DataFrame")

    t0 = time.perf_counter()
    out = persons.copy().reset_index(drop=True)
    out[str(group_col)] = out[str(group_col)].astype(str)
    out[str(work_group_col)] = out[str(work_group_col)].astype(str)
    n = int(out.shape[0])
    rng = np.random.default_rng(int(seed))

    eligible = _resolve_work_eligible_mask(
        persons=out,
        work_eligible_col=(str(work_eligible_col) if work_eligible_col else None),
        work_eligible_values=work_eligible_values,
    ).to_numpy(dtype=bool)

    home_candidate_id = np.full(n, None, dtype=object)
    home_x = np.full(n, np.nan, dtype=float)
    home_y = np.full(n, np.nan, dtype=float)
    home_stage = np.full(n, None, dtype=object)
    home_mode_out = np.full(n, None, dtype=object)
    home_fallback = np.full(n, False, dtype=bool)

    work_candidate_id = np.full(n, None, dtype=object)
    work_x = np.full(n, np.nan, dtype=float)
    work_y = np.full(n, np.nan, dtype=float)
    work_stage = np.full(n, None, dtype=object)
    work_mode_out = np.full(n, None, dtype=object)
    work_fallback = np.full(n, False, dtype=bool)
    work_mode_out[~eligible] = "ineligible"

    home_groups = {str(g): np.asarray(idx, dtype=int) for g, idx in out.groupby(str(group_col), sort=False).groups.items()}
    work_groups = {str(g): np.asarray(idx, dtype=int) for g, idx in out.groupby(str(work_group_col), sort=False).groups.items()}
    person_groups = set(home_groups) | set(work_groups)

    road_sindex = None
    try:
        road_sindex = roads.sindex
    except Exception:
        road_sindex = None
    state = _road_candidate_state(
        road_g=roads,
        road_mtfcc_col=str(road_mtfcc_col),
        road_component_col=(str(road_component_col) if road_component_col else None),
        road_sindex=road_sindex,
        home_primary=_home_codes(str(home_mode)),
        home_compat=["S1400", "S1740"],
        work_mtfcc=[str(v) for v in work_mtfcc_values],
        work_gap_exception_mtfcc=[str(v) for v in work_gap_exception_mtfcc_values],
        allow_home_fallback=bool(allow_home_fallback),
        allow_work_fallback=bool(allow_work_fallback),
        legalization_fraction=float(legalization_fraction),
        home_interpolation_density=float(home_interpolation_density),
        work_interpolation_density=float(work_interpolation_density),
        dedupe_precision=int(dedupe_precision),
    )

    candidate_rows: list[dict[str, Any]] = []
    home_stage_counts: dict[str, int] = {}
    work_stage_counts: dict[str, int] = {}
    home_geometry_meta = {"input_points": 0, "kept_points": 0, "legalized_points": 0, "dropped_points": 0}
    work_geometry_meta = {"input_points": 0, "kept_points": 0, "legalized_points": 0, "dropped_points": 0}
    processed_groups: set[str] = set()
    candidate_build_seconds = 0.0
    assignment_seconds = 0.0

    use_household = bool(household_col) and str(household_col) in out.columns and out[str(household_col)].notna().any()

    def _accumulate_meta(total: dict[str, int], inc: dict[str, int]) -> None:
        for k, v in inc.items():
            total[k] = int(total.get(k, 0) + int(v))

    def _candidate_xy(points: list[dict[str, Any]], chosen: np.ndarray) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for i in chosen.tolist():
            pt = points[int(i)]["geometry"]
            xs.append(float(pt.x))
            ys.append(float(pt.y))
        return xs, ys

    def _assign_home_group(g: str, points: list[dict[str, Any]], stage: str) -> None:
        nonlocal assignment_seconds
        idx = home_groups.get(str(g))
        if idx is None or int(idx.shape[0]) == 0:
            return
        t = time.perf_counter()
        if not points:
            home_stage[idx] = "no_candidates"
            home_mode_out[idx] = "unassigned_no_candidates"
            assignment_seconds += float(time.perf_counter() - t)
            return
        fallback_value = _stage_uses_fallback(stage=str(stage), allowed_non_primary_stages=set())

        def assign_rows(rows: np.ndarray, mode: str) -> None:
            if int(rows.shape[0]) == 0:
                return
            chosen = rng.integers(0, int(len(points)), size=int(rows.shape[0]))
            xs, ys = _candidate_xy(points, chosen)
            home_candidate_id[rows] = [f"{g}:home:{int(i):06d}" for i in chosen.tolist()]
            home_x[rows] = xs
            home_y[rows] = ys
            home_stage[rows] = str(stage)
            home_mode_out[rows] = str(mode)
            home_fallback[rows] = bool(fallback_value)

        if use_household:
            hh_values = out.loc[idx, str(household_col)].to_numpy(dtype=object)
            has_hh = pd.notna(hh_values)
            if bool(has_hh.any()):
                hh_df = pd.DataFrame(
                    {
                        "row": idx[has_hh],
                        "household": [str(v) for v in hh_values[has_hh].tolist()],
                    }
                )
                unique_hh = hh_df["household"].drop_duplicates().tolist()
                chosen = rng.integers(0, int(len(points)), size=int(len(unique_hh)))
                hh_choice = {hh: int(ci) for hh, ci in zip(unique_hh, chosen.tolist())}
                rows = hh_df["row"].to_numpy(dtype=int)
                picked = np.asarray([hh_choice[str(hh)] for hh in hh_df["household"].tolist()], dtype=int)
                xs, ys = _candidate_xy(points, picked)
                home_candidate_id[rows] = [f"{g}:home:{int(i):06d}" for i in picked.tolist()]
                home_x[rows] = xs
                home_y[rows] = ys
                home_stage[rows] = str(stage)
                home_mode_out[rows] = "household"
                home_fallback[rows] = bool(fallback_value)
            assign_rows(idx[~has_hh], "person_proxy")
        else:
            assign_rows(idx, "person_proxy")
        assignment_seconds += float(time.perf_counter() - t)

    def _assign_work_group(g: str, points: list[dict[str, Any]], stage: str) -> None:
        nonlocal assignment_seconds
        idx = work_groups.get(str(g))
        if idx is None or int(idx.shape[0]) == 0:
            return
        idx = idx[eligible[idx]]
        if int(idx.shape[0]) == 0:
            return
        t = time.perf_counter()
        if not points:
            work_stage[idx] = "no_candidates"
            work_mode_out[idx] = "unassigned_no_candidates"
            assignment_seconds += float(time.perf_counter() - t)
            return
        allowed = {"arterial_missing_exception"} if str(stage) == "arterial_missing_exception" else set()
        fallback_value = _stage_uses_fallback(stage=str(stage), allowed_non_primary_stages=allowed)
        chosen = rng.integers(0, int(len(points)), size=int(idx.shape[0]))
        xs, ys = _candidate_xy(points, chosen)
        work_candidate_id[idx] = [f"{g}:work:{int(i):06d}" for i in chosen.tolist()]
        work_x[idx] = xs
        work_y[idx] = ys
        work_stage[idx] = str(stage)
        work_mode_out[idx] = "worker"
        work_fallback[idx] = bool(fallback_value)
        assignment_seconds += float(time.perf_counter() - t)

    area_records = areas[[str(group_col), "geometry"]].dropna(subset=["geometry"]).to_dict(orient="records")
    for area in area_records:
        g = str(area[str(group_col)])
        if g not in person_groups:
            continue
        processed_groups.add(g)
        t = time.perf_counter()
        item = _build_candidates_for_area(area=area, group_col=str(group_col), state=state)
        candidate_build_seconds += float(time.perf_counter() - t)

        home_group = item["home_group"]
        work_group = item["work_group"]
        home_points = list(home_group["points"])
        work_points = list(work_group["points"])
        home_stage_value = str(home_group["source_stage"])
        work_stage_value = str(work_group["source_stage"])
        home_stage_counts[home_stage_value] = int(home_stage_counts.get(home_stage_value, 0) + 1)
        work_stage_counts[work_stage_value] = int(work_stage_counts.get(work_stage_value, 0) + 1)
        _accumulate_meta(home_geometry_meta, item["home_geom_meta"])
        _accumulate_meta(work_geometry_meta, item["work_geom_meta"])
        candidate_rows.append({str(group_col): g, "candidate_role": "home", "source_stage": home_stage_value, "n_candidates": int(len(home_points))})
        candidate_rows.append({str(group_col): g, "candidate_role": "work", "source_stage": work_stage_value, "n_candidates": int(len(work_points))})
        _assign_home_group(g, home_points, home_stage_value)
        _assign_work_group(g, work_points, work_stage_value)

    for g in sorted(person_groups - processed_groups):
        candidate_rows.append({str(group_col): g, "candidate_role": "home", "source_stage": "no_area", "n_candidates": 0})
        candidate_rows.append({str(group_col): g, "candidate_role": "work", "source_stage": "no_area", "n_candidates": 0})
        _assign_home_group(g, [], "no_area")
        _assign_work_group(g, [], "no_area")

    out["home_candidate_id"] = home_candidate_id.tolist()
    out["home_source_stage"] = home_stage.tolist()
    out["home_assignment_mode"] = home_mode_out.tolist()
    out["home_fallback_flag"] = home_fallback.tolist()
    out["work_candidate_id"] = work_candidate_id.tolist()
    out["work_source_stage"] = work_stage.tolist()
    out["work_assignment_mode"] = work_mode_out.tolist()
    out["work_fallback_flag"] = work_fallback.tolist()
    out["home_x"] = home_x
    out["home_y"] = home_y
    out["work_x"] = work_x
    out["work_y"] = work_y

    group_counts = pd.DataFrame(candidate_rows)
    if group_counts.empty:
        group_counts = pd.DataFrame(columns=[str(group_col), "candidate_role", "source_stage", "n_candidates"])

    tmp = out[[str(group_col), str(work_group_col), str(person_id_col)]].copy()
    tmp["_is_worker"] = eligible
    tmp["_is_home_assigned"] = pd.notna(home_candidate_id)
    tmp["_is_work_assigned"] = pd.notna(work_candidate_id)
    tmp["_is_work_unassigned"] = tmp["_is_worker"] & (~tmp["_is_work_assigned"])
    home_person_counts = (
        tmp.groupby(str(group_col), as_index=False, sort=False)
        .agg(
            n_persons=(str(person_id_col), "size"),
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
    groups = pd.DataFrame({str(group_col): sorted(person_groups)})

    def _role_counts(role: str) -> pd.DataFrame:
        sub = group_counts[group_counts["candidate_role"].astype(str) == role].copy()
        if sub.empty:
            return pd.DataFrame(columns=[str(group_col), f"n_{role}_candidates", f"{role}_source_stage"])
        return sub.rename(
            columns={
                "n_candidates": f"n_{role}_candidates",
                "source_stage": f"{role}_source_stage",
            }
        )[[str(group_col), f"n_{role}_candidates", f"{role}_source_stage"]]

    group_diag = groups.merge(home_person_counts, on=str(group_col), how="left")
    group_diag = group_diag.merge(work_person_counts, on=str(group_col), how="left")
    group_diag = group_diag.merge(_role_counts("home"), on=str(group_col), how="left")
    group_diag = group_diag.merge(_role_counts("work"), on=str(group_col), how="left")
    for col in ["n_persons", "n_home_assigned", "n_home_unassigned", "n_workers", "n_work_assigned", "n_work_unassigned", "n_home_candidates", "n_work_candidates"]:
        if col in group_diag.columns:
            group_diag[col] = group_diag[col].fillna(0).astype(int)
    group_diag["home_has_candidates"] = group_diag["n_home_candidates"] > 0
    group_diag["work_has_candidates"] = group_diag["n_work_candidates"] > 0
    group_diag["home_candidates_per_person"] = group_diag["n_home_candidates"] / group_diag["n_persons"].clip(lower=1)
    group_diag["work_candidates_per_worker"] = group_diag["n_work_candidates"] / group_diag["n_workers"].clip(lower=1)
    group_diag["work_assignment_pressure"] = group_diag["n_workers"] / group_diag["n_work_candidates"].clip(lower=1)
    group_diag["home_assignment_pressure"] = group_diag["n_persons"] / group_diag["n_home_candidates"].clip(lower=1)
    no_candidate_groups = group_diag[
        (~group_diag["home_has_candidates"]) | (~group_diag["work_has_candidates"]) | (group_diag["n_work_unassigned"] > 0)
    ].copy()

    candidate_meta = {
        "group_col": str(group_col),
        "home_mode": str(home_mode),
        "work_mtfcc_values": [str(v) for v in work_mtfcc_values],
        "work_gap_exception_mtfcc_values": [str(v) for v in work_gap_exception_mtfcc_values],
        "allow_home_fallback": bool(allow_home_fallback),
        "allow_work_fallback": bool(allow_work_fallback),
        "home_allowed_non_primary_stages": [],
        "work_allowed_non_primary_stages": ["arterial_missing_exception"] if work_gap_exception_mtfcc_values else [],
        "legalization_fraction": float(legalization_fraction),
        "home_interpolation_density": float(home_interpolation_density),
        "work_interpolation_density": float(work_interpolation_density),
        "n_jobs": 1,
        "parallel_chunksize": 1,
        "parallel_used": False,
        "low_memory": True,
        "n_groups": int(len(processed_groups)),
        "n_home_candidates": int(group_counts.loc[group_counts["candidate_role"] == "home", "n_candidates"].sum()),
        "n_work_candidates": int(group_counts.loc[group_counts["candidate_role"] == "work", "n_candidates"].sum()),
        "home_stage_counts": home_stage_counts,
        "work_stage_counts": work_stage_counts,
        "home_geometry_meta": home_geometry_meta,
        "work_geometry_meta": work_geometry_meta,
    }
    assignment_meta = {
        "group_col": str(group_col),
        "work_group_col": str(work_group_col),
        "seed": int(seed),
        "home_assignment_mode": ("household" if use_household else "person_proxy"),
        "n_persons": int(n),
        "home_assigned": int(pd.notna(home_candidate_id).sum()),
        "home_unassigned": int(pd.isna(home_candidate_id).sum()),
        "home_fallback_assignments": int(pd.Series(home_fallback).sum()),
        "work_eligible": int(eligible.sum()),
        "work_assigned": int(pd.notna(work_candidate_id).sum()),
        "work_unassigned": int(eligible.sum() - pd.notna(work_candidate_id).sum()),
        "work_fallback_assignments": int(pd.Series(work_fallback).sum()),
        "low_memory": True,
    }
    return {
        "assigned": out,
        "candidate_meta": candidate_meta,
        "assignment_meta": assignment_meta,
        "candidate_group_counts": group_counts,
        "group_diagnostics": group_diag.sort_values([str(group_col)], kind="stable").reset_index(drop=True),
        "no_candidate_groups": no_candidate_groups.sort_values([str(group_col)], kind="stable").reset_index(drop=True),
        "timing_seconds": {
            "candidate_build": float(candidate_build_seconds),
            "assignment": float(assignment_seconds),
            "total_core": float(time.perf_counter() - t0),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase3_road_locations")
    ap.add_argument("--persons_path", required=True)
    ap.add_argument("--areas_path", required=True)
    ap.add_argument("--roads_path", required=True)
    ap.add_argument("--cross_state_asset_inventory_csv", default="")
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
    ap.add_argument(
        "--low_memory",
        action="store_true",
        help=(
            "Build and assign road candidates tract-by-tract. This avoids materializing "
            "statewide home/work candidate point tables and is intended for large states."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="phase3_road_locations")
    args = ap.parse_args()

    persons_path = pathlib.Path(args.persons_path).expanduser().resolve()
    areas_path = pathlib.Path(args.areas_path).expanduser().resolve()
    roads_path = pathlib.Path(args.roads_path).expanduser().resolve()
    cross_state_asset_inventory_csv = pathlib.Path(args.cross_state_asset_inventory_csv).expanduser().resolve() if str(args.cross_state_asset_inventory_csv).strip() else None
    for p in [persons_path, areas_path, roads_path]:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")
    if cross_state_asset_inventory_csv is not None and not cross_state_asset_inventory_csv.exists():
        raise SystemExit(f"cross_state_asset_inventory_csv not found: {cross_state_asset_inventory_csv}")

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
    persons[str(args.group_col)] = _normalize_geoid_text(persons[str(args.group_col)], width=11)
    persons[work_group_col] = _normalize_geoid_text(persons[work_group_col], width=11)
    cross_state_support_meta = {"enabled": False}
    if cross_state_asset_inventory_csv is not None:
        areas, roads, cross_state_support_meta = _append_cross_state_support(
            areas=areas,
            roads=roads,
            persons=persons,
            group_col=str(args.group_col),
            work_group_col=work_group_col,
            areas_group_col=areas_group_col,
            asset_inventory_csv=cross_state_asset_inventory_csv,
        )
    person_groups = set(persons[str(args.group_col)].dropna().astype(str).tolist()) | set(persons[work_group_col].dropna().astype(str).tolist())
    areas[str(args.group_col)] = _normalize_geoid_text(areas[str(args.group_col)], width=11)
    areas = areas[areas[str(args.group_col)].isin(sorted(person_groups))].copy()

    if bool(args.low_memory):
        low = _assign_locations_low_memory(
            persons=persons,
            areas=areas,
            roads=roads,
            group_col=str(args.group_col),
            work_group_col=work_group_col,
            person_id_col=str(args.person_id_col),
            household_col=(str(args.household_col) if args.household_col else None),
            work_eligible_col=(str(args.work_eligible_col) if args.work_eligible_col else None),
            work_eligible_values=_parse_csv_list(args.work_eligible_values),
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
            seed=int(args.seed),
        )
        assigned = low["assigned"]
        candidate_meta = low["candidate_meta"]
        assignment_meta = low["assignment_meta"]
        group_counts = low["candidate_group_counts"]
        group_diag = low["group_diagnostics"]
        no_candidate_groups = low["no_candidate_groups"]
        candidate_build_seconds = float(low["timing_seconds"]["candidate_build"])
        assignment_seconds = float(low["timing_seconds"]["assignment"])
        home_frame = pd.DataFrame()
        work_frame = pd.DataFrame()
    else:
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

    person_frame = _persons_to_frame(assigned)

    home_csv = synthetic_dir / "home_candidates.csv"
    work_csv = synthetic_dir / "work_candidates.csv"
    persons_csv = synthetic_dir / "person_locations.csv"
    group_csv = metrics_dir / "candidate_group_counts.csv"
    group_diag_csv = metrics_dir / "group_diagnostics.csv"
    no_candidate_csv = metrics_dir / "no_candidate_groups.csv"
    if bool(args.low_memory):
        pd.DataFrame().to_csv(home_csv, index=False)
        pd.DataFrame().to_csv(work_csv, index=False)
    else:
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
        "low_memory": bool(args.low_memory),
        "n_jobs": int(args.n_jobs),
        "parallel_chunksize": int(args.parallel_chunksize),
        "input_paths": {
            "persons_path": str(persons_path),
            "areas_path": str(areas_path),
            "roads_path": str(roads_path),
            "cross_state_asset_inventory_csv": str(cross_state_asset_inventory_csv) if cross_state_asset_inventory_csv else None,
        },
        "cross_state_work_support": cross_state_support_meta,
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
            "candidate_detail_csvs_empty": bool(args.low_memory),
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
        "low_memory": bool(args.low_memory),
        "n_jobs": int(args.n_jobs),
        "parallel_chunksize": int(args.parallel_chunksize),
        "home_interpolation_density": float(args.home_interpolation_density),
        "work_interpolation_density": float(args.work_interpolation_density),
        "candidate_meta": candidate_meta,
        "assignment_meta": assignment_meta,
        "cross_state_work_support": cross_state_support_meta,
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
