#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import sys
import urllib.request
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.data.lodes import (
    assign_job_center_membership,
    aggregate_lodes_to_tract_od,
    aggregate_lodes_wac_to_tract,
    build_tract_area_crosswalk,
    build_tract_centroid_table,
    compute_gravity_accessibility,
    compute_job_center_accessibility,
    ensure_lodes_od_file,
    enrich_tract_od_with_geometry_and_wac,
    load_lodes_rac_or_wac,
    load_lodes_od,
    prepare_internal_study_tract_od,
    remap_tract_od_geoids,
    remap_tract_wac_geoids,
)
from src.synthpop.paths import ensure_dir, project_root

CT_PLANNING_REGION_COUNTYFPS = {"110", "120", "130", "140", "150", "160", "170", "180", "190"}


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


def _canon_geoid_text(s: pd.Series, *, width: int = 11) -> pd.Series:
    out = s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    missing = out.isna() | out.str.lower().isin({"", "nan", "none", "<na>"})
    numeric = out.str.fullmatch(r"\d+").fillna(False)
    out.loc[numeric] = out.loc[numeric].str.zfill(int(width))
    out.loc[missing] = pd.NA
    return out


def _load_asset_inventory(path: pathlib.Path) -> pd.DataFrame:
    inv = pd.read_csv(path, dtype={"statefp": str}, low_memory=False)
    inv["statefp"] = inv["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    if "status" in inv.columns:
        inv = inv[inv["status"].astype(str).str.lower() == "ready"].copy()
    return inv


def _study_tracts_canonical(study_tracts: set[str] | list[str]) -> set[str]:
    return {str(x).replace(".0", "").strip().zfill(11) for x in study_tracts}


def _looks_like_ct_planning_region_tracts(study_tracts: set[str] | list[str]) -> bool:
    study = _study_tracts_canonical(study_tracts)
    return any(t[:2] == "09" and t[2:5] in CT_PLANNING_REGION_COUNTYFPS for t in study)


def _download_tiger_tract_zip(*, statefp: str, year: int, out_path: pathlib.Path) -> pathlib.Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    statefp = str(statefp).zfill(2)
    year = int(year)
    urls = [
        f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{statefp}_tract.zip",
    ]
    if year == 2020:
        # Some Census mirrors expose 2020 tract files through the PL hierarchy.
        urls.extend(
            [
                f"https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TRACT/2020/tl_2020_{statefp}_tract20.zip",
                f"https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/{statefp}_CONNECTICUT/{statefp}/tl_2020_{statefp}_tract20.zip"
                if statefp == "09"
                else "",
            ]
        )
    errors: list[str] = []
    for url in [u for u in urls if u]:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Synthetic-City research data preparation"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, out_path.open("wb") as f:
                shutil.copyfileobj(resp, f)
            return out_path
        except Exception as e:
            if out_path.exists():
                out_path.unlink()
            errors.append(f"{url}: {e}")
    raise SystemExit("failed to download TIGER tract zip:\n" + "\n".join(errors))
    return out_path


def _load_or_build_lodes_geoid_crosswalk(
    *,
    statefp: str,
    study_tracts: set[str] | list[str],
    areas: Any,
    areas_group_col: str,
    areas_path: pathlib.Path,
    run_dir: pathlib.Path,
    explicit_crosswalk_csv: pathlib.Path | None,
    legacy_tract_areas_path: pathlib.Path | None,
    legacy_tract_group_col: str,
    legacy_tract_year: int,
    disable_auto_ct_crosswalk: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": False,
        "reason": "not_requested",
        "statefp": str(statefp).zfill(2),
    }
    if explicit_crosswalk_csv is not None:
        if not explicit_crosswalk_csv.exists():
            raise SystemExit(f"tract_geoid_crosswalk_csv not found: {explicit_crosswalk_csv}")
        cw = pd.read_csv(explicit_crosswalk_csv, dtype=str, low_memory=False)
        meta.update(
            {
                "enabled": True,
                "reason": "explicit_crosswalk_csv",
                "crosswalk_csv": str(explicit_crosswalk_csv),
                "n_rows": int(cw.shape[0]),
            }
        )
        return cw, meta

    if disable_auto_ct_crosswalk or str(statefp).zfill(2) != "09" or not _looks_like_ct_planning_region_tracts(study_tracts):
        meta["reason"] = "auto_ct_crosswalk_not_applicable"
        return None, meta

    if legacy_tract_areas_path is None:
        legacy_tract_areas_path = areas_path.parent / f"tl_{int(legacy_tract_year)}_09_tract.zip"
    legacy_tract_areas_path = _download_tiger_tract_zip(
        statefp="09",
        year=int(legacy_tract_year),
        out_path=legacy_tract_areas_path,
    )
    legacy_areas = _read_geodata(legacy_tract_areas_path)
    cw = build_tract_area_crosswalk(
        legacy_areas=legacy_areas,
        current_areas=areas,
        legacy_group_col=str(legacy_tract_group_col),
        current_group_col=str(areas_group_col),
    )
    if cw.empty:
        raise SystemExit(
            "Connecticut LODES tract GEOID crosswalk is empty. "
            "Check legacy_tract_areas_path and current areas_path."
        )
    out_csv = ensure_dir(run_dir / "metrics") / "lodes_legacy_to_current_tract_crosswalk.csv"
    cw.to_csv(out_csv, index=False)
    meta.update(
        {
            "enabled": True,
            "reason": "auto_ct_legacy_to_planning_region_crosswalk",
            "legacy_tract_areas_path": str(legacy_tract_areas_path),
            "legacy_tract_year": int(legacy_tract_year),
            "crosswalk_csv": str(out_csv),
            "n_rows": int(cw.shape[0]),
            "n_legacy_tracts": int(cw["legacy_tract_geoid"].nunique()),
            "n_current_tracts": int(cw["tract_geoid"].nunique()),
        }
    )
    return cw, meta


def _prepare_home_outbound_tract_od(
    *,
    tract_od: pd.DataFrame,
    study_tracts: set[str] | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    study = {str(x).replace(".0", "").strip().zfill(11) for x in study_tracts}
    od = tract_od.copy()
    od["home_tract_geoid"] = _canon_geoid_text(od["home_tract_geoid"], width=11)
    od["work_tract_geoid"] = _canon_geoid_text(od["work_tract_geoid"], width=11)
    od = od.dropna(subset=["home_tract_geoid", "work_tract_geoid"]).copy()
    od["S000"] = pd.to_numeric(od["S000"], errors="coerce").fillna(0.0)
    value_cols = [c for c in od.columns if c not in {"home_tract_geoid", "work_tract_geoid"}]
    for col in value_cols:
        od[col] = pd.to_numeric(od[col], errors="coerce").fillna(0.0)

    outbound = od[od["home_tract_geoid"].isin(sorted(study))].copy()
    outbound = (
        outbound.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values(["home_tract_geoid", "work_tract_geoid"], kind="stable")
        .reset_index(drop=True)
    )
    origin_total = (
        outbound.groupby("home_tract_geoid", as_index=False, sort=False)["S000"]
        .sum()
        .rename(columns={"S000": "total_jobs_from_origin"})
    )
    same_state = outbound["home_tract_geoid"].astype(str).str.slice(0, 2) == outbound["work_tract_geoid"].astype(str).str.slice(0, 2)
    internal_by_origin = (
        outbound.loc[same_state]
        .groupby("home_tract_geoid", as_index=False, sort=False)["S000"]
        .sum()
        .rename(columns={"S000": "internal_jobs_from_origin"})
    )
    origin_stats = pd.DataFrame({"home_tract_geoid": sorted(study)}).merge(origin_total, on="home_tract_geoid", how="left")
    origin_stats = origin_stats.merge(internal_by_origin, on="home_tract_geoid", how="left")
    origin_stats["total_jobs_from_origin"] = origin_stats["total_jobs_from_origin"].fillna(0.0)
    origin_stats["internal_jobs_from_origin"] = origin_stats["internal_jobs_from_origin"].fillna(0.0)
    origin_stats["cross_state_jobs_from_origin"] = origin_stats["total_jobs_from_origin"] - origin_stats["internal_jobs_from_origin"]
    origin_stats["internal_share"] = origin_stats["internal_jobs_from_origin"] / origin_stats["total_jobs_from_origin"].replace(0.0, 1.0)
    origin_stats["cross_state_share"] = origin_stats["cross_state_jobs_from_origin"] / origin_stats["total_jobs_from_origin"].replace(0.0, 1.0)
    origin_stats["has_internal_destination"] = origin_stats["internal_jobs_from_origin"] > 0.0
    origin_stats["has_cross_state_destination"] = origin_stats["cross_state_jobs_from_origin"] > 0.0

    total_jobs = float(origin_stats["total_jobs_from_origin"].sum())
    internal_jobs = float(origin_stats["internal_jobs_from_origin"].sum())
    cross_jobs = float(origin_stats["cross_state_jobs_from_origin"].sum())
    summary = {
        "n_study_tracts": int(len(study)),
        "n_origin_tracts_with_any_jobs": int((origin_stats["total_jobs_from_origin"] > 0.0).sum()),
        "n_origin_tracts_with_internal_dest": int(origin_stats["has_internal_destination"].sum()),
        "n_origin_tracts_with_cross_state_dest": int(origin_stats["has_cross_state_destination"].sum()),
        "share_origin_tracts_with_internal_dest": float(origin_stats["has_internal_destination"].mean()) if len(origin_stats) else float("nan"),
        "share_origin_tracts_with_cross_state_dest": float(origin_stats["has_cross_state_destination"].mean()) if len(origin_stats) else float("nan"),
        "total_jobs_from_study_origins": total_jobs,
        "total_internal_jobs": internal_jobs,
        "total_cross_state_jobs": cross_jobs,
        "overall_internal_share": float(internal_jobs / max(total_jobs, 1.0)),
        "overall_cross_state_share": float(cross_jobs / max(total_jobs, 1.0)),
    }
    return outbound, origin_stats.sort_values("home_tract_geoid", kind="stable").reset_index(drop=True), summary


def _load_cross_state_tract_od(
    *,
    asset_inventory: pd.DataFrame,
    study_tracts: set[str] | list[str],
    tract_geoid_crosswalk: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    study = _study_tracts_canonical(study_tracts)
    frames: list[pd.DataFrame] = []
    states_scanned = 0
    states_with_outbound = 0
    missing: list[str] = []
    remapped_states: list[str] = []
    for row in asset_inventory.sort_values("statefp").to_dict("records"):
        statefp = str(row.get("statefp", "")).zfill(2)
        main_path = pathlib.Path(str(row.get("lodes_main_path", ""))).expanduser()
        aux_path = pathlib.Path(str(row.get("lodes_aux_path", ""))).expanduser()
        if not main_path.exists() or not aux_path.exists():
            missing.append(statefp)
            continue
        states_scanned += 1
        od_block = load_lodes_od(main_path=main_path, aux_path=aux_path)
        od = aggregate_lodes_to_tract_od(od_block)
        if tract_geoid_crosswalk is not None and not tract_geoid_crosswalk.empty:
            before_home = set(_canon_geoid_text(od["home_tract_geoid"], width=11).dropna().astype(str).tolist())
            before_work = set(_canon_geoid_text(od["work_tract_geoid"], width=11).dropna().astype(str).tolist())
            od = remap_tract_od_geoids(od, tract_geoid_crosswalk)
            after_home = set(_canon_geoid_text(od["home_tract_geoid"], width=11).dropna().astype(str).tolist())
            after_work = set(_canon_geoid_text(od["work_tract_geoid"], width=11).dropna().astype(str).tolist())
            if before_home != after_home or before_work != after_work:
                remapped_states.append(statefp)
        od["home_tract_geoid"] = _canon_geoid_text(od["home_tract_geoid"], width=11)
        od["work_tract_geoid"] = _canon_geoid_text(od["work_tract_geoid"], width=11)
        od = od[od["home_tract_geoid"].isin(study)].copy()
        if od.empty:
            continue
        states_with_outbound += 1
        frames.append(od)
    if not frames:
        return pd.DataFrame(columns=["home_tract_geoid", "work_tract_geoid", "S000"]), {
            "states_scanned": states_scanned,
            "states_with_outbound": states_with_outbound,
            "missing_lodes_statefps": missing,
            "geoid_remapped_statefps": sorted(set(remapped_states)),
        }
    out = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in out.columns if c not in {"home_tract_geoid", "work_tract_geoid"}]
    out = (
        out.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values(["home_tract_geoid", "work_tract_geoid"], kind="stable")
        .reset_index(drop=True)
    )
    return out, {
        "states_scanned": states_scanned,
        "states_with_outbound": states_with_outbound,
        "missing_lodes_statefps": missing,
        "geoid_remapped_statefps": sorted(set(remapped_states)),
    }


def _cross_state_home_outbound_cache_path(cache_dir: pathlib.Path, home_statefp: str) -> pathlib.Path:
    return cache_dir / f"home_statefp={str(home_statefp).zfill(2)}" / "tract_od.parquet"


def _load_cross_state_tract_od_from_home_cache(
    *,
    cache_dir: pathlib.Path,
    home_statefp: str,
    study_tracts: set[str] | list[str],
    tract_geoid_crosswalk: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    path = _cross_state_home_outbound_cache_path(cache_dir.expanduser().resolve(), home_statefp)
    meta: dict[str, Any] = {
        "cache_dir": str(cache_dir.expanduser().resolve()),
        "cache_path": str(path),
        "cache_hit": False,
    }
    if not path.exists():
        return None, meta
    od = pd.read_parquet(path)
    if tract_geoid_crosswalk is not None and not tract_geoid_crosswalk.empty:
        od = remap_tract_od_geoids(od, tract_geoid_crosswalk)
        meta["geoid_remapped_from_cache"] = True
    od["home_tract_geoid"] = _canon_geoid_text(od["home_tract_geoid"], width=11)
    od["work_tract_geoid"] = _canon_geoid_text(od["work_tract_geoid"], width=11)
    study = _study_tracts_canonical(study_tracts)
    od = od[od["home_tract_geoid"].isin(study)].copy()
    meta.update(
        {
            "cache_hit": True,
            "rows_after_study_filter": int(od.shape[0]),
        }
    )
    return od, meta


def _build_centroids_and_wac_for_states(
    *,
    statefps: set[str],
    needed_tracts: set[str],
    base_areas: Any,
    base_group_col: str,
    asset_inventory: pd.DataFrame | None,
    home_statefp: str,
    wac_path_override: pathlib.Path | None,
    accessibility_beta: float,
    job_center_beta: float,
    job_center_top_quantile: float,
    job_center_min_centers: int,
    job_center_min_centers_per_county: int,
    tract_geoid_crosswalk: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    inv_by_state = {}
    if asset_inventory is not None and not asset_inventory.empty:
        inv_by_state = {str(r["statefp"]).zfill(2): r for r in asset_inventory.to_dict("records")}
    centroid_parts: list[pd.DataFrame] = []
    wac_parts: list[pd.DataFrame] = []
    missing: list[str] = []
    loaded_states: list[str] = []

    for statefp in sorted(statefps):
        if statefp == home_statefp:
            areas = base_areas.copy()
            group_col = base_group_col
        else:
            row = inv_by_state.get(statefp)
            if row is None or not str(row.get("tract_zip", "")).strip():
                missing.append(f"{statefp}:tract_zip")
                continue
            tract_path = pathlib.Path(str(row.get("tract_zip"))).expanduser()
            if not tract_path.exists():
                missing.append(f"{statefp}:tract_zip")
                continue
            areas = _read_geodata(tract_path)
            group_col = "GEOID" if "GEOID" in areas.columns else base_group_col
        if group_col not in areas.columns:
            missing.append(f"{statefp}:group_col")
            continue
        cent = build_tract_centroid_table(areas=areas, group_col=group_col).rename(columns={group_col: "tract_geoid"})
        cent["tract_geoid"] = _canon_geoid_text(cent["tract_geoid"], width=11)
        cent = cent[cent["tract_geoid"].isin(needed_tracts)].copy()
        if cent.empty:
            continue
        centroid_parts.append(cent)
        loaded_states.append(statefp)

        wac_path = None
        if statefp == home_statefp and wac_path_override is not None and wac_path_override.exists():
            wac_path = wac_path_override
        else:
            row = inv_by_state.get(statefp)
            if row is not None and str(row.get("wac_path", "")).strip():
                cand = pathlib.Path(str(row.get("wac_path"))).expanduser()
                if cand.exists():
                    wac_path = cand
        if wac_path is None:
            missing.append(f"{statefp}:wac_path")
            continue
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
        if tract_geoid_crosswalk is not None and not tract_geoid_crosswalk.empty:
            tract_wac = remap_tract_wac_geoids(tract_wac, tract_geoid_crosswalk)
        tract_wac["tract_geoid"] = _canon_geoid_text(tract_wac["tract_geoid"], width=11)
        tract_wac = tract_wac[tract_wac["tract_geoid"].isin(needed_tracts)].copy()
        if tract_wac.empty:
            continue
        state_cent = cent[["tract_geoid", "centroid_x", "centroid_y"]].copy()
        if float(accessibility_beta) > 0.0:
            access = compute_gravity_accessibility(
                tract_centroids=state_cent,
                tract_mass=tract_wac,
                tract_col="tract_geoid",
                mass_col="C000",
                distance_beta=float(accessibility_beta),
                out_col="access_jobs_gravity",
            )
            tract_wac = tract_wac.merge(access, on="tract_geoid", how="left")
            tract_wac["access_jobs_gravity"] = pd.to_numeric(tract_wac["access_jobs_gravity"], errors="coerce").fillna(0.0)
        if float(job_center_beta) > 0.0:
            center_access = compute_job_center_accessibility(
                tract_centroids=state_cent,
                tract_mass=tract_wac,
                tract_col="tract_geoid",
                mass_col="C000",
                distance_beta=float(job_center_beta),
                top_quantile=float(job_center_top_quantile),
                min_centers=int(job_center_min_centers),
                out_col="access_job_centers_gravity",
            )
            tract_wac = tract_wac.merge(center_access, on="tract_geoid", how="left")
            tract_wac["access_job_centers_gravity"] = pd.to_numeric(
                tract_wac["access_job_centers_gravity"], errors="coerce"
            ).fillna(0.0)
        center_membership = assign_job_center_membership(
            tract_centroids=state_cent,
            tract_mass=tract_wac,
            tract_col="tract_geoid",
            mass_col="C000",
            county_col="county_geoid",
            top_quantile=float(job_center_top_quantile),
            min_centers_per_county=int(job_center_min_centers_per_county),
        )
        tract_wac = tract_wac.merge(center_membership, on="tract_geoid", how="left")
        tract_wac["center_geoid"] = tract_wac["center_geoid"].fillna(tract_wac["tract_geoid"]).astype(str)
        tract_wac["center_county_geoid"] = tract_wac["center_county_geoid"].fillna(tract_wac["tract_geoid"].astype(str).str.slice(0, 5)).astype(str)
        tract_wac["center_distance_km"] = pd.to_numeric(tract_wac["center_distance_km"], errors="coerce").fillna(0.0)
        tract_wac["center_mass"] = pd.to_numeric(tract_wac["center_mass"], errors="coerce").fillna(0.0)
        wac_parts.append(tract_wac)

    centroids = pd.concat(centroid_parts, ignore_index=True).drop_duplicates("tract_geoid", keep="first") if centroid_parts else pd.DataFrame()
    wac = pd.concat(wac_parts, ignore_index=True).drop_duplicates("tract_geoid", keep="first") if wac_parts else None
    return centroids, wac, {
        "statefps_requested": sorted(statefps),
        "statefps_loaded_centroids": sorted(set(loaded_states)),
        "missing_assets": missing,
        "n_centroid_tracts": int(centroids.shape[0]),
        "n_wac_tracts": int(wac.shape[0]) if wac is not None else 0,
    }


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
    ap.add_argument("--cross_state_asset_inventory_csv", default="")
    ap.add_argument("--cross_state_home_outbound_cache_dir", default="")
    ap.add_argument("--allow_cross_state_work", action="store_true")
    ap.add_argument("--tract_geoid_crosswalk_csv", default="")
    ap.add_argument("--legacy_tract_areas_path", default="")
    ap.add_argument("--legacy_tract_group_col", default="GEOID")
    ap.add_argument("--legacy_tract_year", type=int, default=2020)
    ap.add_argument("--disable_auto_ct_lodes_geoid_crosswalk", action="store_true")
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
    home_statefp = str(next(iter(study_tracts))).replace(".0", "").strip().zfill(11)[:2] if study_tracts else ""

    explicit_crosswalk_csv = pathlib.Path(args.tract_geoid_crosswalk_csv).expanduser().resolve() if args.tract_geoid_crosswalk_csv else None
    legacy_tract_areas_path = pathlib.Path(args.legacy_tract_areas_path).expanduser().resolve() if args.legacy_tract_areas_path else None
    tract_geoid_crosswalk, geoid_crosswalk_meta = _load_or_build_lodes_geoid_crosswalk(
        statefp=home_statefp,
        study_tracts=study_tracts,
        areas=areas,
        areas_group_col=group_col,
        areas_path=areas_path,
        run_dir=run_dir,
        explicit_crosswalk_csv=explicit_crosswalk_csv,
        legacy_tract_areas_path=legacy_tract_areas_path,
        legacy_tract_group_col=str(args.legacy_tract_group_col),
        legacy_tract_year=int(args.legacy_tract_year),
        disable_auto_ct_crosswalk=bool(args.disable_auto_ct_lodes_geoid_crosswalk),
    )

    raw_dir = pathlib.Path(args.raw_dir).expanduser().resolve() if args.raw_dir else (project_root() / "dataset" / "lodes")
    if args.main_path:
        main_path = pathlib.Path(args.main_path).expanduser().resolve()
    else:
        main_path = ensure_lodes_od_file(state_postal=args.state_postal, year=int(args.year), part="main", out_dir=raw_dir)
    if args.aux_path:
        aux_path = pathlib.Path(args.aux_path).expanduser().resolve()
    else:
        aux_path = ensure_lodes_od_file(state_postal=args.state_postal, year=int(args.year), part="aux", out_dir=raw_dir)

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

    cross_state_meta: dict[str, Any] = {"enabled": bool(args.allow_cross_state_work)}
    asset_inventory = None
    if args.allow_cross_state_work:
        inv_path = pathlib.Path(args.cross_state_asset_inventory_csv).expanduser().resolve() if args.cross_state_asset_inventory_csv else None
        if inv_path is None or not inv_path.exists():
            raise SystemExit("--allow_cross_state_work requires --cross_state_asset_inventory_csv")
        asset_inventory = _load_asset_inventory(inv_path)
        cache_dir = pathlib.Path(args.cross_state_home_outbound_cache_dir).expanduser().resolve() if str(args.cross_state_home_outbound_cache_dir).strip() else None
        cache_meta: dict[str, Any] = {"enabled": cache_dir is not None}
        tract_od = None
        if cache_dir is not None:
            tract_od, cache_meta = _load_cross_state_tract_od_from_home_cache(
                cache_dir=cache_dir,
                home_statefp=home_statefp,
                study_tracts=study_tracts,
                tract_geoid_crosswalk=tract_geoid_crosswalk,
            )
        if tract_od is None:
            tract_od, od_scan_meta = _load_cross_state_tract_od(
                asset_inventory=asset_inventory,
                study_tracts=study_tracts,
                tract_geoid_crosswalk=tract_geoid_crosswalk,
            )
        else:
            od_scan_meta = {
                "states_scanned": 0,
                "states_with_outbound": 0,
                "missing_lodes_statefps": [],
                "geoid_remapped_statefps": [],
            }
        internal_od, origin_stats, summary = _prepare_home_outbound_tract_od(tract_od=tract_od, study_tracts=study_tracts)
        cross_state_meta.update({"asset_inventory_csv": str(inv_path), "home_outbound_cache": cache_meta, **od_scan_meta})
    else:
        od_block = load_lodes_od(main_path=main_path, aux_path=aux_path)
        tract_od = aggregate_lodes_to_tract_od(od_block)
        if tract_geoid_crosswalk is not None and not tract_geoid_crosswalk.empty:
            tract_od = remap_tract_od_geoids(tract_od, tract_geoid_crosswalk)
        internal_od, origin_stats, summary = prepare_internal_study_tract_od(tract_od=tract_od, study_tracts=study_tracts)

    if (
        int(summary.get("n_study_tracts", 0)) > 0
        and int(summary.get("n_origin_tracts_with_any_jobs", 0)) == 0
        and _looks_like_ct_planning_region_tracts(study_tracts)
        and (tract_geoid_crosswalk is None or tract_geoid_crosswalk.empty)
    ):
        raise SystemExit(
            "Connecticut study tracts use 2022+ planning-region GEOIDs, but no LODES tract crosswalk was applied. "
            "Provide --tract_geoid_crosswalk_csv or allow the automatic CT TIGER2020-to-current crosswalk."
        )

    needed_tracts = set(_canon_geoid_text(internal_od["home_tract_geoid"], width=11).dropna().astype(str).tolist())
    needed_tracts |= set(_canon_geoid_text(internal_od["work_tract_geoid"], width=11).dropna().astype(str).tolist())
    statefps_needed = {str(x)[:2] for x in needed_tracts if str(x) and str(x).lower() != "nan"}
    tract_centroids, tract_wac, support_meta = _build_centroids_and_wac_for_states(
        statefps=statefps_needed if args.allow_cross_state_work else {home_statefp},
        needed_tracts=needed_tracts if args.allow_cross_state_work else {str(x).replace(".0", "").strip().zfill(11) for x in study_tracts},
        base_areas=areas,
        base_group_col=group_col,
        asset_inventory=asset_inventory,
        home_statefp=home_statefp,
        wac_path_override=wac_path if wac_path is not None else None,
        accessibility_beta=float(args.accessibility_beta),
        job_center_beta=float(args.job_center_beta),
        job_center_top_quantile=float(args.job_center_top_quantile),
        job_center_min_centers=int(args.job_center_min_centers),
        job_center_min_centers_per_county=int(args.job_center_min_centers_per_county),
        tract_geoid_crosswalk=tract_geoid_crosswalk,
    )
    cross_state_meta["support"] = support_meta

    internal_od = enrich_tract_od_with_geometry_and_wac(
        tract_od=internal_od,
        tract_centroids=tract_centroids.rename(columns={group_col: "tract_geoid"}) if group_col in tract_centroids.columns else tract_centroids,
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
        "allow_cross_state_work": bool(args.allow_cross_state_work),
        "cross_state_work": cross_state_meta,
        "lodes_geoid_compatibility": geoid_crosswalk_meta,
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
