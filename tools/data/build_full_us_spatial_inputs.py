#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.build_acs_targets_long_michigan import (  # noqa: E402
    _b01001_records,
    _b15003_records,
    _b20001_records,
    _b23025_records,
)


STATES: dict[str, tuple[str, str]] = {
    "01": ("al", "Alabama"),
    "02": ("ak", "Alaska"),
    "04": ("az", "Arizona"),
    "05": ("ar", "Arkansas"),
    "06": ("ca", "California"),
    "08": ("co", "Colorado"),
    "09": ("ct", "Connecticut"),
    "10": ("de", "Delaware"),
    "11": ("dc", "District_of_Columbia"),
    "12": ("fl", "Florida"),
    "13": ("ga", "Georgia"),
    "15": ("hi", "Hawaii"),
    "16": ("id", "Idaho"),
    "17": ("il", "Illinois"),
    "18": ("in", "Indiana"),
    "19": ("ia", "Iowa"),
    "20": ("ks", "Kansas"),
    "21": ("ky", "Kentucky"),
    "22": ("la", "Louisiana"),
    "23": ("me", "Maine"),
    "24": ("md", "Maryland"),
    "25": ("ma", "Massachusetts"),
    "26": ("mi", "Michigan"),
    "27": ("mn", "Minnesota"),
    "28": ("ms", "Mississippi"),
    "29": ("mo", "Missouri"),
    "30": ("mt", "Montana"),
    "31": ("ne", "Nebraska"),
    "32": ("nv", "Nevada"),
    "33": ("nh", "New_Hampshire"),
    "34": ("nj", "New_Jersey"),
    "35": ("nm", "New_Mexico"),
    "36": ("ny", "New_York"),
    "37": ("nc", "North_Carolina"),
    "38": ("nd", "North_Dakota"),
    "39": ("oh", "Ohio"),
    "40": ("ok", "Oklahoma"),
    "41": ("or", "Oregon"),
    "42": ("pa", "Pennsylvania"),
    "44": ("ri", "Rhode_Island"),
    "45": ("sc", "South_Carolina"),
    "46": ("sd", "South_Dakota"),
    "47": ("tn", "Tennessee"),
    "48": ("tx", "Texas"),
    "49": ("ut", "Utah"),
    "50": ("vt", "Vermont"),
    "51": ("va", "Virginia"),
    "53": ("wa", "Washington"),
    "54": ("wv", "West_Virginia"),
    "55": ("wi", "Wisconsin"),
    "56": ("wy", "Wyoming"),
}

ACS_TABLE_COUNTS = {
    "B01001": 49,
    "B15003": 25,
    "B20001": 43,
    "B23025": 7,
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "SyntheticCity-SpatialInputBuilder/1.0"})
    return s


def _run(cmd: list[str], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] COMMAND {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def _wget(url: str, dest: pathlib.Path, *, log_path: pathlib.Path, overwrite: bool = False) -> pathlib.Path:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-c",
        "-nv",
        "--tries=8",
        "--timeout=60",
        "--waitretry=2",
        "--retry-connrefused",
        "-O",
        str(dest),
        url,
    ]
    rc = _run(cmd, cwd=dest.parent, log_path=log_path)
    if rc != 0 or (not dest.exists()) or dest.stat().st_size == 0:
        raise RuntimeError(f"download failed rc={rc}: {url}")
    return dest


def _url_available(url: str, *, log_path: pathlib.Path) -> bool:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30, headers={"User-Agent": "SyntheticCity-SpatialInputBuilder/1.0"})
        ok = int(resp.status_code) == 200
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_utc_now()}] HEAD {url} status={resp.status_code}\n")
        return ok
    except Exception as e:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_utc_now()}] HEAD {url} error={type(e).__name__}: {e}\n")
        return False


def _fetch_acs_table(
    *,
    statefp: str,
    table: str,
    acs_year: int,
    out_path: pathlib.Path,
    log_path: pathlib.Path,
    overwrite: bool,
    api_key: str,
) -> pathlib.Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return out_path
    vars_list = [f"{table}_{i:03d}E" for i in range(1, ACS_TABLE_COUNTS[table] + 1)]
    url = (
        f"https://api.census.gov/data/{acs_year}/acs/acs5"
        f"?get=NAME,{','.join(vars_list)}"
        f"&for=tract:*&in=state:{statefp}"
    )
    if api_key:
        url += f"&key={api_key}"
    sess = _session()
    t0 = time.perf_counter()
    resp = sess.get(url, timeout=180)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_utc_now()}] ACS {statefp} {table} status={resp.status_code} seconds={time.perf_counter() - t0:.1f}\n")
    resp.raise_for_status()
    if "application/json" not in str(resp.headers.get("content-type", "")).lower() and resp.text.lstrip().startswith("<"):
        raise RuntimeError("Census API returned HTML instead of JSON; set CENSUS_API_KEY or pass --census_api_key.")
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"empty ACS response for state={statefp} table={table}")
    df = pd.DataFrame(data[1:], columns=data[0])
    df["GEOID"] = df["state"].astype(str).str.zfill(2) + df["county"].astype(str).str.zfill(3) + df["tract"].astype(str).str.zfill(6)
    for c in vars_list:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")
    return out_path


def _read_zip_geodata(path: pathlib.Path):
    import geopandas as gpd

    return gpd.read_file(f"zip://{path}")


def _build_targets_long(
    *,
    acs_paths: dict[str, pathlib.Path],
    out_path: pathlib.Path,
    acs_year: int,
    overwrite: bool,
    include_age_sex_cross: bool = False,
) -> pathlib.Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        if not bool(include_age_sex_cross):
            return out_path
        try:
            existing_vars = set(
                pd.read_csv(out_path, usecols=["variable"])["variable"]
                .astype(str)
                .unique()
                .tolist()
            )
            if "AGEP_SEX_cross" in existing_vars:
                return out_path
        except Exception:
            pass
    dfs: dict[str, pd.DataFrame] = {}
    for table, path in acs_paths.items():
        df = pd.read_csv(path, compression="gzip", dtype={"state": str, "county": str, "tract": str}, low_memory=False)
        if "tract_geoid" not in df.columns:
            if "GEOID" in df.columns:
                df["tract_geoid"] = df["GEOID"].astype(str).str.replace(r"[^0-9]", "", regex=True).str[-11:]
            else:
                df["tract_geoid"] = df["state"].astype(str).str.zfill(2) + df["county"].astype(str).str.zfill(3) + df["tract"].astype(str).str.zfill(6)
        dfs[table] = df

    records: list[dict[str, Any]] = []
    records.extend(
        _b01001_records(
            dfs["B01001"],
            group_col="tract_geoid",
            include_age_sex_cross=bool(include_age_sex_cross),
        )
    )
    records.extend(_b23025_records(dfs["B23025"], group_col="tract_geoid"))
    records.extend(_b15003_records(dfs["B15003"], group_col="tract_geoid"))
    records.extend(_b20001_records(dfs["B20001"], group_col="tract_geoid"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tract_geoid", "variable", "category", "target", "table_id", "source", "acs_year", "geo_level"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in records:
            var = r.get("variable")
            if var in {"SEX", "AGEP_bin", "AGEP_SEX_cross"}:
                table_id = "B01001"
            elif var == "ESR_16p":
                table_id = "B23025"
            elif var == "SCHL_25p":
                table_id = "B15003"
            elif var == "PINCP_16p_bin":
                table_id = "B20001"
            else:
                table_id = ""
            writer.writerow(
                {
                    "tract_geoid": r.get("tract_geoid"),
                    "variable": var,
                    "category": r.get("category"),
                    "target": float(r.get("target", 0.0)),
                    "table_id": table_id,
                    "source": "acs5",
                    "acs_year": int(acs_year),
                    "geo_level": "tract",
                }
            )
    return out_path


def _build_roads_parquet(
    *,
    statefp: str,
    countyfps: list[str],
    roads_raw_dir: pathlib.Path,
    out_path: pathlib.Path,
    log_path: pathlib.Path,
    county_workers: int,
    overwrite: bool,
) -> pathlib.Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return out_path
    import geopandas as gpd
    import pandas as pd

    roads_raw_dir.mkdir(parents=True, exist_ok=True)

    def download_county(countyfp: str) -> pathlib.Path:
        geoid = f"{statefp}{str(countyfp).zfill(3)}"
        dest = roads_raw_dir / f"tl_2023_{geoid}_roads.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER2023/ROADS/tl_2023_{geoid}_roads.zip"
        return _wget(url, dest, log_path=log_path, overwrite=overwrite)

    road_paths: list[pathlib.Path] = []
    failures: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(county_workers))) as ex:
        futs = {ex.submit(download_county, c): c for c in countyfps}
        for fut in cf.as_completed(futs):
            countyfp = futs[fut]
            try:
                road_paths.append(fut.result())
            except Exception as e:
                failures.append(f"{statefp}{countyfp}:{e}")

    if failures:
        raise RuntimeError("road county download failures: " + "; ".join(failures[:20]))

    frames = []
    read_failures: list[str] = []
    keep_mtfcc = {"S1100", "S1200", "S1400", "S1740"}
    for p in sorted(road_paths):
        if (not p.exists()) or p.stat().st_size < 1024:
            read_failures.append(f"{p.name}:missing_or_too_small")
            continue
        try:
            gdf = gpd.read_file(f"zip://{p}")
        except Exception as e:
            read_failures.append(f"{p.name}:{type(e).__name__}")
            continue
        cols = [c for c in ["LINEARID", "FULLNAME", "MTFCC", "RTTYP", "geometry"] if c in gdf.columns]
        gdf = gdf[cols].copy()
        if "MTFCC" in gdf.columns:
            gdf = gdf[gdf["MTFCC"].astype(str).isin(keep_mtfcc)].copy()
        if not gdf.empty:
            frames.append(gdf)
    if not frames:
        detail = "; ".join(read_failures[:20])
        raise RuntimeError(f"no usable road segments for state={statefp}; road read failures: {detail}")
    if read_failures:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"[{_utc_now()}] ROADS {statefp} skipped_invalid_zips={len(read_failures)} "
                f"examples={'; '.join(read_failures[:10])}\n"
            )
    roads = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(out_path, index=False)
    return out_path


def _download_lodes_file(
    *,
    state_postal: str,
    subdir: str,
    name_template: str,
    requested_year: int,
    fallback_years: list[int],
    dest_dir: pathlib.Path,
    log_path: pathlib.Path,
    overwrite: bool,
) -> tuple[pathlib.Path, int]:
    st = state_postal.lower()
    years = [int(requested_year), *[int(y) for y in fallback_years if int(y) != int(requested_year)]]
    last_url = ""
    for year in years:
        name = name_template.format(st=st, year=int(year))
        url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{st}/{subdir}/{name}"
        last_url = url
        if not _url_available(url, log_path=log_path):
            continue
        return _wget(url, dest_dir / name, log_path=log_path, overwrite=overwrite), int(year)
    raise RuntimeError(f"no available LODES file for {st}/{subdir}; requested_year={requested_year}; last_url={last_url}")


def _download_lodes(*, state_postal: str, year_od: int, year_wac: int, lodes_root: pathlib.Path, log_path: pathlib.Path, overwrite: bool) -> dict[str, pathlib.Path | int]:
    st = state_postal.lower()
    od_dir = lodes_root / "lodes8_od_puma_functional_2020_v0" / "raw" / st
    wac_dir = lodes_root / "lodes8_wac_tract_jobs_2021_v0" / "raw"
    od_dir.mkdir(parents=True, exist_ok=True)
    wac_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pathlib.Path | int] = {}
    # Alaska's public LODES8 OD/WAC directory currently stops at 2016. The
    # fallback is explicit and recorded in the state inventory, rather than
    # silently marking the state as a 2020/2021 LODES product.
    fallback_years = [2016] if st == "ak" else []
    for part in ["main", "aux"]:
        path, actual_year = _download_lodes_file(
            state_postal=st,
            subdir="od",
            name_template=f"{{st}}_od_{part}_JT00_{{year}}.csv.gz",
            requested_year=year_od,
            fallback_years=fallback_years,
            dest_dir=od_dir,
            log_path=log_path,
            overwrite=overwrite,
        )
        out[f"lodes_{part}_path"] = path
        out[f"lodes_{part}_year"] = actual_year
    wac_path, wac_year = _download_lodes_file(
        state_postal=st,
        subdir="wac",
        name_template="{st}_wac_S000_JT00_{year}.csv.gz",
        requested_year=year_wac,
        fallback_years=fallback_years,
        dest_dir=wac_dir,
        log_path=log_path,
        overwrite=overwrite,
    )
    out["wac_path"] = wac_path
    out["wac_year"] = wac_year
    return out


def _build_state(
    *,
    statefp: str,
    base_data_dir: pathlib.Path,
    lodes_root: pathlib.Path,
    run_dir: pathlib.Path,
    repo_root: pathlib.Path,
    acs_year: int,
    census_api_key: str,
    overwrite: bool,
    county_workers: int,
    include_age_sex_cross: bool,
) -> dict[str, Any]:
    statefp = str(statefp).zfill(2)
    state_postal, state_name = STATES[statefp]
    log_path = run_dir / "run.log"
    state_dir = base_data_dir / f"state={statefp}"
    raw_dir = state_dir / "raw"
    out_dir = state_dir / "processed"
    geo_dir = raw_dir / "geo"
    acs_dir = raw_dir / "acs" / f"acs5_{acs_year}"
    roads_dir = raw_dir / "roads"

    row: dict[str, Any] = {
        "statefp": statefp,
        "state_postal": state_postal,
        "state_name": state_name,
        "status": "started",
        "error": "",
        "include_age_sex_cross": bool(include_age_sex_cross),
        "created_utc": _utc_now(),
    }
    try:
        tract_zip = _wget(
            f"https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_{statefp}_tract.zip",
            geo_dir / f"tl_2023_{statefp}_tract.zip",
            log_path=log_path,
            overwrite=overwrite,
        )
        puma_zip = _wget(
            f"https://www2.census.gov/geo/tiger/TIGER2023/PUMA/tl_2023_{statefp}_puma20.zip",
            geo_dir / f"tl_2023_{statefp}_puma20.zip",
            log_path=log_path,
            overwrite=overwrite,
        )

        mapping_csv = out_dir / f"tract_puma_mapping_state{statefp}.csv"
        if not mapping_csv.exists() or overwrite:
            rc = _run(
                [
                    sys.executable,
                    "tools/data/create_tract_puma_mapping.py",
                    "--tract_zip",
                    str(tract_zip),
                    "--puma_zip",
                    str(puma_zip),
                    "--out_csv",
                    str(mapping_csv),
                    "--statefp",
                    statefp,
                ],
                cwd=repo_root,
                log_path=log_path,
            )
            if rc != 0:
                raise RuntimeError(f"tract-puma mapping failed rc={rc}")

        acs_paths = {
            table: _fetch_acs_table(
                statefp=statefp,
                table=table,
                acs_year=acs_year,
                out_path=acs_dir / f"acs5_{acs_year}_{table}_tract_state{statefp}.csv.gz",
                log_path=log_path,
                overwrite=overwrite,
                api_key=census_api_key,
            )
            for table in ["B01001", "B15003", "B20001", "B23025"]
        }
        target_suffix = "_agesex" if bool(include_age_sex_cross) else ""
        targets_csv = _build_targets_long(
            acs_paths=acs_paths,
            out_path=out_dir / f"acs5_{acs_year}_marginals_long_tract_state{statefp}{target_suffix}.csv",
            acs_year=acs_year,
            overwrite=overwrite,
            include_age_sex_cross=bool(include_age_sex_cross),
        )

        tracts = _read_zip_geodata(tract_zip)
        county_col = "COUNTYFP" if "COUNTYFP" in tracts.columns else "COUNTYFP20"
        countyfps = sorted(tracts[county_col].astype(str).str.zfill(3).unique().tolist())
        roads_parquet = _build_roads_parquet(
            statefp=statefp,
            countyfps=countyfps,
            roads_raw_dir=roads_dir,
            out_path=out_dir / f"roads_state{statefp}_tiger2023_support.parquet",
            log_path=log_path,
            county_workers=county_workers,
            overwrite=overwrite,
        )
        lodes = _download_lodes(
            state_postal=state_postal,
            year_od=2020,
            year_wac=2021,
            lodes_root=lodes_root,
            log_path=log_path,
            overwrite=overwrite,
        )
        row.update(
            {
                "status": "ready",
                "tract_zip": str(tract_zip),
                "puma_zip": str(puma_zip),
                "targets_long_csv": str(targets_csv),
                "tract_puma_csv": str(mapping_csv),
                "roads_path": str(roads_parquet),
                "lodes_main_path": str(lodes["lodes_main_path"]),
                "lodes_aux_path": str(lodes["lodes_aux_path"]),
                "wac_path": str(lodes["wac_path"]),
                "lodes_main_year": int(lodes["lodes_main_year"]),
                "lodes_aux_year": int(lodes["lodes_aux_year"]),
                "wac_year": int(lodes["wac_year"]),
                "include_age_sex_cross": bool(include_age_sex_cross),
                "n_tracts": int(len(tracts)),
                "n_counties": int(len(countyfps)),
                "finished_utc": _utc_now(),
            }
        )
    except Exception as e:
        row.update({"status": "failed", "error": str(e), "finished_utc": _utc_now()})
    _write_json(out_dir / "state_spatial_input_summary.json", row)
    return row


def _parse_states(value: str) -> list[str]:
    v = value.strip().lower()
    if v in {"all", "all50", "all51"}:
        return sorted(STATES)
    states = []
    for x in value.split(","):
        x = x.strip()
        if not x:
            continue
        states.append(x.zfill(2))
    unknown = [s for s in states if s not in STATES]
    if unknown:
        raise SystemExit(f"unknown statefp(s): {unknown}")
    return states


def main() -> int:
    ap = argparse.ArgumentParser(prog="build_full_us_spatial_inputs")
    ap.add_argument("--repo_root", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--run_dir", required=True, type=pathlib.Path)
    ap.add_argument("--base_data_dir", type=pathlib.Path, default=None)
    ap.add_argument("--lodes_root", type=pathlib.Path, default=None)
    ap.add_argument("--states", default="all")
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--census_api_key", default=os.environ.get("CENSUS_API_KEY", ""))
    ap.add_argument("--state_workers", type=int, default=4)
    ap.add_argument("--county_workers", type=int, default=8)
    ap.add_argument("--include_age_sex_cross", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = pathlib.Path(os.environ.get("SYNTHETIC_CITY_DATA_ROOT", repo_root / "data")).expanduser().resolve()
    base_data_dir = (
        args.base_data_dir.expanduser().resolve()
        if args.base_data_dir is not None
        else data_root / "us" / "processed" / "spatial_inputs_2023"
    )
    lodes_root = args.lodes_root.expanduser().resolve() if args.lodes_root is not None else data_root / "lodes"
    run_dir = _ensure_dir(args.run_dir.expanduser().resolve())
    metrics_dir = _ensure_dir(run_dir / "metrics")
    states = _parse_states(str(args.states))
    _write_json(
        run_dir / "run_summary.json",
        {
            "label": "full_us_spatial_inputs_2023",
            "created_utc": _utc_now(),
            "run_dir": str(run_dir),
            "repo_root": str(repo_root),
            "base_data_dir": str(base_data_dir),
            "lodes_root": str(lodes_root),
            "states_requested": states,
            "acs_year": int(args.acs_year),
            "include_age_sex_cross": bool(args.include_age_sex_cross),
            "state_workers": int(args.state_workers),
            "county_workers": int(args.county_workers),
            "status": "running",
        },
    )

    rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.state_workers))) as ex:
        futs = {
            ex.submit(
                _build_state,
                statefp=s,
                base_data_dir=base_data_dir,
                lodes_root=lodes_root,
                run_dir=run_dir,
                repo_root=repo_root,
                acs_year=int(args.acs_year),
                census_api_key=str(args.census_api_key),
                overwrite=bool(args.overwrite),
                county_workers=int(args.county_workers),
                include_age_sex_cross=bool(args.include_age_sex_cross),
            ): s
            for s in states
        }
        for fut in cf.as_completed(futs):
            row = fut.result()
            rows.append(row)
            pd.DataFrame(rows).sort_values("statefp").to_csv(metrics_dir / "state_asset_inventory.csv", index=False)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    ready = [r for r in rows if r.get("status") == "ready"]
    failed = [r for r in rows if r.get("status") != "ready"]
    pd.DataFrame(rows).sort_values("statefp").to_csv(metrics_dir / "state_asset_inventory.csv", index=False)
    if failed:
        pd.DataFrame(failed).sort_values("statefp").to_csv(metrics_dir / "state_asset_failure_summary.csv", index=False)
    payload = {
        "label": "full_us_spatial_inputs_2023",
        "created_utc": _utc_now(),
        "run_dir": str(run_dir),
        "base_data_dir": str(base_data_dir),
        "lodes_root": str(lodes_root),
        "states_requested": states,
        "include_age_sex_cross": bool(args.include_age_sex_cross),
        "states_ready": int(len(ready)),
        "states_failed": int(len(failed)),
        "failed_statefps": [str(r.get("statefp")) for r in failed],
        "status": "completed" if not failed else "completed_with_failures",
        "inventory_csv": str(metrics_dir / "state_asset_inventory.csv"),
    }
    _write_json(run_dir / "run_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
