#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.experimental.data.build_lodes_functional_latent import main as build_role_main


STATE_FIPS_TO_POSTAL = {
    "01": "al",
    "02": "ak",
    "04": "az",
    "05": "ar",
    "06": "ca",
    "08": "co",
    "09": "ct",
    "10": "de",
    "12": "fl",
    "13": "ga",
    "15": "hi",
    "16": "id",
    "17": "il",
    "18": "in",
    "19": "ia",
    "20": "ks",
    "21": "ky",
    "22": "la",
    "23": "me",
    "24": "md",
    "25": "ma",
    "26": "mi",
    "27": "mn",
    "28": "ms",
    "29": "mo",
    "30": "mt",
    "31": "ne",
    "32": "nv",
    "33": "nh",
    "34": "nj",
    "35": "nm",
    "36": "ny",
    "37": "nc",
    "38": "nd",
    "39": "oh",
    "40": "ok",
    "41": "or",
    "42": "pa",
    "44": "ri",
    "45": "sc",
    "46": "sd",
    "47": "tn",
    "48": "tx",
    "49": "ut",
    "50": "vt",
    "51": "va",
    "53": "wa",
    "54": "wv",
    "55": "wi",
    "56": "wy",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digits(v: object) -> str:
    return "".join(ch for ch in str(v).strip() if ch.isdigit())


def _canon_statefp(v: object) -> str:
    d = _digits(v)
    return str(int(d)).zfill(2) if d else ""


def _canon_puma5(v: object) -> str:
    d = _digits(v)
    return str(int(d)).zfill(5) if d else ""


def _canon_tract(v: object) -> str:
    d = _digits(v)
    return d[:11] if len(d) >= 11 else ""


def _puma_uid(statefp: object, puma: object) -> str:
    s = _canon_statefp(statefp)
    p = _canon_puma5(puma)
    return s + p if s and p else ""


def _read_target_states_and_uids(target_wide_csv: pathlib.Path) -> tuple[list[str], set[str]]:
    header = pd.read_csv(target_wide_csv, nrows=0)
    usecols = [c for c in ["statefp", "puma", "puma5", "puma_uid", "puma_uid_key"] if c in header.columns]
    df = pd.read_csv(target_wide_csv, usecols=usecols, dtype=str, low_memory=False)
    if "statefp" not in df.columns:
        raise SystemExit("target_wide_csv must contain statefp")
    states = sorted(df["statefp"].map(_canon_statefp).dropna().unique().tolist())
    if "puma_uid_key" in df.columns:
        uids = set(df["puma_uid_key"].map(lambda x: _digits(x)[-7:].zfill(7) if _digits(x) else "").tolist())
    elif "puma_uid" in df.columns:
        uids = set(df["puma_uid"].map(lambda x: _digits(x)[-7:].zfill(7) if _digits(x) else "").tolist())
    elif "puma5" in df.columns:
        uids = set(df.apply(lambda r: _puma_uid(r["statefp"], r["puma5"]), axis=1).tolist())
    elif "puma" in df.columns:
        uids = set(df.apply(lambda r: _puma_uid(r["statefp"], r["puma"]), axis=1).tolist())
    else:
        raise SystemExit("target_wide_csv must contain puma_uid/puma_uid_key or puma/puma5")
    uids = {u for u in uids if u}
    return states, uids


def _url_for_lodes(state_postal: str, year: int, part: str) -> str:
    st = str(state_postal).strip().lower()
    return f"http://lehd.ces.census.gov/data/lodes/LODES8/{st}/od/{st}_od_{part}_JT00_{int(year)}.csv.gz"


def _is_valid_gzip(path: pathlib.Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with gzip.open(path, "rb") as f:
            f.read(1)
        return True
    except Exception:
        return False


def _download_one(url: str, dest: pathlib.Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _is_valid_gzip(dest):
        return {"path": str(dest), "status": "exists", "bytes": int(dest.stat().st_size)}
    if dest.exists() and not _is_valid_gzip(dest):
        backup = dest.with_suffix(dest.suffix + f".invalid_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}")
        dest.rename(backup)
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    rc = 1
    for attempt in range(1, 6):
        if part.exists():
            part.unlink()
        cmd = ["curl", "--silent", "--show-error", "-L", "--fail", "-o", str(part), url]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and _is_valid_gzip(part):
            part.replace(dest)
            break
        if attempt < 5:
            time.sleep(2 * attempt)
    if rc != 0 or not _is_valid_gzip(dest):
        return {"path": str(dest), "status": "failed", "url": url, "returncode": int(rc)}
    return {"path": str(dest), "status": "downloaded", "bytes": int(dest.stat().st_size)}


def _ensure_lodes_files(
    *,
    states: list[str],
    year: int,
    raw_dir: pathlib.Path,
    workers: int,
    skip_download: bool,
    allow_missing: bool,
) -> tuple[list[pathlib.Path], list[dict[str, Any]]]:
    jobs: list[tuple[str, pathlib.Path]] = []
    for statefp in states:
        st = STATE_FIPS_TO_POSTAL.get(statefp)
        if not st:
            continue
        for part in ["main", "aux"]:
            dest = raw_dir / st / f"{st}_od_{part}_JT00_{int(year)}.csv.gz"
            jobs.append((_url_for_lodes(st, int(year), part), dest))

    records: list[dict[str, Any]] = []
    if skip_download:
        for url, dest in jobs:
            ok = _is_valid_gzip(dest)
            records.append({"path": str(dest), "status": "exists" if ok else "missing", "url": url})
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = [ex.submit(_download_one, url, dest) for url, dest in jobs]
            for fut in as_completed(futs):
                records.append(fut.result())
                rec = records[-1]
                print(f"[download] {rec.get('status')} {rec.get('path')}", flush=True)

    failed = [r for r in records if r.get("status") in {"failed", "missing"}]
    if failed and not allow_missing:
        raise SystemExit(f"{len(failed)} LODES files are unavailable. First failure={failed[0]}")
    return [pathlib.Path(r["path"]) for r in records if r.get("status") not in {"failed", "missing"}], records


def _is_valid_zip(path: pathlib.Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
        return bad is None
    except Exception:
        return False


def _ensure_state_tract_zip(*, statefp: str, tract_dir: pathlib.Path) -> pathlib.Path:
    statefp = _canon_statefp(statefp)
    tract_dir.mkdir(parents=True, exist_ok=True)
    tract_zip = tract_dir / f"cb_2020_{statefp}_tract_500k.zip"
    if _is_valid_zip(tract_zip):
        return tract_zip
    if tract_zip.exists() and not _is_valid_zip(tract_zip):
        backup = tract_zip.with_suffix(tract_zip.suffix + f".invalid_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}")
        tract_zip.rename(backup)
    url = f"https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_{statefp}_tract_500k.zip"
    part = tract_zip.with_suffix(tract_zip.suffix + ".part")
    if part.exists():
        part.unlink()
    cmd = ["curl", "--silent", "--show-error", "-L", "--fail", "-o", str(part), url]
    rc = subprocess.run(cmd).returncode
    if rc == 0 and _is_valid_zip(part):
        part.replace(tract_zip)
    if rc != 0 or not _is_valid_zip(tract_zip):
        raise SystemExit(f"Failed to download tract ZIP: {url}")
    return tract_zip


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def _build_or_load_mapping(
    *,
    mapping_csv: pathlib.Path,
    tract_dir: pathlib.Path,
    puma_path: pathlib.Path,
    states: list[str],
    target_uids: set[str],
) -> pd.DataFrame:
    if mapping_csv.exists():
        mapping = pd.read_csv(mapping_csv, dtype=str)
        if {"tract_geoid", "puma", "statefp", "puma_uid"} <= set(mapping.columns):
            return mapping

    import geopandas as gpd

    tract_frames = []
    for statefp in states:
        tract_zip = _ensure_state_tract_zip(statefp=statefp, tract_dir=tract_dir)
        print(f"[mapping] loading tracts: {tract_zip}", flush=True)
        tract_frames.append(gpd.read_file(f"zip://{tract_zip}"))
    if not tract_frames:
        raise SystemExit("No tract shapefiles loaded.")
    tracts = pd.concat(tract_frames, ignore_index=True)
    tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs=tract_frames[0].crs)
    print(f"[mapping] loading pumas: {puma_path}", flush=True)
    pumas = gpd.read_file(str(puma_path))

    tract_geoid_col = _find_col(list(tracts.columns), ["GEOID", "GEOID20"])
    tract_state_col = _find_col(list(tracts.columns), ["STATEFP", "STATEFP20"])
    puma_col = _find_col(list(pumas.columns), ["PUMACE20", "PUMACE", "PUMA", "GEOID20", "GEOID"])
    puma_state_col = _find_col(list(pumas.columns), ["STATEFP20", "STATEFP"])
    if tract_geoid_col is None or tract_state_col is None or puma_col is None or puma_state_col is None:
        raise SystemExit("Cannot identify required tract/PUMA columns for national mapping.")

    state_set = set(states)
    tracts = tracts[tracts[tract_state_col].astype(str).str.zfill(2).isin(state_set)].copy()
    pumas = pumas[pumas[puma_state_col].astype(str).str.zfill(2).isin(state_set)].copy()
    pumas["statefp_norm"] = pumas[puma_state_col].astype(str).str.zfill(2)
    pumas["puma_norm"] = pumas[puma_col].astype(str).str.replace(r"\\.0$", "", regex=True).str.zfill(5)
    pumas["puma_uid"] = pumas["statefp_norm"] + pumas["puma_norm"]
    pumas = pumas[pumas["puma_uid"].isin(target_uids)].copy()

    tracts_pts = tracts[[tract_geoid_col, tract_state_col, "geometry"]].copy()
    tracts_pts["geometry"] = tracts_pts.geometry.representative_point()
    if tracts_pts.crs != pumas.crs:
        pumas = pumas.to_crs(tracts_pts.crs)

    print(f"[mapping] spatial join: {len(tracts_pts)} tracts x {len(pumas)} target PUMAs", flush=True)
    joined = gpd.sjoin(
        tracts_pts,
        pumas[["puma_norm", "puma_uid", "geometry"]],
        how="left",
        predicate="within",
    )
    out = pd.DataFrame(
        {
            "tract_geoid": joined[tract_geoid_col].map(_canon_tract),
            "statefp": joined[tract_state_col].map(_canon_statefp),
            "puma": joined["puma_norm"].map(_canon_puma5),
            "puma_uid": joined["puma_uid"].astype(str),
        }
    )
    out = out[(out["tract_geoid"] != "") & (out["puma_uid"].isin(target_uids))].copy()
    out = out.drop_duplicates("tract_geoid", keep="first").sort_values("tract_geoid").reset_index(drop=True)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(mapping_csv, index=False)
    print(f"[mapping] wrote {len(out)} rows, pumas={out['puma_uid'].nunique()} -> {mapping_csv}", flush=True)
    return out


def _aggregate_state_files(
    *,
    statefp: str,
    lodes_paths: list[pathlib.Path],
    tract_to_puma: dict[str, str],
    out_csv: pathlib.Path,
    chunksize: int,
    count_col: str,
) -> dict[str, Any]:
    if out_csv.exists():
        df = pd.read_csv(out_csv, dtype={"home_puma_uid": str, "work_puma_uid": str})
        return {
            "statefp": statefp,
            "out_csv": str(out_csv),
            "status": "exists",
            "n_edges": int(df.shape[0]),
            "od_count": float(pd.to_numeric(df["od_count"], errors="coerce").fillna(0.0).sum()),
        }

    frames: list[pd.DataFrame] = []
    raw_rows = 0
    mapped_rows = 0
    for path in lodes_paths:
        for chunk in pd.read_csv(
            path,
            usecols=["w_geocode", "h_geocode", count_col],
            dtype={"w_geocode": str, "h_geocode": str},
            chunksize=int(chunksize),
            compression="gzip" if path.suffix == ".gz" else "infer",
            low_memory=False,
        ):
            raw_rows += int(chunk.shape[0])
            chunk[count_col] = pd.to_numeric(chunk[count_col], errors="coerce").fillna(0.0)
            chunk = chunk[chunk[count_col] > 0].copy()
            if chunk.empty:
                continue
            chunk["home_tract_geoid"] = chunk["h_geocode"].map(_canon_tract)
            chunk["work_tract_geoid"] = chunk["w_geocode"].map(_canon_tract)
            chunk["home_puma_uid"] = chunk["home_tract_geoid"].map(tract_to_puma)
            chunk["work_puma_uid"] = chunk["work_tract_geoid"].map(tract_to_puma)
            chunk = chunk.dropna(subset=["home_puma_uid", "work_puma_uid"]).copy()
            mapped_rows += int(chunk.shape[0])
            if chunk.empty:
                continue
            frames.append(
                chunk.groupby(["home_puma_uid", "work_puma_uid"], as_index=False, sort=False)[count_col].sum()
            )

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.groupby(["home_puma_uid", "work_puma_uid"], as_index=False, sort=False)[count_col].sum()
        out = out.rename(columns={count_col: "od_count"})
        out["origin_total"] = out.groupby("home_puma_uid")["od_count"].transform("sum")
        out["dest_total"] = out.groupby("work_puma_uid")["od_count"].transform("sum")
        out["origin_share"] = out["od_count"] / np.clip(out["origin_total"], 1e-12, None)
        out["destination_share"] = out["od_count"] / np.clip(out["dest_total"], 1e-12, None)
        out["log1p_count"] = np.log1p(out["od_count"].to_numpy(dtype=float))
        out = out.drop(columns=["origin_total", "dest_total"])
        out = out.sort_values(["home_puma_uid", "origin_share"], ascending=[True, False]).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["home_puma_uid", "work_puma_uid", "od_count", "origin_share", "destination_share", "log1p_count"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return {
        "statefp": statefp,
        "out_csv": str(out_csv),
        "status": "built",
        "raw_rows": int(raw_rows),
        "mapped_rows": int(mapped_rows),
        "n_edges": int(out.shape[0]),
        "od_count": float(out["od_count"].sum()) if not out.empty else 0.0,
    }


def _build_national_edges(
    *,
    states: list[str],
    raw_dir: pathlib.Path,
    state_dir: pathlib.Path,
    out_csv: pathlib.Path,
    tract_to_puma: dict[str, str],
    year: int,
    chunksize: int,
    count_col: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    state_files: list[pathlib.Path] = []
    for statefp in states:
        st = STATE_FIPS_TO_POSTAL[statefp]
        paths = [
            raw_dir / st / f"{st}_od_main_JT00_{int(year)}.csv.gz",
            raw_dir / st / f"{st}_od_aux_JT00_{int(year)}.csv.gz",
        ]
        paths = [p for p in paths if _is_valid_gzip(p)]
        if not paths:
            state_out = state_dir / st / f"puma_od_{st}_{int(year)}.csv"
            state_out.parent.mkdir(parents=True, exist_ok=True)
            empty = pd.DataFrame(columns=["home_puma_uid", "work_puma_uid", "od_count", "origin_share", "destination_share", "log1p_count"])
            empty.to_csv(state_out, index=False)
            rec = {
                "statefp": statefp,
                "out_csv": str(state_out),
                "status": "missing_lodes",
                "n_edges": 0,
                "od_count": 0.0,
            }
            records.append(rec)
            state_files.append(state_out)
            print(f"[state] {st} missing_lodes edges=0 od=0", flush=True)
            continue
        state_out = state_dir / st / f"puma_od_{st}_{int(year)}.csv"
        rec = _aggregate_state_files(
            statefp=statefp,
            lodes_paths=paths,
            tract_to_puma=tract_to_puma,
            out_csv=state_out,
            chunksize=int(chunksize),
            count_col=count_col,
        )
        records.append(rec)
        state_files.append(state_out)
        print(f"[state] {st} {rec['status']} edges={rec['n_edges']} od={rec['od_count']:.0f}", flush=True)

    if out_csv.exists():
        directed = pd.read_csv(out_csv, dtype={"home_puma_uid": str, "work_puma_uid": str})
        return directed, records

    frames = [pd.read_csv(p, dtype={"home_puma_uid": str, "work_puma_uid": str}) for p in state_files if p.exists()]
    directed = pd.concat(frames, ignore_index=True)
    directed["od_count"] = pd.to_numeric(directed["od_count"], errors="coerce").fillna(0.0)
    directed = directed.groupby(["home_puma_uid", "work_puma_uid"], as_index=False, sort=False)["od_count"].sum()
    origin_total = directed.groupby("home_puma_uid", sort=False)["od_count"].transform("sum")
    dest_total = directed.groupby("work_puma_uid", sort=False)["od_count"].transform("sum")
    directed["origin_share"] = directed["od_count"] / np.clip(origin_total, 1e-12, None)
    directed["destination_share"] = directed["od_count"] / np.clip(dest_total, 1e-12, None)
    directed["log1p_count"] = np.log1p(directed["od_count"].to_numpy(dtype=float))
    directed = directed.sort_values(["home_puma_uid", "origin_share"], ascending=[True, False]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    directed.to_csv(out_csv, index=False)
    return directed, records


def _write_json(path: pathlib.Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(prog="build_national_lodes_functional_role")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, required=True)
    ap.add_argument("--out_dir", type=pathlib.Path, required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--raw_dir", type=pathlib.Path, default=pathlib.Path("/home/jinlin/data/geoexplicit_data/lodes/lodes8_od_puma_functional_2020_v0/raw"))
    ap.add_argument("--tract_dir", type=pathlib.Path, default=pathlib.Path("/home/jinlin/data/geoexplicit_data/lodes/geo/tract2020_500k"))
    ap.add_argument("--puma_path", type=pathlib.Path, default=pathlib.Path("data/geo_cache/cb_2020_us_puma20_500k/cb_2020_us_puma20_500k.shp"))
    ap.add_argument("--mapping_csv", type=pathlib.Path, default=None)
    ap.add_argument("--states", default="target", help="Comma-separated FIPS; default 'target' uses states in target_wide_csv.")
    ap.add_argument("--download_workers", type=int, default=6)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--count_col", default="S000")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--allow_missing_lodes", action="store_true")
    args = ap.parse_args()

    target_csv = args.target_wide_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    tract_dir = args.tract_dir.expanduser().resolve()
    puma_path = args.puma_path.expanduser().resolve()
    mapping_csv = (
        args.mapping_csv.expanduser().resolve()
        if args.mapping_csv is not None
        else out_dir / "inputs" / "tract_to_puma2020_national_target_states.csv"
    )

    target_states, target_uids = _read_target_states_and_uids(target_csv)
    if str(args.states).strip().lower() == "target":
        states = target_states
    else:
        states = sorted({_canon_statefp(x) for x in str(args.states).split(",") if _canon_statefp(x)})
    missing_state_map = [s for s in states if s not in STATE_FIPS_TO_POSTAL]
    if missing_state_map:
        raise SystemExit(f"Unsupported state FIPS for LODES download: {missing_state_map}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "run_config.json",
        {
            "created_utc": _utc_now(),
            "target_wide_csv": str(target_csv),
            "states": states,
            "year": int(args.year),
            "raw_dir": str(raw_dir),
            "tract_dir": str(tract_dir),
            "puma_path": str(puma_path),
            "mapping_csv": str(mapping_csv),
            "count_col": str(args.count_col),
            "chunksize": int(args.chunksize),
        },
    )

    _, download_records = _ensure_lodes_files(
        states=states,
        year=int(args.year),
        raw_dir=raw_dir,
        workers=int(args.download_workers),
        skip_download=bool(args.skip_download),
        allow_missing=bool(args.allow_missing_lodes),
    )
    _write_json(out_dir / "metrics" / "download_manifest.json", {"records": download_records})

    mapping = _build_or_load_mapping(
        mapping_csv=mapping_csv,
        tract_dir=tract_dir,
        puma_path=puma_path,
        states=states,
        target_uids=target_uids,
    )
    tract_to_puma = dict(zip(mapping["tract_geoid"].astype(str), mapping["puma_uid"].astype(str), strict=False))

    directed_path = out_dir / "puma_lodes_directed_od_edges.csv"
    directed, state_records = _build_national_edges(
        states=states,
        raw_dir=raw_dir,
        state_dir=out_dir / "states",
        out_csv=directed_path,
        tract_to_puma=tract_to_puma,
        year=int(args.year),
        chunksize=int(args.chunksize),
        count_col=str(args.count_col),
    )

    role_csv = out_dir / "puma_lodes_functional_role_summary_national.csv"
    old_argv = sys.argv
    try:
        sys.argv = [
            "build_lodes_functional_latent.py",
            "--target_wide_csv",
            str(target_csv),
            "--directed_edges_csv",
            str(directed_path),
            "--n_components",
            "0",
            "--out_csv",
            str(role_csv),
        ]
        build_role_main()
    finally:
        sys.argv = old_argv

    summary = {
        "created_utc": _utc_now(),
        "target_wide_csv": str(target_csv),
        "states": states,
        "n_target_pumas": int(len(target_uids)),
        "n_mapping_tracts": int(mapping["tract_geoid"].nunique()),
        "n_mapping_pumas": int(mapping["puma_uid"].nunique()),
        "n_puma_od_edges": int(directed.shape[0]),
        "n_home_pumas": int(directed["home_puma_uid"].nunique()),
        "n_work_pumas": int(directed["work_puma_uid"].nunique()),
        "total_od_count": float(pd.to_numeric(directed["od_count"], errors="coerce").fillna(0.0).sum()),
        "directed_edges_csv": str(directed_path),
        "functional_role_csv": str(role_csv),
        "state_records": state_records,
    }
    _write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
