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
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

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


def _states_from_target(path: pathlib.Path) -> list[str]:
    df = pd.read_csv(path, usecols=["statefp"], dtype=str, low_memory=False)
    return sorted(df["statefp"].map(_canon_statefp).dropna().unique().tolist())


def _is_valid_gzip(path: pathlib.Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with gzip.open(path, "rb") as f:
            f.read(1)
        return True
    except Exception:
        return False


def _is_valid_zip(path: pathlib.Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except Exception:
        return False


def _download(url: str, dest: pathlib.Path, kind: str, proxy: str | None = None) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    valid = _is_valid_gzip if kind == "gzip" else _is_valid_zip
    if valid(dest):
        return {"status": "exists", "url": url, "path": str(dest), "bytes": int(dest.stat().st_size)}
    if dest.exists() and not valid(dest):
        backup = dest.with_suffix(dest.suffix + f".invalid_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}")
        dest.rename(backup)

    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()

    rc = 1
    status = "failed"
    for attempt in range(1, 6):
        if part.exists():
            part.unlink()
        cmd = ["curl", "--silent", "--show-error", "-L", "--fail", "-o", str(part), url]
        if proxy:
            cmd[1:1] = ["--proxy", proxy]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and valid(part):
            part.replace(dest)
            status = "downloaded"
            break
        if attempt < 5:
            time.sleep(2 * attempt)
    return {
        "status": status,
        "url": url,
        "path": str(dest),
        "bytes": int(dest.stat().st_size) if dest.exists() else 0,
        "returncode": int(rc),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="download_lodes_functional_assets")
    ap.add_argument("--target_wide_csv", type=pathlib.Path)
    ap.add_argument("--raw_dir", type=pathlib.Path, required=True)
    ap.add_argument("--tract_dir", type=pathlib.Path, required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--states", default="target", help="'target', 'all50', or comma-separated state FIPS codes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--proxy", default="", help="Optional curl proxy, e.g. http://127.0.0.1:7890")
    ap.add_argument("--allow_missing", action="store_true", help="Finish successfully even if some assets are unavailable.")
    ap.add_argument("--manifest_json", type=pathlib.Path, required=True)
    args = ap.parse_args()

    states_arg = str(args.states).strip().lower()
    if states_arg == "target":
        if args.target_wide_csv is None:
            raise SystemExit("--target_wide_csv is required when --states target")
        states = _states_from_target(args.target_wide_csv.expanduser().resolve())
    elif states_arg in {"all", "all50", "us50"}:
        states = sorted(STATE_FIPS_TO_POSTAL)
    else:
        states = sorted({_canon_statefp(x) for x in str(args.states).split(",") if _canon_statefp(x)})
    missing = [s for s in states if s not in STATE_FIPS_TO_POSTAL]
    if missing:
        raise SystemExit(f"Unsupported states: {missing}")

    raw_dir = args.raw_dir.expanduser().resolve()
    tract_dir = args.tract_dir.expanduser().resolve()
    proxy = str(args.proxy).strip() or None
    jobs: list[tuple[str, pathlib.Path, str]] = []
    for statefp in states:
        st = STATE_FIPS_TO_POSTAL[statefp]
        for part in ["main", "aux"]:
            url = f"http://lehd.ces.census.gov/data/lodes/LODES8/{st}/od/{st}_od_{part}_JT00_{int(args.year)}.csv.gz"
            dest = raw_dir / st / f"{st}_od_{part}_JT00_{int(args.year)}.csv.gz"
            jobs.append((url, dest, "gzip"))
        tract_url = f"https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_{statefp}_tract_500k.zip"
        tract_dest = tract_dir / f"cb_2020_{statefp}_tract_500k.zip"
        jobs.append((tract_url, tract_dest, "zip"))

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_download, url, dest, kind, proxy) for url, dest, kind in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            print(f"[{rec['status']}] {rec['bytes']} {rec['path']}", flush=True)

    summary = {
        "created_utc": _utc_now(),
        "target_wide_csv": str(args.target_wide_csv) if args.target_wide_csv else None,
        "states": states,
        "year": int(args.year),
        "proxy": proxy,
        "raw_dir": str(raw_dir),
        "tract_dir": str(tract_dir),
        "n_jobs": len(jobs),
        "n_failed": int(sum(1 for r in records if r["status"] == "failed")),
        "records": sorted(records, key=lambda r: r["path"]),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))
    return 0 if args.allow_missing or not summary["n_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
