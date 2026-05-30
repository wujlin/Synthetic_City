#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import pathlib
import sys
import time
import zipfile
from typing import Any

import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.build_full_us_spatial_inputs import STATES  # noqa: E402


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 SyntheticCity-TIGER-cache/1.0",
            "Referer": "https://www.census.gov/",
            "Accept": "application/zip,application/octet-stream,*/*",
        }
    )
    return s


def _is_valid_zip(path: pathlib.Path) -> bool:
    if (not path.exists()) or path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def _candidate_urls(url: str) -> list[str]:
    # Census occasionally returns a 200 HTML "Request Rejected" page for
    # direct TIGER zip URLs. The query variant currently bypasses that edge
    # case while serving the same file.
    return [url, f"{url}?download=1"] if "?" not in url else [url]


def _download(url: str, dest: pathlib.Path, *, overwrite: bool = False, max_attempts: int = 8) -> pathlib.Path:
    if dest.exists() and _is_valid_zip(dest) and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, int(max_attempts) + 1):
        for candidate_url in _candidate_urls(url):
            try:
                sess = _session()
                with sess.get(candidate_url, stream=True, timeout=180) as resp:
                    if resp.status_code in {403, 429, 500, 502, 503, 504}:
                        raise requests.HTTPError(f"{resp.status_code} response for {candidate_url}", response=resp)
                    resp.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if not _is_valid_zip(tmp):
                    raise RuntimeError(f"downloaded file is not a valid zip: {candidate_url}")
                tmp.replace(dest)
                return dest
            except Exception as e:
                last_error = e
        sleep_s = min(120.0, 4.0 * attempt * attempt)
        time.sleep(sleep_s)
    raise RuntimeError(f"download failed after {max_attempts} attempts: {url}; last_error={last_error}")


def _parse_states(value: str) -> list[str]:
    if value.strip().lower() in {"all", "all50", "all51"}:
        return sorted(STATES)
    out = [x.strip().zfill(2) for x in value.split(",") if x.strip()]
    unknown = [x for x in out if x not in STATES]
    if unknown:
        raise SystemExit(f"unknown statefp(s): {unknown}")
    return out


def _state_cache(*, statefp: str, root: pathlib.Path, county_workers: int, overwrite: bool) -> dict[str, Any]:
    statefp = statefp.zfill(2)
    state_postal, state_name = STATES[statefp]
    state_dir = root / f"state={statefp}" / "raw"
    geo_dir = state_dir / "geo"
    roads_dir = state_dir / "roads"
    row: dict[str, Any] = {
        "statefp": statefp,
        "state_postal": state_postal,
        "state_name": state_name,
        "status": "started",
        "error": "",
        "created_utc": _utc_now(),
    }
    try:
        tract = _download(
            f"https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_{statefp}_tract.zip",
            geo_dir / f"tl_2023_{statefp}_tract.zip",
            overwrite=overwrite,
        )
        puma = _download(
            f"https://www2.census.gov/geo/tiger/TIGER2023/PUMA/tl_2023_{statefp}_puma20.zip",
            geo_dir / f"tl_2023_{statefp}_puma20.zip",
            overwrite=overwrite,
        )
        tracts = gpd.read_file(f"zip://{tract}")
        county_col = "COUNTYFP" if "COUNTYFP" in tracts.columns else "COUNTYFP20"
        countyfps = sorted(tracts[county_col].astype(str).str.zfill(3).unique().tolist())

        def county_road(countyfp: str) -> pathlib.Path:
            geoid = f"{statefp}{countyfp}"
            return _download(
                f"https://www2.census.gov/geo/tiger/TIGER2023/ROADS/tl_2023_{geoid}_roads.zip",
                roads_dir / f"tl_2023_{geoid}_roads.zip",
                overwrite=overwrite,
            )

        failures: list[str] = []
        done = 0
        with cf.ThreadPoolExecutor(max_workers=max(1, int(county_workers))) as ex:
            futs = {ex.submit(county_road, c): c for c in countyfps}
            for fut in cf.as_completed(futs):
                countyfp = futs[fut]
                try:
                    fut.result()
                    done += 1
                except Exception as e:
                    failures.append(f"{statefp}{countyfp}:{e}")
        if failures:
            raise RuntimeError("; ".join(failures[:20]))
        row.update(
            {
                "status": "ready",
                "tract_zip": str(tract),
                "puma_zip": str(puma),
                "n_tracts": int(len(tracts)),
                "n_counties": int(len(countyfps)),
                "n_road_zips": int(done),
                "finished_utc": _utc_now(),
            }
        )
    except Exception as e:
        row.update({"status": "failed", "error": str(e), "finished_utc": _utc_now()})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(prog="download_tiger2023_cache")
    ap.add_argument("--cache_root", type=pathlib.Path, required=True)
    ap.add_argument("--states", default="all")
    ap.add_argument("--state_workers", type=int, default=4)
    ap.add_argument("--county_workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.cache_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    states = _parse_states(str(args.states))
    rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.state_workers))) as ex:
        futs = {
            ex.submit(
                _state_cache,
                statefp=s,
                root=root,
                county_workers=int(args.county_workers),
                overwrite=bool(args.overwrite),
            ): s
            for s in states
        }
        for fut in cf.as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with (root / "tiger2023_cache_inventory.csv").open("w", encoding="utf-8", newline="") as f:
                fields = sorted({k for r in rows for k in r})
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        "created_utc": _utc_now(),
        "cache_root": str(root),
        "states_requested": states,
        "states_ready": sum(1 for r in rows if r.get("status") == "ready"),
        "states_failed": sum(1 for r in rows if r.get("status") != "ready"),
        "failed_statefps": [r["statefp"] for r in rows if r.get("status") != "ready"],
    }
    (root / "tiger2023_cache_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["states_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
