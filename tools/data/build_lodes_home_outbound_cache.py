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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.data.lodes import (
    aggregate_lodes_to_tract_od,
    build_tract_area_crosswalk,
    load_lodes_od,
    remap_tract_od_geoids,
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canon_geoid_text(s: pd.Series, *, width: int) -> pd.Series:
    out = s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    missing = out.isna() | out.str.lower().isin({"", "nan", "none", "<na>"})
    numeric = out.str.fullmatch(r"\d+").fillna(False)
    out.loc[numeric] = out.loc[numeric].str.zfill(int(width))
    out.loc[missing] = pd.NA
    return out


def _cache_path(cache_dir: pathlib.Path, statefp: str) -> pathlib.Path:
    return cache_dir / f"home_statefp={str(statefp).zfill(2)}" / "tract_od.parquet"


def _read_geodata(path: pathlib.Path):
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("build_lodes_home_outbound_cache requires geopandas.") from e
    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _download_tiger_tract_zip(*, statefp: str, year: int, out_path: pathlib.Path) -> pathlib.Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    statefp = str(statefp).zfill(2)
    urls = [
        f"https://www2.census.gov/geo/tiger/TIGER{int(year)}/TRACT/tl_{int(year)}_{statefp}_tract.zip",
    ]
    if int(year) == 2020 and statefp == "09":
        urls.extend(
            [
                "https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TRACT/2020/tl_2020_09_tract20.zip",
                "https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/09_CONNECTICUT/09/tl_2020_09_tract20.zip",
            ]
        )
    errors: list[str] = []
    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": "Synthetic-City research data preparation"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, out_path.open("wb") as f:
                shutil.copyfileobj(resp, f)
            return out_path
        except Exception as e:
            if out_path.exists():
                out_path.unlink()
            errors.append(f"{url}: {e}")
    raise SystemExit("failed to download TIGER tract zip:\n" + "\n".join(errors))


def _load_asset_inventory(path: pathlib.Path) -> pd.DataFrame:
    inv = pd.read_csv(path, dtype={"statefp": str}, low_memory=False)
    inv["statefp"] = inv["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    if "status" in inv.columns:
        inv = inv[inv["status"].astype(str).str.lower() == "ready"].copy()
    return inv


def _selected_home_statefps(text: str, inv: pd.DataFrame) -> list[str]:
    value = str(text).strip().lower()
    if value in {"", "ready", "all"}:
        return sorted(inv["statefp"].astype(str).str.zfill(2).unique().tolist())
    return sorted({part.strip().zfill(2) for part in str(text).split(",") if part.strip()})


def _build_ct_crosswalk(
    *,
    inv: pd.DataFrame,
    cache_dir: pathlib.Path,
    legacy_tract_path: pathlib.Path | None,
    legacy_tract_year: int,
    disable: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    meta: dict[str, Any] = {"enabled": False, "reason": "disabled" if disable else "not_applicable"}
    if disable:
        return None, meta
    ct = inv.loc[inv["statefp"].astype(str).str.zfill(2) == "09"]
    if ct.empty or not str(ct.iloc[0].get("tract_zip", "")).strip():
        meta["reason"] = "missing_current_ct_tract_zip"
        return None, meta
    current_path = pathlib.Path(str(ct.iloc[0]["tract_zip"])).expanduser()
    if not current_path.exists():
        meta["reason"] = "current_ct_tract_zip_not_found"
        meta["current_tract_zip"] = str(current_path)
        return None, meta
    if legacy_tract_path is not None and legacy_tract_path.exists():
        legacy_path = legacy_tract_path.expanduser().resolve()
    else:
        legacy_path = _download_tiger_tract_zip(
            statefp="09",
            year=int(legacy_tract_year),
            out_path=cache_dir / "raw" / f"tl_{int(legacy_tract_year)}_09_tract.zip",
        )
    current = _read_geodata(current_path)
    legacy = _read_geodata(legacy_path)
    cw = build_tract_area_crosswalk(
        legacy_areas=legacy,
        current_areas=current,
        legacy_group_col="GEOID",
        current_group_col="GEOID",
    )
    cw_path = cache_dir / "metrics" / "ct_legacy_to_current_tract_crosswalk.csv"
    cw_path.parent.mkdir(parents=True, exist_ok=True)
    cw.to_csv(cw_path, index=False)
    meta.update(
        {
            "enabled": True,
            "reason": "legacy_tract_to_current_planning_region",
            "legacy_tract_zip": str(legacy_path),
            "current_tract_zip": str(current_path),
            "crosswalk_csv": str(cw_path),
            "n_rows": int(cw.shape[0]),
            "n_legacy_tracts": int(cw["legacy_tract_geoid"].nunique()) if "legacy_tract_geoid" in cw.columns else 0,
            "n_current_tracts": int(cw["tract_geoid"].nunique()) if "tract_geoid" in cw.columns else 0,
        }
    )
    return cw, meta


def _write_work_state_parts(
    *,
    row: dict[str, Any],
    cache_dir: pathlib.Path,
    requested_home_statefps: set[str],
    ct_crosswalk: pd.DataFrame | None,
    overwrite_parts: bool,
) -> dict[str, Any]:
    statefp = str(row.get("statefp", "")).zfill(2)
    main_path = pathlib.Path(str(row.get("lodes_main_path", ""))).expanduser()
    aux_path = pathlib.Path(str(row.get("lodes_aux_path", ""))).expanduser()
    if not main_path.exists() or not aux_path.exists():
        return {"statefp": statefp, "status": "missing_lodes", "parts_written": 0}

    od_block = load_lodes_od(main_path=main_path, aux_path=aux_path)
    od = aggregate_lodes_to_tract_od(od_block)
    if ct_crosswalk is not None and not ct_crosswalk.empty:
        od = remap_tract_od_geoids(od, ct_crosswalk)
    od["home_tract_geoid"] = _canon_geoid_text(od["home_tract_geoid"], width=11)
    od["work_tract_geoid"] = _canon_geoid_text(od["work_tract_geoid"], width=11)
    od = od.dropna(subset=["home_tract_geoid", "work_tract_geoid"]).copy()
    od["home_statefp"] = od["home_tract_geoid"].astype(str).str.slice(0, 2)
    od = od[od["home_statefp"].isin(requested_home_statefps)].copy()

    parts_written = 0
    rows_written = 0
    for home_statefp, grp in od.groupby("home_statefp", sort=False):
        part_path = cache_dir / "_parts" / f"work_statefp={statefp}" / f"home_statefp={str(home_statefp).zfill(2)}.parquet"
        if part_path.exists() and not overwrite_parts:
            continue
        part_path.parent.mkdir(parents=True, exist_ok=True)
        out = grp.drop(columns=["home_statefp"]).reset_index(drop=True)
        out.to_parquet(part_path, index=False)
        parts_written += 1
        rows_written += int(out.shape[0])
    return {
        "statefp": statefp,
        "status": "completed",
        "rows_after_home_filter": int(od.shape[0]),
        "parts_written": int(parts_written),
        "rows_written": int(rows_written),
    }


def _finalize_home_state(
    *,
    cache_dir: pathlib.Path,
    statefp: str,
    overwrite: bool,
) -> dict[str, Any]:
    statefp = str(statefp).zfill(2)
    final_path = _cache_path(cache_dir, statefp)
    if final_path.exists() and not overwrite:
        return {
            "statefp": statefp,
            "status": "skipped_existing",
            "output_path": str(final_path),
            "rows": int(pd.read_parquet(final_path, columns=["home_tract_geoid"]).shape[0]),
            "output_file_size_bytes": int(final_path.stat().st_size),
        }
    part_paths = sorted((cache_dir / "_parts").glob(f"work_statefp=*/home_statefp={statefp}.parquet"))
    if not part_paths:
        return {"statefp": statefp, "status": "missing_parts", "output_path": str(final_path), "rows": 0}
    frames = [pd.read_parquet(p) for p in part_paths]
    out = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in out.columns if c not in {"home_tract_geoid", "work_tract_geoid"}]
    out = (
        out.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values(["home_tract_geoid", "work_tract_geoid"], kind="stable")
        .reset_index(drop=True)
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(final_path, index=False)
    return {
        "statefp": statefp,
        "status": "completed",
        "output_path": str(final_path),
        "rows": int(out.shape[0]),
        "output_file_size_bytes": int(final_path.stat().st_size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="build_lodes_home_outbound_cache")
    ap.add_argument("--asset_inventory_csv", required=True, type=pathlib.Path)
    ap.add_argument("--cache_dir", required=True, type=pathlib.Path)
    ap.add_argument("--home_statefps", default="ready")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--overwrite_parts", action="store_true")
    ap.add_argument("--disable_ct_crosswalk", action="store_true")
    ap.add_argument("--ct_legacy_tract_zip", default="", type=pathlib.Path)
    ap.add_argument("--legacy_tract_year", type=int, default=2020)
    args = ap.parse_args()

    inv = _load_asset_inventory(args.asset_inventory_csv.expanduser().resolve())
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    home_statefps = _selected_home_statefps(str(args.home_statefps), inv)
    requested = set(home_statefps)

    ct_crosswalk, ct_meta = _build_ct_crosswalk(
        inv=inv,
        cache_dir=cache_dir,
        legacy_tract_path=args.ct_legacy_tract_zip.expanduser().resolve() if str(args.ct_legacy_tract_zip).strip() else None,
        legacy_tract_year=int(args.legacy_tract_year),
        disable=bool(args.disable_ct_crosswalk),
    )

    part_rows = []
    for row in inv.sort_values("statefp").to_dict("records"):
        status = _write_work_state_parts(
            row=row,
            cache_dir=cache_dir,
            requested_home_statefps=requested,
            ct_crosswalk=ct_crosswalk,
            overwrite_parts=bool(args.overwrite_parts),
        )
        part_rows.append(status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    final_rows = []
    for statefp in home_statefps:
        status = _finalize_home_state(cache_dir=cache_dir, statefp=statefp, overwrite=bool(args.overwrite))
        final_rows.append(status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    metrics_dir = cache_dir / "metrics"
    pd.DataFrame(part_rows).to_csv(metrics_dir / "work_state_part_status.csv", index=False)
    pd.DataFrame(final_rows).to_csv(metrics_dir / "home_state_cache_status.csv", index=False)
    payload = {
        "created_utc": _utc_now(),
        "asset_inventory_csv": str(args.asset_inventory_csv.expanduser().resolve()),
        "cache_dir": str(cache_dir),
        "home_statefps_requested": home_statefps,
        "ct_crosswalk": ct_meta,
        "work_states_scanned": int(len(part_rows)),
        "home_states_finalized": int(len(final_rows)),
        "home_states_completed_or_existing": int(
            sum(str(r.get("status")) in {"completed", "skipped_existing"} for r in final_rows)
        ),
        "home_states_missing_parts": [str(r.get("statefp")) for r in final_rows if str(r.get("status")) == "missing_parts"],
    }
    _write_json(metrics_dir / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if not payload["home_states_missing_parts"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
