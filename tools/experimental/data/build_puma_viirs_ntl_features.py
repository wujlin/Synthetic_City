#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as exc:
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in the WSA dpl environment.") from exc


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canon_statefp(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return str(int(float(s))).zfill(2) if s.replace(".", "", 1).isdigit() else s.zfill(2)


def _canon_puma5(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return str(int(float(s))).zfill(5) if s.replace(".", "", 1).isdigit() else s.zfill(5)


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _safe_stats(raw: np.ndarray, area_km2: float) -> dict[str, float | int]:
    raw = np.asarray(raw, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return {
            "ntl_pixel_count": 0,
            "ntl_raw_mean": float("nan"),
            "ntl_raw_std": float("nan"),
            "ntl_raw_min": float("nan"),
            "ntl_raw_p50": float("nan"),
            "ntl_raw_p90": float("nan"),
            "ntl_raw_p95": float("nan"),
            "ntl_raw_p99": float("nan"),
            "ntl_raw_max": float("nan"),
            "ntl_pos_mean": float("nan"),
            "ntl_pos_std": float("nan"),
            "ntl_pos_p50": float("nan"),
            "ntl_pos_p90": float("nan"),
            "ntl_pos_p95": float("nan"),
            "ntl_pos_p99": float("nan"),
            "ntl_pos_max": float("nan"),
            "ntl_log1p_pos_mean": float("nan"),
            "ntl_log1p_pos_std": float("nan"),
            "ntl_log1p_pos_p90": float("nan"),
            "ntl_lit_share_gt0": float("nan"),
            "ntl_lit_share_gt5": float("nan"),
            "ntl_lit_share_gt10": float("nan"),
            "ntl_lit_share_gt50": float("nan"),
            "ntl_lit_share_gt100": float("nan"),
            "ntl_pos_sum": float("nan"),
            "ntl_pos_sum_per_km2": float("nan"),
            "ntl_top10_pos_share": float("nan"),
        }

    pos = np.clip(raw, 0.0, None)
    logp = np.log1p(pos)
    sorted_pos = np.sort(pos)
    top_n = max(1, int(math.ceil(0.10 * sorted_pos.size)))
    pos_sum = float(np.sum(pos))
    return {
        "ntl_pixel_count": int(raw.size),
        "ntl_raw_mean": float(np.mean(raw)),
        "ntl_raw_std": float(np.std(raw)),
        "ntl_raw_min": float(np.min(raw)),
        "ntl_raw_p50": float(np.percentile(raw, 50)),
        "ntl_raw_p90": float(np.percentile(raw, 90)),
        "ntl_raw_p95": float(np.percentile(raw, 95)),
        "ntl_raw_p99": float(np.percentile(raw, 99)),
        "ntl_raw_max": float(np.max(raw)),
        "ntl_pos_mean": float(np.mean(pos)),
        "ntl_pos_std": float(np.std(pos)),
        "ntl_pos_p50": float(np.percentile(pos, 50)),
        "ntl_pos_p90": float(np.percentile(pos, 90)),
        "ntl_pos_p95": float(np.percentile(pos, 95)),
        "ntl_pos_p99": float(np.percentile(pos, 99)),
        "ntl_pos_max": float(np.max(pos)),
        "ntl_log1p_pos_mean": float(np.mean(logp)),
        "ntl_log1p_pos_std": float(np.std(logp)),
        "ntl_log1p_pos_p90": float(np.percentile(logp, 90)),
        "ntl_lit_share_gt0": float(np.mean(pos > 0)),
        "ntl_lit_share_gt5": float(np.mean(pos > 5)),
        "ntl_lit_share_gt10": float(np.mean(pos > 10)),
        "ntl_lit_share_gt50": float(np.mean(pos > 50)),
        "ntl_lit_share_gt100": float(np.mean(pos > 100)),
        "ntl_pos_sum": pos_sum,
        "ntl_pos_sum_per_km2": float(pos_sum / area_km2) if np.isfinite(area_km2) and area_km2 > 0 else float("nan"),
        "ntl_top10_pos_share": float(np.sum(sorted_pos[-top_n:]) / pos_sum) if pos_sum > 0 else 0.0,
    }


def main() -> int:
    gpd = _require("geopandas")

    ap = argparse.ArgumentParser(
        description="Aggregate annual VIIRS nighttime-light raster to PUMA-level external-view features."
    )
    ap.add_argument(
        "--raster_tif",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/jinlin/data/geoexplicit_data/nighttime_light/viirs_openearthmonitor_2021_v0/"
            "rasters/viirs_ntl_openearthmonitor_2021_500m_epsg4326.tif"
        ),
    )
    ap.add_argument(
        "--puma_shp",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/jinlin/projects/Synthetic_City/data/geo_cache/"
            "cb_2020_us_puma20_500k/cb_2020_us_puma20_500k.shp"
        ),
    )
    ap.add_argument("--statefps", default="", help="Optional comma-separated state FIPS subset.")
    ap.add_argument("--out_csv", type=pathlib.Path, required=True)
    ap.add_argument("--out_metadata_json", type=pathlib.Path, default=None)
    ap.add_argument("--all_touched", action="store_true", help="Include pixels touched by PUMA boundaries.")
    args = ap.parse_args()

    raster_tif = args.raster_tif.expanduser().resolve()
    puma_shp = args.puma_shp.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_meta = (
        args.out_metadata_json.expanduser().resolve()
        if args.out_metadata_json is not None
        else out_csv.with_suffix(".metadata.json")
    )
    statefps = [_canon_statefp(x.strip()) for x in str(args.statefps).split(",") if x.strip()]

    pumas = gpd.read_file(puma_shp)
    state_col = _pick_col(list(pumas.columns), ("STATEFP20", "STATEFP", "STATEFP10"))
    puma_col = _pick_col(list(pumas.columns), ("PUMACE20", "PUMA", "PUMACE10"))
    geoid_col = _pick_col(list(pumas.columns), ("GEOID20", "GEOID", "GEOID10"))
    if state_col is None or puma_col is None:
        raise SystemExit(f"Cannot find state/PUMA columns in {puma_shp}: {list(pumas.columns)}")
    pumas["statefp"] = pumas[state_col].map(_canon_statefp)
    pumas["puma5"] = pumas[puma_col].map(_canon_puma5)
    if statefps:
        pumas = pumas[pumas["statefp"].isin(statefps)].copy()
    if pumas.empty:
        raise SystemExit(f"No PUMA rows selected from {puma_shp}")
    pumas["puma_uid"] = pumas["statefp"] + pumas["puma5"]
    pumas["geoid"] = pumas[geoid_col].astype(str) if geoid_col is not None else pumas["puma_uid"]
    if pumas.crs is None:
        pumas = pumas.set_crs(4269, allow_override=True)
    pumas_ll = pumas.to_crs(4326).sort_values(["statefp", "puma5"]).reset_index(drop=True)
    pumas_area = pumas.to_crs(5070).copy()
    area_km2 = dict(zip(pumas["puma_uid"], (pumas_area.geometry.area / 1_000_000.0).astype(float)))

    rows: list[dict[str, Any]] = []
    with rasterio.open(raster_tif) as src:
        raster_bounds = src.bounds
        for i, r in enumerate(pumas_ll.itertuples(index=False), start=1):
            geom = r.geometry
            minx, miny, maxx, maxy = geom.bounds
            if maxx < raster_bounds.left or minx > raster_bounds.right or maxy < raster_bounds.bottom or miny > raster_bounds.top:
                values = np.array([], dtype=np.float64)
            else:
                win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                win = win.round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                arr = src.read(1, window=win, masked=True)
                transform = src.window_transform(win)
                inside = geometry_mask(
                    [mapping(geom)],
                    out_shape=arr.shape,
                    transform=transform,
                    invert=True,
                    all_touched=bool(args.all_touched),
                )
                valid = inside & ~np.ma.getmaskarray(arr)
                if src.nodata is not None:
                    valid &= np.asarray(arr) != src.nodata
                values = np.asarray(arr, dtype=np.float64)[valid]
            puma_uid = str(r.puma_uid)
            row = {
                "puma_uid": puma_uid,
                "statefp": str(r.statefp),
                "puma5": str(r.puma5),
                "geoid": str(r.geoid),
                "puma_area_km2": float(area_km2.get(puma_uid, float("nan"))),
            }
            row.update(_safe_stats(values, row["puma_area_km2"]))
            rows.append(row)
            if i % 250 == 0 or i == len(pumas_ll):
                print(f"[progress] {i}/{len(pumas_ll)} PUMAs", flush=True)

        meta = {
            "dataset": "PUMA-level VIIRS nighttime-light features",
            "raster_tif": str(raster_tif),
            "puma_shp": str(puma_shp),
            "statefps": statefps,
            "out_csv": str(out_csv),
            "out_metadata_json": str(out_meta),
            "n_pumas": int(len(rows)),
            "all_touched": bool(args.all_touched),
            "raster_profile": {
                "crs": str(src.crs),
                "width": int(src.width),
                "height": int(src.height),
                "bounds": list(map(float, src.bounds)),
                "nodata": None if src.nodata is None else float(src.nodata),
                "dtype": str(src.dtypes[0]),
                "res": list(map(float, src.res)),
            },
            "feature_columns": [c for c in rows[0].keys() if c not in {"puma_uid", "statefp", "puma5", "geoid"}] if rows else [],
            "notes": [
                "Values are from Open-Earth-Monitor annual VIIRS nighttime lights, scaled to an intensity index.",
                "Negative raster values are retained in raw statistics but clipped to zero for *_pos features.",
                "PUMA areas are computed in EPSG:5070.",
            ],
        }

    out = pd.DataFrame(rows).sort_values(["statefp", "puma5"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    _write_json(out_meta, meta)
    print(f"[ok] wrote {out_csv} rows={out.shape[0]} cols={out.shape[1]}")
    print(f"[ok] wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
