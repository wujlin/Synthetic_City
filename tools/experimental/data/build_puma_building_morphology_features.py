#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import box


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {pkg}. Install it in the WSA dpl environment."
        ) from e


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canon_statefp(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(int(float(x))).zfill(2) if str(x).replace(".", "", 1).isdigit() else str(x).zfill(2)


def _canon_puma5(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(int(float(x))).zfill(5) if str(x).replace(".", "", 1).isdigit() else str(x).zfill(5)


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _hist_quantile(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    total = float(np.sum(hist))
    if total <= 0:
        return float("nan")
    target = float(q) * total
    csum = np.cumsum(hist.astype(float))
    idx = int(np.searchsorted(csum, target, side="left"))
    idx = max(0, min(idx, len(edges) - 2))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den and den > 0 else float("nan")


def _tile_slug(path: object) -> str:
    name = pathlib.Path(str(path)).stem
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


@dataclass
class Accumulator:
    n: int = 0
    n_source_osm: int = 0
    n_source_other: int = 0
    height_sum: float = 0.0
    height_sumsq: float = 0.0
    height_max: float = float("nan")
    area_sum: float = 0.0
    area_sumsq: float = 0.0
    area_max: float = float("nan")
    cap_sum: float = 0.0
    cap_sumsq: float = 0.0
    cap_max: float = float("nan")
    compactness_sum: float = 0.0
    compactness_sumsq: float = 0.0
    var_sum: float = 0.0
    var_sumsq: float = 0.0
    tile_ids: set[str] = field(default_factory=set)
    height_hist: np.ndarray | None = None
    log_area_hist: np.ndarray | None = None
    log_cap_hist: np.ndarray | None = None


def _update_acc(
    acc: Accumulator,
    *,
    n: int,
    source_osm: int,
    source_other: int,
    height: np.ndarray,
    area: np.ndarray,
    cap: np.ndarray,
    compact: np.ndarray,
    var: np.ndarray,
    tile_slug: str,
    height_hist: np.ndarray,
    log_area_hist: np.ndarray,
    log_cap_hist: np.ndarray,
) -> None:
    acc.n += int(n)
    acc.n_source_osm += int(source_osm)
    acc.n_source_other += int(source_other)
    acc.height_sum += float(np.nansum(height))
    acc.height_sumsq += float(np.nansum(height * height))
    acc.height_max = float(np.nanmax([acc.height_max, np.nanmax(height) if height.size else np.nan]))
    acc.area_sum += float(np.nansum(area))
    acc.area_sumsq += float(np.nansum(area * area))
    acc.area_max = float(np.nanmax([acc.area_max, np.nanmax(area) if area.size else np.nan]))
    acc.cap_sum += float(np.nansum(cap))
    acc.cap_sumsq += float(np.nansum(cap * cap))
    acc.cap_max = float(np.nanmax([acc.cap_max, np.nanmax(cap) if cap.size else np.nan]))
    acc.compactness_sum += float(np.nansum(compact))
    acc.compactness_sumsq += float(np.nansum(compact * compact))
    acc.var_sum += float(np.nansum(var))
    acc.var_sumsq += float(np.nansum(var * var))
    acc.tile_ids.add(tile_slug)
    if acc.height_hist is None:
        acc.height_hist = height_hist.astype(np.int64)
    else:
        acc.height_hist += height_hist.astype(np.int64)
    if acc.log_area_hist is None:
        acc.log_area_hist = log_area_hist.astype(np.int64)
    else:
        acc.log_area_hist += log_area_hist.astype(np.int64)
    if acc.log_cap_hist is None:
        acc.log_cap_hist = log_cap_hist.astype(np.int64)
    else:
        acc.log_cap_hist += log_cap_hist.astype(np.int64)


def _select_manifest_tiles(manifest: pd.DataFrame, puma_ll) -> pd.DataFrame:
    sindex = puma_ll.sindex
    keep: list[int] = []
    for idx, row in manifest.iterrows():
        tile_box = box(float(row["minLon"]), float(row["minLat"]), float(row["maxLon"]), float(row["maxLat"]))
        try:
            hits = sindex.query(tile_box, predicate="intersects")
        except TypeError:
            hits = sindex.query(tile_box)
        if len(hits) == 0:
            continue
        # Some spatial-index backends only use bbox tests; confirm exact intersection.
        if bool(puma_ll.iloc[hits].intersects(tile_box).any()):
            keep.append(idx)
    return manifest.loc[keep].copy()


def main() -> int:
    gpd = _require("geopandas")

    ap = argparse.ArgumentParser(
        description="Aggregate building-level ShadowMap/LoD1 GeoParquet assets to PUMA-level morphology features."
    )
    ap.add_argument(
        "--manifest_csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "/mnt/data_hdd/wellspace_v2/shadowmap/buildings_geoparquet/"
            "deg05_missing3121_20260424T031000Z/export_manifest_deg05_full_required3698.csv"
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
    ap.add_argument("--statefps", default="26,12,48,55", help="Comma-separated state FIPS codes.")
    ap.add_argument("--out_csv", type=pathlib.Path, required=True)
    ap.add_argument("--out_metadata_json", type=pathlib.Path, default=None)
    ap.add_argument("--max_tiles", type=int, default=0, help="Optional smoke-test cap; 0 means all selected tiles.")
    ap.add_argument(
        "--partition_regex",
        default="",
        help="Optional regex matched against partitionId/outputParquet for targeted smoke tests.",
    )
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    manifest_csv = args.manifest_csv.expanduser().resolve()
    puma_shp = args.puma_shp.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_meta = (
        args.out_metadata_json.expanduser().resolve()
        if args.out_metadata_json is not None
        else out_csv.with_suffix(".metadata.json")
    )
    if bool(args.skip_existing) and out_csv.exists() and out_meta.exists():
        print(f"[skip] existing outputs: {out_csv}")
        return 0

    statefps = [_canon_statefp(x.strip()) for x in str(args.statefps).split(",") if x.strip()]
    if not statefps:
        raise SystemExit("--statefps cannot be empty")

    pumas = gpd.read_file(puma_shp)
    state_col = _pick_col(list(pumas.columns), ("STATEFP20", "STATEFP", "STATEFP10"))
    puma_col = _pick_col(list(pumas.columns), ("PUMACE20", "PUMA", "PUMACE10"))
    geoid_col = _pick_col(list(pumas.columns), ("GEOID20", "GEOID", "GEOID10"))
    if state_col is None or puma_col is None:
        raise SystemExit(f"Cannot find state/PUMA columns in {puma_shp}: {list(pumas.columns)}")
    pumas = pumas[pumas[state_col].map(_canon_statefp).isin(statefps)].copy()
    if pumas.empty:
        raise SystemExit(f"No PUMA rows for statefps={statefps}")
    pumas["statefp"] = pumas[state_col].map(_canon_statefp)
    pumas["puma5"] = pumas[puma_col].map(_canon_puma5)
    pumas["puma_uid"] = pumas["statefp"] + pumas["puma5"]
    pumas["geoid"] = pumas[geoid_col].astype(str) if geoid_col is not None else pumas["puma_uid"]
    if pumas.crs is None:
        pumas = pumas.set_crs(4269, allow_override=True)
    pumas_ll = pumas.to_crs(4326)
    pumas_area = pumas.to_crs(5070).copy()
    puma_area_m2 = dict(zip(pumas_area["puma_uid"], pumas_area.geometry.area.astype(float)))
    puma_lookup = pumas_ll[["puma_uid", "statefp", "puma5", "geoid", "geometry"]].copy()

    manifest = pd.read_csv(manifest_csv)
    required_cols = {"outputParquet", "rowCount", "minLon", "minLat", "maxLon", "maxLat"}
    missing = sorted(required_cols - set(manifest.columns))
    if missing:
        raise SystemExit(f"manifest missing columns {missing}: {manifest_csv}")
    selected = _select_manifest_tiles(manifest, pumas_ll)
    selected = selected[selected["rowCount"].fillna(0).astype(float) > 0].copy()
    if str(args.partition_regex).strip():
        pat = str(args.partition_regex).strip()
        hay = selected["partitionId"].astype(str) + " " + selected["outputParquet"].astype(str)
        selected = selected[hay.str.contains(pat, regex=True, na=False)].copy()
    selected = selected.sort_values(["minLon", "minLat", "outputParquet"]).reset_index(drop=True)
    if int(args.max_tiles) > 0:
        selected = selected.head(int(args.max_tiles)).copy()
    if selected.empty:
        raise SystemExit("No non-empty building tiles intersect selected PUMA bounds.")

    height_edges = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 45, 70, 110, 180], dtype=float)
    log_area_edges = np.linspace(0.0, 12.0, 61)
    log_cap_edges = np.linspace(0.0, 16.0, 81)
    accs: dict[str, Accumulator] = defaultdict(Accumulator)
    processed_tiles: list[dict[str, Any]] = []
    total_input_rows = 0
    total_assigned_rows = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    for tile_idx, row in enumerate(selected.itertuples(index=False), start=1):
        path = pathlib.Path(str(row.outputParquet))
        slug = _tile_slug(path)
        tile_box = box(float(row.minLon), float(row.minLat), float(row.maxLon), float(row.maxLat))
        tile_pumas = puma_lookup[puma_lookup.intersects(tile_box)].copy()
        if tile_pumas.empty:
            processed_tiles.append({"tile": str(path), "status": "no_puma_overlap_before_read"})
            continue
        if not path.exists():
            processed_tiles.append({"tile": str(path), "status": "missing"})
            continue
        try:
            b = gpd.read_parquet(path, columns=["source", "height", "var", "geometry"])
        except TypeError:
            b = gpd.read_parquet(path)
            keep = [c for c in ["source", "height", "var", "geometry"] if c in b.columns]
            b = b[keep].copy()
        if b.empty:
            processed_tiles.append({"tile": str(path), "status": "empty"})
            continue
        if b.crs is None:
            b = b.set_crs(4326, allow_override=True)
        else:
            b = b.to_crs(4326)
        total_input_rows += int(b.shape[0])

        # Narrow buildings by the local PUMA bbox. This keeps projection and sjoin small.
        minx, miny, maxx, maxy = tile_pumas.total_bounds
        b = b.cx[minx:maxx, miny:maxy].copy()
        if b.empty:
            processed_tiles.append({"tile": str(path), "status": "bbox_filtered_empty"})
            continue

        b_proj = b.to_crs(5070)
        area = b_proj.geometry.area.to_numpy(dtype=float)
        perimeter = b_proj.geometry.length.to_numpy(dtype=float)
        compact = np.where(perimeter > 0, 4.0 * math.pi * area / np.clip(perimeter * perimeter, 1e-12, None), np.nan)
        height = pd.to_numeric(b.get("height", np.nan), errors="coerce").to_numpy(dtype=float)
        height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
        height = np.clip(height, 0.0, None)
        cap = area * np.clip(height / 3.0, 1.0, None)
        var = pd.to_numeric(b.get("var", np.nan), errors="coerce").to_numpy(dtype=float)
        var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
        source = b.get("source", pd.Series([""] * len(b), index=b.index)).astype(str).str.lower().to_numpy()

        cent_ll = gpd.GeoDataFrame(
            {"row_id": np.arange(len(b), dtype=np.int64)},
            geometry=b_proj.geometry.centroid,
            crs=5070,
        ).to_crs(4326)
        pts = gpd.GeoDataFrame({"row_id": cent_ll["row_id"].to_numpy(dtype=np.int64)}, geometry=cent_ll.geometry, crs=4326)
        joined = gpd.sjoin(pts, tile_pumas[["puma_uid", "statefp", "puma5", "geometry"]], how="inner", predicate="within")
        if joined.empty:
            processed_tiles.append({"tile": str(path), "status": "unassigned", "rows": int(b.shape[0])})
            continue
        row_ids = joined["row_id"].to_numpy(dtype=np.int64)
        puma_ids = joined["puma_uid"].astype(str).to_numpy()
        total_assigned_rows += int(row_ids.size)

        for puma_uid in np.unique(puma_ids):
            idx = row_ids[puma_ids == puma_uid]
            if idx.size == 0:
                continue
            h = height[idx]
            a = area[idx]
            c = cap[idx]
            comp = compact[idx]
            vv = var[idx]
            src = source[idx]
            height_hist, _ = np.histogram(np.clip(h, height_edges[0], height_edges[-1]), bins=height_edges)
            log_area_hist, _ = np.histogram(np.clip(np.log1p(np.clip(a, 0.0, None)), log_area_edges[0], log_area_edges[-1]), bins=log_area_edges)
            log_cap_hist, _ = np.histogram(np.clip(np.log1p(np.clip(c, 0.0, None)), log_cap_edges[0], log_cap_edges[-1]), bins=log_cap_edges)
            _update_acc(
                accs[puma_uid],
                n=int(idx.size),
                source_osm=int(np.sum(src == "osm")),
                source_other=int(np.sum(src != "osm")),
                height=h,
                area=a,
                cap=c,
                compact=comp,
                var=vv,
                tile_slug=slug,
                height_hist=height_hist,
                log_area_hist=log_area_hist,
                log_cap_hist=log_cap_hist,
            )

        processed_tiles.append(
            {
                "tile": str(path),
                "status": "ok",
                "input_rows": int(b.shape[0]),
                "assigned_rows": int(row_ids.size),
                "puma_count": int(len(np.unique(puma_ids))),
            }
        )
        if tile_idx % 20 == 0 or tile_idx == len(selected):
            print(
                f"[progress] {tile_idx}/{len(selected)} tiles | "
                f"assigned={total_assigned_rows:,} | pumas={len(accs)}",
                flush=True,
            )

    key_df = pumas_ll[["puma_uid", "statefp", "puma5", "geoid"]].drop_duplicates("puma_uid")
    rows: list[dict[str, Any]] = []
    for r in key_df.itertuples(index=False):
        puma_uid = str(r.puma_uid)
        acc = accs.get(puma_uid, Accumulator())
        n = int(acc.n)
        puma_area = float(puma_area_m2.get(puma_uid, float("nan")))
        area_km2 = puma_area / 1_000_000.0 if np.isfinite(puma_area) else float("nan")
        height_mean = _safe_div(acc.height_sum, n)
        area_mean = _safe_div(acc.area_sum, n)
        cap_mean = _safe_div(acc.cap_sum, n)
        compact_mean = _safe_div(acc.compactness_sum, n)
        var_mean = _safe_div(acc.var_sum, n)
        rows.append(
            {
                "puma_uid": puma_uid,
                "statefp": str(r.statefp),
                "puma5": str(r.puma5),
                "geoid": str(r.geoid),
                "puma_area_m2": puma_area,
                "puma_area_km2": area_km2,
                "bldg_count": n,
                "bldg_count_per_km2": _safe_div(n, area_km2),
                "bldg_source_osm_share": _safe_div(acc.n_source_osm, n),
                "bldg_source_other_share": _safe_div(acc.n_source_other, n),
                "footprint_area_sum_m2": acc.area_sum,
                "footprint_area_share": _safe_div(acc.area_sum, puma_area),
                "footprint_area_mean_m2": area_mean,
                "footprint_area_std_m2": math.sqrt(max(_safe_div(acc.area_sumsq, n) - area_mean * area_mean, 0.0)) if n else float("nan"),
                "footprint_area_max_m2": acc.area_max,
                "footprint_log1p_area_p50": math.expm1(_hist_quantile(acc.log_area_hist, log_area_edges, 0.50)) if acc.log_area_hist is not None else float("nan"),
                "footprint_log1p_area_p90": math.expm1(_hist_quantile(acc.log_area_hist, log_area_edges, 0.90)) if acc.log_area_hist is not None else float("nan"),
                "height_mean_m": height_mean,
                "height_std_m": math.sqrt(max(_safe_div(acc.height_sumsq, n) - height_mean * height_mean, 0.0)) if n else float("nan"),
                "height_max_m": acc.height_max,
                "height_p50_m": _hist_quantile(acc.height_hist, height_edges, 0.50) if acc.height_hist is not None else float("nan"),
                "height_p90_m": _hist_quantile(acc.height_hist, height_edges, 0.90) if acc.height_hist is not None else float("nan"),
                "cap_proxy_sum": acc.cap_sum,
                "cap_proxy_per_km2": _safe_div(acc.cap_sum, area_km2),
                "cap_proxy_mean": cap_mean,
                "cap_proxy_std": math.sqrt(max(_safe_div(acc.cap_sumsq, n) - cap_mean * cap_mean, 0.0)) if n else float("nan"),
                "cap_proxy_max": acc.cap_max,
                "cap_proxy_log1p_p50": math.expm1(_hist_quantile(acc.log_cap_hist, log_cap_edges, 0.50)) if acc.log_cap_hist is not None else float("nan"),
                "cap_proxy_log1p_p90": math.expm1(_hist_quantile(acc.log_cap_hist, log_cap_edges, 0.90)) if acc.log_cap_hist is not None else float("nan"),
                "compactness_mean": compact_mean,
                "compactness_std": math.sqrt(max(_safe_div(acc.compactness_sumsq, n) - compact_mean * compact_mean, 0.0)) if n else float("nan"),
                "height_var_mean": var_mean,
                "height_var_std": math.sqrt(max(_safe_div(acc.var_sumsq, n) - var_mean * var_mean, 0.0)) if n else float("nan"),
                "source_tile_count": int(len(acc.tile_ids)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["statefp", "puma5"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    meta = {
        "dataset": "PUMA-level built-environment morphology aggregated from ShadowMap LoD1 building GeoParquet",
        "manifest_csv": str(manifest_csv),
        "puma_shp": str(puma_shp),
        "statefps": statefps,
        "out_csv": str(out_csv),
        "out_metadata_json": str(out_meta),
        "selected_tile_count": int(selected.shape[0]),
        "processed_tile_status_counts": pd.Series([x["status"] for x in processed_tiles]).value_counts().to_dict(),
        "total_input_rows_after_bbox_filter": int(total_input_rows),
        "total_assigned_buildings": int(total_assigned_rows),
        "n_pumas": int(out.shape[0]),
        "n_pumas_with_buildings": int((out["bldg_count"] > 0).sum()),
        "feature_columns": [c for c in out.columns if c not in {"puma_uid", "statefp", "puma5", "geoid"}],
        "height_hist_edges": height_edges.tolist(),
        "log_area_hist_edges": log_area_edges.tolist(),
        "log_cap_hist_edges": log_cap_edges.tolist(),
        "notes": [
            "Assignment uses building centroids within 2020 PUMA polygons.",
            "Area and compactness are computed in EPSG:5070.",
            "Quantile-like features are approximate from fixed histograms.",
        ],
        "tiles": processed_tiles,
    }
    _write_json(out_meta, meta)
    print(f"[ok] wrote {out_csv} rows={out.shape[0]} cols={out.shape[1]}")
    print(f"[ok] wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
